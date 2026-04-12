from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .maxsim_scorer import _get_maxsim, _pad_shard_on_device, score_shards_and_topk

logger = logging.getLogger(__name__)
ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]


@dataclass(slots=True)
class ColBanditConfig:
    relaxation_eps: float = 0.01
    max_rounds: int = 8
    reveal_query_tokens_per_round: int = 2
    min_candidates_for_bandit: int = 64
    exact_survivor_cap: int = 256
    fallback_full_maxsim: bool = True


class ColBanditReranker:
    """Adaptive query-time pruning wrapper for late interaction reranking.

    This implementation is designed for benchmark integration: it keeps the
    existing MaxSim kernel as the source of truth and only uses bounds to reduce
    how many full exact scores are computed.
    """

    def __init__(self, config: ColBanditConfig | None = None) -> None:
        self.config = config or ColBanditConfig()

    def rerank_shard_chunks(
        self,
        query: torch.Tensor,
        shard_chunks: List[ShardChunk],
        k: int = 10,
        device: torch.device | None = None,
    ) -> tuple[List[int], List[float], Dict[str, float]]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_docs = sum(len(doc_ids) for _, _, doc_ids in shard_chunks)
        if total_docs < self.config.min_candidates_for_bandit:
            ids, scores = score_shards_and_topk(query, shard_chunks, k=k, device=device)
            return ids, scores, {
                "bandit_rounds": 0,
                "bandit_docs_pruned": 0,
                "bandit_docs_survived": float(total_docs),
            }

        try:
            return self._rerank(query, shard_chunks, k=k, device=device)
        except Exception:
            if not self.config.fallback_full_maxsim:
                raise
            logger.exception("Col-Bandit reranker failed; falling back to full MaxSim")
            ids, scores = score_shards_and_topk(query, shard_chunks, k=k, device=device)
            return ids, scores, {
                "bandit_rounds": -1,
                "bandit_docs_pruned": 0,
                "bandit_docs_survived": float(total_docs),
            }

    def _rerank(
        self,
        query: torch.Tensor,
        shard_chunks: List[ShardChunk],
        k: int,
        device: torch.device,
    ) -> tuple[List[int], List[float], Dict[str, float]]:
        maxsim = _get_maxsim()
        q = query.to(device, dtype=torch.float16)
        if q.dim() == 2:
            q = q.unsqueeze(0)
        q_norms = torch.linalg.norm(q.squeeze(0).float(), dim=-1)
        token_order = torch.argsort(q_norms, descending=True).tolist()

        flat_doc_ids: List[int] = []
        flat_doc_shard_index: List[Tuple[int, int]] = []
        doc_upper = []
        doc_lower = []
        doc_max_norm = []

        for shard_idx, (flat_emb, offsets, doc_ids) in enumerate(shard_chunks):
            for local_idx, (start, end) in enumerate(offsets):
                vecs = flat_emb[start:end].float()
                mx = torch.linalg.norm(vecs, dim=-1).max().item() if vecs.numel() else 0.0
                flat_doc_ids.append(doc_ids[local_idx])
                flat_doc_shard_index.append((shard_idx, local_idx))
                doc_max_norm.append(mx)
                doc_lower.append(0.0)
                doc_upper.append(float(torch.sum(q_norms).item() * mx))

        lower = torch.tensor(doc_lower, dtype=torch.float32)
        upper = torch.tensor(doc_upper, dtype=torch.float32)
        doc_max_norm_t = torch.tensor(doc_max_norm, dtype=torch.float32)
        active = torch.ones(len(flat_doc_ids), dtype=torch.bool)

        rounds = 0
        seen_tokens = 0
        while rounds < self.config.max_rounds and seen_tokens < len(token_order):
            block = token_order[seen_tokens:seen_tokens + self.config.reveal_query_tokens_per_round]
            if not block:
                break
            q_sub = q[:, block, :]
            remaining_norm = float(torch.sum(q_norms[token_order[seen_tokens + len(block):]]).item())

            # Score the revealed query-token block shard by shard.
            global_doc_base = 0
            for flat_emb, offsets, doc_ids in shard_chunks:
                n_docs = len(doc_ids)
                if n_docs == 0:
                    continue
                mask_slice = active[global_doc_base:global_doc_base + n_docs]
                if not bool(mask_slice.any()):
                    global_doc_base += n_docs
                    continue
                doc_emb, doc_mask = _pad_shard_on_device(flat_emb, offsets, device)
                contrib = maxsim(
                    queries_embeddings=q_sub,
                    documents_embeddings=doc_emb,
                    documents_mask=doc_mask,
                ).squeeze(0).float().cpu()
                lower[global_doc_base:global_doc_base + n_docs] += contrib
                upper[global_doc_base:global_doc_base + n_docs] = (
                    lower[global_doc_base:global_doc_base + n_docs] + remaining_norm * doc_max_norm_t[global_doc_base:global_doc_base + n_docs]
                )
                global_doc_base += n_docs

            rounds += 1
            seen_tokens += len(block)
            active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if active_indices.numel() <= k:
                break
            active_lower = lower[active_indices]
            kth_lower = float(torch.topk(active_lower, min(k, active_lower.numel())).values[-1].item())
            prune_mask = upper < (kth_lower - self.config.relaxation_eps)
            active &= ~prune_mask
            if int(active.sum().item()) <= self.config.exact_survivor_cap:
                break

        survivors = [flat_doc_ids[i] for i in torch.nonzero(active, as_tuple=False).squeeze(-1).tolist()]
        if not survivors:
            ids, scores = score_shards_and_topk(query, shard_chunks, k=k, device=device)
            return ids, scores, {
                "bandit_rounds": float(rounds),
                "bandit_docs_pruned": float(len(flat_doc_ids)),
                "bandit_docs_survived": 0.0,
            }

        docs_by_shard: Dict[int, set[int]] = {}
        for idx in torch.nonzero(active, as_tuple=False).squeeze(-1).tolist():
            shard_idx, local_idx = flat_doc_shard_index[idx]
            docs_by_shard.setdefault(shard_idx, set()).add(local_idx)

        survivor_chunks: List[ShardChunk] = []
        for shard_idx, (flat_emb, offsets, doc_ids) in enumerate(shard_chunks):
            keep_local = sorted(docs_by_shard.get(shard_idx, set()))
            if not keep_local:
                continue
            new_offsets: List[Tuple[int, int]] = []
            new_ids: List[int] = []
            pieces: List[torch.Tensor] = []
            pos = 0
            for lid in keep_local:
                start, end = offsets[lid]
                piece = flat_emb[start:end]
                pieces.append(piece)
                new_offsets.append((pos, pos + piece.shape[0]))
                new_ids.append(doc_ids[lid])
                pos += piece.shape[0]
            survivor_chunks.append((torch.cat(pieces, dim=0), new_offsets, new_ids))

        ids, scores = score_shards_and_topk(query, survivor_chunks, k=k, device=device)
        return ids, scores, {
            "bandit_rounds": float(rounds),
            "bandit_docs_pruned": float(len(flat_doc_ids) - len(survivors)),
            "bandit_docs_survived": float(len(survivors)),
        }
