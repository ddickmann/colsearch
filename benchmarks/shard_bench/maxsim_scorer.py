"""
MaxSim scorer: thin wrapper over voyager-index Triton kernels.

Supports two paths:
- score_and_topk: legacy padded-tensor interface
- score_shards_and_topk: per-shard scoring with GPU top-k merge (fast path)
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_maxsim_fn = None


def _get_maxsim():
    global _maxsim_fn
    if _maxsim_fn is not None:
        return _maxsim_fn
    try:
        from voyager_index._internal.kernels.maxsim import fast_colbert_scores
        _maxsim_fn = fast_colbert_scores
        logger.info("Using voyager-index Triton MaxSim kernel")
    except ImportError:
        _maxsim_fn = _fallback_maxsim
        logger.warning("Triton MaxSim unavailable, using PyTorch fallback")
    return _maxsim_fn


def _fallback_maxsim(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask=None,
    documents_mask=None,
    **kwargs,
) -> torch.Tensor:
    """Pure PyTorch MaxSim for environments without Triton."""
    Q = queries_embeddings.float()
    D = documents_embeddings.float()
    sim = torch.einsum("ash,bth->abst", Q, D)
    if documents_mask is not None:
        dm = documents_mask.bool()
        sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))
    max_sim = sim.max(dim=-1).values
    if queries_mask is not None:
        qm = queries_mask.float()
        max_sim = max_sim * qm[:, None, :]
    max_sim = torch.nan_to_num(max_sim, neginf=0.0)
    return max_sim.sum(dim=-1)


def _pad_shard_on_device(
    flat_emb: torch.Tensor,
    offsets: List[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a single shard's docs on GPU.  Much faster than cross-shard CPU padding
    because (a) per-shard max_tokens is small, (b) the copy runs on device.

    If all docs have the same token count, uses zero-copy view() instead.
    """
    if not offsets:
        dim = flat_emb.shape[1]
        return (
            torch.empty(0, 1, dim, dtype=flat_emb.dtype, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    lengths = [e - s for s, e in offsets]
    n_docs = len(offsets)
    max_tok = max(lengths)
    min_tok = min(lengths)
    dim = flat_emb.shape[1]

    if flat_emb.device != device:
        flat_emb = flat_emb.to(device, non_blocking=True)

    # Fast path: all docs same length -> zero-copy view
    if min_tok == max_tok:
        padded = flat_emb[: n_docs * max_tok].view(n_docs, max_tok, dim)
        mask = torch.ones(n_docs, max_tok, dtype=torch.float32, device=device)
        return padded, mask

    padded = torch.zeros(n_docs, max_tok, dim, dtype=flat_emb.dtype, device=device)
    mask = torch.zeros(n_docs, max_tok, dtype=torch.float32, device=device)

    for i, (s, e) in enumerate(offsets):
        length = e - s
        padded[i, :length] = flat_emb[s:e]
        mask[i, :length] = 1.0

    return padded, mask


ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]


def score_shards_and_topk(
    query: torch.Tensor,
    shard_chunks: List[ShardChunk],
    k: int = 10,
    device: torch.device = None,
) -> Tuple[List[int], List[float]]:
    """
    Score per-shard and merge top-k on GPU.  Avoids cross-shard padding entirely.

    Each shard is padded independently (small max_tokens) and scored with one
    fast_colbert_scores call.  A running top-k heap merges results across shards.

    Args:
        query: (n_query_tokens, dim) or (1, n_query_tokens, dim)
        shard_chunks: list of (flat_emb, offsets, doc_ids) per shard.
            flat_emb is (total_tokens, dim) on CPU or GPU.
        k: final top-k
        device: scoring device (defaults to cuda if available)

    Returns:
        (top_k_ids, top_k_scores) merged across all shards
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maxsim = _get_maxsim()

    q = query.to(device, dtype=torch.float16)
    if q.dim() == 2:
        q = q.unsqueeze(0)

    best_scores: List[torch.Tensor] = []
    best_ids: List[List[int]] = []

    for flat_emb, offsets, doc_ids in shard_chunks:
        if not doc_ids:
            continue

        # Pad this shard on device (small max_tokens, fast)
        doc_emb, doc_mask = _pad_shard_on_device(flat_emb, offsets, device)

        scores = maxsim(
            queries_embeddings=q,
            documents_embeddings=doc_emb,
            documents_mask=doc_mask,
        ).squeeze(0)  # (n_docs,)

        shard_k = min(k, len(doc_ids))
        top_sc, top_idx = scores.topk(shard_k)
        best_scores.append(top_sc)
        best_ids.append([doc_ids[i] for i in top_idx.cpu().tolist()])

    if not best_scores:
        return [], []

    # Merge across shards on GPU
    all_scores = torch.cat(best_scores)
    all_ids_flat: List[int] = []
    for id_list in best_ids:
        all_ids_flat.extend(id_list)

    final_k = min(k, len(all_ids_flat))
    top_sc, top_idx = all_scores.topk(final_k)

    result_ids = [all_ids_flat[i] for i in top_idx.cpu().tolist()]
    result_scores = top_sc.cpu().tolist()
    return result_ids, result_scores


def score_and_topk(
    query: torch.Tensor,
    doc_embeddings: torch.Tensor,
    doc_mask: torch.Tensor,
    doc_ids: List[int],
    k: int = 10,
    use_quantization: bool = False,
    quantization_mode: str = "int8",
) -> Tuple[List[int], List[float]]:
    """
    Score documents against query, return top-k IDs and scores.
    Legacy interface — prefers score_shards_and_topk for new code.
    """
    maxsim = _get_maxsim()

    if query.dim() == 2:
        query = query.unsqueeze(0)

    n_docs = doc_embeddings.shape[0]
    if n_docs == 0:
        return [], []

    scores = maxsim(
        queries_embeddings=query,
        documents_embeddings=doc_embeddings,
        documents_mask=doc_mask,
        use_quantization=use_quantization,
        quantization_mode=quantization_mode,
    ).squeeze(0)

    actual_k = min(k, n_docs)
    top_scores, top_indices = scores.topk(actual_k)

    top_ids = [doc_ids[i] for i in top_indices.cpu().tolist()]
    top_sc = top_scores.cpu().tolist()
    return top_ids, top_sc


def brute_force_maxsim(
    query: torch.Tensor,
    all_doc_vecs: list,
    doc_ids: List[int],
    dim: int,
    k: int = 100,
    device: str = "cuda",
    batch_size: int = 2000,
) -> Tuple[List[int], List[float]]:
    """
    Brute-force MaxSim over an entire corpus for ground-truth computation.

    Args:
        query: (n_tokens, dim) numpy or tensor
        all_doc_vecs: list of per-doc numpy arrays, each (n_tokens_i, dim)
        doc_ids: external IDs
        dim: embedding dimension
        k: top-k to return
        device: scoring device
        batch_size: docs per GPU batch

    Returns:
        (top_k_ids, top_k_scores) sorted best-first
    """
    import numpy as np
    maxsim = _get_maxsim()

    if isinstance(query, np.ndarray):
        query = torch.from_numpy(query)
    q = query.float().unsqueeze(0).to(device)

    all_scores = []
    for start in range(0, len(doc_ids), batch_size):
        end = min(start + batch_size, len(doc_ids))
        batch = all_doc_vecs[start:end]
        max_tok = max(v.shape[0] for v in batch)

        D = np.zeros((end - start, max_tok, dim), dtype=np.float32)
        M = np.zeros((end - start, max_tok), dtype=np.float32)
        for i, v in enumerate(batch):
            fv = v.astype(np.float32) if v.dtype != np.float32 else v
            D[i, : fv.shape[0]] = fv
            M[i, : fv.shape[0]] = 1.0

        scores = maxsim(
            q,
            torch.from_numpy(D).to(device),
            documents_mask=torch.from_numpy(M).to(device),
        ).squeeze(0)
        all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores)
    topk = all_scores.topk(min(k, len(doc_ids)))
    top_ids = [doc_ids[j] for j in topk.indices.tolist()]
    top_sc = topk.values.tolist()
    return top_ids, top_sc
