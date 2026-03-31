"""
GPU-Resident Centroid Screener for Ultra-Fast MaxSim Candidate Generation.

Maintains a flat (N_docs, K, H) FP16 tensor on GPU with K centroid vectors
per document. Screening is a single batched matmul — no HNSW, no Python loops.

Two modes:
  - Single centroid (K=1): Fastest, ~0.13ms for 100K docs
  - Multi-centroid MaxSim (K=4): Better recall, ~0.9ms for 100K docs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ScreenerProfile:
    """Timing and quality profile from a screening call."""
    mode: str = ""
    n_docs: int = 0
    n_centroids_per_doc: int = 0
    candidate_budget: int = 0
    elapsed_ms: float = 0.0
    tier: str = ""


class CentroidScreener:
    """
    GPU-resident centroid index for ultra-fast approximate MaxSim screening.

    Build: Extract K centroids per document (mean + coverage medoids).
    Search: Batched matmul to compute approximate MaxSim scores.

    Memory: ~N_docs * K * H * 2 bytes (FP16). For 100K docs, K=4, H=128: ~100MB.
    """

    def __init__(
        self,
        dim: int,
        max_centroids_per_doc: int = 4,
        device: str = "cuda",
    ):
        self.dim = dim
        self.max_centroids_per_doc = max_centroids_per_doc
        self.device = device

        # GPU-resident storage
        self._centroids: Optional[torch.Tensor] = None  # (N, K, H) FP16
        self._centroid_mask: Optional[torch.Tensor] = None  # (N, K) bool
        self._doc_ids: List[Any] = []
        self._n_docs: int = 0
        self.last_profile: ScreenerProfile = ScreenerProfile()

    @property
    def n_docs(self) -> int:
        return self._n_docs

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

    @classmethod
    def extract_centroids(
        cls,
        embedding: np.ndarray,
        max_centroids: int,
    ) -> np.ndarray:
        """
        Extract representative centroid vectors from a multi-vector embedding.

        Uses mean + farthest-point-sampling for coverage medoids.
        Matches the logic in PrototypeScreeningIndex.extract_prototypes().

        Args:
            embedding: (T, H) token embeddings for one document
            max_centroids: Maximum number of centroids to extract

        Returns:
            (K, H) centroid vectors, L2 normalized
        """
        if embedding.ndim != 2:
            raise ValueError(f"Expected 2D, got {embedding.shape}")

        matrix = cls._normalize_rows(embedding.astype(np.float32))
        mean = cls._normalize_rows(matrix.mean(axis=0, keepdims=True))[0]

        if max_centroids <= 1 or matrix.shape[0] == 1:
            return mean.reshape(1, -1)

        target = min(max_centroids, matrix.shape[0] + 1)
        selected_indices: List[int] = []
        used = np.zeros(matrix.shape[0], dtype=bool)

        # Start with token farthest from mean
        sim_to_mean = matrix @ mean
        first_idx = int(np.argmin(sim_to_mean))
        selected_indices.append(first_idx)
        used[first_idx] = True

        # Farthest-point sampling
        while len(selected_indices) < target - 1:
            selected = matrix[selected_indices]
            similarities = matrix @ selected.T
            closest = similarities.max(axis=1)
            closest[used] = 1.0
            next_idx = int(np.argmin(closest))
            if used[next_idx]:
                break
            selected_indices.append(next_idx)
            used[next_idx] = True

        coverage = matrix[selected_indices] if selected_indices else np.empty((0, matrix.shape[1]), dtype=np.float32)
        result = np.concatenate([mean.reshape(1, -1), coverage], axis=0)
        return result.astype(np.float32)

    def build(
        self,
        doc_ids: Sequence[Any],
        embeddings: Sequence[np.ndarray],
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Build the GPU-resident centroid index.

        Args:
            doc_ids: Document identifiers
            embeddings: List of (T_i, H) arrays or a single (N, T, H) array
            lengths: Optional actual token counts per doc
        """
        t0 = time.perf_counter()

        # Handle both list and single-array inputs
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 3:
            doc_list = [embeddings[i] for i in range(embeddings.shape[0])]
        elif isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
            doc_list = [embeddings[i].cpu().numpy() for i in range(embeddings.shape[0])]
        else:
            doc_list = [np.asarray(e, dtype=np.float32) for e in embeddings]

        if lengths is not None:
            doc_list = [
                d[:max(1, min(int(l), d.shape[0]))]
                for d, l in zip(doc_list, lengths)
            ]

        n = len(doc_list)
        K = self.max_centroids_per_doc
        H = self.dim

        # Extract centroids for all docs
        all_centroids = np.zeros((n, K, H), dtype=np.float32)
        all_mask = np.zeros((n, K), dtype=np.float32)

        for i, doc in enumerate(doc_list):
            centroids = self.extract_centroids(doc, max_centroids=K)
            k_actual = min(centroids.shape[0], K)
            all_centroids[i, :k_actual] = centroids[:k_actual]
            all_mask[i, :k_actual] = 1.0

        # Move to GPU as FP16
        self._centroids = torch.from_numpy(all_centroids).to(
            device=self.device, dtype=torch.float16
        )
        self._centroid_mask = torch.from_numpy(all_mask).to(
            device=self.device, dtype=torch.float16
        )
        self._doc_ids = list(doc_ids)
        self._n_docs = n

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"CentroidScreener.build: {n} docs, {K} centroids/doc, "
            f"{elapsed:.1f}ms, "
            f"{self._centroids.element_size() * self._centroids.nelement() / 1e6:.1f}MB GPU"
        )

    def search_single_centroid(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        candidate_budget: int,
        allowed_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ultra-fast screening using only the mean centroid (slot 0).

        Computes: score = sum_s max(query_token_s · centroid_0)
        This is a single matmul + reduction.

        Args:
            query_embedding: (S, H) query token embeddings
            candidate_budget: Number of candidates to return
            allowed_indices: Optional (M,) tensor of allowed doc indices

        Returns:
            (topk_indices, topk_scores) — indices into the doc array
        """
        if self._centroids is None:
            raise RuntimeError("Index not built")

        t0 = time.perf_counter()

        if isinstance(query_embedding, np.ndarray):
            q = torch.from_numpy(query_embedding).to(device=self.device, dtype=torch.float16)
        else:
            q = query_embedding.to(device=self.device, dtype=torch.float16)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        # F5 fix: L2-normalize queries to match reference MaxSim behavior
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).half()

        # Use only mean centroid (slot 0): (N, H)
        centroids_0 = self._centroids[:, 0, :]

        if allowed_indices is not None:
            centroids_0 = centroids_0[allowed_indices]

        # (S, H) @ (H, N) -> (S, N)
        sim = q @ centroids_0.t()

        # For single centroid: max over doc "tokens" (only 1) is the value itself
        # Sum over query tokens
        scores = sim.sum(dim=0)  # (N,)

        k = min(candidate_budget, scores.shape[0])
        topk_scores, topk_idx = scores.topk(k)

        if allowed_indices is not None:
            topk_idx = allowed_indices[topk_idx]

        elapsed = (time.perf_counter() - t0) * 1000
        self.last_profile = ScreenerProfile(
            mode="single_centroid",
            n_docs=centroids_0.shape[0],
            n_centroids_per_doc=1,
            candidate_budget=k,
            elapsed_ms=elapsed,
            tier="tier1",
        )
        return topk_idx, topk_scores

    def search_multi_centroid(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        candidate_budget: int,
        allowed_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate MaxSim screening using all K centroids per doc.

        For each query token, finds max similarity across all K centroids,
        then sums across query tokens. This is an approximation of full MaxSim
        that uses K centroids instead of T (256→4 = 64x fewer computations).

        Args:
            query_embedding: (S, H) query token embeddings
            candidate_budget: Number of candidates to return
            allowed_indices: Optional (M,) tensor of allowed doc indices

        Returns:
            (topk_indices, topk_scores)
        """
        if self._centroids is None:
            raise RuntimeError("Index not built")

        t0 = time.perf_counter()

        if isinstance(query_embedding, np.ndarray):
            q = torch.from_numpy(query_embedding).to(device=self.device, dtype=torch.float16)
        else:
            q = query_embedding.to(device=self.device, dtype=torch.float16)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        # F5 fix: L2-normalize queries to match reference MaxSim behavior
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).half()

        S, H = q.shape

        centroids = self._centroids  # (N, K, H)
        mask = self._centroid_mask  # (N, K)

        if allowed_indices is not None:
            centroids = centroids[allowed_indices]
            mask = mask[allowed_indices]

        N, K, _ = centroids.shape

        # Flatten centroids: (N*K, H)
        flat_centroids = centroids.reshape(N * K, H)

        # (S, H) @ (H, N*K) -> (S, N*K)
        sim = q @ flat_centroids.t()

        # Reshape to (S, N, K)
        sim = sim.view(S, N, K)

        # Apply mask: set masked centroids to -inf
        mask_expanded = mask.unsqueeze(0)  # (1, N, K)
        sim = sim.masked_fill(mask_expanded < 0.5, float("-inf"))

        # MaxSim: max over centroids, sum over query tokens
        max_sim = sim.max(dim=2).values  # (S, N)
        # Replace -inf with 0 for masked positions
        max_sim = torch.nan_to_num(max_sim, neginf=0.0)
        scores = max_sim.sum(dim=0)  # (N,)

        k = min(candidate_budget, scores.shape[0])
        topk_scores, topk_idx = scores.topk(k)

        if allowed_indices is not None:
            topk_idx = allowed_indices[topk_idx]

        elapsed = (time.perf_counter() - t0) * 1000
        self.last_profile = ScreenerProfile(
            mode="multi_centroid_maxsim",
            n_docs=N,
            n_centroids_per_doc=K,
            candidate_budget=k,
            elapsed_ms=elapsed,
            tier="tier2",
        )
        return topk_idx, topk_scores

    def search(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        candidate_budget: int,
        mode: str = "multi",
        allowed_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Any], ScreenerProfile]:
        """
        Screen documents and return candidate doc_ids.

        Args:
            query_embedding: (S, H) query token embeddings
            candidate_budget: Number of candidates to return
            mode: "single" or "multi" centroid
            allowed_indices: Optional filter

        Returns:
            (candidate_doc_ids, profile)
        """
        if mode == "single":
            idx, scores = self.search_single_centroid(
                query_embedding, candidate_budget, allowed_indices
            )
        else:
            idx, scores = self.search_multi_centroid(
                query_embedding, candidate_budget, allowed_indices
            )

        # Map indices to doc_ids
        indices = idx.cpu().tolist()
        doc_ids = [self._doc_ids[i] for i in indices]
        return doc_ids, self.last_profile

    def get_doc_indices(self, doc_ids: List[Any]) -> torch.Tensor:
        """Map doc_ids to internal indices for filtering."""
        id_to_idx = {did: i for i, did in enumerate(self._doc_ids)}
        indices = [id_to_idx[did] for did in doc_ids if did in id_to_idx]
        return torch.tensor(indices, device=self.device, dtype=torch.long)
