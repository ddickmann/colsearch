"""
Hybrid Funnel Screener — Multi-tier screening pipeline.

Combines binary screening, single-centroid, and multi-centroid MaxSim
into a funnel that progressively refines candidates.

Tier 0 (Binary):          1M docs → 10K     (~0.1ms)
Tier 1 (1-centroid):      10K → 1K          (~0.1ms)
Tier 2 (4-centroid MaxSim): 1K → budget     (~0.3ms)
Total: ~0.5ms for 1M candidate screening
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from centroid_screener import CentroidScreener, ScreenerProfile
from binary_screener import BinaryScreener

logger = logging.getLogger(__name__)


@dataclass
class FunnelProfile:
    """Profile for the full funnel pipeline."""
    total_elapsed_ms: float = 0.0
    n_docs_input: int = 0
    candidate_budget: int = 0
    tiers: List[Dict[str, Any]] = field(default_factory=list)


class HybridFunnelScreener:
    """
    Multi-tier screening funnel combining binary and centroid approaches.

    Architecture:
    - Tier 0 (Binary Hamming): Sign-hash of mean centroid, XOR+popcount
    - Tier 1 (Single centroid): FP16 matmul with mean centroid only
    - Tier 2 (Multi-centroid MaxSim): FP16 matmul with K centroids, MaxSim aggregation

    The funnel fans-in: each tier reduces candidates by ~10x.
    Tiers are skipped when the input count is already small enough.
    """

    def __init__(
        self,
        dim: int,
        max_centroids_per_doc: int = 4,
        device: str = "cuda",
        # Funnel ratios: how many candidates survive each tier
        tier0_survival: int = 5000,    # binary → top 5K
        tier1_survival: int = 500,     # single centroid → top 500
        # tier2 returns candidate_budget
        # F9 fix: tiers activate when input > survival * 1.5 (was *2)
        tier_activation_ratio: float = 1.5,
    ):
        self.dim = dim
        self.device = device
        self.tier0_survival = tier0_survival
        self.tier1_survival = tier1_survival
        self.tier_activation_ratio = tier_activation_ratio

        self.centroid_screener = CentroidScreener(
            dim=dim,
            max_centroids_per_doc=max_centroids_per_doc,
            device=device,
        )
        self.binary_screener = BinaryScreener(dim=dim, device=device)

        self._doc_ids: List[Any] = []
        self._n_docs: int = 0
        self.last_profile: FunnelProfile = FunnelProfile()

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def build(
        self,
        doc_ids: Sequence[Any],
        embeddings: Sequence[np.ndarray],
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Build all tiers of the funnel index.

        Args:
            doc_ids: Document identifiers
            embeddings: List of (T_i, H) arrays or (N, T, H) array
            lengths: Optional actual token counts per doc
        """
        # Build centroid index (extracts centroids internally)
        self.centroid_screener.build(doc_ids, embeddings, lengths)

        # Build binary index from mean centroids (slot 0)
        centroids_gpu = self.centroid_screener._centroids[:, 0, :]  # (N, H)
        centroids_np = centroids_gpu.cpu().numpy().astype(np.float32)
        self.binary_screener.build(doc_ids, centroids_np)

        self._doc_ids = list(doc_ids)
        self._n_docs = len(doc_ids)

    def search(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        candidate_budget: int,
    ) -> Tuple[List[Any], FunnelProfile]:
        """
        Run the multi-tier screening funnel.

        Automatically skips unnecessary tiers when the doc count is small.

        Args:
            query_embedding: (S, H) query token embeddings
            candidate_budget: Final number of candidates to return

        Returns:
            (candidate_doc_ids, profile)
        """
        t0_total = time.perf_counter()
        profile = FunnelProfile(n_docs_input=self._n_docs, candidate_budget=candidate_budget)

        N = self._n_docs
        allowed_indices = None  # Initially all docs

        # --- Tier 0: Binary Hamming pre-filter ---
        # F9 fix: use configurable activation ratio
        if N > self.tier0_survival * self.tier_activation_ratio:
            t0 = time.perf_counter()
            tier0_budget = min(self.tier0_survival, N)
            tier0_idx, tier0_scores = self.binary_screener.search(
                query_embedding, tier0_budget
            )
            allowed_indices = tier0_idx
            t1 = time.perf_counter()
            profile.tiers.append({
                "tier": "binary_centroid",
                "input_count": N,
                "output_count": tier0_idx.shape[0],
                "elapsed_ms": (t1 - t0) * 1000,
            })
        else:
            profile.tiers.append({"tier": "binary_centroid", "skipped": True, "reason": f"N={N} < {self.tier0_survival}*{self.tier_activation_ratio}"})

        # --- Tier 1: Single centroid screening ---
        current_count = allowed_indices.shape[0] if allowed_indices is not None else N
        if current_count > self.tier1_survival * self.tier_activation_ratio:
            t0 = time.perf_counter()
            tier1_budget = min(self.tier1_survival, current_count)
            tier1_idx, tier1_scores = self.centroid_screener.search_single_centroid(
                query_embedding, tier1_budget, allowed_indices
            )
            allowed_indices = tier1_idx
            t1 = time.perf_counter()
            profile.tiers.append({
                "tier": "single_centroid",
                "input_count": current_count,
                "output_count": tier1_idx.shape[0],
                "elapsed_ms": (t1 - t0) * 1000,
            })
        else:
            profile.tiers.append({"tier": "single_centroid", "skipped": True, "reason": f"count={current_count} < threshold"})

        # --- Tier 2: Multi-centroid MaxSim ---
        t0 = time.perf_counter()
        tier2_idx, tier2_scores = self.centroid_screener.search_multi_centroid(
            query_embedding, candidate_budget, allowed_indices
        )
        t1 = time.perf_counter()
        current_count = allowed_indices.shape[0] if allowed_indices is not None else N
        profile.tiers.append({
            "tier": "multi_centroid_maxsim",
            "input_count": current_count,
            "output_count": tier2_idx.shape[0],
            "elapsed_ms": (t1 - t0) * 1000,
        })

        # Map indices to doc_ids
        indices = tier2_idx.cpu().tolist()
        doc_ids = [self._doc_ids[i] for i in indices]

        profile.total_elapsed_ms = (time.perf_counter() - t0_total) * 1000
        self.last_profile = profile
        return doc_ids, profile
