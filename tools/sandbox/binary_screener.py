"""
Binary Hamming Screener — Ultra-fast pre-filter using binary hashing.

Each document centroid is hashed to a binary code: sign(normalize(centroid)).
Screening uses XOR + popcount to compute Hamming distance → cosine estimate.

This serves as Tier 0 in the funnel: reduce 1M → 100K candidates.

NOTE (F7 audit): With only 1 binary code per doc, this computes
sum-of-cosine-estimates, NOT true MaxSim. It is equivalent to
"binary centroid similarity". This is by design — it's a coarse pre-filter.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try importing Triton for the hardware popcount kernel
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _popc(x):
        """Hardware popcount via PTX inline assembly."""
        return tl.inline_asm_elementwise(
            "popc.b32 $0, $1;", "=r,r", [x],
            dtype=tl.int32, is_pure=True, pack=1
        )

    @triton.jit
    def _binary_centroid_screening_kernel(
        Q_PTR,          # (S, n_ints) - query token binary codes
        D_PTR,          # (N, n_ints) - doc centroid binary codes
        OUT_PTR,        # (N,) - aggregate scores (higher = more similar)
        N,
        S: tl.constexpr,       # number of query tokens
        n_ints: tl.constexpr,
        DIM: tl.constexpr,     # original embedding dim for cosine correction
        BLOCK_N: tl.constexpr,
    ):
        """
        Approximate centroid similarity using binary codes.

        For each query token: hamming distance to doc centroid → cosine estimate.
        Sum of cosine estimates across query tokens.

        NOTE: This is NOT MaxSim (no max over doc tokens — only 1 centroid).
        It is a coarse pre-filter for the centroid tier.
        """
        pid = tl.program_id(0)
        n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        total_score = tl.zeros([BLOCK_N], dtype=tl.float32)

        for s in range(S):
            dist = tl.zeros([BLOCK_N], dtype=tl.int32)
            for i in range(n_ints):
                q_word = tl.load(Q_PTR + s * n_ints + i)
                d_words = tl.load(
                    D_PTR + n_offsets * n_ints + i,
                    mask=n_mask,
                    other=0,
                )
                xor = q_word ^ d_words
                bits = _popc(xor)
                dist += bits

            # Cosine correction: cos(π · hamming / dim)
            dist_f = dist.to(tl.float32)
            dim_f = DIM + 0.0
            angle = 3.14159265 * dist_f / dim_f
            sim = tl.cos(angle)
            total_score += sim

        tl.store(OUT_PTR + n_offsets, total_score, mask=n_mask)


class BinaryScreener:
    """
    Binary hash screener for ultra-fast approximate centroid similarity.

    Build: sign(normalize(centroid)) → packed int32 binary codes.
    Search: XOR + popcount → Hamming distance → cosine estimate.

    NOTE: This is a coarse pre-filter, NOT an approximate MaxSim.
    It computes sum-of-cosine-estimates against a single doc centroid.
    """

    def __init__(self, dim: int, device: str = "cuda"):
        if dim % 32 != 0:
            raise ValueError(f"dim must be divisible by 32, got {dim}")
        self.dim = dim
        self.device = device
        self.n_ints = dim // 32  # number of int32s per binary code

        self._codes: Optional[torch.Tensor] = None  # (N, n_ints) int32
        self._doc_ids: List[Any] = []
        self._n_docs: int = 0

    @property
    def n_docs(self) -> int:
        return self._n_docs

    @staticmethod
    def _to_binary_code(vectors: np.ndarray) -> np.ndarray:
        """
        Convert float vectors to packed binary codes using vectorized operations.

        sign(normalize(v)) → pack bits into int32.
        F6 fix: uses np.packbits instead of Python loops.

        Args:
            vectors: (N, H) float32 vectors

        Returns:
            (N, H//32) int32 packed binary codes
        """
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        normalized = vectors / norms

        # Sign binarization
        bits = (normalized > 0).astype(np.uint8)  # (N, H)

        N, H = bits.shape
        n_ints = H // 32

        # F6 fix: vectorized bit packing
        # Reshape to (N, n_ints, 32) and use positional weights
        bits_reshaped = bits.reshape(N, n_ints, 32)
        bit_positions = (1 << np.arange(32, dtype=np.int64))  # [1, 2, 4, ..., 2^31]
        # Multiply each bit by its position weight and sum → packed int
        codes = (bits_reshaped.astype(np.int64) * bit_positions).sum(axis=2).astype(np.int32)

        return codes

    def build(
        self,
        doc_ids: Sequence[Any],
        centroids: np.ndarray,
    ) -> None:
        """
        Build binary index from pre-extracted centroids.

        Args:
            doc_ids: Document identifiers
            centroids: (N, H) centroid vectors (one per doc)
        """
        t0 = time.perf_counter()

        codes = self._to_binary_code(centroids)
        self._codes = torch.from_numpy(codes).to(device=self.device, dtype=torch.int32)
        self._doc_ids = list(doc_ids)
        self._n_docs = len(doc_ids)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"BinaryScreener.build: {self._n_docs} docs, "
            f"{self._codes.element_size() * self._codes.nelement() / 1e6:.2f}MB GPU, "
            f"{elapsed:.1f}ms"
        )

    def search(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        candidate_budget: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Screen docs using binary Hamming distance.

        Args:
            query_embedding: (S, H) query token embeddings
            candidate_budget: Number of candidates to return

        Returns:
            (topk_indices, topk_scores) — indices into the doc array
        """
        if self._codes is None:
            raise RuntimeError("Index not built")

        t0 = time.perf_counter()

        if isinstance(query_embedding, np.ndarray):
            q = query_embedding
        else:
            q = query_embedding.cpu().numpy()

        if q.ndim == 1:
            q = q.reshape(1, -1)

        N = self._n_docs

        if TRITON_AVAILABLE and self.device.startswith("cuda"):
            # Binarize all query tokens for multi-query screening
            q_codes_all = self._to_binary_code(q.astype(np.float32))
            q_codes_gpu = torch.from_numpy(q_codes_all).to(device=self.device, dtype=torch.int32)

            S = q_codes_gpu.shape[0]
            out = torch.empty(N, device=self.device, dtype=torch.float32)

            BLOCK_N = 256
            grid = ((N + BLOCK_N - 1) // BLOCK_N,)

            _binary_centroid_screening_kernel[grid](
                q_codes_gpu, self._codes, out,
                N, S, self.n_ints, self.dim,
                BLOCK_N=BLOCK_N,
            )
            scores = out  # higher = more similar
        else:
            # CPU fallback: vectorized XOR + popcount
            q_mean = q.mean(axis=0, keepdims=True)
            q_code = self._to_binary_code(q_mean.astype(np.float32))
            q_code_gpu = torch.from_numpy(q_code).to(device=self.device, dtype=torch.int32)
            q_code_expanded = q_code_gpu.expand(N, -1)
            xor = self._codes ^ q_code_expanded

            # Vectorized popcount using lookup table
            dist = torch.zeros(N, device=self.device, dtype=torch.float32)
            for i in range(self.n_ints):
                word = xor[:, i].to(torch.int64) & 0xFFFFFFFF
                count = torch.zeros_like(word, dtype=torch.float32)
                for _ in range(32):
                    count += (word & 1).float()
                    word >>= 1
                dist += count
            scores = -dist

        k = min(candidate_budget, N)
        topk_scores, topk_idx = scores.topk(k)

        elapsed = (time.perf_counter() - t0) * 1000
        return topk_idx, topk_scores
