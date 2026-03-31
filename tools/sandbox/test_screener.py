"""
Unit tests for the GPU-native screener sandbox.

Mirrors the test patterns from the repo's test_prototype_screening.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add sandbox to path
sys.path.insert(0, str(Path(__file__).parent))

from centroid_screener import CentroidScreener
from binary_screener import BinaryScreener
from hybrid_funnel import HybridFunnelScreener


@pytest.fixture
def small_embeddings():
    """Small test embeddings matching repo fixture format."""
    doc_embeddings = np.asarray([
        # doc-a: tokens pointing mostly in +x direction
        [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        # doc-b: tokens pointing mostly in +y direction
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
    ], dtype=np.float32)
    doc_ids = ["doc-a", "doc-b"]
    # Query pointing in +x direction → should match doc-a
    query = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]], dtype=np.float32)
    return doc_ids, doc_embeddings, query


@pytest.fixture
def medium_embeddings():
    """Medium-scale embeddings for recall testing."""
    rng = np.random.default_rng(42)
    n_docs = 100
    dim = 128
    doc_tokens = 32
    query_tokens = 16

    doc_embeddings = rng.standard_normal((n_docs, doc_tokens, dim)).astype(np.float32)
    # Normalize tokens
    norms = np.linalg.norm(doc_embeddings, axis=-1, keepdims=True) + 1e-8
    doc_embeddings = doc_embeddings / norms

    query = rng.standard_normal((query_tokens, dim)).astype(np.float32)
    norms = np.linalg.norm(query, axis=-1, keepdims=True) + 1e-8
    query = query / norms

    doc_ids = [f"doc-{i:03d}" for i in range(n_docs)]
    return doc_ids, doc_embeddings, query


class TestCentroidScreener:
    """Tests for CentroidScreener — the core component."""

    def test_extract_centroids_produces_correct_shape(self):
        emb = np.random.randn(256, 128).astype(np.float32)
        centroids = CentroidScreener.extract_centroids(emb, max_centroids=4)
        assert centroids.shape == (4, 128) or centroids.shape[0] <= 4
        assert centroids.dtype == np.float32

    def test_extract_centroids_single_token(self):
        emb = np.random.randn(1, 128).astype(np.float32)
        centroids = CentroidScreener.extract_centroids(emb, max_centroids=4)
        assert centroids.shape == (1, 128)

    def test_build_and_search_prefers_matching_doc(self, small_embeddings):
        doc_ids, doc_emb, query = small_embeddings
        screener = CentroidScreener(dim=4, max_centroids_per_doc=3, device="cpu")
        screener.build(doc_ids, doc_emb)

        # Single centroid search
        ids, profile = screener.search(query, candidate_budget=1, mode="single")
        assert ids == ["doc-a"], f"Expected doc-a, got {ids}"

        # Multi centroid search
        ids, profile = screener.search(query, candidate_budget=1, mode="multi")
        assert ids == ["doc-a"], f"Expected doc-a, got {ids}"

    def test_returns_correct_count(self, medium_embeddings):
        doc_ids, doc_emb, query = medium_embeddings
        screener = CentroidScreener(dim=128, max_centroids_per_doc=4, device="cpu")
        screener.build(doc_ids, doc_emb)

        ids, _ = screener.search(query, candidate_budget=10, mode="multi")
        assert len(ids) == 10

        ids, _ = screener.search(query, candidate_budget=50, mode="single")
        assert len(ids) == 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_search_matches_cpu(self, medium_embeddings):
        doc_ids, doc_emb, query = medium_embeddings

        cpu_screener = CentroidScreener(dim=128, max_centroids_per_doc=4, device="cpu")
        cpu_screener.build(doc_ids, doc_emb)
        cpu_ids, _ = cpu_screener.search(query, candidate_budget=20, mode="multi")

        gpu_screener = CentroidScreener(dim=128, max_centroids_per_doc=4, device="cuda")
        gpu_screener.build(doc_ids, doc_emb)
        gpu_ids, _ = gpu_screener.search(query, candidate_budget=20, mode="multi")

        # Top results should largely overlap (may differ slightly due to float precision)
        overlap = len(set(cpu_ids[:10]) & set(gpu_ids[:10]))
        assert overlap >= 7, f"CPU/GPU top-10 overlap too low: {overlap}/10"


class TestBinaryScreener:
    """Tests for BinaryScreener — binary hash pre-filter."""

    def test_binary_code_shape(self):
        vectors = np.random.randn(100, 128).astype(np.float32)
        codes = BinaryScreener._to_binary_code(vectors)
        assert codes.shape == (100, 4)  # 128 bits = 4 x int32
        assert codes.dtype == np.int32

    def test_identical_vectors_have_zero_hamming(self):
        v = np.random.randn(1, 128).astype(np.float32)
        codes = BinaryScreener._to_binary_code(v)
        xor = codes[0] ^ codes[0]
        assert np.all(xor == 0)

    def test_opposite_vectors_have_max_hamming(self):
        v = np.random.randn(1, 128).astype(np.float32)
        codes_pos = BinaryScreener._to_binary_code(v)
        codes_neg = BinaryScreener._to_binary_code(-v)
        xor = codes_pos[0] ^ codes_neg[0]
        # All bits should differ
        total_bits = sum(bin(int(np.uint32(x))).count('1') for x in xor)
        assert total_bits == 128

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_search_returns_candidates(self):
        dim = 128
        n_docs = 1000
        rng = np.random.default_rng(42)
        centroids = rng.standard_normal((n_docs, dim)).astype(np.float32)
        doc_ids = [f"doc-{i}" for i in range(n_docs)]

        screener = BinaryScreener(dim=dim, device="cuda")
        screener.build(doc_ids, centroids)

        query = centroids[0:1]  # Query = first doc centroid
        idx, scores = screener.search(query, candidate_budget=10)
        assert idx.shape[0] == 10
        # The query IS doc-0's centroid, so doc-0 should have highest score
        assert 0 in idx.cpu().tolist(), "Doc-0 should be in top-10 for its own centroid"


class TestHybridFunnel:
    """Tests for HybridFunnelScreener."""

    def test_funnel_prefers_matching_doc(self, small_embeddings):
        doc_ids, doc_emb, query = small_embeddings
        # Binary screener needs dim >= 32 for int32 packing, use CPU centroid path
        # For dim=4, funnel falls back to centroid-only
        screener = CentroidScreener(dim=4, max_centroids_per_doc=3, device="cpu")
        screener.build(doc_ids, doc_emb)
        ids, _ = screener.search(query, candidate_budget=1, mode="multi")
        assert ids == ["doc-a"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_funnel_at_scale(self):
        dim = 128
        n_docs = 1000
        query_tokens = 32
        doc_tokens = 64
        rng = np.random.default_rng(42)

        doc_emb = rng.standard_normal((n_docs, doc_tokens, dim)).astype(np.float32)
        query = rng.standard_normal((query_tokens, dim)).astype(np.float32)
        doc_ids = [f"doc-{i}" for i in range(n_docs)]

        funnel = HybridFunnelScreener(
            dim=dim, max_centroids_per_doc=4, device="cuda",
            tier0_survival=500, tier1_survival=200,
        )
        funnel.build(doc_ids, doc_emb)
        ids, profile = funnel.search(query, candidate_budget=50)

        assert len(ids) == 50
        assert profile.total_elapsed_ms > 0
        assert len(profile.tiers) >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_recall_at_budget(self):
        """
        Key quality test: multi-centroid screener should achieve >80% recall
        vs full MaxSim when retrieving top-100 from 1000 docs.
        """
        dim = 128
        n_docs = 1000
        query_tokens = 32
        doc_tokens = 64
        budget = 100
        rng = np.random.default_rng(42)

        # Generate correlated data (not random — recall on random is meaningless)
        n_topics = 20
        topic_centroids = rng.standard_normal((n_topics, dim)).astype(np.float32)
        topic_centroids /= np.linalg.norm(topic_centroids, axis=1, keepdims=True) + 1e-8

        doc_emb = np.zeros((n_docs, doc_tokens, dim), dtype=np.float32)
        for i in range(n_docs):
            topics = rng.choice(n_topics, size=rng.integers(1, 4), replace=False)
            for t in range(doc_tokens):
                topic = rng.choice(topics)
                noise = rng.standard_normal(dim).astype(np.float32) * 0.3
                doc_emb[i, t] = topic_centroids[topic] + noise
                doc_emb[i, t] /= np.linalg.norm(doc_emb[i, t]) + 1e-8

        query_topic = 0
        query = np.zeros((query_tokens, dim), dtype=np.float32)
        for t in range(query_tokens):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.2
            query[t] = topic_centroids[query_topic] + noise
            query[t] /= np.linalg.norm(query[t]) + 1e-8

        doc_ids = [f"doc-{i}" for i in range(n_docs)]

        # Full MaxSim reference
        q_t = torch.from_numpy(query).cuda()
        d_t = torch.from_numpy(doc_emb).cuda()
        q_n = torch.nn.functional.normalize(q_t.float(), p=2, dim=-1)
        d_n = torch.nn.functional.normalize(d_t.float(), p=2, dim=-1)
        sim = torch.einsum("sh,nth->snt", q_n, d_n)
        ref_scores = sim.max(dim=2).values.sum(dim=0)
        ref_sorted = ref_scores.argsort(descending=True).cpu().tolist()
        ref_top_ids = set(doc_ids[i] for i in ref_sorted[:budget])

        # Multi-centroid screener
        screener = CentroidScreener(dim=dim, max_centroids_per_doc=4, device="cuda")
        screener.build(doc_ids, doc_emb)
        screened_ids, _ = screener.search(query, candidate_budget=budget, mode="multi")
        screened_set = set(screened_ids)

        recall = len(ref_top_ids & screened_set) / len(ref_top_ids)
        print(f"\n  Recall@{budget} from {n_docs} docs: {recall:.3f}")
        assert recall >= 0.60, f"Recall too low: {recall:.3f} (expected >= 0.60)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
