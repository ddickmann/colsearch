"""
Comprehensive benchmark harness for the GPU-native screener sandbox.

Compares screening approaches against full-precision MaxSim at various scales.
Uses the same data format as existing repo benchmarks for apple-to-apple comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add sandbox to path
sys.path.insert(0, str(Path(__file__).parent))

from centroid_screener import CentroidScreener
from binary_screener import BinaryScreener
from hybrid_funnel import HybridFunnelScreener


def generate_synthetic_data(
    n_docs: int,
    dim: int = 128,
    query_tokens: int = 32,
    doc_tokens: int = 256,
    seed: int = 42,
    n_query_topics: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic ColBERT-style embeddings with realistic diversity.

    Creates clustered data with realistic structure:
    - Documents have tokens from a mixture of 1-3 topics
    - Queries span 3-5 topics (matching real ColPali multi-aspect queries)
    - Query tokens have varying noise levels for realistic dispersion
    """
    rng = np.random.default_rng(seed)

    n_topics = min(50, n_docs // 2 + 1)

    # Generate topic centroids
    topic_centroids = rng.standard_normal((n_topics, dim)).astype(np.float32)
    topic_centroids /= np.linalg.norm(topic_centroids, axis=1, keepdims=True) + 1e-8

    # Generate document embeddings
    doc_embeddings = np.zeros((n_docs, doc_tokens, dim), dtype=np.float32)
    doc_topic_assignments = []

    for i in range(n_docs):
        # Each doc is a mixture of 1-3 topics
        n_doc_topics = rng.integers(1, min(4, n_topics + 1))
        topics = rng.choice(n_topics, size=n_doc_topics, replace=False)
        doc_topic_assignments.append(topics.tolist())

        for t in range(doc_tokens):
            topic = rng.choice(topics)
            noise = rng.standard_normal(dim).astype(np.float32) * 0.3
            token = topic_centroids[topic] + noise
            token /= np.linalg.norm(token) + 1e-8
            doc_embeddings[i, t] = token

    # F1 FIX: Generate multi-topic queries (realistic ColPali-style)
    # Real queries span 3-5 semantic aspects, not just one topic.
    if n_query_topics <= 0:
        n_query_topics = min(rng.integers(3, 6), n_topics)
    query_topics = rng.choice(n_topics, size=n_query_topics, replace=False)
    query_embeddings = np.zeros((query_tokens, dim), dtype=np.float32)

    for t in range(query_tokens):
        # Distribute tokens across query topics
        topic = query_topics[t % len(query_topics)]
        # Varying noise levels for realistic dispersion (0.15-0.35)
        noise_scale = 0.15 + rng.random() * 0.20
        noise = rng.standard_normal(dim).astype(np.float32) * noise_scale
        token = topic_centroids[topic] + noise
        token /= np.linalg.norm(token) + 1e-8
        query_embeddings[t] = token

    doc_ids = [f"doc-{i:06d}" for i in range(n_docs)]

    return query_embeddings, doc_embeddings, doc_ids


def reference_maxsim(
    query: torch.Tensor,
    documents: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact MaxSim scores (reference implementation).

    Args:
        query: (S, H) query token embeddings
        documents: (N, T, H) document token embeddings

    Returns:
        scores: (N,) MaxSim scores
    """
    # Normalize
    q = torch.nn.functional.normalize(query.float(), p=2, dim=-1)  # (S, H)
    d = torch.nn.functional.normalize(documents.float(), p=2, dim=-1)  # (N, T, H)

    # Compute similarities: (S, N, T)
    # For each query token, similarity to each doc token
    sim = torch.einsum("sh,nth->snt", q, d)  # (S, N, T)

    # MaxSim: for each query token, max over doc tokens
    max_sim = sim.max(dim=2).values  # (S, N)

    # Sum over query tokens
    scores = max_sim.sum(dim=0)  # (N,)

    return scores


def timed_call(fn, warmup: int = 1, runs: int = 5, sync_cuda: bool = True):
    """Time a function with warmup and multiple runs."""
    for _ in range(warmup):
        result = fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return result, {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "runs_ms": times,
    }


def compute_recall(
    screened_ids: List[str],
    reference_ids: List[str],
    k: int,
) -> Dict[str, float]:
    """Compute recall metrics: recall@k, top-1 agreement, MRR."""
    ref_set = set(reference_ids[:k])
    screened_set = set(screened_ids[:k])

    recall_at_k = len(ref_set & screened_set) / len(ref_set) if ref_set else 0.0
    top_1_match = 1.0 if screened_ids[0] == reference_ids[0] else 0.0

    # MRR: rank of the reference top-1 in screened results
    mrr = 0.0
    ref_top1 = reference_ids[0]
    for rank, sid in enumerate(screened_ids):
        if sid == ref_top1:
            mrr = 1.0 / (rank + 1)
            break

    return {
        "recall_at_k": recall_at_k,
        "top_1_agreement": top_1_match,
        "mrr": mrr,
    }


def benchmark_at_scale(
    n_docs: int,
    dim: int = 128,
    query_tokens: int = 32,
    doc_tokens: int = 256,
    candidate_budget: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Run comprehensive benchmark at a given scale.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_docs} docs, dim={dim}, S={query_tokens}, T={doc_tokens}")
    print(f"{'='*60}")

    query_emb, doc_emb, doc_ids = generate_synthetic_data(
        n_docs, dim, query_tokens, doc_tokens
    )

    result = {
        "n_docs": n_docs,
        "dim": dim,
        "query_tokens": query_tokens,
        "doc_tokens": doc_tokens,
        "candidate_budget": candidate_budget,
    }

    # --- Reference: Full MaxSim ---
    if n_docs <= 20000:
        print(f"  Computing reference MaxSim ({n_docs} docs)...")
        q_tensor = torch.from_numpy(query_emb).to(device)
        d_tensor = torch.from_numpy(doc_emb).to(device)

        ref_scores, ref_timing = timed_call(
            lambda: reference_maxsim(q_tensor, d_tensor),
            warmup=1, runs=3,
        )
        ref_sorted = ref_scores.argsort(descending=True).cpu().tolist()
        ref_top_ids = [doc_ids[i] for i in ref_sorted[:candidate_budget]]

        result["reference_maxsim"] = {
            "timing": ref_timing,
            "top_1_id": ref_top_ids[0],
        }
        print(f"    Reference: {ref_timing['median_ms']:.2f}ms")

        del d_tensor
        torch.cuda.empty_cache()
    else:
        print(f"  Skipping reference MaxSim (too many docs for einsum)")
        ref_top_ids = None
        result["reference_maxsim"] = {"timing": None, "skipped": True}

    # --- Strategy 1: Single centroid screener ---
    print(f"  Building CentroidScreener...")
    screener = CentroidScreener(dim=dim, max_centroids_per_doc=4, device=device)

    _, build_timing = timed_call(
        lambda: screener.build(doc_ids, doc_emb),
        warmup=0, runs=1, sync_cuda=False,
    )

    print(f"  Single centroid search...")
    single_result, single_timing = timed_call(
        lambda: screener.search(query_emb, candidate_budget, mode="single"),
        warmup=1, runs=5,
    )
    single_ids, single_profile = single_result
    result["single_centroid"] = {
        "timing": single_timing,
        "profile": {
            "mode": single_profile.mode,
            "elapsed_ms": single_profile.elapsed_ms,
        },
        "build_ms": build_timing["median_ms"],
    }
    print(f"    Single centroid: {single_timing['median_ms']:.3f}ms")

    if ref_top_ids:
        metrics = compute_recall(single_ids, ref_top_ids, candidate_budget)
        result["single_centroid"]["recall"] = metrics
        print(f"    Recall@{candidate_budget}: {metrics['recall_at_k']:.3f}, Top-1: {metrics['top_1_agreement']:.1f}")

    # --- Strategy 2: Multi-centroid MaxSim screener ---
    print(f"  Multi-centroid MaxSim search...")
    multi_result, multi_timing = timed_call(
        lambda: screener.search(query_emb, candidate_budget, mode="multi"),
        warmup=1, runs=5,
    )
    multi_ids, multi_profile = multi_result
    result["multi_centroid"] = {
        "timing": multi_timing,
        "profile": {
            "mode": multi_profile.mode,
            "elapsed_ms": multi_profile.elapsed_ms,
        },
    }
    print(f"    Multi-centroid: {multi_timing['median_ms']:.3f}ms")

    if ref_top_ids:
        metrics = compute_recall(multi_ids, ref_top_ids, candidate_budget)
        result["multi_centroid"]["recall"] = metrics
        print(f"    Recall@{candidate_budget}: {metrics['recall_at_k']:.3f}, Top-1: {metrics['top_1_agreement']:.1f}")

    # --- Strategy 3: Hybrid Funnel ---
    print(f"  Building HybridFunnelScreener...")
    funnel = HybridFunnelScreener(
        dim=dim,
        max_centroids_per_doc=4,
        device=device,
        tier0_survival=min(10000, n_docs),
        tier1_survival=min(1000, n_docs),
    )

    _, funnel_build_timing = timed_call(
        lambda: funnel.build(doc_ids, doc_emb),
        warmup=0, runs=1, sync_cuda=False,
    )

    print(f"  Hybrid funnel search...")
    funnel_result, funnel_timing = timed_call(
        lambda: funnel.search(query_emb, candidate_budget),
        warmup=1, runs=5,
    )
    funnel_ids, funnel_profile = funnel_result
    result["hybrid_funnel"] = {
        "timing": funnel_timing,
        "profile": {
            "total_elapsed_ms": funnel_profile.total_elapsed_ms,
            "tiers": funnel_profile.tiers,
        },
        "build_ms": funnel_build_timing["median_ms"],
    }
    print(f"    Hybrid funnel: {funnel_timing['median_ms']:.3f}ms")

    if ref_top_ids:
        metrics = compute_recall(funnel_ids, ref_top_ids, candidate_budget)
        result["hybrid_funnel"]["recall"] = metrics
        print(f"    Recall@{candidate_budget}: {metrics['recall_at_k']:.3f}, Top-1: {metrics['top_1_agreement']:.1f}")

    # --- Speedup summary ---
    if result["reference_maxsim"].get("timing"):
        ref_ms = result["reference_maxsim"]["timing"]["median_ms"]
        for strat in ["single_centroid", "multi_centroid", "hybrid_funnel"]:
            strat_ms = result[strat]["timing"]["median_ms"]
            result[strat]["speedup_vs_reference"] = ref_ms / strat_ms if strat_ms > 0 else float("inf")
            print(f"    {strat} speedup: {result[strat]['speedup_vs_reference']:.1f}x")

    # --- End-to-end: screening + full MaxSim rerank on candidates ---
    if n_docs <= 50000 and ref_top_ids:
        print(f"  End-to-end (screen + rerank)...")
        q_tensor = torch.from_numpy(query_emb).to(device)
        d_tensor = torch.from_numpy(doc_emb).to(device)

        def screen_and_rerank():
            # Screen
            ids, _ = screener.search(query_emb, candidate_budget, mode="multi")
            # Get indices for reranking
            id_to_idx = {did: i for i, did in enumerate(doc_ids)}
            indices = [id_to_idx[did] for did in ids]
            idx_tensor = torch.tensor(indices, device=device)
            # Rerank with full MaxSim on candidates only
            candidate_docs = d_tensor[idx_tensor]  # (budget, T, H)
            scores = reference_maxsim(q_tensor, candidate_docs)
            reranked_order = scores.argsort(descending=True).cpu().tolist()
            return [ids[i] for i in reranked_order]

        e2e_result, e2e_timing = timed_call(
            screen_and_rerank, warmup=1, runs=5,
        )
        e2e_metrics = compute_recall(e2e_result, ref_top_ids, min(10, candidate_budget))

        result["end_to_end"] = {
            "timing": e2e_timing,
            "recall_at_10": e2e_metrics["recall_at_k"],
            "top_1_agreement": e2e_metrics["top_1_agreement"],
            "mrr": e2e_metrics["mrr"],
            "speedup_vs_reference": ref_ms / e2e_timing["median_ms"] if e2e_timing["median_ms"] > 0 else float("inf"),
        }
        print(f"    End-to-end: {e2e_timing['median_ms']:.2f}ms")
        print(f"    E2E Speedup: {result['end_to_end']['speedup_vs_reference']:.1f}x")
        print(f"    E2E Recall@10: {e2e_metrics['recall_at_k']:.3f}, Top-1: {e2e_metrics['top_1_agreement']:.1f}")

        del q_tensor, d_tensor
        torch.cuda.empty_cache()

    return result


def multi_query_benchmark(
    n_docs: int,
    dim: int = 128,
    n_queries: int = 8,
    query_tokens: int = 32,
    doc_tokens: int = 256,
    candidate_budget: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Benchmark with multiple diverse queries for reliable recall statistics.
    F1 fix: queries span 3-5 topics each. F9 fix: funnel thresholds exercised.
    """
    print(f"\n{'='*60}")
    print(f"Multi-Query Benchmark: {n_docs} docs, {n_queries} queries")
    print(f"{'='*60}")

    all_recalls = {"single": [], "multi": [], "funnel": [], "e2e": []}
    all_top1 = {"single": [], "multi": [], "funnel": [], "e2e": []}
    all_latency = {"single": [], "multi": [], "funnel": [], "e2e": []}

    # Generate documents (shared across queries)
    rng = np.random.default_rng(42)
    n_topics = min(50, n_docs // 2 + 1)
    topic_centroids = rng.standard_normal((n_topics, dim)).astype(np.float32)
    topic_centroids /= np.linalg.norm(topic_centroids, axis=1, keepdims=True) + 1e-8

    doc_emb = np.zeros((n_docs, doc_tokens, dim), dtype=np.float32)
    for i in range(n_docs):
        n_doc_topics = rng.integers(1, min(4, n_topics + 1))
        topics = rng.choice(n_topics, size=n_doc_topics, replace=False)
        for t in range(doc_tokens):
            topic = rng.choice(topics)
            noise = rng.standard_normal(dim).astype(np.float32) * 0.3
            token = topic_centroids[topic] + noise
            token /= np.linalg.norm(token) + 1e-8
            doc_emb[i, t] = token

    doc_ids = [f"doc-{i:06d}" for i in range(n_docs)]

    # Build screeners once
    screener = CentroidScreener(dim=dim, max_centroids_per_doc=4, device=device)
    screener.build(doc_ids, doc_emb)

    # F9 fix: use lower thresholds so funnel tiers actually fire
    funnel = HybridFunnelScreener(
        dim=dim, max_centroids_per_doc=4, device=device,
        tier0_survival=min(5000, n_docs),
        tier1_survival=min(500, n_docs),
        tier_activation_ratio=1.5,
    )
    funnel.build(doc_ids, doc_emb)

    # Prepare tensors
    d_tensor = torch.from_numpy(doc_emb).to(device) if n_docs <= 20000 else None

    for qi in range(n_queries):
        # F1 fix: multi-topic queries (3-5 topics per query)
        n_qtopics = min(rng.integers(3, 6), n_topics)
        query_topics = rng.choice(n_topics, size=n_qtopics, replace=False)

        query_emb = np.zeros((query_tokens, dim), dtype=np.float32)
        for t in range(query_tokens):
            topic = query_topics[t % len(query_topics)]
            noise_scale = 0.15 + rng.random() * 0.20
            noise = rng.standard_normal(dim).astype(np.float32) * noise_scale
            query_emb[t] = topic_centroids[topic] + noise
            query_emb[t] /= np.linalg.norm(query_emb[t]) + 1e-8

        # Reference
        if d_tensor is not None:
            q_tensor = torch.from_numpy(query_emb).to(device)
            ref_scores = reference_maxsim(q_tensor, d_tensor)
            ref_sorted = ref_scores.argsort(descending=True).cpu().tolist()
            ref_top_ids = [doc_ids[i] for i in ref_sorted[:candidate_budget]]
        else:
            ref_top_ids = None

        # Single centroid
        t0 = time.perf_counter()
        single_ids, _ = screener.search(query_emb, candidate_budget, mode="single")
        torch.cuda.synchronize()
        all_latency["single"].append((time.perf_counter() - t0) * 1000)

        # Multi centroid
        t0 = time.perf_counter()
        multi_ids, _ = screener.search(query_emb, candidate_budget, mode="multi")
        torch.cuda.synchronize()
        all_latency["multi"].append((time.perf_counter() - t0) * 1000)

        # Funnel
        t0 = time.perf_counter()
        funnel_ids, _ = funnel.search(query_emb, candidate_budget)
        torch.cuda.synchronize()
        all_latency["funnel"].append((time.perf_counter() - t0) * 1000)

        if ref_top_ids:
            # Single recall
            m = compute_recall(single_ids, ref_top_ids, candidate_budget)
            all_recalls["single"].append(m["recall_at_k"])
            all_top1["single"].append(m["top_1_agreement"])

            # Multi recall
            m = compute_recall(multi_ids, ref_top_ids, candidate_budget)
            all_recalls["multi"].append(m["recall_at_k"])
            all_top1["multi"].append(m["top_1_agreement"])

            # Funnel recall
            m = compute_recall(funnel_ids, ref_top_ids, candidate_budget)
            all_recalls["funnel"].append(m["recall_at_k"])
            all_top1["funnel"].append(m["top_1_agreement"])

            # End-to-end (screen + rerank)
            id_to_idx = {did: i for i, did in enumerate(doc_ids)}
            t0 = time.perf_counter()
            multi_ids2, _ = screener.search(query_emb, candidate_budget, mode="multi")
            indices = [id_to_idx[did] for did in multi_ids2]
            idx_tensor = torch.tensor(indices, device=device)
            candidate_docs = d_tensor[idx_tensor]
            scores = reference_maxsim(q_tensor, candidate_docs)
            reranked = scores.argsort(descending=True).cpu().tolist()
            e2e_ids = [multi_ids2[i] for i in reranked]
            torch.cuda.synchronize()
            all_latency["e2e"].append((time.perf_counter() - t0) * 1000)

            m = compute_recall(e2e_ids, ref_top_ids, min(10, candidate_budget))
            all_recalls["e2e"].append(m["recall_at_k"])
            all_top1["e2e"].append(m["top_1_agreement"])

    # Aggregate
    summary = {}
    for strat in ["single", "multi", "funnel", "e2e"]:
        summary[strat] = {
            "median_latency_ms": statistics.median(all_latency[strat]) if all_latency[strat] else None,
        }
        if all_recalls[strat]:
            summary[strat].update({
                "mean_recall": statistics.mean(all_recalls[strat]),
                "mean_top1": statistics.mean(all_top1[strat]),
                "min_recall": min(all_recalls[strat]),
                "max_recall": max(all_recalls[strat]),
            })

    print(f"\n  Summary ({n_queries} queries, {n_docs} docs, budget={candidate_budget}):")
    for strat, s in summary.items():
        lat = s.get("median_latency_ms", 0)
        rec = s.get("mean_recall", "N/A")
        top1 = s.get("mean_top1", "N/A")
        if isinstance(rec, float):
            print(f"    {strat:12s}: {lat:.3f}ms  recall={rec:.3f}  top1={top1:.3f}")
        else:
            print(f"    {strat:12s}: {lat:.3f}ms")

    return {"n_docs": n_docs, "n_queries": n_queries, "candidate_budget": candidate_budget, "strategies": summary}


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU-native screener sandbox")
    parser.add_argument("--doc-counts", default="100,1000,5000,10000", help="Comma-separated doc counts")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--doc-tokens", type=int, default=256)
    parser.add_argument("--candidate-budget", type=int, default=100)
    parser.add_argument("--n-queries", type=int, default=8, help="Multi-query benchmark")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    doc_counts = [int(x) for x in args.doc_counts.split(",")]

    results = {"benchmark": "screener_sandbox", "config": vars(args), "runs": []}

    for n_docs in doc_counts:
        # Single-query detailed benchmark
        run = benchmark_at_scale(
            n_docs=n_docs,
            dim=args.dim,
            query_tokens=args.query_tokens,
            doc_tokens=args.doc_tokens,
            candidate_budget=args.candidate_budget,
            device=args.device,
        )
        results["runs"].append(run)

    # Multi-query benchmark on largest feasible scale
    for n_docs in [d for d in doc_counts if d <= 20000]:
        mq = multi_query_benchmark(
            n_docs=n_docs,
            dim=args.dim,
            n_queries=args.n_queries,
            query_tokens=args.query_tokens,
            doc_tokens=args.doc_tokens,
            candidate_budget=args.candidate_budget,
            device=args.device,
        )
        results.setdefault("multi_query", []).append(mq)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
