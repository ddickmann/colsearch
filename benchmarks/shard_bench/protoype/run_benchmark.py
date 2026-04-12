"""Main benchmark harness for shard_bench."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.shard_bench.baselines import BaselineDenseSingleVector, BaselineGpuMaxSim
from benchmarks.shard_bench.build_index import DEFAULT_NPZ, _index_dir, build, load_corpus
from benchmarks.shard_bench.centroid_router import CentroidRouter
from benchmarks.shard_bench.colbandit_reranker import ColBanditConfig, ColBanditReranker
from benchmarks.shard_bench.config import (
    BenchmarkConfig,
    BuildConfig,
    Compression,
    RouterType,
    SearchConfig,
    StorageLayout,
    SWEEP_MAX_DOCS_EXACT,
    SWEEP_TOP_SHARDS,
    SWEEP_TRANSFER,
    TransferMode,
)
from benchmarks.shard_bench.fetch_pipeline import FetchPipeline, PinnedBufferPool
from benchmarks.shard_bench.lemur_router import LemurRouter
from benchmarks.shard_bench.maxsim_scorer import brute_force_maxsim, score_and_topk, score_shards_and_topk
from benchmarks.shard_bench.metrics import compute_all_metrics
from benchmarks.shard_bench.profiler import QueryProfile, Timer, aggregate_profiles, memory_snapshot
from benchmarks.shard_bench.shard_store import ShardStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _mem_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0


def compute_ground_truth(
    query_vecs: list,
    doc_vecs: list,
    doc_ids: List[int],
    dim: int,
    n_eval: int,
    k: int = 100,
    cache_path: Optional[Path] = None,
    device: str = "cuda",
) -> List[List[int]]:
    if cache_path and cache_path.exists():
        log.info("Loading cached ground truth from %s", cache_path)
        data = np.load(str(cache_path), allow_pickle=True)
        gt = [row.tolist() for row in data["gt_ids"][:n_eval]]
        if len(gt) >= n_eval:
            return gt

    log.info("Computing brute-force ground truth for %d queries (k=%d)...", n_eval, k)
    gts = []
    for qi in range(n_eval):
        ids, _scores = brute_force_maxsim(query_vecs[qi], doc_vecs, doc_ids, dim, k=k, device=device)
        gts.append(ids)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        max_len = max(len(g) for g in gts)
        gt_arr = np.full((len(gts), max_len), -1, dtype=np.int64)
        for i, g in enumerate(gts):
            gt_arr[i, :len(g)] = g
        np.savez_compressed(str(cache_path), gt_ids=gt_arr)
    return gts


def search_shard_routed(
    query: torch.Tensor,
    router,
    pipeline: FetchPipeline,
    search_cfg: SearchConfig,
    router_type: RouterType,
    k: int = 10,
    device: str = "cuda",
) -> QueryProfile:
    prof = QueryProfile()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    with Timer(sync_cuda=True) as t_route:
        if router_type == RouterType.CENTROID:
            routed = router.route(query, top_shards=search_cfg.top_shards, max_docs=search_cfg.max_docs_exact)
        else:
            routed = router.route(query, k_candidates=search_cfg.k_candidates, prefetch_doc_cap=search_cfg.max_docs_exact)
    prof.routing_ms = t_route.elapsed_ms

    if router_type == RouterType.CENTROID:
        prof.num_shards_fetched = len(routed)
        shard_chunks, fetch_stats = pipeline.fetch_per_shard(routed, max_docs=search_cfg.max_docs_exact)
    else:
        prof.num_shards_fetched = len(routed.shard_ids)
        shard_chunks, fetch_stats = pipeline.fetch_candidate_docs(routed.by_shard, max_docs=search_cfg.max_docs_exact)
    prof.fetch_ms = fetch_stats.get("fetch_ms", 0.0)
    prof.h2d_bytes = fetch_stats.get("h2d_bytes", 0)
    prof.num_docs_scored = fetch_stats.get("num_docs", 0)

    if prof.num_docs_scored > 0:
        with Timer(sync_cuda=True) as t_score:
            if search_cfg.use_colbandit and router_type == RouterType.LEMUR:
                reranker = ColBanditReranker(search_cfg.colbandit)
                ids, scores, _stats = reranker.rerank_shard_chunks(query, shard_chunks, k=k, device=dev)
            else:
                ids, scores = score_shards_and_topk(query, shard_chunks, k=k, device=dev)
        prof.maxsim_ms = t_score.elapsed_ms
        prof.h2d_ms = t_score.elapsed_ms
        prof.retrieved_ids = ids
        prof.retrieved_scores = scores
    prof.total_ms = prof.routing_ms + prof.fetch_ms + prof.maxsim_ms
    return prof


def run_single_config(
    cfg: BenchmarkConfig,
    query_vecs: list,
    doc_vecs: list,
    doc_ids: List[int],
    ground_truth: List[List[int]],
    dim: int,
    device: str = "cuda",
) -> dict:
    bcfg = cfg.build
    scfg = cfg.search
    index_dir = _index_dir(bcfg)
    if not (index_dir / "manifest.json").exists():
        build(bcfg, device=device)

    store = ShardStore(index_dir)
    if bcfg.router_type == RouterType.CENTROID:
        router = CentroidRouter.load(index_dir / "router", device=device)
    else:
        router = LemurRouter(index_dir / "lemur", ann_backend=bcfg.lemur.ann_backend.value, device=bcfg.lemur.device)
        router.load()

    pool = None
    if scfg.transfer_mode in (TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED):
        pool = PinnedBufferPool(max_tokens=scfg.pinned_buffer_max_tokens, dim=dim, n_buffers=scfg.pinned_pool_buffers)
    pipeline = FetchPipeline(store=store, mode=scfg.transfer_mode, pinned_pool=pool, device=device)

    n_eval = min(cfg.n_eval_queries, len(query_vecs), len(ground_truth))
    profiles: List[QueryProfile] = []
    for qi in range(n_eval):
        qv = torch.from_numpy(query_vecs[qi]).float()
        prof = search_shard_routed(qv, router, pipeline, scfg, bcfg.router_type, k=cfg.top_k_recall, device=device)
        profiles.append(prof)

    all_retrieved = [p.retrieved_ids for p in profiles]
    quality = compute_all_metrics(all_retrieved, ground_truth[:n_eval], ks=(10, 100))
    latency = aggregate_profiles(profiles)
    total_time_s = sum(p.total_ms for p in profiles) / 1000.0
    result = {
        "router_type": bcfg.router_type.value,
        "corpus_size": bcfg.corpus_size,
        "compression": bcfg.compression.value,
        "layout": bcfg.layout.value,
        "n_shards": bcfg.n_shards,
        "top_shards": scfg.top_shards if bcfg.router_type == RouterType.CENTROID else None,
        "k_candidates": scfg.k_candidates if bcfg.router_type == RouterType.LEMUR else None,
        "max_docs_exact": scfg.max_docs_exact,
        "transfer_mode": scfg.transfer_mode.value,
        "use_colbandit": scfg.use_colbandit,
        **quality,
        **latency,
        "qps": n_eval / total_time_s if total_time_s > 0 else 0.0,
        **memory_snapshot(),
    }
    return result


def run_baselines(
    query_vecs: list,
    doc_vecs: list,
    doc_ids: List[int],
    ground_truth: List[List[int]],
    dim: int,
    n_eval: int,
    max_docs_gpu: int = 10_000,
    device: str = "cuda",
) -> List[dict]:
    results = []
    log.info("=== Baseline A: GPU-only MaxSim ===")
    try:
        baseline_a = BaselineGpuMaxSim(doc_vecs, doc_ids, dim, device=device, max_docs=max_docs_gpu)
        profiles_a = []
        for qi in range(n_eval):
            qv = torch.from_numpy(query_vecs[qi]).float()
            profiles_a.append(baseline_a.search(qv, k=100))
        results.append({
            "pipeline": "baseline_a_gpu_maxsim",
            **compute_all_metrics([p.retrieved_ids for p in profiles_a], ground_truth[:n_eval], ks=(10, 100)),
            **aggregate_profiles(profiles_a),
            **memory_snapshot(),
        })
    except Exception as e:
        log.warning("Baseline A failed: %s", e)

    log.info("=== Baseline C: Dense single-vector ===")
    try:
        baseline_c = BaselineDenseSingleVector(doc_vecs, doc_ids, dim)
        profiles_c = []
        for qi in range(n_eval):
            qv = torch.from_numpy(query_vecs[qi]).float()
            profiles_c.append(baseline_c.search(qv, k=100))
        results.append({
            "pipeline": "baseline_c_dense_single_vector",
            **compute_all_metrics([p.retrieved_ids for p in profiles_c], ground_truth[:n_eval], ks=(10, 100)),
            **aggregate_profiles(profiles_c),
            **memory_snapshot(),
        })
    except Exception as e:
        log.warning("Baseline C failed: %s", e)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard Benchmark Harness")
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--router", choices=[x.value for x in RouterType], default=RouterType.LEMUR.value)
    parser.add_argument("--enable-pooling", action="store_true")
    parser.add_argument("--pool-factor", type=int, default=2)
    parser.add_argument("--use-colbandit", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    all_vectors, doc_offsets, doc_ids, query_vecs, _qrels, dim = load_corpus(Path(args.npz), max_docs=args.corpus_size)
    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    n_eval = min(args.n_eval, len(query_vecs))

    gt_cache = Path.home() / ".cache" / "shard-bench" / f"gt_{args.corpus_size}.npz"
    ground_truth = compute_ground_truth(query_vecs, doc_vecs, doc_ids, dim, n_eval, k=100, cache_path=None if not args.skip_gt else gt_cache, device=device)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"bench_{args.corpus_size}.jsonl"
    all_results = []

    if not args.skip_baselines:
        all_results.extend(run_baselines(query_vecs, doc_vecs, doc_ids, ground_truth, dim, n_eval, max_docs_gpu=min(args.corpus_size, 20_000), device=device))

    if args.quick:
        sweep_shards = [8]
        sweep_docs = [5_000]
        sweep_transfer = [TransferMode.PINNED]
        sweep_compression = [Compression.FP16]
        sweep_layout = [StorageLayout.PROXY_GROUPED]
    else:
        sweep_shards = SWEEP_TOP_SHARDS
        sweep_docs = SWEEP_MAX_DOCS_EXACT
        sweep_transfer = SWEEP_TRANSFER
        sweep_compression = [Compression.FP16, Compression.INT8]
        sweep_layout = [StorageLayout.PROXY_GROUPED, StorageLayout.RANDOM]

    for compression in sweep_compression:
        for layout in sweep_layout:
            bcfg = BuildConfig(corpus_size=args.corpus_size, compression=compression, layout=layout, router_type=RouterType(args.router))
            bcfg.pooling.enabled = bool(args.enable_pooling)
            bcfg.pooling.pool_factor = int(args.pool_factor)
            bcfg.lemur.enabled = bcfg.router_type == RouterType.LEMUR
            build(bcfg, npz_path=Path(args.npz), device=device)

            for top_shards in sweep_shards:
                for max_docs in sweep_docs:
                    for transfer in sweep_transfer:
                        scfg = SearchConfig(top_shards=top_shards, max_docs_exact=max_docs, transfer_mode=transfer, use_colbandit=args.use_colbandit)
                        if args.use_colbandit:
                            scfg.colbandit = ColBanditConfig(enabled=True)
                        cfg = BenchmarkConfig(build=bcfg, search=scfg, n_eval_queries=n_eval)
                        try:
                            result = run_single_config(cfg, query_vecs, doc_vecs, doc_ids, ground_truth, dim, device=device)
                            result["pipeline"] = f"shard_routed_{bcfg.router_type.value}"
                            all_results.append(result)
                        except Exception as e:
                            log.error("Config failed: %s", e, exc_info=True)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Results written to %s (%d entries)", results_file, len(all_results))
    log.info("Final RSS: %.1f GB", _mem_gb())


if __name__ == "__main__":
    main()
