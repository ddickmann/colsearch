"""
Offline index build pipeline.

1. Load BEIR-style NPZ corpus (reuses voyager-index eval_100k format)
2. Train centroids via k-means
3. Assign documents to shards by dominant centroid
4. Pack safetensors shards
5. Save router state

Usage:
    python -m benchmarks.shard_bench.build_index --corpus-size 100000
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.shard_bench.config import (
    BuildConfig, Compression, StorageLayout,
)
from benchmarks.shard_bench.shard_store import ShardStore
from benchmarks.shard_bench.centroid_router import CentroidRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_NPZ = Path.home() / ".cache" / "voyager-qa" / "beir_100k.npz"


def _mem_gb():
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1


def load_corpus(npz_path: Path, max_docs: int = 0):
    """Load corpus from NPZ, return (all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim)."""
    log.info("Loading corpus from %s ...", npz_path)
    npz = np.load(str(npz_path), allow_pickle=True)

    doc_offsets_arr = npz["doc_offsets"]
    n_docs = int(npz["n_docs"])
    dim = int(npz["dim"])

    if 0 < max_docs < n_docs:
        n_docs = max_docs
        doc_offsets_arr = doc_offsets_arr[:n_docs]
        log.info("Truncating to %d docs", n_docs)

    last_vec = int(doc_offsets_arr[-1][1])
    all_vectors = npz["doc_vectors"][:last_vec]

    # Keep as float16 for memory efficiency; k-means will sample and upcast
    doc_offsets = [(int(s), int(e)) for s, e in doc_offsets_arr]
    doc_ids = list(range(n_docs))

    # Queries
    query_offsets = npz["query_offsets"]
    all_q = npz["query_vectors"]
    query_vecs = [all_q[int(s):int(e)].astype(np.float32) for s, e in query_offsets]

    # Qrels
    qrels_mat = npz["qrels"]
    qrels = {}
    for qi in range(qrels_mat.shape[0]):
        rels = [int(x) for x in qrels_mat[qi] if 0 <= x < n_docs]
        if rels:
            qrels[qi] = rels

    tok_counts = [e - s for s, e in doc_offsets]
    log.info(
        "Corpus loaded: %d docs, %d vectors, dim=%d, tokens/doc mean=%.0f p50=%.0f p95=%.0f, RSS=%.1f GB",
        n_docs, all_vectors.shape[0], dim,
        np.mean(tok_counts), np.median(tok_counts), np.percentile(tok_counts, 95),
        _mem_gb(),
    )

    return all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim


def build(cfg: BuildConfig, npz_path: Path = DEFAULT_NPZ, device: str = "cuda") -> Path:
    """
    Run the full build pipeline. Returns the index directory path.
    """
    index_dir = Path(cfg.corpus_size.__str__()).parent
    cache_base = Path.home() / ".cache" / "shard-bench"
    suffix = "_uniform" if cfg.uniform_shard_tokens else ""
    index_dir = cache_base / f"index_{cfg.corpus_size}_{cfg.compression.value}_{cfg.layout.value}{suffix}"

    if (index_dir / "manifest.json").exists() and (index_dir / "router" / "router_state.json").exists():
        log.info("Index already exists at %s, skipping build", index_dir)
        return index_dir

    index_dir.mkdir(parents=True, exist_ok=True)

    all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim = load_corpus(
        npz_path, max_docs=cfg.corpus_size,
    )

    # Upcast for k-means (only the sample will be float32)
    all_vectors_f32 = all_vectors.astype(np.float32)

    # Train router
    t0 = time.time()
    router, shard_assignments, centroid_to_shard = CentroidRouter.train(
        all_vectors=all_vectors_f32,
        doc_offsets=doc_offsets,
        n_centroids=cfg.n_centroids,
        n_shards=cfg.n_shards,
        sample_fraction=cfg.kmeans_sample_fraction,
        max_iter=cfg.max_kmeans_iter,
        seed=cfg.seed,
        device=device,
    )
    train_s = time.time() - t0
    log.info("Router training done in %.1fs", train_s)

    # Optionally randomize layout for comparison
    if cfg.layout == StorageLayout.RANDOM:
        log.info("Randomizing shard assignments (layout=random)")
        rng = np.random.RandomState(cfg.seed + 1)
        shard_assignments = rng.randint(0, cfg.n_shards, size=len(doc_ids)).astype(np.int32)

    # Build shard store
    t0 = time.time()
    store = ShardStore(index_dir)
    manifest = store.build(
        all_vectors=all_vectors,
        doc_offsets=doc_offsets,
        doc_ids=doc_ids,
        shard_assignments=shard_assignments,
        n_shards=cfg.n_shards,
        dim=dim,
        compression=cfg.compression,
        centroid_to_shard=centroid_to_shard,
        uniform_shard_tokens=cfg.uniform_shard_tokens,
    )
    build_s = time.time() - t0
    log.info("Shard store built in %.1fs", build_s)

    # Save router
    router.save(index_dir / "router")
    log.info("Router saved to %s", index_dir / "router")

    # Save corpus metadata for benchmark reuse
    meta = {
        "corpus_size": cfg.corpus_size,
        "dim": dim,
        "n_centroids": cfg.n_centroids,
        "n_shards": cfg.n_shards,
        "compression": cfg.compression.value,
        "layout": cfg.layout.value,
        "train_time_s": train_s,
        "build_time_s": build_s,
        "total_tokens": int(all_vectors.shape[0]),
        "avg_tokens_per_doc": float(np.mean([e - s for s, e in doc_offsets])),
    }
    import json
    with open(index_dir / "build_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Free big arrays
    del all_vectors, all_vectors_f32, doc_offsets, doc_ids
    gc.collect()
    log.info("Build complete. Index at %s, RSS=%.1f GB", index_dir, _mem_gb())

    return index_dir


def main():
    parser = argparse.ArgumentParser(description="Build shard index")
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--n-centroids", type=int, default=1024)
    parser.add_argument("--n-shards", type=int, default=256)
    parser.add_argument("--compression", choices=["fp16", "int8", "roq4"], default="fp16")
    parser.add_argument("--layout", choices=["random", "centroid_grouped"], default="centroid_grouped")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = BuildConfig(
        corpus_size=args.corpus_size,
        n_centroids=args.n_centroids,
        n_shards=args.n_shards,
        compression=Compression(args.compression),
        layout=StorageLayout(args.layout),
    )

    build(cfg, npz_path=Path(args.npz), device=args.device)


if __name__ == "__main__":
    main()
