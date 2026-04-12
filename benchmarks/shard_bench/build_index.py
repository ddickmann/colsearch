"""Offline index build pipeline for shard_bench."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.shard_bench.config import (
    AnnBackend,
    BuildConfig,
    Compression,
    RouterType,
    StorageLayout,
)
from benchmarks.shard_bench.centroid_router import CentroidRouter
from benchmarks.shard_bench.lemur_router import LemurRouter
from benchmarks.shard_bench.pooling import TokenPooler
from benchmarks.shard_bench.shard_store import ShardStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DEFAULT_NPZ = Path.home() / ".cache" / "voyager-qa" / "beir_100k.npz"


def _mem_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0


def load_corpus(npz_path: Path, max_docs: int = 0):
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
    doc_offsets = [(int(s), int(e)) for s, e in doc_offsets_arr]
    doc_ids = list(range(n_docs))
    query_offsets = npz["query_offsets"]
    all_q = npz["query_vectors"]
    query_vecs = [all_q[int(s):int(e)].astype(np.float32) for s, e in query_offsets]
    qrels_mat = npz["qrels"]
    qrels = {}
    for qi in range(qrels_mat.shape[0]):
        rels = [int(x) for x in qrels_mat[qi] if 0 <= x < n_docs]
        if rels:
            qrels[qi] = rels
    tok_counts = [e - s for s, e in doc_offsets]
    log.info(
        "Corpus loaded: %d docs, %d vectors, dim=%d, tokens/doc mean=%.0f p50=%.0f p95=%.0f, RSS=%.1f GB",
        n_docs, int(all_vectors.shape[0]), dim,
        np.mean(tok_counts), np.median(tok_counts), np.percentile(tok_counts, 95),
        _mem_gb(),
    )
    return all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim


def _index_dir(cfg: BuildConfig) -> Path:
    cache_base = Path.home() / ".cache" / "shard-bench"
    suffix = "_uniform" if cfg.uniform_shard_tokens else ""
    pool_suffix = f"_pool{cfg.pooling.pool_factor}" if cfg.pooling.enabled else ""
    router_suffix = f"_{cfg.router_type.value}"
    return cache_base / f"index_{cfg.corpus_size}_{cfg.compression.value}_{cfg.layout.value}{router_suffix}{pool_suffix}{suffix}"


def assign_storage_shards(
    pooled_offsets: List[Tuple[int, int]],
    n_shards: int,
    seed: int,
    layout: StorageLayout,
    proxy_weights: torch.Tensor | None = None,
) -> np.ndarray:
    lengths = np.array([e - s for s, e in pooled_offsets], dtype=np.int64)
    n_docs = len(pooled_offsets)
    rng = np.random.RandomState(seed)

    if layout == StorageLayout.RANDOM:
        return rng.randint(0, n_shards, size=n_docs).astype(np.int32)

    if proxy_weights is not None and proxy_weights.numel() > 0 and layout in (
        StorageLayout.PROXY_GROUPED, StorageLayout.CENTROID_GROUPED,
    ):
        W = proxy_weights.detach().cpu().numpy().astype(np.float32)
        order = _proxy_order(W, seed)
    else:
        order = np.argsort(-lengths)

    assignments = np.zeros(n_docs, dtype=np.int32)

    if layout == StorageLayout.TOKEN_BALANCED:
        shard_loads = np.zeros(n_shards, dtype=np.int64)
        for doc_idx in order:
            sid = int(np.argmin(shard_loads))
            assignments[doc_idx] = sid
            shard_loads[sid] += lengths[doc_idx]
        return assignments

    total_tokens = max(1, int(lengths.sum()))
    target = max(1, int(np.ceil(total_tokens / n_shards)))
    sid = 0
    current = 0
    for doc_idx in order:
        if sid < n_shards - 1 and current >= target:
            sid += 1
            current = 0
        assignments[doc_idx] = sid
        current += int(lengths[doc_idx])
    return assignments


def _proxy_order(weights: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(256, max(16, int(np.sqrt(weights.shape[0]))))
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=seed)
        labels = km.fit_predict(weights)
        centroid_norm = np.linalg.norm(km.cluster_centers_, axis=1)
        return np.lexsort((centroid_norm[labels], labels)).astype(np.int64)
    except Exception:
        proj = weights[:, 0] if weights.shape[1] > 0 else np.zeros(weights.shape[0], dtype=np.float32)
        return np.argsort(proj, kind="mergesort")


def build(cfg: BuildConfig, npz_path: Path = DEFAULT_NPZ, device: str = "cuda") -> Path:
    index_dir = _index_dir(cfg)

    if (index_dir / "manifest.json").exists():
        router_path = index_dir / ("lemur" if cfg.router_type == RouterType.LEMUR else "router")
        if (router_path / "router_state.json").exists():
            log.info("Index already exists at %s, skipping build", index_dir)
            return index_dir

    index_dir.mkdir(parents=True, exist_ok=True)
    all_vectors, doc_offsets, doc_ids, *_rest, dim = load_corpus(npz_path, max_docs=cfg.corpus_size)

    active_vectors = all_vectors
    active_offsets = doc_offsets
    active_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int32)
    centroid_to_shard = None

    # Optional token pooling
    if cfg.pooling.enabled:
        pooler = TokenPooler(
            method=cfg.pooling.method,
            pool_factor=cfg.pooling.pool_factor,
            protected_tokens=cfg.pooling.protected_tokens,
        )
        pooled_vectors, pooled_offsets, pooled_counts = pooler.pool_docs(all_vectors, doc_offsets)
        active_vectors = pooled_vectors.numpy()
        active_offsets = pooled_offsets
        active_counts = pooled_counts.numpy()
        log.info(
            "Token pooling reduced corpus from %d to %d token vectors (%.2fx)",
            int(all_vectors.shape[0]), int(active_vectors.shape[0]),
            float(all_vectors.shape[0]) / max(1, int(active_vectors.shape[0])),
        )

    # Train router
    if cfg.router_type == RouterType.CENTROID:
        t0 = time.time()
        router, shard_assignments, centroid_to_shard = CentroidRouter.train(
            all_vectors=np.asarray(active_vectors, dtype=np.float32),
            doc_offsets=active_offsets,
            n_centroids=cfg.n_centroids,
            n_shards=cfg.n_shards,
            sample_fraction=cfg.kmeans_sample_fraction,
            max_iter=cfg.max_kmeans_iter,
            seed=cfg.seed,
            device=device,
        )
        train_s = time.time() - t0
        log.info("Centroid router training done in %.1fs", train_s)
    else:
        t0 = time.time()
        lemur_device = device if device != "cpu" else cfg.lemur.device
        router = LemurRouter(
            index_dir=index_dir / "lemur",
            ann_backend=cfg.lemur.ann_backend.value,
            device=lemur_device,
        )
        # Build FP16 tensor once, reuse for both passes
        doc_vecs_f16 = torch.from_numpy(np.asarray(active_vectors)).to(torch.float16)
        doc_counts_t = torch.from_numpy(active_counts)

        # First pass: temporary token-balanced shards for LEMUR training
        shard_assignments = assign_storage_shards(
            active_offsets, cfg.n_shards, cfg.seed, StorageLayout.TOKEN_BALANCED,
        )
        doc_id_to_shard = {doc_id: int(shard_assignments[i]) for i, doc_id in enumerate(doc_ids)}
        router.fit_initial(
            pooled_doc_vectors=doc_vecs_f16,
            pooled_doc_counts=doc_counts_t,
            doc_ids=doc_ids,
            doc_id_to_shard=doc_id_to_shard,
            epochs=cfg.lemur.epochs,
        )
        # Second pass: use learned proxy weights for better shard layout
        proxy_weights = router._weights.clone()
        shard_assignments = assign_storage_shards(
            active_offsets, cfg.n_shards, cfg.seed, cfg.layout, proxy_weights=proxy_weights,
        )
        doc_id_to_shard = {doc_id: int(shard_assignments[i]) for i, doc_id in enumerate(doc_ids)}
        router.fit_initial(
            pooled_doc_vectors=doc_vecs_f16,
            pooled_doc_counts=doc_counts_t,
            doc_ids=doc_ids,
            doc_id_to_shard=doc_id_to_shard,
            epochs=cfg.lemur.epochs,
        )
        del doc_vecs_f16, doc_counts_t
        gc.collect()
        train_s = time.time() - t0
        log.info("LEMUR router training done in %.1fs", train_s)

    if cfg.layout == StorageLayout.RANDOM:
        rng = np.random.RandomState(cfg.seed + 1)
        shard_assignments = rng.randint(0, cfg.n_shards, size=len(doc_ids)).astype(np.int32)
        if cfg.router_type == RouterType.LEMUR:
            doc_id_to_shard = {doc_id: int(shard_assignments[i]) for i, doc_id in enumerate(doc_ids)}
            router._doc_id_to_shard = doc_id_to_shard
            router.save()

    # ROQ4: train quantizer and pre-encode all documents
    roq_quantizer = None
    roq_doc_codes = None
    roq_doc_meta = None
    if cfg.compression == Compression.ROQ4:
        try:
            from voyager_index._internal.inference.quantization.rotational import (
                RotationalQuantizer, RoQConfig,
            )
            log.info("Training ROQ 4-bit quantizer ...")
            roq_quantizer = RotationalQuantizer(RoQConfig(dim=dim, num_bits=4, seed=cfg.seed))
            roq_doc_codes = []
            roq_doc_meta = []
            t_roq = time.time()
            for i, (s, e) in enumerate(active_offsets):
                vecs = np.asarray(active_vectors[s:e], dtype=np.float32)
                q = roq_quantizer.quantize(vecs, store=False)
                roq_doc_codes.append(np.asarray(q["codes"], dtype=np.uint8))
                roq_doc_meta.append(roq_quantizer.build_triton_meta(q, include_norm_sq=True))
                if (i + 1) % 2000 == 0:
                    log.info("  ROQ encoded %d/%d docs (%.1fs)", i + 1, len(active_offsets), time.time() - t_roq)
            log.info("ROQ encoding done in %.1fs", time.time() - t_roq)

            # Save quantizer for query-time use
            import pickle
            with open(index_dir / "roq_quantizer.pkl", "wb") as f:
                pickle.dump(roq_quantizer, f)
        except ImportError:
            log.warning("ROQ quantizer not available, falling back to FP16 storage for ROQ4")
            cfg.compression = Compression.FP16

    # Build shard store
    t0 = time.time()
    store = ShardStore(index_dir)
    store.build(
        all_vectors=np.asarray(active_vectors),
        doc_offsets=active_offsets,
        doc_ids=doc_ids,
        shard_assignments=shard_assignments,
        n_shards=cfg.n_shards,
        dim=dim,
        compression=cfg.compression,
        centroid_to_shard=centroid_to_shard,
        uniform_shard_tokens=cfg.uniform_shard_tokens,
        roq_doc_codes=roq_doc_codes,
        roq_doc_meta=roq_doc_meta,
    )
    build_s = time.time() - t0
    log.info("Shard store built in %.1fs", build_s)

    # Save router (LEMUR already saved itself via fit_initial; centroid needs explicit save)
    if cfg.router_type == RouterType.CENTROID:
        router.save(index_dir / "router")
        log.info("Router saved to %s", index_dir / "router")

    with open(index_dir / "build_meta.json", "w") as f:
        json.dump({
            "corpus_size": cfg.corpus_size,
            "dim": dim,
            "n_centroids": cfg.n_centroids,
            "n_shards": cfg.n_shards,
            "compression": cfg.compression.value,
            "layout": cfg.layout.value,
            "router_type": cfg.router_type.value,
            "pooling_enabled": cfg.pooling.enabled,
            "pool_factor": cfg.pooling.pool_factor,
            "train_time_s": train_s,
            "build_time_s": build_s,
            "total_tokens": int(np.asarray(active_vectors).shape[0]),
            "avg_tokens_per_doc": float(np.mean([e - s for s, e in active_offsets])),
        }, f, indent=2)

    gc.collect()
    log.info("Build complete. Index at %s, RSS=%.1f GB", index_dir, _mem_gb())
    return index_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shard index")
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--n-centroids", type=int, default=1024)
    parser.add_argument("--n-shards", type=int, default=256)
    parser.add_argument("--compression", choices=["fp16", "int8", "roq4"], default="fp16")
    parser.add_argument("--layout", choices=[x.value for x in StorageLayout], default=StorageLayout.PROXY_GROUPED.value)
    parser.add_argument("--router", choices=[x.value for x in RouterType], default=RouterType.LEMUR.value)
    parser.add_argument("--enable-pooling", action="store_true")
    parser.add_argument("--pool-factor", type=int, default=2)
    parser.add_argument("--lemur-epochs", type=int, default=10)
    parser.add_argument("--ann-backend", choices=[x.value for x in AnnBackend], default=AnnBackend.FAISS_HNSW_IP.value)
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = BuildConfig(
        corpus_size=args.corpus_size,
        n_centroids=args.n_centroids,
        n_shards=args.n_shards,
        compression=Compression(args.compression),
        layout=StorageLayout(args.layout),
        router_type=RouterType(args.router),
    )
    cfg.pooling.enabled = bool(args.enable_pooling)
    cfg.pooling.pool_factor = int(args.pool_factor)
    cfg.lemur.enabled = cfg.router_type == RouterType.LEMUR
    cfg.lemur.epochs = int(args.lemur_epochs)
    cfg.lemur.ann_backend = AnnBackend(args.ann_backend)
    build(cfg, npz_path=Path(args.npz), device=args.device)


if __name__ == "__main__":
    main()
