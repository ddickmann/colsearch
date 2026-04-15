"""CLI entrypoint for offline shard builds."""
from __future__ import annotations

import argparse

from .corpus import DEFAULT_NPZ
from .pipeline import build
from ..config import AnnBackend, BuildConfig, Compression, RouterType, StorageLayout

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

