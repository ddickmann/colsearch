"""
Configuration dataclasses for the shard benchmark sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Compression(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    ROQ4 = "roq4"


class StorageLayout(str, Enum):
    RANDOM = "random"
    CENTROID_GROUPED = "centroid_grouped"


class TransferMode(str, Enum):
    PAGEABLE = "pageable"
    PINNED = "pinned"
    DOUBLE_BUFFERED = "double_buffered"


@dataclass
class BuildConfig:
    corpus_size: int = 100_000
    n_centroids: int = 1024
    n_shards: int = 256
    dim: int = 128
    compression: Compression = Compression.FP16
    layout: StorageLayout = StorageLayout.CENTROID_GROUPED
    kmeans_sample_fraction: float = 0.1
    max_kmeans_iter: int = 50
    seed: int = 42
    uniform_shard_tokens: bool = True


@dataclass
class SearchConfig:
    top_shards: int = 8
    max_docs_exact: int = 10_000
    transfer_mode: TransferMode = TransferMode.PINNED
    pinned_pool_buffers: int = 3
    pinned_buffer_max_tokens: int = 50_000
    batch_size: int = 1


@dataclass
class BenchmarkConfig:
    build: BuildConfig = field(default_factory=BuildConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    n_eval_queries: int = 300
    n_stress_queries: int = 1_000
    top_k: int = 10
    top_k_recall: int = 100
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "shard-bench")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "results")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["results_dir"] = str(self.results_dir)
        return d


# Default sweep matrix
SWEEP_CORPUS_SIZES = [100_000, 300_000, 1_000_000]
SWEEP_TOP_SHARDS = [4, 8, 16, 32]
SWEEP_MAX_DOCS_EXACT = [1_000, 5_000, 10_000, 25_000]
SWEEP_COMPRESSION = [Compression.FP16, Compression.INT8, Compression.ROQ4]
SWEEP_LAYOUT = [StorageLayout.RANDOM, StorageLayout.CENTROID_GROUPED]
SWEEP_TRANSFER = [TransferMode.PAGEABLE, TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED]
