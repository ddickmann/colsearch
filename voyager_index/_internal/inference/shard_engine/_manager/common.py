"""Shared imports and configuration for shard-engine manager internals."""
from __future__ import annotations

import gc
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

try:
    from voyager_index._internal.inference.index_core.io_utils import (
        FileLock as _FileLock,
    )
    from voyager_index._internal.inference.index_core.io_utils import (
        atomic_json_write as _atomic_json_write,
    )
except ImportError:
    _FileLock = None
    _atomic_json_write = None

from ..checkpoint import ShardCheckpointManager
from ..colbandit_reranker import ColBanditReranker
from ..config import (
    AnnBackend,
    BuildConfig,
    Compression,
    LemurConfig,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
)
from ..fetch_pipeline import FetchPipeline, PinnedBufferPool
from ..lemur_router import CandidatePlan, LemurRouter
from ..memtable import MemTable
from ..profiler import Timer
from ..scorer import (
    PreloadedGpuCorpus,
    proxy_score_candidates,
    score_all_docs_topk,
    score_roq4_topk,
    warmup_maxsim,
)
from ..shard_store import ShardStore
from ..wal import WalOp, WalReader, WalWriter

logger = logging.getLogger(__name__)

def atomic_json_write(path: Path, data: Any) -> None:
    """Atomic JSON write with graceful fallback to plain json.dump."""
    if _atomic_json_write is not None:
        _atomic_json_write(path, data)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

class ShardEngineConfig:
    """Production configuration for the shard engine.

    Wraps BuildConfig + SearchConfig with sensible defaults
    for the LEMUR-routed path.
    """

    def __init__(
        self,
        *,
        n_shards: int = 256,
        dim: int = 128,
        compression: Compression = Compression.FP16,
        layout: StorageLayout = StorageLayout.PROXY_GROUPED,
        router_type: RouterType = RouterType.LEMUR,
        ann_backend: AnnBackend = AnnBackend.FAISS_FLAT_IP,
        lemur_epochs: int = 10,
        k_candidates: int = 2000,
        max_docs_exact: int = 10_000,
        lemur_search_k_cap: int | None = 2048,
        n_full_scores: int = 4096,
        transfer_mode: TransferMode = TransferMode.PINNED,
        pinned_pool_buffers: int = 3,
        pinned_buffer_max_tokens: int = 50_000,
        use_colbandit: bool = False,
        uniform_shard_tokens: bool = True,
        quantization_mode: str = "",
        variable_length_strategy: str = "bucketed",
        gpu_corpus_rerank_topn: int = 16,
        n_centroids: int = 1024,
        n_centroid_approx: int = 0,
        router_device: str | None = "cpu",
        seed: int = 42,
    ):
        self.n_shards = n_shards
        self.dim = dim
        self.compression = compression
        self.layout = layout
        self.router_type = router_type
        self.ann_backend = ann_backend
        self.lemur_epochs = lemur_epochs
        self.k_candidates = k_candidates
        self.max_docs_exact = max_docs_exact
        self.lemur_search_k_cap = lemur_search_k_cap
        self.n_full_scores = n_full_scores
        self.transfer_mode = transfer_mode
        self.pinned_pool_buffers = pinned_pool_buffers
        self.pinned_buffer_max_tokens = pinned_buffer_max_tokens
        self.use_colbandit = use_colbandit
        self.uniform_shard_tokens = uniform_shard_tokens
        self.quantization_mode = quantization_mode
        self.variable_length_strategy = variable_length_strategy
        self.gpu_corpus_rerank_topn = gpu_corpus_rerank_topn
        self.n_centroids = n_centroids
        self.n_centroid_approx = n_centroid_approx
        self.router_device = router_device
        self.seed = seed

    def to_build_config(self, corpus_size: int) -> BuildConfig:
        cfg = BuildConfig(
            corpus_size=corpus_size,
            n_shards=self.n_shards,
            dim=self.dim,
            compression=self.compression,
            layout=self.layout,
            router_type=self.router_type,
            uniform_shard_tokens=self.uniform_shard_tokens,
            seed=self.seed,
        )
        cfg.lemur = LemurConfig(
            enabled=self.router_type == RouterType.LEMUR,
            device=self.router_device or "cuda",
            ann_backend=self.ann_backend,
            epochs=self.lemur_epochs,
            k_candidates=self.k_candidates,
            search_k_cap=self.lemur_search_k_cap,
        )
        return cfg

    def to_search_config(self) -> SearchConfig:
        return SearchConfig(
            k_candidates=self.k_candidates,
            max_docs_exact=self.max_docs_exact,
            lemur_search_k_cap=self.lemur_search_k_cap,
            n_full_scores=self.n_full_scores,
            n_centroid_approx=self.n_centroid_approx,
            transfer_mode=self.transfer_mode,
            pinned_pool_buffers=self.pinned_pool_buffers,
            pinned_buffer_max_tokens=self.pinned_buffer_max_tokens,
            use_colbandit=self.use_colbandit,
            quantization_mode=self.quantization_mode,
            variable_length_strategy=self.variable_length_strategy,
            gpu_corpus_rerank_topn=self.gpu_corpus_rerank_topn,
        )

