"""FetchPipeline: tiered data movement from disk -> CPU -> pinned -> GPU."""
from __future__ import annotations

import logging
import queue
from typing import Dict, List, Optional, Tuple

import torch

from .config import TransferMode
from .profiler import Timer
from .shard_store import ShardStore

logger = logging.getLogger(__name__)
ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]


class PinnedBufferPool:
    def __init__(self, max_tokens: int, dim: int, n_buffers: int = 3):
        self.max_tokens = max_tokens
        self.dim = dim
        self._pool: queue.Queue[torch.Tensor] = queue.Queue()
        for _ in range(n_buffers):
            self._pool.put(torch.empty(max_tokens, dim, dtype=torch.float16, pin_memory=True))

    def get(self) -> torch.Tensor:
        return self._pool.get()

    def release(self, buf: torch.Tensor) -> None:
        self._pool.put(buf)

    @property
    def available(self) -> int:
        return self._pool.qsize()


class FetchPipeline:
    def __init__(
        self,
        store: ShardStore,
        mode: TransferMode = TransferMode.PINNED,
        pinned_pool: Optional[PinnedBufferPool] = None,
        device: str = "cuda",
    ):
        self.store = store
        self.mode = mode
        self.pool = pinned_pool
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def fetch_per_shard(self, shard_ids: List[int], max_docs: int = 0) -> Tuple[List[ShardChunk], dict]:
        chunks: List[ShardChunk] = []
        total_docs = 0
        total_h2d_bytes = 0
        with Timer() as t_fetch:
            for sid in shard_ids:
                emb, offsets, doc_ids = self.store.load_shard(sid, device="cpu")
                if max_docs and total_docs + len(doc_ids) > max_docs:
                    keep = max_docs - total_docs
                    emb, offsets, doc_ids = self._trim_flat_docs(emb, offsets, doc_ids, keep)
                if not doc_ids:
                    continue
                chunks.append((emb, offsets, doc_ids))
                total_docs += len(doc_ids)
                total_h2d_bytes += emb.nelement() * emb.element_size()
                if max_docs and total_docs >= max_docs:
                    break
        return chunks, {
            "fetch_ms": t_fetch.elapsed_ms,
            "h2d_bytes": total_h2d_bytes,
            "num_shards": len(chunks),
            "num_docs": total_docs,
        }

    def fetch_candidate_docs(
        self,
        docs_by_shard: Dict[int, List[int]],
        max_docs: int = 0,
    ) -> Tuple[List[ShardChunk], dict]:
        chunks: List[ShardChunk] = []
        total_docs = 0
        total_h2d_bytes = 0
        with Timer() as t_fetch:
            for shard_id, doc_ids in docs_by_shard.items():
                if max_docs and total_docs >= max_docs:
                    break
                request_ids = doc_ids
                if max_docs:
                    request_ids = doc_ids[: max(0, max_docs - total_docs)]
                emb, offsets, loaded_ids = self.store.load_docs_from_shard(shard_id, request_ids, device="cpu")
                if not loaded_ids:
                    continue
                chunks.append((emb, offsets, loaded_ids))
                total_docs += len(loaded_ids)
                total_h2d_bytes += emb.nelement() * emb.element_size()
        return chunks, {
            "fetch_ms": t_fetch.elapsed_ms,
            "h2d_bytes": total_h2d_bytes,
            "num_shards": len(chunks),
            "num_docs": total_docs,
        }

    def fetch(
        self,
        shard_ids: List[int],
        max_docs: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], dict]:
        if self.mode == TransferMode.DOUBLE_BUFFERED:
            return self._fetch_double_buffered(shard_ids, max_docs)
        if self.mode == TransferMode.PINNED:
            return self._fetch_pinned(shard_ids, max_docs)
        return self._fetch_pageable(shard_ids, max_docs)

    def _fetch_pageable(self, shard_ids: List[int], max_docs: int = 0):
        with Timer() as t_fetch:
            emb_flat, offsets, doc_ids = self.store.load_shards(shard_ids, device="cpu")
            if max_docs and len(doc_ids) > max_docs:
                emb_flat, offsets, doc_ids = self._trim_flat_docs(emb_flat, offsets, doc_ids, max_docs)
        with Timer() as t_pad:
            doc_emb, doc_mask = _pad_docs(emb_flat, offsets)
        h2d_bytes = doc_emb.nelement() * doc_emb.element_size()
        with Timer(sync_cuda=True) as t_h2d:
            doc_emb_gpu = doc_emb.to(self.device)
            doc_mask_gpu = doc_mask.to(self.device)
        return doc_emb_gpu, doc_mask_gpu, doc_ids, {
            "fetch_ms": t_fetch.elapsed_ms,
            "pad_ms": t_pad.elapsed_ms,
            "h2d_ms": t_h2d.elapsed_ms,
            "h2d_bytes": h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(doc_ids),
        }

    def _fetch_pinned(self, shard_ids: List[int], max_docs: int = 0):
        with Timer() as t_fetch:
            emb_flat, offsets, doc_ids = self.store.load_shards(shard_ids, device="cpu")
            if max_docs and len(doc_ids) > max_docs:
                emb_flat, offsets, doc_ids = self._trim_flat_docs(emb_flat, offsets, doc_ids, max_docs)
        with Timer() as t_pad:
            doc_emb, doc_mask = _pad_docs(emb_flat, offsets)
        h2d_bytes = doc_emb.nelement() * doc_emb.element_size()
        with Timer(sync_cuda=True) as t_h2d:
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    doc_emb_gpu = doc_emb.to(self.device, non_blocking=True)
                    doc_mask_gpu = doc_mask.to(self.device, non_blocking=True)
                self._stream.synchronize()
            else:
                doc_emb_gpu = doc_emb.to(self.device)
                doc_mask_gpu = doc_mask.to(self.device)
        return doc_emb_gpu, doc_mask_gpu, doc_ids, {
            "fetch_ms": t_fetch.elapsed_ms,
            "pad_ms": t_pad.elapsed_ms,
            "h2d_ms": t_h2d.elapsed_ms,
            "h2d_bytes": h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(doc_ids),
        }

    def _fetch_double_buffered(self, shard_ids: List[int], max_docs: int = 0):
        return self._fetch_pinned(shard_ids, max_docs)

    @staticmethod
    def _trim_flat_docs(
        flat_emb: torch.Tensor,
        offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        keep_docs: int,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        keep_docs = max(0, min(keep_docs, len(doc_ids)))
        if keep_docs == 0:
            dim = flat_emb.shape[1] if flat_emb.ndim == 2 else 0
            return torch.empty((0, dim), dtype=flat_emb.dtype), [], []
        offsets = offsets[:keep_docs]
        doc_ids = doc_ids[:keep_docs]
        last_end = offsets[-1][1]
        flat_emb = flat_emb[:last_end]
        return flat_emb, offsets, doc_ids


def _pad_docs(flat_emb: torch.Tensor, offsets: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not offsets:
        dim = flat_emb.shape[1] if flat_emb.dim() == 2 else 128
        return torch.empty(0, 1, dim, dtype=flat_emb.dtype), torch.empty(0, 1, dtype=torch.float32)
    n_docs = len(offsets)
    max_tokens = max(e - s for s, e in offsets)
    dim = flat_emb.shape[1]
    padded = torch.zeros(n_docs, max_tokens, dim, dtype=flat_emb.dtype)
    mask = torch.zeros(n_docs, max_tokens, dtype=torch.float32)
    for i, (s, e) in enumerate(offsets):
        length = e - s
        padded[i, :length] = flat_emb[s:e]
        mask[i, :length] = 1.0
    return padded, mask
