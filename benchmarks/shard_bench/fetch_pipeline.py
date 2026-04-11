"""
FetchPipeline: tiered data movement from disk -> CPU -> pinned -> GPU.

Two APIs:
- fetch():  legacy — returns one merged padded tensor (slow, kept for baselines)
- fetch_per_shard():  new — returns list of per-shard (flat_emb, offsets, doc_ids)
- pipelined_search():  new — overlaps fetch of shard N+1 with GPU scoring of shard N

Modes:
- PAGEABLE:  load to CPU, .to('cuda') (default PyTorch, simplest)
- PINNED:    load to pinned host buffer, async H2D via CUDA stream
- DOUBLE_BUFFERED: overlap shard N+1 fetch with shard N's H2D transfer
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, List, Optional, Tuple

import torch

from .config import TransferMode
from .shard_store import ShardStore
from .profiler import Timer

logger = logging.getLogger(__name__)

ShardChunk = Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]


class PinnedBufferPool:
    """Pool of pre-allocated pinned host memory buffers."""

    def __init__(self, max_tokens: int, dim: int, n_buffers: int = 3):
        self.max_tokens = max_tokens
        self.dim = dim
        self._pool: queue.Queue[torch.Tensor] = queue.Queue()
        for _ in range(n_buffers):
            buf = torch.empty(max_tokens, dim, dtype=torch.float16, pin_memory=True)
            self._pool.put(buf)

    def get(self) -> torch.Tensor:
        return self._pool.get()

    def release(self, buf: torch.Tensor):
        self._pool.put(buf)

    @property
    def available(self) -> int:
        return self._pool.qsize()


class FetchPipeline:
    """
    Orchestrates shard fetching from disk/CPU to GPU.

    Primary API for the fast path is fetch_per_shard(), which returns a list
    of per-shard chunks that can be scored independently (no cross-shard padding).

    pipelined_search() overlaps fetch with scoring for maximum throughput.
    """

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

    # ------------------------------------------------------------------
    # New fast path: per-shard chunks
    # ------------------------------------------------------------------

    def fetch_per_shard(
        self,
        shard_ids: List[int],
        max_docs: int = 0,
    ) -> Tuple[List[ShardChunk], dict]:
        """
        Fetch shards and return per-shard (flat_emb, offsets, doc_ids) chunks.

        Always uses CPU load (no pinned staging) because per-shard tensors are
        small enough that pinned allocation overhead exceeds H2D benefit.
        The caller's scoring loop handles H2D inside _pad_shard_on_device().

        Returns:
            chunks: list of (flat_emb, offsets, doc_ids)
            stats:  dict with fetch_ms, h2d_bytes, num_shards, num_docs
        """
        chunks: List[ShardChunk] = []
        total_docs = 0
        total_bytes = 0

        with Timer() as t_fetch:
            for sid in shard_ids:
                if max_docs and total_docs >= max_docs:
                    break

                emb, offsets, dids = self.store.load_shard(sid, device="cpu")

                if max_docs and total_docs + len(dids) > max_docs:
                    n_take = max_docs - total_docs
                    offsets = offsets[:n_take]
                    dids = dids[:n_take]
                    last_end = offsets[-1][1] if offsets else 0
                    emb = emb[:last_end]

                total_docs += len(dids)
                total_bytes += emb.nelement() * emb.element_size()
                chunks.append((emb, offsets, dids))

        stats = {
            "fetch_ms": t_fetch.elapsed_ms,
            "h2d_bytes": total_bytes,
            "num_shards": len(chunks),
            "num_docs": total_docs,
        }
        return chunks, stats

    def pipelined_search(
        self,
        shard_ids: List[int],
        score_fn: Callable[[ShardChunk], None],
        max_docs: int = 0,
    ) -> dict:
        """
        Overlap disk fetch of shard N+1 with GPU scoring of shard N.

        score_fn receives one (flat_emb, offsets, doc_ids) chunk at a time.
        The caller accumulates scores in its own top-k heap.

        Returns stats dict.
        """
        total_docs = 0
        total_bytes = 0
        total_fetch_ms = 0.0

        ready_q: queue.Queue[Optional[ShardChunk]] = queue.Queue(maxsize=2)

        def _fetcher():
            nonlocal total_docs, total_bytes, total_fetch_ms
            for sid in shard_ids:
                if max_docs and total_docs >= max_docs:
                    break
                with Timer() as t:
                    if self.mode in (TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED):
                        emb, offsets, dids = self.store.load_shard_to_pinned(sid)
                    else:
                        emb, offsets, dids = self.store.load_shard(sid, device="cpu")
                total_fetch_ms += t.elapsed_ms

                if max_docs and total_docs + len(dids) > max_docs:
                    n_take = max_docs - total_docs
                    offsets = offsets[:n_take]
                    dids = dids[:n_take]
                    last_end = offsets[-1][1] if offsets else 0
                    emb = emb[:last_end]

                total_docs += len(dids)
                total_bytes += emb.nelement() * emb.element_size()
                ready_q.put((emb, offsets, dids))

            ready_q.put(None)  # sentinel

        fetch_thread = threading.Thread(target=_fetcher, daemon=True)
        fetch_thread.start()

        while True:
            chunk = ready_q.get()
            if chunk is None:
                break
            score_fn(chunk)

        fetch_thread.join()

        return {
            "fetch_ms": total_fetch_ms,
            "h2d_bytes": total_bytes,
            "num_shards": len(shard_ids),
            "num_docs": total_docs,
        }

    # ------------------------------------------------------------------
    # Legacy merged-tensor API (kept for baselines / backward compat)
    # ------------------------------------------------------------------

    def fetch(
        self,
        shard_ids: List[int],
        max_docs: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], dict]:
        """
        Legacy: fetch shards and deliver padded documents to GPU.
        Prefer fetch_per_shard() + score_shards_and_topk() for new code.
        """
        if self.mode == TransferMode.PAGEABLE:
            return self._fetch_pageable(shard_ids, max_docs)
        elif self.mode == TransferMode.PINNED:
            return self._fetch_pinned(shard_ids, max_docs)
        elif self.mode == TransferMode.DOUBLE_BUFFERED:
            return self._fetch_double_buffered(shard_ids, max_docs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _fetch_pageable(self, shard_ids, max_docs):
        with Timer() as t_fetch:
            emb_flat, offsets, doc_ids = self.store.load_shards(shard_ids, device="cpu")

        if max_docs and len(doc_ids) > max_docs:
            offsets = offsets[:max_docs]
            doc_ids = doc_ids[:max_docs]

        with Timer() as t_pad:
            doc_emb, doc_mask = _pad_docs(emb_flat, offsets)

        h2d_bytes = doc_emb.nelement() * doc_emb.element_size()
        with Timer(sync_cuda=True) as t_h2d:
            doc_emb = doc_emb.to(self.device)
            doc_mask = doc_mask.to(self.device)

        stats = {
            "fetch_ms": t_fetch.elapsed_ms,
            "pad_ms": t_pad.elapsed_ms,
            "h2d_ms": t_h2d.elapsed_ms,
            "h2d_bytes": h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(doc_ids),
        }
        return doc_emb, doc_mask, doc_ids, stats

    def _fetch_pinned(self, shard_ids, max_docs):
        with Timer() as t_fetch:
            all_emb = []
            all_offsets = []
            all_ids = []
            global_offset = 0

            for sid in shard_ids:
                if self.pool:
                    emb, offsets, dids = self.store.load_shard_to_pinned(sid)
                else:
                    emb, offsets, dids = self.store.load_shard(sid, device="cpu")

                for s, e in offsets:
                    all_offsets.append((global_offset + s, global_offset + e))
                global_offset += emb.shape[0]
                all_emb.append(emb)
                all_ids.extend(dids)

            if not all_emb:
                dim = self.store.manifest.dim if self.store.manifest else 128
                emb_flat = torch.empty(0, dim, dtype=torch.float16)
            else:
                emb_flat = torch.cat(all_emb, dim=0)

        if max_docs and len(all_ids) > max_docs:
            all_offsets = all_offsets[:max_docs]
            all_ids = all_ids[:max_docs]

        with Timer() as t_pad:
            doc_emb, doc_mask = _pad_docs(emb_flat, all_offsets)

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

        stats = {
            "fetch_ms": t_fetch.elapsed_ms,
            "pad_ms": t_pad.elapsed_ms,
            "h2d_ms": t_h2d.elapsed_ms,
            "h2d_bytes": h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(all_ids),
        }
        return doc_emb_gpu, doc_mask_gpu, all_ids, stats

    def _fetch_double_buffered(self, shard_ids, max_docs):
        """Overlap fetch of shard N+1 with H2D transfer of shard N."""
        if not self._stream:
            return self._fetch_pinned(shard_ids, max_docs)

        all_gpu_emb = []
        all_gpu_mask = []
        all_ids: List[int] = []
        total_fetch_ms = 0.0
        total_h2d_ms = 0.0
        total_h2d_bytes = 0

        pending_h2d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        for i, sid in enumerate(shard_ids):
            with Timer() as t_fetch:
                if self.pool:
                    emb, offsets, dids = self.store.load_shard_to_pinned(sid)
                else:
                    emb, offsets, dids = self.store.load_shard(sid, device="cpu")

            total_fetch_ms += t_fetch.elapsed_ms

            if max_docs and len(all_ids) + len(dids) > max_docs:
                n_take = max_docs - len(all_ids)
                offsets = offsets[:n_take]
                dids = dids[:n_take]

            doc_emb, doc_mask = _pad_docs(emb, offsets)
            h2d_bytes = doc_emb.nelement() * doc_emb.element_size()

            if pending_h2d is not None:
                self._stream.synchronize()

            with Timer(sync_cuda=False) as t_h2d:
                with torch.cuda.stream(self._stream):
                    gpu_emb = doc_emb.to(self.device, non_blocking=True)
                    gpu_mask = doc_mask.to(self.device, non_blocking=True)

            pending_h2d = (gpu_emb, gpu_mask)
            total_h2d_ms += t_h2d.elapsed_ms
            total_h2d_bytes += h2d_bytes

            all_gpu_emb.append(gpu_emb)
            all_gpu_mask.append(gpu_mask)
            all_ids.extend(dids)

            if max_docs and len(all_ids) >= max_docs:
                break

        if pending_h2d is not None:
            self._stream.synchronize()

        if not all_gpu_emb:
            dim = self.store.manifest.dim if self.store.manifest else 128
            empty = torch.empty(0, 1, dim, dtype=torch.float16, device=self.device)
            mask = torch.empty(0, 1, dtype=torch.float32, device=self.device)
            return empty, mask, [], {"fetch_ms": 0, "h2d_ms": 0, "h2d_bytes": 0, "num_shards": 0, "num_docs": 0}

        max_t = max(e.shape[1] for e in all_gpu_emb)
        dim = all_gpu_emb[0].shape[2]
        combined_emb = []
        combined_mask = []
        for e, m in zip(all_gpu_emb, all_gpu_mask):
            if e.shape[1] < max_t:
                pad = torch.zeros(e.shape[0], max_t - e.shape[1], dim, dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=1)
                mpad = torch.zeros(m.shape[0], max_t - m.shape[1], dtype=m.dtype, device=m.device)
                m = torch.cat([m, mpad], dim=1)
            combined_emb.append(e)
            combined_mask.append(m)

        doc_emb_gpu = torch.cat(combined_emb, dim=0)
        doc_mask_gpu = torch.cat(combined_mask, dim=0)

        stats = {
            "fetch_ms": total_fetch_ms,
            "h2d_ms": total_h2d_ms,
            "h2d_bytes": total_h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(all_ids),
        }
        return doc_emb_gpu, doc_mask_gpu, all_ids, stats


# ------------------------------------------------------------------
# Padding utility (legacy, used by merged-tensor paths only)
# ------------------------------------------------------------------

def _pad_docs(
    flat_emb: torch.Tensor,
    offsets: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length documents to a uniform tensor.
    Only used by the legacy merged-tensor code paths.
    """
    if not offsets:
        dim = flat_emb.shape[1] if flat_emb.dim() == 2 else 128
        return (
            torch.empty(0, 1, dim, dtype=flat_emb.dtype),
            torch.empty(0, 1, dtype=torch.float32),
        )

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
