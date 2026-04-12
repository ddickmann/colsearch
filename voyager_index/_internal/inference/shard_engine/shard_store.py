"""ShardStore: safetensors-backed shard storage with doc-selective fetch."""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from safetensors.numpy import save_file as st_save_np
    from safetensors.torch import load_file as st_load
    SAFETENSORS_AVAILABLE = True
except Exception:
    SAFETENSORS_AVAILABLE = False
    st_save_np = None
    st_load = None

from .config import Compression

logger = logging.getLogger(__name__)


@dataclass
class ShardMeta:
    shard_id: int
    num_docs: int
    total_tokens: int
    centroid_ids: List[int]
    byte_size: int
    file_name: str
    compression: str
    p50_tokens: float = 0.0
    p95_tokens: float = 0.0
    shard_max_tokens: int = 0


@dataclass
class StoreManifest:
    num_shards: int
    num_docs: int
    dim: int
    total_tokens: int
    avg_tokens_per_chunk: float
    p50_tokens: float
    p95_tokens: float
    compression: str
    shards: List[ShardMeta]
    global_target_len: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StoreManifest":
        with open(path) as f:
            d = json.load(f)
        d["shards"] = [ShardMeta(**s) for s in d["shards"]]
        d.setdefault("global_target_len", 0)
        return cls(**d)


@dataclass
class DocMeta:
    doc_id: int
    shard_id: int
    local_offset_start: int
    local_offset_end: int
    row_index: int


class _ShardLRU:
    """Thread-safe LRU cache for decoded shard data."""

    def __init__(self, max_shards: int = 512):
        self._max = max_shards
        self._data: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, shard_id: int):
        with self._lock:
            if shard_id in self._data:
                self._data.move_to_end(shard_id)
                return self._data[shard_id]
            return None

    def put(self, shard_id: int, emb: torch.Tensor, offsets_t: torch.Tensor, ids_t: torch.Tensor):
        with self._lock:
            if shard_id in self._data:
                self._data.move_to_end(shard_id)
                return
            if len(self._data) >= self._max:
                self._data.popitem(last=False)
            self._data[shard_id] = (emb, offsets_t, ids_t)


class ShardStore:
    def __init__(self, root_path: Path, lru_max_shards: int = 512):
        self.root = Path(root_path)
        self.shard_dir = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"
        self.doc_index_path = self.root / "doc_index.npz"
        self.manifest: Optional[StoreManifest] = None
        self._meta_cache: Dict[int, ShardMeta] = {}
        self._doc_index: Dict[int, DocMeta] = {}
        self._shard_cache = _ShardLRU(max_shards=lru_max_shards)

        if self.manifest_path.exists():
            self.manifest = StoreManifest.load(self.manifest_path)
            self._meta_cache = {m.shard_id: m for m in self.manifest.shards}
        if self.doc_index_path.exists():
            self._load_doc_index()

    def all_doc_ids(self) -> List[int]:
        """Return all document IDs stored in the index."""
        return sorted(self._doc_index.keys())

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        all_vectors: np.ndarray,
        doc_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        shard_assignments: np.ndarray,
        n_shards: int,
        dim: int,
        compression: Compression = Compression.FP16,
        centroid_to_shard: Optional[Dict[int, int]] = None,
        uniform_shard_tokens: bool = False,
        roq_doc_codes: Optional[list] = None,
        roq_doc_meta: Optional[list] = None,
    ) -> StoreManifest:
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required: pip install safetensors")

        self.shard_dir.mkdir(parents=True, exist_ok=True)

        shard_to_centroid: Dict[int, List[int]] = {}
        if centroid_to_shard:
            for cid, sid in centroid_to_shard.items():
                shard_to_centroid.setdefault(int(sid), []).append(int(cid))

        all_token_counts = [e - s for s, e in doc_offsets]
        shard_metas: List[ShardMeta] = []
        doc_index_rows: List[Tuple[int, int, int, int, int]] = []

        # Compute one global target_len for the entire corpus
        global_target_len = 0
        if uniform_shard_tokens:
            tc_global = np.array(all_token_counts, dtype=np.float64)
            raw_p95 = int(np.ceil(np.percentile(tc_global, 95))) if len(tc_global) else 1
            raw_p95 = max(raw_p95, 1)
            global_target_len = 1
            while global_target_len < raw_p95:
                global_target_len <<= 1
            logger.info(
                "Global uniform target_len = %d (p95=%d, rounded to next pow2, %d docs)",
                global_target_len, raw_p95, len(tc_global),
            )

        for shard_id in range(n_shards):
            shard_doc_indices = np.where(shard_assignments == shard_id)[0]
            if len(shard_doc_indices) == 0:
                continue

            shard_doc_ids = [int(doc_ids[i]) for i in shard_doc_indices]
            shard_offsets = [doc_offsets[i] for i in shard_doc_indices]
            token_counts = [int(e - s) for s, e in shard_offsets]
            tc_arr = np.array(token_counts, dtype=np.float64)

            shard_max_tokens = 0

            if uniform_shard_tokens:
                target_len = global_target_len
                shard_max_tokens = target_len

                uniform_chunks = []
                local_offsets = []
                pos = 0
                for idx in shard_doc_indices:
                    s, e = doc_offsets[idx]
                    doc_vec = all_vectors[s:e]
                    n_tok = doc_vec.shape[0]
                    if n_tok >= target_len:
                        piece = doc_vec[:target_len]
                    else:
                        pad = np.zeros((target_len - n_tok, dim), dtype=doc_vec.dtype)
                        piece = np.concatenate([doc_vec, pad], axis=0)
                    uniform_chunks.append(piece)
                    local_offsets.append((pos, pos + target_len))
                    pos += target_len
                shard_vectors = np.concatenate(uniform_chunks, axis=0)
                token_counts = [target_len] * len(shard_doc_ids)
            else:
                chunks = [all_vectors[s:e] for s, e in shard_offsets]
                shard_vectors = np.concatenate(chunks, axis=0) if chunks else np.empty((0, dim), dtype=all_vectors.dtype)
                local_offsets = []
                pos = 0
                for tc in token_counts:
                    local_offsets.append((pos, pos + tc))
                    pos += tc

            file_name = f"shard_{shard_id:05d}.safetensors"
            shard_path = self.shard_dir / file_name

            if compression == Compression.ROQ4 and roq_doc_codes is not None:
                # Build ROQ4 shard with pre-encoded codes and meta
                shard_codes_list = []
                shard_meta_list = []
                roq_offsets = []
                pos = 0
                for idx_in_shard, doc_global_idx in enumerate(shard_doc_indices):
                    codes_i = roq_doc_codes[doc_global_idx]
                    meta_i = roq_doc_meta[doc_global_idx]
                    n_tok = codes_i.shape[0]
                    if uniform_shard_tokens:
                        target = global_target_len
                        if n_tok >= target:
                            codes_i = codes_i[:target]
                            meta_i = meta_i[:target]
                        else:
                            pad_c = np.zeros((target - n_tok,) + codes_i.shape[1:], dtype=codes_i.dtype)
                            pad_m = np.zeros((target - n_tok,) + meta_i.shape[1:], dtype=meta_i.dtype)
                            codes_i = np.concatenate([codes_i, pad_c], axis=0)
                            meta_i = np.concatenate([meta_i, pad_m], axis=0)
                        n_tok = target
                    shard_codes_list.append(codes_i)
                    shard_meta_list.append(meta_i)
                    roq_offsets.append((pos, pos + n_tok))
                    pos += n_tok
                tensors = ShardStore.pack_shard_roq4(
                    shard_codes_list, shard_meta_list, roq_offsets, shard_doc_ids,
                )
            else:
                tensors = self._pack_shard(shard_vectors, local_offsets, shard_doc_ids, compression)
            st_save_np(tensors, str(shard_path))
            byte_size = shard_path.stat().st_size

            for row_index, (doc_id, (start, end)) in enumerate(zip(shard_doc_ids, local_offsets)):
                doc_index_rows.append((doc_id, shard_id, int(start), int(end), row_index))

            meta = ShardMeta(
                shard_id=shard_id,
                num_docs=len(shard_doc_ids),
                total_tokens=int(shard_vectors.shape[0]),
                centroid_ids=shard_to_centroid.get(shard_id, []),
                byte_size=byte_size,
                file_name=file_name,
                compression=compression.value,
                p50_tokens=float(np.median(tc_arr)) if len(tc_arr) else 0.0,
                p95_tokens=float(np.percentile(tc_arr, 95)) if len(tc_arr) else 0.0,
                shard_max_tokens=shard_max_tokens,
            )
            shard_metas.append(meta)

        tc_all = np.array(all_token_counts, dtype=np.float64)
        self.manifest = StoreManifest(
            num_shards=len(shard_metas),
            num_docs=len(doc_ids),
            dim=dim,
            total_tokens=int(all_vectors.shape[0]),
            avg_tokens_per_chunk=float(np.mean(tc_all)) if len(tc_all) else 0.0,
            p50_tokens=float(np.median(tc_all)) if len(tc_all) else 0.0,
            p95_tokens=float(np.percentile(tc_all, 95)) if len(tc_all) else 0.0,
            compression=compression.value,
            shards=shard_metas,
            global_target_len=global_target_len,
        )
        self.manifest.save(self.manifest_path)
        self._meta_cache = {m.shard_id: m for m in shard_metas}
        self._save_doc_index(doc_index_rows)

        logger.info(
            "ShardStore built: %d shards, %d docs, %d tokens, compression=%s, uniform=%s",
            len(shard_metas), len(doc_ids), int(all_vectors.shape[0]),
            compression.value, uniform_shard_tokens,
        )
        return self.manifest

    def _pack_shard(
        self,
        vectors: np.ndarray,
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        compression: Compression,
    ) -> Dict[str, np.ndarray]:
        offsets_arr = np.array(local_offsets, dtype=np.int64)
        ids_arr = np.array(doc_ids, dtype=np.int64)

        if compression == Compression.FP16:
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        if compression == Compression.INT8:
            vf = vectors.astype(np.float32)
            abs_max = np.abs(vf).max(axis=-1, keepdims=True)
            abs_max = np.where(abs_max == 0, 1.0, abs_max)
            scales = (abs_max / 127.0).astype(np.float32)
            quantized = np.clip(np.round(vf / scales), -127, 127).astype(np.int8)
            return {
                "embeddings": quantized,
                "scales": scales.squeeze(-1).astype(np.float32),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        if compression == Compression.ROQ4:
            logger.warning(
                "ROQ4 shard packing requires pre-encoded codes via pack_shard_roq4(); "
                "falling back to FP16 storage for this shard"
            )
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        raise ValueError(f"Unknown compression: {compression}")

    @staticmethod
    def pack_shard_roq4(
        codes_list: list,
        meta_list: list,
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
    ) -> Dict[str, np.ndarray]:
        """Pack ROQ4-encoded tokens into a shard file.

        codes_list/meta_list are per-doc arrays produced by RotationalQuantizer.
        Each codes entry is (n_tok, NB) uint8, each meta entry is (n_tok, 4) float32.
        """
        all_codes = np.concatenate(codes_list, axis=0) if codes_list else np.empty((0, 0), dtype=np.uint8)
        all_meta = np.concatenate(meta_list, axis=0) if meta_list else np.empty((0, 4), dtype=np.float32)
        return {
            "roq_codes": all_codes,
            "roq_meta": all_meta.astype(np.float32),
            "doc_offsets": np.array(local_offsets, dtype=np.int64),
            "doc_ids": np.array(doc_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _load_raw_shard(self, shard_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and cache decoded shard data. Returns (emb, offsets_tensor, ids_tensor)."""
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            return cached

        meta = self._meta_by_id(shard_id)
        data = st_load(str(self.shard_dir / meta.file_name), device="cpu")
        emb = self._decode_embeddings(meta, data)
        offsets_t = data["doc_offsets"]  # (N, 2) int64 tensor
        ids_t = data["doc_ids"]  # (N,) int64 tensor
        self._shard_cache.put(shard_id, emb, offsets_t, ids_t)
        return emb, offsets_t, ids_t

    def load_shard(self, shard_id: int, device: str = "cpu") -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        emb, offsets_t, ids_t = self._load_raw_shard(shard_id)
        if device != "cpu":
            emb = emb.to(device)
        offsets = [(int(offsets_t[i, 0]), int(offsets_t[i, 1])) for i in range(offsets_t.shape[0])]
        doc_ids = ids_t.tolist()
        return emb, offsets, doc_ids

    def load_docs_from_shard(
        self,
        shard_id: int,
        doc_ids: List[int],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """Load only specific documents from a shard (doc-selective access)."""
        if not doc_ids:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []

        emb, _offsets_t, _ids_t = self._load_raw_shard(shard_id)

        selected_ids = [
            int(d) for d in doc_ids
            if int(d) in self._doc_index and self._doc_index[int(d)].shard_id == shard_id
        ]
        if not selected_ids:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []

        pieces: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        pos = 0
        for doc_id in selected_ids:
            meta_doc = self._doc_index[doc_id]
            piece = emb[meta_doc.local_offset_start:meta_doc.local_offset_end]
            pieces.append(piece)
            offsets.append((pos, pos + piece.shape[0]))
            pos += int(piece.shape[0])

        merged = torch.cat(pieces, dim=0)
        if device != "cpu":
            merged = merged.to(device)
        return merged, offsets, selected_ids

    def load_shards(self, shard_ids: List[int], device: str = "cpu") -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        all_emb = []
        all_offsets = []
        all_ids = []
        global_offset = 0
        for sid in shard_ids:
            emb, offsets, doc_ids = self.load_shard(sid, device=device)
            for s, e in offsets:
                all_offsets.append((global_offset + s, global_offset + e))
            global_offset += emb.shape[0]
            all_emb.append(emb)
            all_ids.extend(doc_ids)
        if not all_emb:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []
        return torch.cat(all_emb, dim=0), all_offsets, all_ids

    def load_shard_to_pinned(
        self,
        shard_id: int,
        pinned_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        emb, offsets, doc_ids = self.load_shard(shard_id, device="cpu")
        n_tokens = emb.shape[0]
        if pinned_buffer is not None and pinned_buffer.shape[0] >= n_tokens:
            pinned_buffer[:n_tokens].copy_(emb)
            return pinned_buffer[:n_tokens], offsets, doc_ids
        pinned = torch.empty_like(emb, pin_memory=True)
        pinned.copy_(emb)
        return pinned, offsets, doc_ids

    def shard_ids(self) -> List[int]:
        return [s.shard_id for s in self.manifest.shards] if self.manifest else []

    def shard_doc_count(self, shard_id: int) -> int:
        return self._meta_by_id(shard_id).num_docs

    def doc_shard_id(self, doc_id: int) -> int:
        return self._doc_index[int(doc_id)].shard_id

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _meta_by_id(self, shard_id: int) -> ShardMeta:
        shard_id = int(shard_id)
        if shard_id not in self._meta_cache:
            raise KeyError(f"unknown shard_id={shard_id}")
        return self._meta_cache[shard_id]

    def _decode_embeddings(self, meta: ShardMeta, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if meta.compression == "int8":
            emb = data["embeddings"].float()
            scales = data["scales"].unsqueeze(-1)
            return (emb * scales).to(torch.float16)
        if meta.compression == "roq4" and "roq_codes" in data:
            # For ROQ4 we still return FP16 for the standard path (GpuCorpus, fetch).
            # The raw codes/meta are accessed via load_shard_roq4() for the ROQ scoring path.
            return data["roq_codes"].to(torch.float16)
        return data["embeddings"].to(torch.float16)

    def load_shard_roq4(self, shard_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load ROQ4-specific codes and meta for a shard.

        Returns (codes, meta, offsets_tensor, ids_tensor) or None if not ROQ4.
        """
        meta = self._meta_by_id(shard_id)
        if meta.compression != "roq4":
            return None
        data = st_load(str(self.shard_dir / meta.file_name), device="cpu")
        if "roq_codes" not in data:
            return None
        return data["roq_codes"], data["roq_meta"], data["doc_offsets"], data["doc_ids"]

    def _save_doc_index(self, rows: List[Tuple[int, int, int, int, int]]) -> None:
        rows = sorted(rows, key=lambda x: x[0])
        np.savez_compressed(
            self.doc_index_path,
            doc_ids=np.array([r[0] for r in rows], dtype=np.int64),
            shard_ids=np.array([r[1] for r in rows], dtype=np.int32),
            local_starts=np.array([r[2] for r in rows], dtype=np.int64),
            local_ends=np.array([r[3] for r in rows], dtype=np.int64),
            row_indices=np.array([r[4] for r in rows], dtype=np.int32),
        )
        self._load_doc_index()

    def _load_doc_index(self) -> None:
        data = np.load(self.doc_index_path)
        self._doc_index = {}
        for doc_id, shard_id, start, end, row_index in zip(
            data["doc_ids"], data["shard_ids"], data["local_starts"], data["local_ends"], data["row_indices"],
        ):
            self._doc_index[int(doc_id)] = DocMeta(
                doc_id=int(doc_id),
                shard_id=int(shard_id),
                local_offset_start=int(start),
                local_offset_end=int(end),
                row_index=int(row_index),
            )

    # ------------------------------------------------------------------
    # Disk-tier helpers
    # ------------------------------------------------------------------

    def drop_page_cache(self, shard_id: int) -> None:
        meta = self._meta_by_id(shard_id)
        path = self.shard_dir / meta.file_name
        try:
            fd = os.open(str(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass

    def drop_all_page_cache(self) -> None:
        if not self.manifest:
            return
        for meta in self.manifest.shards:
            self.drop_page_cache(meta.shard_id)
