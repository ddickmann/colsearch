"""
ShardStore: safetensors-backed shard storage with contiguous centroid-grouped layout.

Each shard is one safetensors file containing all token embeddings for a group
of documents that share similar centroid assignments. Documents within a shard
are packed contiguously so that fetching a shard is a single sequential read.

Storage modes:
- FP16: raw float16 embeddings (2 bytes/value)
- INT8: per-document absmax quantized int8 (1 byte/value + scales)
- ROQ4: 4-bit rotational quantization codes (0.5 bytes/value + metadata)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from safetensors.numpy import save_file as st_save_np
    from safetensors.torch import save_file as st_save, load_file as st_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

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
    shard_max_tokens: int = 0  # 0 = variable length; >0 = all docs padded to this


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

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StoreManifest":
        with open(path) as f:
            d = json.load(f)
        d["shards"] = [ShardMeta(**s) for s in d["shards"]]
        return cls(**d)


class ShardStore:
    """
    Write and read safetensors shard files.

    Build-time: pack documents into shards grouped by centroid assignment.
    Query-time: load specific shards by ID, optionally into pinned memory.
    """

    def __init__(self, root_path: Path):
        self.root = Path(root_path)
        self.shard_dir = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"
        self.manifest: Optional[StoreManifest] = None

        if self.manifest_path.exists():
            self.manifest = StoreManifest.load(self.manifest_path)

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
    ) -> StoreManifest:
        """
        Pack documents into shard files.

        Args:
            all_vectors: (total_tokens, dim) flattened token embeddings
            doc_offsets: per-doc (start, end) into all_vectors
            doc_ids: external document IDs
            shard_assignments: per-doc shard ID array of length len(doc_ids)
            n_shards: total shard count
            dim: embedding dimension
            compression: storage compression mode
            centroid_to_shard: mapping from centroid_id to shard_id (for metadata)
            uniform_shard_tokens: if True, truncate/pad all docs within a shard
                to the shard's p95 token count for zero-copy view() at query time
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required: pip install safetensors")

        self.shard_dir.mkdir(parents=True, exist_ok=True)

        shard_to_centroid: Dict[int, List[int]] = {}
        if centroid_to_shard:
            for cid, sid in centroid_to_shard.items():
                shard_to_centroid.setdefault(sid, []).append(cid)

        all_token_counts = [e - s for s, e in doc_offsets]
        shard_metas: List[ShardMeta] = []

        for shard_id in range(n_shards):
            mask = shard_assignments == shard_id
            shard_doc_indices = np.where(mask)[0]

            if len(shard_doc_indices) == 0:
                continue

            shard_doc_ids = [doc_ids[i] for i in shard_doc_indices]
            shard_offsets = [doc_offsets[i] for i in shard_doc_indices]
            token_counts = [e - s for s, e in shard_offsets]
            tc_arr = np.array(token_counts, dtype=np.float64)

            shard_max_tokens = 0

            if uniform_shard_tokens:
                # Truncate/pad all docs to shard p95 for zero-copy view()
                target_len = int(np.ceil(np.percentile(tc_arr, 95))) if len(tc_arr) else 1
                target_len = max(target_len, 1)
                shard_max_tokens = target_len

                uniform_chunks = []
                for idx in shard_doc_indices:
                    s, e = doc_offsets[idx]
                    doc_vec = all_vectors[s:e]
                    n_tok = doc_vec.shape[0]
                    if n_tok >= target_len:
                        uniform_chunks.append(doc_vec[:target_len])
                    else:
                        pad = np.zeros((target_len - n_tok, dim), dtype=doc_vec.dtype)
                        uniform_chunks.append(np.concatenate([doc_vec, pad], axis=0))
                shard_vectors = np.concatenate(uniform_chunks, axis=0)

                local_offsets = []
                for i in range(len(shard_doc_ids)):
                    s = i * target_len
                    local_offsets.append((s, s + target_len))
                token_counts = [target_len] * len(shard_doc_ids)
            else:
                chunks = [all_vectors[s:e] for s, e in shard_offsets]
                shard_vectors = np.concatenate(chunks, axis=0)

                local_offsets = []
                pos = 0
                for tc in token_counts:
                    local_offsets.append((pos, pos + tc))
                    pos += tc

            file_name = f"shard_{shard_id:05d}.safetensors"
            shard_path = self.shard_dir / file_name

            tensors = self._pack_shard(
                shard_vectors, local_offsets, shard_doc_ids, dim, compression,
            )
            st_save_np(tensors, str(shard_path))

            byte_size = shard_path.stat().st_size

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
            avg_tokens_per_chunk=float(np.mean(tc_all)),
            p50_tokens=float(np.median(tc_all)),
            p95_tokens=float(np.percentile(tc_all, 95)),
            compression=compression.value,
            shards=shard_metas,
        )
        self.manifest.save(self.manifest_path)
        logger.info(
            "ShardStore built: %d shards, %d docs, %d tokens, compression=%s, uniform=%s",
            len(shard_metas), len(doc_ids), all_vectors.shape[0],
            compression.value, uniform_shard_tokens,
        )
        return self.manifest

    def _pack_shard(
        self,
        vectors: np.ndarray,
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        dim: int,
        compression: Compression,
    ) -> Dict[str, np.ndarray]:
        """Pack a single shard's data into a dict for safetensors serialization."""
        offsets_arr = np.array(local_offsets, dtype=np.int64)
        ids_arr = np.array(doc_ids, dtype=np.int64)

        if compression == Compression.FP16:
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        elif compression == Compression.INT8:
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
        elif compression == Compression.ROQ4:
            # ROQ4 requires external quantizer; store raw FP16 as fallback
            # and let the build pipeline provide pre-quantized codes
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        else:
            raise ValueError(f"Unknown compression: {compression}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_shard(
        self,
        shard_id: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """
        Load a shard's embeddings, doc offsets, and doc IDs.

        Returns:
            (embeddings, doc_offsets, doc_ids) where embeddings is
            (total_tokens, dim) on the target device.
        """
        meta = self._meta_by_id(shard_id)
        shard_path = self.shard_dir / meta.file_name

        # Use torch loader directly — avoids numpy intermediary + copy
        data = st_load(str(shard_path), device="cpu")

        if meta.compression == "int8":
            emb = data["embeddings"].float()
            scales = data["scales"].unsqueeze(-1)
            emb = (emb * scales).to(torch.float16)
        else:
            emb = data["embeddings"].to(torch.float16)

        if device != "cpu":
            emb = emb.to(device)

        offset_tuples = [(int(s), int(e)) for s, e in data["doc_offsets"].tolist()]
        doc_ids = data["doc_ids"].tolist()
        return emb, offset_tuples, doc_ids

    def load_shards(
        self,
        shard_ids: List[int],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """Load multiple shards, concatenating their contents."""
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
            return torch.empty(0, dim, dtype=torch.float16), [], []

        return torch.cat(all_emb, dim=0), all_offsets, all_ids

    def load_shard_to_pinned(
        self,
        shard_id: int,
        pinned_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """Load shard into a pre-allocated pinned memory buffer."""
        meta = self._meta_by_id(shard_id)
        shard_path = self.shard_dir / meta.file_name
        data = st_load(str(shard_path), device="cpu")

        offset_tuples = [(int(s), int(e)) for s, e in data["doc_offsets"].tolist()]
        doc_ids = data["doc_ids"].tolist()

        if meta.compression == "int8":
            emb = data["embeddings"].float()
            scales = data["scales"].unsqueeze(-1)
            emb = (emb * scales).to(torch.float16)
        else:
            emb = data["embeddings"].to(torch.float16)

        n_tokens = emb.shape[0]

        if pinned_buffer is not None and pinned_buffer.shape[0] >= n_tokens:
            pinned_buffer[:n_tokens].copy_(emb)
            return pinned_buffer[:n_tokens], offset_tuples, doc_ids

        pinned = torch.empty_like(emb, pin_memory=True)
        pinned.copy_(emb)
        return pinned, offset_tuples, doc_ids

    def shard_ids(self) -> List[int]:
        if not self.manifest:
            return []
        return [s.shard_id for s in self.manifest.shards]

    def shard_doc_count(self, shard_id: int) -> int:
        return self._meta_by_id(shard_id).num_docs

    def shard_byte_size(self, shard_id: int) -> int:
        return self._meta_by_id(shard_id).byte_size

    def _meta_by_id(self, shard_id: int) -> ShardMeta:
        if not self.manifest:
            raise RuntimeError("No manifest loaded")
        if not hasattr(self, "_meta_cache"):
            self._meta_cache = {s.shard_id: s for s in self.manifest.shards}
        meta = self._meta_cache.get(shard_id)
        if meta is None:
            raise KeyError(f"Shard {shard_id} not found in manifest ({len(self._meta_cache)} shards)")
        return meta

    # ------------------------------------------------------------------
    # Disk-tier helpers
    # ------------------------------------------------------------------

    def drop_page_cache(self, shard_id: int):
        """Advise kernel to drop page cache for a shard file (cold-read testing)."""
        meta = self._meta_by_id(shard_id)
        path = self.shard_dir / meta.file_name
        try:
            fd = os.open(str(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass

    def drop_all_page_cache(self):
        if not self.manifest:
            return
        for meta in self.manifest.shards:
            self.drop_page_cache(meta.shard_id)
