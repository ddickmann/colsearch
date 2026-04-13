"""Add centroid codes + packed residuals to an existing shard index.

Reads all embeddings, clusters token embeddings via K-means, assigns
per-token centroid codes, computes per-dimension scalar-quantized residuals,
and rewrites each shard file with 'centroid_codes' and 'packed_residuals'
tensors alongside the existing 'embeddings'.

Codec metadata (bucket_cutoffs, bucket_weights, nbits) is saved as
codec_meta.npz in the index directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open
from safetensors.numpy import save_file as st_save_np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX = "/root/.cache/shard-bench/index_100000_fp16_proxy_grouped_lemur_uniform"
N_CENTROIDS = 1024
MAX_TRAIN_TOKENS = 500_000
MAX_RESIDUAL_SAMPLE = 2_000_000


def quantize_residuals(residuals_flat: np.ndarray, bucket_cutoffs: np.ndarray, nbits: int) -> np.ndarray:
    """Quantize a flat residual array into bucket indices (0..2^nbits-1)."""
    indices = np.searchsorted(bucket_cutoffs, residuals_flat).astype(np.uint8)
    return indices


def pack_residual_indices(indices_2d: np.ndarray, nbits: int, dim: int) -> np.ndarray:
    """Pack per-dimension bucket indices into bytes (MSB-first within each byte)."""
    n_tokens = indices_2d.shape[0]
    packed_dim = dim * nbits // 8
    packed = np.zeros((n_tokens, packed_dim), dtype=np.uint8)

    keys_per_byte = 8 // nbits
    for token_i in range(n_tokens):
        row = indices_2d[token_i]
        for byte_j in range(packed_dim):
            val = np.uint8(0)
            for k in range(keys_per_byte):
                d_idx = byte_j * keys_per_byte + k
                bucket = row[d_idx]
                shift = (keys_per_byte - 1 - k) * nbits
                val |= np.uint8(bucket) << np.uint8(shift)
            packed[token_i, byte_j] = val
    return packed


def pack_residual_indices_fast(indices_2d: np.ndarray, nbits: int, dim: int) -> np.ndarray:
    """Vectorized packing of per-dimension bucket indices into bytes."""
    n_tokens = indices_2d.shape[0]
    keys_per_byte = 8 // nbits
    packed_dim = dim * nbits // 8

    reshaped = indices_2d.reshape(n_tokens, packed_dim, keys_per_byte).astype(np.uint32)
    shifts = np.array([(keys_per_byte - 1 - k) * nbits for k in range(keys_per_byte)], dtype=np.uint32)
    packed = np.zeros((n_tokens, packed_dim), dtype=np.uint8)
    for k in range(keys_per_byte):
        packed |= (reshaped[:, :, k] << shifts[k]).astype(np.uint8)
    return packed


def main():
    parser = argparse.ArgumentParser(description="Add centroid codes + packed residuals")
    parser.add_argument("index_dir", nargs="?", default=DEFAULT_INDEX, help="Index directory")
    parser.add_argument("--nbits", type=int, default=2, choices=[1, 2, 4], help="Bits per residual dimension")
    parser.add_argument("--n-centroids", type=int, default=N_CENTROIDS)
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip K-means if centroids.npy exists")
    args = parser.parse_args()

    INDEX_DIR = Path(args.index_dir)
    SHARD_DIR = INDEX_DIR / "shards"
    nbits = args.nbits
    n_centroids = args.n_centroids

    t0 = time.perf_counter()

    shard_files = sorted(SHARD_DIR.glob("shard_*.safetensors"))
    logger.info("Found %d shard files in %s", len(shard_files), SHARD_DIR)

    centroids_path = INDEX_DIR / "centroids.npy"
    if args.skip_kmeans and centroids_path.exists():
        logger.info("Loading existing centroids from %s", centroids_path)
        centroids = np.load(str(centroids_path)).astype(np.float32)
        dim = centroids.shape[1]
        total_tokens = 0
        for sf in shard_files:
            with safe_open(str(sf), framework="numpy") as f:
                total_tokens += f.get_tensor("embeddings").shape[0]
    else:
        logger.info("Pass 1: sampling tokens for K-means...")
        all_sample_vecs = []
        sampled_so_far = 0
        total_tokens = 0
        for sf in shard_files:
            with safe_open(str(sf), framework="numpy") as f:
                emb = f.get_tensor("embeddings")
                n = emb.shape[0]
                total_tokens += n
                if sampled_so_far >= MAX_TRAIN_TOKENS:
                    continue
                need = MAX_TRAIN_TOKENS - sampled_so_far
                if n <= need:
                    all_sample_vecs.append(emb.astype(np.float32))
                    sampled_so_far += n
                else:
                    idx = np.random.RandomState(42).choice(n, need, replace=False)
                    all_sample_vecs.append(emb[idx].astype(np.float32))
                    sampled_so_far += need

        train_data = np.concatenate(all_sample_vecs, axis=0)
        dim = train_data.shape[1]
        logger.info("Training K-means: %d tokens (sampled from %d total), dim=%d, K=%d",
                    len(train_data), total_tokens, dim, n_centroids)

        kmeans = faiss.Kmeans(dim, n_centroids, niter=20, verbose=True,
                              gpu=torch.cuda.is_available())
        kmeans.train(train_data)
        centroids = kmeans.centroids.copy()
        np.save(str(centroids_path), centroids)
        logger.info("Centroids saved to %s", centroids_path)
        del train_data, all_sample_vecs

    assign_index = faiss.IndexFlatIP(dim)
    assign_index.add(centroids)

    # --- Compute bucket cutoffs and weights from sampled residuals ---
    logger.info("Sampling residuals for bucket table computation (nbits=%d)...", nbits)
    residual_samples = []
    residual_sampled = 0
    for sf in shard_files:
        if residual_sampled >= MAX_RESIDUAL_SAMPLE:
            break
        with safe_open(str(sf), framework="numpy") as f:
            emb = f.get_tensor("embeddings").astype(np.float32)
        _, I = assign_index.search(emb, 1)
        codes_i = I[:, 0]
        residuals = emb - centroids[codes_i]
        residual_samples.append(residuals.ravel())
        residual_sampled += emb.shape[0]

    all_residuals_flat = np.concatenate(residual_samples)
    logger.info("Computing bucket tables from %d residual scalars...", len(all_residuals_flat))

    n_buckets = 1 << nbits
    cutoff_quantiles = np.array([i / n_buckets for i in range(1, n_buckets)], dtype=np.float64)
    weight_quantiles = np.array([(i + 0.5) / n_buckets for i in range(n_buckets)], dtype=np.float64)

    bucket_cutoffs = np.quantile(all_residuals_flat, cutoff_quantiles).astype(np.float32)
    bucket_weights = np.quantile(all_residuals_flat, weight_quantiles).astype(np.float32)

    logger.info("Bucket cutoffs (%d): %s", len(bucket_cutoffs), bucket_cutoffs)
    logger.info("Bucket weights (%d): %s", len(bucket_weights), bucket_weights)

    np.savez(
        str(INDEX_DIR / "codec_meta.npz"),
        bucket_cutoffs=bucket_cutoffs,
        bucket_weights=bucket_weights,
        nbits=np.array([nbits], dtype=np.int32),
    )
    logger.info("Codec metadata saved to %s", INDEX_DIR / "codec_meta.npz")
    del all_residuals_flat, residual_samples

    packed_dim = dim * nbits // 8
    logger.info("Pass 2: assigning codes + packing residuals (packed_dim=%d)...", packed_dim)
    for i, sf in enumerate(shard_files):
        with safe_open(str(sf), framework="numpy") as f:
            tensor_names = list(f.keys())
            tensors = {}
            for name in tensor_names:
                tensors[name] = f.get_tensor(name)

        emb = tensors["embeddings"]
        emb_f32 = emb.astype(np.float32)
        n_tokens = emb_f32.shape[0]

        _, I = assign_index.search(emb_f32, 1)
        codes = I[:, 0].astype(np.uint16)
        tensors["centroid_codes"] = codes

        residuals = emb_f32 - centroids[codes.astype(np.int64)]
        bucket_indices = quantize_residuals(residuals.ravel(), bucket_cutoffs, nbits)
        bucket_indices_2d = bucket_indices.reshape(n_tokens, dim)
        packed = pack_residual_indices_fast(bucket_indices_2d, nbits, dim)
        tensors["packed_residuals"] = packed

        st_save_np(tensors, str(sf))

        if (i + 1) % 50 == 0 or i == len(shard_files) - 1:
            logger.info("  Processed %d/%d shards (%d tokens in this shard)",
                        i + 1, len(shard_files), n_tokens)

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1fs. Centroid codes + packed residuals added to all %d shards.", elapsed, len(shard_files))

    sf0 = shard_files[0]
    with safe_open(str(sf0), framework="numpy") as f:
        codes_check = f.get_tensor("centroid_codes")
        emb_check = f.get_tensor("embeddings")
        packed_check = f.get_tensor("packed_residuals")
    logger.info("Verification: shard_0 embeddings %s, centroid_codes %s (dtype=%s), packed_residuals %s (dtype=%s)",
                emb_check.shape, codes_check.shape, codes_check.dtype,
                packed_check.shape, packed_check.dtype)
    assert codes_check.dtype == np.uint16
    assert codes_check.shape[0] == emb_check.shape[0]
    assert packed_check.dtype == np.uint8
    assert packed_check.shape == (emb_check.shape[0], packed_dim)
    logger.info("Verification passed!")


if __name__ == "__main__":
    main()
