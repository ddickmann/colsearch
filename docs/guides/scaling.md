# Scaling GEM Index to Millions of Documents

This guide explains how the GEM index scales, what the current limits are,
and the concrete engineering roadmap to support million-scale and beyond.

## Current Architecture (v1.0)

The GEM index is a graph-based multi-vector retrieval system.  Each document
is represented by multiple token-level vectors (e.g., 128 vectors from
ColBERT-Zero, or 1024 patch vectors from ColPali).  The index stores:

1. **Codebook** — a two-stage quantizer mapping each token vector to a
   centroid ID (u16).  Size: `O(n_fine * dim + n_fine^2)`.
2. **Flat codes** — one u16 centroid ID per token.  Size: `2 * n_tokens` bytes.
3. **Document profiles** — per-document centroid ID lists and routing info.
   Size: `4 * n_tokens + 4 * n_docs * ctop_r` bytes.
4. **Navigable graph** — CSR-compressed adjacency lists, one node per document.
   Size: `~4 * n_docs * max_degree` bytes.
5. **Raw vectors (optional)** — full float32 vectors for exact MaxSim
   reranking.  Size: `4 * n_tokens * dim` bytes.

### Memory Formulas

The **persistent index** (what stays in RAM after build, without raw vectors):

```
index_bytes ≈ 6 * n_tokens              # flat_codes + centroid_ids
            + 4 * n_docs * max_degree    # graph
            + 4 * n_fine^2               # codebook pairwise distances
            + 4 * n_fine * dim           # codebook centroids
            + 24 * n_docs               # doc_ids + offsets
```

The **build-time peak** (zero-copy builder, v1.1+):

```
build_bytes ≈ 4 * n_tokens * dim        # Python float32 array (zero-copy, no Rust copy)
            + index_bytes               # structures being constructed
            + ~0.5 * n_docs * max_degree # NN-Descent working memory (batched)
```

### Scaling Table

Assuming ColBERT-style embeddings (~128 tokens/doc, dim=128), n_fine=2048,
max_degree=48:

| Corpus Size | Tokens     | Persistent Index | Build-Time RAM | Status     |
|-------------|------------|------------------|----------------|------------|
| 7,500       | 950K       | ~9 MB            | ~1 GB          | Verified   |
| 75,000      | 9.5M       | ~91 MB           | ~8 GB          | Verified   |
| 100,000     | 13.3M      | ~127 MB          | ~10 GB         | Verified   |
| 1,000,000   | 128M       | ~1.2 GB          | ~68 GB         | v1.2       |
| 10,000,000  | 1.28B      | ~12 GB           | ~660 GB        | v2.0       |

The persistent index is remarkably compact: **~1.2 GB for 1M documents**.
The bottleneck is entirely build-time memory — the monolithic builder must
hold all float32 vectors in RAM simultaneously.

## Hard Limits

- **Document IDs**: stored as `u64` — no practical limit.
- **Token centroid codes**: stored as `u16` — max 65,535 fine centroids.
  Recommended: `n_fine = sqrt(n_tokens)`, so this supports up to ~4.3B tokens
  (~33M docs at 128 tokens/doc) before saturating.
- **Document count in graph**: stored as `u32` node IDs — max ~4.3B documents.
- **Posting list offsets**: stored as `u32` — max ~4.3B entries.

None of these limits are a concern below 10M documents.

## Recommended Hyperparameters by Corpus Size

| Parameter          | < 10K docs | 10K-100K   | 100K-1M    | > 1M        |
|--------------------|------------|------------|------------|-------------|
| `n_fine`           | 256        | 1024-2048  | 2048-4096  | 4096-8192   |
| `n_coarse`         | 16-32      | 64-128     | 128-256    | 256-512     |
| `max_degree`       | 32         | 32-48      | 48-64      | 64          |
| `ef_construction`  | 200        | 200-400    | 400        | 400-600     |
| `ctop_r`           | 2-3        | 3-4        | 4          | 4-6         |
| `max_kmeans_iter`  | 10         | 15-20      | 20         | 20          |

**Rule of thumb for n_fine**: `n_fine ≈ sqrt(total_tokens)`.  The auto-tuner
applies this when `n_fine=0` is passed to `build()`.

**Rule of thumb for n_coarse**: `n_coarse ≈ sqrt(n_docs)`.  The auto-tuner
applies this when `n_coarse=0` is passed to `build()`.  This is critical for
build speed: the graph construction algorithm (NN-Descent) runs per-cluster,
and its cost is roughly O(cluster_size^2).  Doubling `n_coarse` halves the
largest cluster size, giving ~4x faster graph construction.

## Production Configuration

For maximum quality at scale, use the full acceleration stack:

```python
seg.build(
    vectors, doc_ids, offsets,
    n_fine=0,             # auto: sqrt(n_tokens), clamped [64, 2048]
    n_coarse=0,           # auto: sqrt(n_docs), clamped [16, 1024]
    max_degree=48,
    ef_construction=400,
    ctop_r=4,
    use_emd=False,        # qCH proxy for construction (fast)
    dual_graph=True,      # GEM paper Algorithm 1
    store_raw_vectors=False,  # save memory; use GPU reranking instead
)
```

For search with GPU-accelerated reranking:

```python
# 1. Proxy search on graph (CPU)
candidates = seg.search(query_vecs, k=100, ef=2000, n_probes=8)

# 2. GPU MaxSim reranking (Triton kernel)
from voyager_index.kernels.triton_maxsim import fast_colbert_scores
reranked = rerank_with_triton(candidates, query_vecs, doc_vectors)
```

For storage-efficient deployment, ROQ 4-bit quantization reduces raw vector
storage by 8x while maintaining > 99% of MaxSim fidelity.

---

## Roadmap: Scaling Beyond 100K

### v1.0 (Current Release)

Monolithic in-memory build.  Verified at 75K documents (9.5M tokens) with
aggressive hyperparameters on a 25 GB container.

Key optimizations shipped:
- In-place L2 normalization (avoids 6.8 GB copy during codebook training)
- Reusable buffer in fine k-means (eliminates ~5 GB RSS bloat from glibc)
- Memory-aware evaluation pipeline with float16 storage and batched GPU
  ground truth

### v1.1 — Zero-Copy Build (Next Patch)

**Goal**: Cut build-time RAM in half by eliminating the Rust vector copy.

Currently, `seg.build()` copies all float32 vectors from Python (numpy) into
a Rust-owned `Vec<f32>`.  Both copies coexist during the entire build.
By borrowing the numpy buffer directly via a raw pointer, the Rust side
operates on the same memory without allocating a second copy.

| Metric              | v1.0           | v1.1          |
|---------------------|----------------|---------------|
| 75K build RAM       | ~10 GB         | ~5 GB         |
| 100K build RAM      | ~14 GB         | ~7 GB         |
| 1M build RAM        | ~131 GB        | ~66 GB        |

This is a single-file change in `lib.rs` (replace `.to_vec()` with unsafe
borrow when `store_raw_vectors=False`).  The numpy array is immutable during
the build call, so this is sound.

### v1.2 — Streaming Build (Next Minor Release)

**Goal**: Build million-scale indexes with < 5 GB working memory.

The key insight: the persistent index does not store raw vectors.  It only
needs centroid codes (u16 per token) and graph adjacency.  Therefore, we do
not need all vectors in RAM simultaneously.

**Architecture:**

```
Phase 1: Train codebook on a random sample
  - Load 1-5% of tokens (~1M from a 128M corpus) → ~0.5 GB
  - Run two-stage k-means → codebook (~17 MB)

Phase 2: Stream-assign centroid codes
  - Process documents in batches of 10K-50K
  - Per batch: load float32 vectors, assign codes, free vectors
  - Accumulate FlatDocCodes and DocProfiles incrementally
  - Peak memory: one batch of vectors (~0.5 GB) + accumulated codes

Phase 3: Incremental graph construction
  - Use MutableGemSegment (already implemented) for online insertion
  - Insert documents one-by-one or in small batches
  - Graph connectivity maintained via existing bridge repair
  - No need for all vectors — qCH scoring uses centroid codes only

Phase 4: Optional reranking index
  - Quantize vectors to ROQ 4-bit on the fly during Phase 2
  - Store quantized vectors to disk (memory-mapped at query time)
  - 8x storage reduction: 1M docs → ~8 GB on disk instead of 64 GB
```

**Working set**: ~2-5 GB regardless of corpus size.  The only scaling
factor is the accumulated code index (~6 bytes/token) and graph (~200
bytes/doc).

**What needs to be built:**
- Codebook training on sampled subset (straightforward refactor)
- Batched vector loading with streaming code assignment
- Integration with `MutableGemSegment` for incremental graph insertion
- Memory-mapped ROQ storage for query-time reranking

### v2.0 — Sharded Index (Major Release)

**Goal**: Support 10M+ documents across multiple index shards with
query-time merging.

**Architecture:**

```
                    ┌─────────────────┐
                    │   Query Router   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Shard 0  │  │ Shard 1  │  │ Shard 2  │
        │ 1M docs  │  │ 1M docs  │  │ 1M docs  │
        └──────────┘  └──────────┘  └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Result Merger   │
                    │  (MaxSim RRF)    │
                    └─────────────────┘
```

Each shard is a self-contained GEM index built via v1.2 streaming.
The query router fans out to all shards in parallel, and results are
merged using score-normalized fusion.

**What needs to be built:**
- Shard-aware `GemCollection` that manages multiple `GemSegment` instances
- Parallel query dispatch with async result collection
- Cross-shard score normalization (MaxSim scores are comparable across
  shards if codebooks are shared or scores are z-normalized)
- Shared codebook option: train one codebook on a global sample, reuse
  across shards for score comparability
- Shard balancing: assign documents to shards by cluster affinity to
  maximize intra-shard recall
- Optional: cross-shard graph edges for hard queries (graph-of-graphs)

**Scaling:**

| Shards | Docs/Shard | Total Docs | RAM/Shard | Total RAM |
|--------|------------|------------|-----------|-----------|
| 3      | 1M         | 3M         | 5 GB      | 15 GB     |
| 10     | 1M         | 10M        | 5 GB      | 50 GB     |
| 10     | 10M        | 100M       | 15 GB     | 150 GB    |

With v1.2 streaming build, each shard needs only ~5 GB to build.
Shards can be built in parallel across machines or sequentially on a
single node.

---

## Summary

The GEM index persistent footprint is inherently compact (~6 bytes per token
+ ~200 bytes per document for the graph).  The scaling challenge is
build-time memory, not runtime.  The v1.2 streaming builder eliminates
this bottleneck entirely by processing vectors in batches and never
holding the full corpus in RAM.  Combined with v2.0 sharding for
query-time parallelism, the architecture scales to hundreds of millions
of documents on commodity hardware.
