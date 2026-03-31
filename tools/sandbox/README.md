# GPU-Native Centroid Screener for MaxSim Candidate Generation

## Overview

A GPU-resident index that approximates MaxSim scoring using centroid vectors extracted from each document's token embeddings. Instead of scoring all 256 tokens per document, it extracts K=4 representative centroids and performs a single batched matrix multiplication across the entire corpus in **<0.4ms** — enabling a screen-then-rerank pipeline that is **8–14x faster** than brute-force MaxSim with 72–92% end-to-end recall.

### Validated Performance (10K docs, 20 diverse multi-topic queries)

| Budget | E2E Latency | Speedup | E2E Recall@10 | Top-1 Agreement |
|--------|-------------|---------|---------------|-----------------|
| 200 | 0.76ms | **14.1x** | 72% | 90% |
| 300 | 0.91ms | **11.8x** | 79% | 90% |
| 500 | 1.22ms | **8.8x** | 91.5% | 95% |
| 1000 | 1.99ms | **5.4x** | 97% | 100% |

Baseline: brute-force `einsum` MaxSim on 10K docs = 10.70ms.  
Screening alone (no rerank): 0.33ms = **32x faster**.

---

## How It Works

### Core Algorithm

```
BUILD (offline, once per corpus):
  For each document (T=256 tokens, H=128 dim):
    1. L2-normalize all tokens
    2. Compute mean centroid (slot 0)
    3. Farthest-point-sampling for K-1 coverage medoids (slots 1..K-1)
    4. Store as (K, H) FP16 tensor on GPU

  Final index: (N_docs, K, H) contiguous FP16 tensor
  Memory: ~10MB per 10K docs (K=4, H=128)

SEARCH (per query):
  Input: query_embedding (S=32 tokens, H=128 dim)

  1. L2-normalize query tokens             → (S, H) FP16
  2. Flatten doc centroids                 → (N*K, H) FP16
  3. Compute sim = query @ centroids.T     → (S, N*K) — one matmul!
  4. Reshape to (S, N, K)
  5. MaxSim: max over K centroids per doc  → (S, N)
  6. Sum over S query tokens               → (N,)
  7. topk(budget)                          → candidate indices

  Time: 0.33ms for 10K docs, 0.6ms for 50K, 1.0ms for 100K

RERANK (on candidates only):
  8. Index full token embeddings: d_t[candidate_indices]
  9. Full-precision MaxSim on budget docs  → final ranking

  Time: 0.3ms (budget=100) to 1.6ms (budget=1000)
```

### Why It Works

MaxSim = for each query token, find the best-matching doc token, then sum. With 4 centroids instead of 256 tokens, we lose granularity but preserve the **ranking signal** well enough that reranking on the screened candidates recovers quality.

Key design choice: **queries stay at full precision** — only the document side is compressed to centroids. This is fundamentally different from RoQ bit-quantization which compresses both sides and loses more information.

### Critical Implementation Detail: Query Normalization

Both queries AND centroids must be L2-normalized before the dot product. The centroid extraction normalizes doc tokens, but the search must also normalize the query:

```python
q = F.normalize(q.float(), p=2, dim=-1).half()
```

Without this, rankings are incorrect (discovered during audit).

---

## File Reference

All files are in `sandbox/` — no production code was modified.

### `centroid_screener.py` — Core Component

The GPU-resident centroid index. This is the file you'll want to integrate.

**Key classes/methods:**

| Method | Purpose | Input → Output |
|--------|---------|----------------|
| `CentroidScreener(dim, max_centroids_per_doc, device)` | Constructor | — |
| `.extract_centroids(embedding, max_centroids)` | Extract K centroids from one doc | `(T, H) ndarray → (K, H) ndarray` |
| `.build(doc_ids, embeddings, lengths)` | Build GPU index from corpus | List of doc embeddings |
| `.search(query, budget, mode)` | High-level search returning doc_ids | `(S, H) → (doc_ids, profile)` |
| `.search_multi_centroid(query, budget, allowed_indices)` | **GPU-only search returning GPU indices** | `(S, H) tensor → (topk_idx, topk_scores)` |

**`search_multi_centroid` is the production-path method** — it takes GPU tensors in and returns GPU tensors out, with zero Python overhead. The high-level `search()` wraps it with doc_id mapping for convenience.

**Internal state:**
- `._centroids`: `(N, K, H)` FP16 tensor on GPU
- `._centroid_mask`: `(N, K)` FP16 mask (for docs with <K centroids)
- `._doc_ids`: list mapping index → doc_id

### `binary_screener.py` — Binary Pre-filter (Tier 0)

Ultra-fast binary hash screener using Triton `popcount`. Reduces 1M→10K candidates in ~0.1ms. Only useful at very large scale (>10K docs). Uses the same `_popc` intrinsic as your `triton_roq.py`.

### `hybrid_funnel.py` — Multi-Tier Pipeline

Chains binary → single-centroid → multi-centroid screening into a funnel. Auto-skips tiers when the corpus is small enough. Good for production where corpus size varies.

### `benchmark_screener.py` — Benchmark Harness

Generates synthetic ColBERT-style data with **multi-topic queries** (3–5 topics per query, matching real ColPali diversity). Measures recall@k, top-1 agreement, MRR, and latency with proper `cuda.synchronize()`.

```bash
# Run the full benchmark
cd sandbox && python benchmark_screener.py \
    --doc-counts 1000,5000,10000 \
    --candidate-budget 100 \
    --n-queries 20 \
    --output results.json
```

### `test_screener.py` — Unit Tests (12 tests, all pass)

```bash
cd sandbox && python -m pytest test_screener.py -v
```

Tests cover: centroid extraction shapes, correct document preference, CPU/GPU parity, binary code packing, funnel pipeline, and recall@100 quality assertion.

---

## Integration Guide

### Step 1: Replace Prototype Sidecar Screening

The screener replaces `PrototypeScreeningIndex.search()` in the inference pipeline. The centroid extraction logic is identical (same farthest-point-sampling as `extract_prototypes()`), but search uses GPU matmul instead of HNSW.

**Current flow** (in `prototype_screening.py`):
```python
# CPU-side HNSW search → Python dict aggregation → sorted ranking
for prototype in query_prototypes:
    results = self.manager.search(prototype, k=budget.per_prototype_k)
    for prototype_id, score in results:
        score_map[doc_id] += score
ranked = sorted(score_map.items(), ...)
```

**New flow** (with centroid screener):
```python
# Single GPU matmul → topk → done
idx, scores = self.centroid_screener.search_multi_centroid(
    query_embedding_gpu,    # (S, H) FP16 tensor on GPU
    candidate_budget=500,   # how many candidates to pass to reranker
)
# idx is a GPU tensor of doc indices → use directly for reranking
candidate_docs = self.doc_embeddings[idx]  # (budget, T, H)
final_scores = fused_maxsim_colbert_scores(query, candidate_docs, ...)
```

### Step 2: Build at Index Time

In `PrototypeScreeningIndex.rebuild()`, replace the HNSW build:

```python
# Instead of:
self.manager.add(stacked_prototypes, ids=ids, payloads=payloads)

# Do:
self.centroid_screener = CentroidScreener(dim=self.dim, max_centroids_per_doc=4)
self.centroid_screener.build(doc_ids, embeddings, lengths)
# Optionally save: torch.save(screener._centroids, "centroids.pt")
```

### Step 3: Use Triton Kernel for Reranking

The E2E benchmark currently uses `einsum` for reranking (slow). For production, use your existing Triton kernel:

```python
from voyager_index import fast_colbert_scores

# After screening:
candidate_docs = doc_store[candidate_indices]
scores = fast_colbert_scores(query.unsqueeze(0), candidate_docs)
```

This should improve E2E speedup by 2–3x since the Triton kernel is much faster than einsum at small doc counts.

### Step 4: Choose Budget Based on Use Case

| Use Case | Budget | Expected Recall@10 | Latency |
|----------|--------|-------------------|---------|
| Speed-critical (autocomplete, typeahead) | 200 | ~72% | <1ms |
| Balanced (search API) | 500 | ~92% | ~1.2ms |
| Quality-critical (single-query RAG) | 1000 | ~97% | ~2ms |

For adaptive budgets, use query dispersion (already computed in `plan_budget()`):
- Low dispersion (focused query) → budget=200
- High dispersion (multi-aspect query) → budget=500–1000

### Step 5: Persistence (TODO)

The centroid tensor needs save/load for production. Minimal implementation:

```python
# Save
torch.save({
    'centroids': screener._centroids,
    'mask': screener._centroid_mask,
    'doc_ids': screener._doc_ids,
    'dim': screener.dim,
    'max_centroids': screener.max_centroids_per_doc,
}, path)

# Load
data = torch.load(path)
screener = CentroidScreener(dim=data['dim'], max_centroids_per_doc=data['max_centroids'])
screener._centroids = data['centroids'].to(device)
screener._centroid_mask = data['mask'].to(device)
screener._doc_ids = data['doc_ids']
screener._n_docs = len(data['doc_ids'])
```

---

## Key Numbers to Remember

- **Screening latency**: 0.33ms / 10K docs, 0.6ms / 50K, 1.0ms / 100K
- **GPU memory**: 10MB / 10K docs (K=4, H=128, FP16)
- **Build time**: ~1.5s / 10K docs (single-threaded, can be parallelized)
- **Sweet spot**: budget=500, 8.8x speedup, 91.5% recall@10
- **Centroid count**: K=4 is optimal; K=8/16 doesn't improve E2E recall

## Audit Log

This prototype went through a rigorous self-audit that caught and fixed:

1. **Missing query normalization** — rankings were wrong without L2-norm on queries
2. **Inflated recall from single-topic synthetic queries** — switched to multi-topic (3–5 topics)
3. **Python overhead in E2E timing** — switched to GPU-tensor-in/out path
4. **Slow binary code packing** — vectorized with positional weights
5. **Funnel tiers never activating** — lowered activation thresholds
6. **Misleading binary screener naming** — clarified it's centroid similarity, not MaxSim

Pre-audit "100% recall" was **fake**. Post-audit 91.5% recall at 8.8x speedup is **real**.
