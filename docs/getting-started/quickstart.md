# Quickstart

## Installation

```bash
pip install voyager-index                # pure Python
pip install voyager-index[native]        # + prebuilt Rust kernels (recommended)
pip install voyager-index[native,gpu]    # + Triton GPU acceleration
```

## 5-Line Minimal Example

```python
import numpy as np
from voyager_index import Index

idx = Index("my_index", dim=128, engine="gem", seed_batch_size=64)
idx.add([np.random.randn(32, 128).astype(np.float32) for _ in range(100)])
results = idx.search(np.random.randn(32, 128).astype(np.float32), k=10)
```

## Create an Index

Each document is a matrix of token embeddings `(n_tokens, dim)`:

```python
import numpy as np
from voyager_index import Index

idx = Index(
    "my_index",
    dim=128,
    engine="gem",              # native GEM graph index
    seed_batch_size=64,        # train codebook after 64 docs
    n_fine=128,                # fine centroids (128–2048 depending on corpus)
    n_coarse=16,               # coarse clusters for routing
)
```

### Engine Selection

| Engine | When to use |
|---|---|
| `"gem"` | Multi-vector workloads (ColBERT, ColPali). Default when native crates are installed. |
| `"hnsw"` | Legacy single-vector workloads. |
| `"auto"` | Auto-detect: uses GEM if available, falls back to HNSW. |

### Builder Pattern (Advanced)

```python
from voyager_index import IndexBuilder

idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64, n_fine=256)
       .with_gpu_rerank(device="cuda")      # exact MaxSim reranking on GPU
       .with_wal(enabled=True)              # crash-safe write-ahead log
       .build())

# Or with ROQ 4-bit compressed reranking (~8x less memory):
idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64)
       .with_roq(bits=4)                    # ROQ 4-bit + fused Triton kernel
       .build())
```

## Add Documents

```python
n_docs = 100
embeddings = [np.random.randn(32, 128).astype(np.float32) for _ in range(n_docs)]
ids = list(range(n_docs))
payloads = [{"title": f"Document {i}"} for i in range(n_docs)]

idx.add(embeddings, ids=ids, payloads=payloads)
```

## Search

```python
query = np.random.randn(32, 128).astype(np.float32)
results = idx.search(query, k=10, ef=200)

for r in results:
    print(f"  Doc {r.doc_id}: score={r.score:.4f}, payload={r.payload}")
```

## Filtered Search

```python
results = idx.search(query, k=10, filters={"title": {"$contains": "science"}})
```

Qdrant-compatible filter operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`,
`$in`, `$nin`, `$exists`, `$contains`, `$and`, `$or`, `$not`.

Filters are applied natively during graph traversal via cluster-level bitmap
pruning — not post-hoc.

## Text-Based Search (with Embedding Function)

If you have a ColBERT model, pass it as `embedding_fn`:

```python
idx = Index("my_index", dim=128, engine="gem", embedding_fn=my_colbert_model)
idx.add_texts(["Document about AI", "Document about biology"])
results = idx.search_text("What is machine learning?", k=5)
```

The `embedding_fn` must implement `embed_documents(texts)` and `embed_query(text)`.

## Multimodal: PDF to Search Results

```python
from voyager_index.preprocessing import enumerate_renderable_documents, render_documents

# Step 1: Discover and render documents to page images
docs = enumerate_renderable_documents("./my_documents/")
pages = render_documents(docs["documents"], "./rendered_pages/")

# Step 2: Embed page images with a ColPali model
# page_embeddings = [model.embed_image(p["image_path"]) for p in pages["rendered"]]

# Step 3: Index and search
idx = Index("multimodal_index", dim=128, engine="gem", seed_batch_size=32)
# idx.add(page_embeddings, payloads=[{"source": p["source"], "page": p["page_number"]} ...])
# results = idx.search(query_embedding, k=10)
```

Supported input formats: PDF, DOCX, XLSX, PNG, JPG, WebP, GIF.

## Update and Delete

```python
idx.update_payload(0, {"title": "Updated Document 0"})
idx.upsert(new_vectors, ids=[0], payloads=[{"title": "Replaced Doc 0"}])
idx.delete([1, 2, 3])
```

## Scroll (Pagination)

```python
page = idx.scroll(limit=20, offset=0)
for r in page.results:
    print(f"  Doc {r.doc_id}: {r.payload}")
if page.next_offset:
    print(f"  Next page at offset {page.next_offset}")
```

## Snapshot and Restore

```python
idx.snapshot("backup.tar.gz")
```

## Cleanup

```python
idx.close()
```

Or use a context manager:

```python
with Index("my_index", dim=128, engine="gem", seed_batch_size=64) as idx:
    idx.add(embeddings, ids=ids)
    results = idx.search(query, k=10)
```

## Next Steps

- **[API Reference](../api/python.md)** — full method signatures and types
- **[GEM Guide](../guides/gem-native.md)** — production configuration and tuning
- **[Scaling Guide](../guides/scaling.md)** — memory planning for large corpora
- **[ColPali Guide](../guides/colpali.md)** — multimodal retrieval end-to-end
