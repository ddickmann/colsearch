# Python API Reference

## `voyager_index.Index`

The primary interface for creating and querying indexes.

### Constructor

```python
Index(
    path: str,
    dim: int,
    *,
    engine: str = "auto",        # "gem", "hnsw", or "auto"
    mode: str | None = None,     # "colbert", "colpali", or None
    embedding_fn: Any = None,    # auto-embedding callable
    n_fine: int = 256,           # fine centroids (codebook size)
    n_coarse: int = 32,          # coarse clusters
    max_degree: int = 32,        # max graph neighbors (M)
    ef_construction: int = 200,  # beam width during build
    n_probes: int = 4,           # clusters to probe during search
    enable_wal: bool = True,     # write-ahead log
    # GEM keyword args (passed through):
    rerank_device: str = None,   # "cuda" for GPU MaxSim reranking
    roq_rerank: bool = False,    # enable ROQ 4-bit reranking
    roq_bits: int = 4,           # ROQ quantization bit width
    use_emd: bool = False,       # qEMD for graph construction
    dual_graph: bool = True,     # per-cluster + cross-cluster bridges
    warmup_kernels: bool = True, # vLLM-style Triton pre-compilation
    seed_batch_size: int = 256,  # docs before codebook training
)
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `add` | `(vectors, *, ids=None, payloads=None)` | Add documents (3D array or list of 2D) |
| `add_batch` | `(vectors, *, ids=None, payloads=None)` | Alias for `add()` |
| `add_texts` | `(texts, *, ids=None, payloads=None)` | Add documents by text using `embedding_fn` |
| `upsert` | `(vectors, *, ids, payloads=None)` | Insert or replace documents by ID |
| `search` | `(query, k=10, *, ef=100, n_probes=4, filters=None, explain=False)` | Multi-vector search → `List[SearchResult]` |
| `search_text` | `(text, k=10, *, ef=100, filters=None, explain=False)` | Text search using `embedding_fn` |
| `search_batch` | `(queries, k=10, *, ef=100, n_probes=4, filters=None)` | Batch search → `List[List[SearchResult]]` |
| `delete` | `(ids)` | Delete documents by ID |
| `update_payload` | `(doc_id, payload)` | Update document metadata |
| `get` | `(ids)` | Retrieve payloads by ID → `List[Optional[Dict]]` |
| `scroll` | `(limit=100, offset=0, *, filters=None)` | Paginated iteration → `ScrollPage` |
| `stats` | `()` | Summary statistics → `IndexStats` |
| `snapshot` | `(output_path)` | Create tarball backup |
| `flush` | `()` | Force pending writes to disk |
| `close` | `()` | Release resources |
| `set_metrics_hook` | `(hook: Callable[[str, float], None])` | Register metrics callback |

**Filter behavior**: `search()` applies filters natively during graph traversal
via cluster-level bitmap pruning (Roaring bitmaps). `search_batch()` applies
filters post-hoc in Python. Use `search()` for production filter workloads.

### Properties

| Property | Type | Description |
|---|---|---|
| `path` | `Path` | Index directory |
| `dim` | `int` | Vector dimension |
| `engine` | `str` | Backend engine name (`"gem"` or `"hnsw"`) |

---

## `voyager_index.IndexBuilder`

Fluent builder for custom Index configuration.

```python
from voyager_index import IndexBuilder

idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64, n_fine=128)
       .with_gpu_rerank(device="cuda")
       .with_wal(enabled=True)
       .with_quantization(n_fine=256, n_coarse=32)
       .build())
```

| Method | Description |
|---|---|
| `with_gem(**kwargs)` | Select GEM engine with optional overrides |
| `with_hnsw(**kwargs)` | Select HNSW engine (legacy) |
| `with_wal(enabled=True)` | Enable/disable write-ahead log |
| `with_quantization(n_fine=256, n_coarse=32)` | Configure codebook |
| `with_gpu_rerank(device="cuda")` | Enable GPU MaxSim reranking |
| `with_roq(bits=4, device="cuda")` | Enable ROQ compressed reranking |
| `build()` | Create and return the `Index` |

---

## Data Classes

### `voyager_index.SearchResult`

```python
@dataclass
class SearchResult:
    doc_id: int                                  # document ID
    score: float                                 # similarity score
    payload: Optional[Dict[str, Any]] = None     # document metadata
    token_scores: Optional[List[float]] = None   # per-query-token attribution (explain=True)
    matched_tokens: Optional[List[int]] = None   # matched token indices (explain=True)
```

### `voyager_index.ScrollPage`

```python
@dataclass
class ScrollPage:
    results: List[SearchResult]
    next_offset: Optional[int] = None  # None when no more pages
```

### `voyager_index.IndexStats`

```python
@dataclass
class IndexStats:
    total_documents: int   # total docs across all segments
    sealed_segments: int   # number of sealed read-only segments
    active_documents: int  # docs in the mutable segment
    dim: int               # vector dimension
    engine: str            # "gem" or "hnsw"
```

---

## Configuration

### `voyager_index.BM25Config`

```python
@dataclass
class BM25Config:
    k1: float = 1.5       # term frequency saturation (1.2–2.0)
    b: float = 0.75        # length normalization
    epsilon: float = 0.25  # IDF floor
```

### `voyager_index.FusionConfig`

```python
@dataclass
class FusionConfig:
    strategy: str = "rrf"                      # "rrf", "weighted", "dbsf"
    weights: Optional[Dict[str, float]] = None # per-engine weights (sum to 1.0)
    normalization: str = "minmax"              # "minmax", "zscore", "none"
    top_k: int = 10
    min_score: float = 0.0
```

### `voyager_index.IndexConfig`

Controls ColBERT index behavior including three modes:

- **Real-time** (< 1K docs): pure Triton, cached in VRAM, 50+ QPS
- **High quality** (1K–50K docs): Triton + mmap streaming, exact search
- **Balanced** (> 50K docs): GEM graph traversal + MaxSim rerank

### `voyager_index.Neo4jConfig`

```python
@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_hop_distance: int = 2
    relationship_types: Optional[List[str]] = None
```

---

## `voyager_index.SearchPipeline`

Hybrid dense+sparse (BM25 + vector) retrieval pipeline with RRF fusion.

```python
from voyager_index import SearchPipeline, BM25Config, FusionConfig

pipeline = SearchPipeline(
    index=idx,
    bm25_config=BM25Config(k1=1.5, b=0.75),
    fusion_config=FusionConfig(strategy="rrf", top_k=10),
)
results = pipeline.search("query text", k=10)
```

---

## `voyager_index.ColbertIndex`

Production-grade ColBERT index with automatic scaling. Selects the optimal
search strategy (Triton-only, mmap, or GEM graph) based on corpus size.

```python
from voyager_index import ColbertIndex

cidx = ColbertIndex(path="colbert_idx", dim=128)
cidx.add_documents(embeddings, ids)
results = cidx.search(query, k=10)
```

---

## `voyager_index.ColPaliEngine`

Visual document search engine using ColPali-family models.

```python
from voyager_index import ColPaliEngine, ColPaliConfig

config = ColPaliConfig(
    screening_backend="gem_router",
    n_fine=128,
)
engine = ColPaliEngine(config=config)
```

---

## `voyager_index.MultiModalEngine`

Multi-modal search combining ColBERT (text) and ColPali (visual) with
shared MaxSim scoring.

---

## Multimodal

### `voyager_index.MultimodalModelSpec`

```python
@dataclass(frozen=True)
class MultimodalModelSpec:
    plugin_name: str          # short alias ("collfm2", "colqwen3")
    model_id: str             # HuggingFace model ID
    architecture: str         # backbone description
    embedding_style: str      # "colpali" or "colbert"
    modalities: tuple[str, ...] # ("text", "image")
    pooling_task: str         # vLLM pooling task name
    serve_command: str        # example vLLM launch command
```

### `voyager_index.VllmPoolingProvider`

```python
from voyager_index import VllmPoolingProvider

provider = VllmPoolingProvider(
    endpoint="http://localhost:8200",
    model="VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
)
result = provider.pool(input_data)             # single request
results = provider.batch_pool([data1, data2])  # sequential batch
```

---

## Preprocessing

### `voyager_index.enumerate_renderable_documents`

```python
enumerate_renderable_documents(
    root: Path | str,
    *,
    exclude_paths: Sequence[Path | str] | None = None,
    recursive: bool = True,
) -> dict[str, Any]
```

Returns: `{"root": str, "documents": List[Path], "skipped": List[{"path", "reason"}]}`

Supported formats: PDF, DOCX, XLSX, PNG, JPG, WebP, GIF.

### `voyager_index.render_documents`

```python
render_documents(
    documents: Sequence[Path | str],
    output_dir: Path | str,
    *,
    source_root: Path | str | None = None,
) -> dict[str, Any]
```

Returns:
- `status`: `"passed"` or `"skipped"`
- `output_dir`: resolved output path
- `bundles`: per-document bundles with `doc_id`, `pages`, metadata
- `rendered`: flat list of page dicts (`image_path`, `page_number`, `doc_id`, `renderer`, `text`, `width`, `height`)
- `skipped`: files that couldn't be rendered
- `summary`: aggregate counts

---

## Triton Kernels

All GPU kernels require `voyager-index[gpu]` (Triton). They are optional —
the system falls back to CPU scoring when unavailable.

### `voyager_index.fast_colbert_scores`

Exact MaxSim late-interaction scoring via fused Triton kernel.

```python
from voyager_index import fast_colbert_scores

scores = fast_colbert_scores(
    queries,           # (n_queries, n_q_tokens, dim) float16
    documents,         # (n_docs, n_d_tokens, dim) float16
    documents_mask=mask,  # (n_docs, n_d_tokens) float — 1.0 for real tokens
)  # → (n_queries, n_docs)
```

### ROQ MaxSim Kernels

```python
from voyager_index import roq_maxsim_4bit, roq_maxsim_8bit, roq_maxsim_2bit, roq_maxsim_1bit

scores = roq_maxsim_4bit(
    query_codes, query_meta,
    doc_codes, doc_meta,
    documents_mask=mask,
)  # → (n_queries, n_docs)
```

Full ROQ ladder: `roq_maxsim_1bit`, `roq_maxsim_2bit`, `roq_maxsim_4bit`, `roq_maxsim_8bit`.

### `voyager_index.TRITON_AVAILABLE`

Boolean flag: `True` if Triton is installed and operational.

---

## Low-Level: `latence_gem_index`

For power users who need direct access to the Rust GEM segments.

### `GemSegment` (sealed, read-only)

| Method | Description |
|---|---|
| `build(all_vectors, doc_ids, doc_offsets, ...)` | Build graph from vectors |
| `search(query_vectors, k=10, ef=100, n_probes=4, filter=None)` | Search → `List[(doc_id, score)]` |
| `search_with_stats(query_vectors, k, ef, n_probes)` | Search + `(nodes_visited, dist_computations)` |
| `search_batch(queries, k, ef, n_probes)` | Batch search |
| `brute_force_proxy(query_vectors, k)` | Exhaustive qCH (oracle baseline) |
| `save(path)` / `load(path)` | Persist with CRC32 integrity |
| `set_doc_payloads(payloads)` | Build filter index for filtered search |
| `graph_connectivity_report()` | BFS connectivity → `(n_components, giant_frac, cross_cluster_ratio)` |
| `get_codebook_centroids()` | → `ndarray (n_fine, dim)` |
| `get_idf()` / `get_flat_codes()` | Codebook internals |
| `n_docs()` / `n_nodes()` / `n_edges()` / `dim()` | Graph stats |

### `PyMutableGemSegment` (writable)

| Method | Description |
|---|---|
| `build(...)` | Build initial graph |
| `search(query_vectors, k, ef, n_probes)` | Search |
| `insert(vectors, doc_id)` | Insert one document |
| `insert_batch(vectors_list, doc_ids)` | Batch insert |
| `delete(doc_id)` → `bool` | Soft-delete |
| `upsert(vectors, doc_id)` | Delete + insert |
| `compact()` | Rebuild without deleted nodes |
| `heal()` | Local graph repair |
| `needs_healing()` → `bool` | Drift detection |
| `graph_quality_metrics()` | → `(delete_ratio, avg_degree, isolated_ratio, stale_rep_ratio)` |

### `PyEnsembleGemSegment` (multi-modal)

| Method | Description |
|---|---|
| `build(..., modality_tags, n_modalities)` | Build per-modality graphs |
| `search(query_vectors, query_modality_tags, k, ef, n_probes)` | Search with RRF fusion |

### `GpuQchScorer`

```python
from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer

scorer = GpuQchScorer.from_gem_segment(segment, device="cuda")
scores = scorer.score_query(query_vecs)            # → (n_docs,) lower = closer
scores = scorer.score_query_filtered(query_vecs, mask)  # masked scoring
```
