# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`voyager-index` is a multi-vector native, shard-first retrieval system for teams
building high-recall on-prem search around token- and patch-level embeddings.
It is designed for ColBERT-style late interaction, ColPali/ColQwen-style
multimodal retrieval, dense lexical fusion, and durable single-host serving
without turning the stack into a control-plane project.

This repo has one supported product story:

- LEMUR-routed shard retrieval as the mainline engine
- exact or quantized MaxSim on CPU or Triton/CUDA
- optional GPU-resident corpus mode
- durable CRUD, WAL, checkpoint, and recovery
- a simple Python SDK plus FastAPI reference server
- base64 as the preferred/default HTTP transport for vectors
- BM25 hybrid search with `rrf` or Tabu Search solver refinement

Legacy `gem` and `hnsw` code remains for compatibility, but the supported docs,
CI, and release surface are shard-first.

## Built For

Use `voyager-index` if you are building:

- an on-prem retrieval system where MaxSim quality is non-negotiable
- multimodal search over PDFs, office docs, images, and page-level corpora
- hybrid retrieval that combines dense semantics with BM25 lexical recall
- an application that needs CRUD, WAL-backed recovery, and straightforward
  single-host operations
- a graph-aware or ontology-aware system where graph construction should stay
  outside the hot retrieval path and attach through stable IDs

It is a strong fit for search, RAG, document intelligence, compliance and
regulated knowledge systems, technical corpora, and teams that want the
serving path to stay technically honest about recall, latency, and hardware
placement.

## Why This Architecture Exists

Most retrieval stacks still behave like ANN systems that occasionally rerank.
`voyager-index` is organized the other way around: late interaction remains the
truth scorer, and the rest of the system exists to make that scorer practical.

- `Routing instead of graph construction`: a learned LEMUR MLP reduces
  multi-vector candidate generation to single-vector MIPS over FAISS.
- `Exact semantics across hardware`: the same collection layout serves CPU
  exact, streamed GPU, or GPU-corpus modes.
- `Optimization without model drift`: `int8`, `fp8`, and `roq4` trim bandwidth
  and kernel cost without changing the retrieval objective.
- `Operational simplicity`: one package, one reference API, one base64 vector
  contract, one durability model.
- `Composable intelligence`: multimodal rendering, BM25 fusion, and
  graph-adjacent sidecars can enrich the system without destabilizing the hot
  path.

## Retrieval Pipeline

```text
query token vectors / patch vectors
  -> LEMUR routing MLP
  -> FAISS ANN over routing representations
  -> candidate document IDs
  -> optional centroid-approx or doc-mean proxy pruning
  -> optional ColBANDIT query-time pruning
  -> exact or quantized MaxSim
       CPU exact
       Triton exact
       Triton INT8 / FP8 / ROQ4
       GPU-corpus gather + rerank
  -> top-K
  -> optional lexical fusion / downstream reasoning
```

### Architecture Layers

| Layer | Core primitives | Why it matters |
|---|---|---|
| Routing | LEMUR MLP, FAISS MIPS, candidate budgets | makes late interaction tractable without graph construction |
| Storage | safetensors shards, selective mmap reads, storage shards, GPU-resident corpus path | supports large corpora with honest CPU and GPU layouts |
| Exact stage | Triton MaxSim, CPU exact fallback, quantized kernels, variable-length scheduling | keeps MaxSim as the truth scorer across deployment shapes |
| Retrieval optimization | ColBANDIT, centroid approximation, doc-mean proxy, GPU rerank frontier | moves the latency/recall frontier without changing the product contract |
| Durability | WAL, memtable, checkpoint, crash recovery, snapshots | lets a retrieval engine behave like a real application system |
| Serving | FastAPI, base64 transport helpers, multi-worker single-host serving | keeps SDK and API integration simple for operators and clients |

## What You Get

| Capability | What it means in practice |
|---|---|
| Shard-first late interaction | storage shards inside one shard engine, routed by LEMUR, then scored by MaxSim |
| CPU and GPU execution shapes | same collection format across CPU exact, streamed GPU, and GPU-corpus serving |
| Triton MaxSim | fused CUDA kernels for exact and quantized late-interaction scoring |
| Quantized serving | `int8`, `fp8`, and `roq4` scoring modes on the CUDA fast path |
| ColBANDIT in the real path | production query-time pruning, not an isolated research branch |
| Durable mutations | add, upsert, delete, payload updates, checkpoint, restart recovery |
| Multimodal ingestion path | PDF, DOCX, XLSX, and images rendered into page assets for patch-level retrieval |
| Hybrid retrieval | BM25 plus dense retrieval with `rrf` or Tabu Search refinement |
| Base64 transport | smaller, faster, shared HTTP payload contract for dense and multivector traffic |
| Admin surface | shard inspection, WAL status, checkpoint, retrieve, scroll, batch search |

## Who Should Use It And Who Should Not

Use it when:

- you already have token- or patch-level embeddings
- you want one system for SDK, API, CPU fallback, and GPU acceleration
- you care about recall, payload size, and hardware placement at the same time
- you want graph-aware semantics as a composable sidecar, not a forced graph DB

Do not use it when:

- you only have pooled dense vectors and do not need multivector retrieval
- you want a distributed control plane more than a retrieval engine
- you want graph-native serving in the core OSS HTTP path
- you want the repo to generate embeddings, run document intelligence, and host
  every upstream model provider for you

## Quickstart

Install the mainline shard path:

```bash
pip install "voyager-index[shard]"
pip install "voyager-index[server,shard]"        # reference API + preprocessing
pip install "voyager-index[server,shard,gpu]"    # Triton MaxSim on CUDA
pip install "voyager-index[server,shard,native]" # adds Tabu Search solver
```

### First Local Search

```python
import numpy as np

from voyager_index import Index

rng = np.random.default_rng(7)
docs = [rng.normal(size=(16, 128)).astype("float32") for _ in range(32)]
query = rng.normal(size=(16, 128)).astype("float32")

idx = Index(
    "demo-index",
    dim=128,
    engine="shard",
    n_shards=32,          # storage shards inside the shard engine
    k_candidates=256,
    compression="fp16",
)
idx.add(docs, ids=list(range(len(docs))))
results = idx.search(query, k=5)
print(results[0])
idx.close()
```

### First HTTP Search With Base64 Transport

```python
import numpy as np
import requests

from voyager_index import encode_vector_payload

query = np.random.default_rng(7).normal(size=(16, 128)).astype("float32")

response = requests.post(
    "http://127.0.0.1:8080/collections/my-shard/search",
    json={
        "vectors": encode_vector_payload(query, dtype="float16"),
        "top_k": 5,
        "quantization_mode": "fp8",
        "use_colbandit": True,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

### Run The Server

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
# OpenAPI docs: http://127.0.0.1:8080/docs
```

## CPU, Streamed GPU, And GPU-Corpus Modes

| Mode | Same retrieval semantics | What changes |
|---|---|---|
| CPU exact | LEMUR routing, MaxSim, CRUD, WAL, collection layout | exact stage stays on CPU; Triton quantized kernels are not active |
| GPU streamed | LEMUR routing, storage shards, API contract, durability | candidate docs are fetched from disk/CPU memory and scored on GPU |
| GPU corpus | LEMUR routing, top-level retrieval flow, request contract | corpus stays resident in VRAM for the lowest exact-stage latency |

Practical rule:

- start with CPU if you want the simplest deployment and easiest observability
- add `gpu` when exact-stage latency matters more than deployment minimalism
- use GPU-corpus mode when the corpus fits comfortably in VRAM and you want the
  shortest exact path

## Engineering Knobs That Actually Move The Needle

### Routing And Candidate Budgets

- `k_candidates`: LEMUR candidate budget before exact scoring
- `lemur_search_k_cap`: upper cap on routed search breadth
- `max_docs_exact`: hard ceiling for the exact-stage document set
- `n_full_scores`: proxy shortlist size before full MaxSim
- `n_centroid_approx`: optional centroid-approx stage before exact full scoring
- `use_colbandit`: query-time pruning in the production shard path

### Storage, Transfer, And Layout

- `n_shards`: number of storage shards inside the shard engine
- `compression`: persisted representation, `fp16`, `int8`, or `roq4`
- `transfer_mode`: `pageable`, `pinned`, or `double_buffered`
- `pinned_pool_buffers`: pinned-memory buffer pool size
- `pinned_buffer_max_tokens`: upper bound for a pinned transfer chunk
- `uniform_shard_tokens`: optional shard packing control

### Scoring And Hardware Placement

- `quantization_mode`: exact, `int8`, `fp8`, or `roq4`
- `router_device`: where LEMUR executes, usually `cpu` or `cuda`
- `gpu_corpus_rerank_topn`: GPU rerank frontier when the corpus is resident
- `variable_length_strategy`: scheduling mode for uneven token counts
- `seed`: reproducible training/layout seed
- `WORKERS`: single-host multi-worker QPS scaling for the reference server

The rule of thumb is simple: tune routing breadth, exact-stage budget, and
transfer mode first; only then reach for more exotic knobs.

## Hybrid, Multimodal, And Graph-Aware Retrieval

### Hybrid lexical + semantic retrieval

- dense collections expose BM25-only, vector-only, or fused BM25+dense search
- set `dense_hybrid_mode="rrf"` for simple fusion
- set `dense_hybrid_mode="tabu"` when `latence_solver` is installed and you want
  Tabu Search refinement
- use `HybridSearchManager` in process when you want BM25 fusion with the shard
  backend itself

### Multimodal retrieval

- late-interaction and multimodal collections share the same base64 vector
  contract
- local preprocessing starts with `enumerate_renderable_documents()` and
  `render_documents()`
- rendered pages can feed ColPali-family or related patch-level retrieval flows
- the reference API handles the document-to-page stage; embedding generation
  remains an external producer/provider role

### Optional graph-aware retrieval

`voyager-index` supports graph-aware and ontology-aware workflows through stable
document IDs and sidecars, not by forcing graph construction into the retrieval
hot path.

- ontology, entity, relation, and concept sidecars can enrich downstream ranking
  or reasoning systems
- graph construction and graph serving stay external to the OSS runtime by design
- the reference HTTP API does not expose a dedicated ontology or graph endpoint

This is intentional. The retrieval engine stays simple and fast; graph-aware
systems can still compose cleanly on top.

## Production-Wired Surface

What is already wired into the shipped shard-first path:

- shard collections for the mainline late-interaction engine
- CRUD, upsert, payload updates, WAL-backed mutation logging, and recovery
- multi-worker single-host serving
- worker-visible collection revisions after mutation
- base64 vector transport as the preferred/default API format
- ColBANDIT in the shard scoring flow
- Triton MaxSim on CUDA
- `int8`, `fp8`, and `roq4` shard scoring modes
- multimodal preprocessing flows
- dense BM25 hybrid search with `rrf` or Tabu Search

Operational truth:

- shard HTTP search is vector-only
- dense BM25 hybrid over HTTP lives on `dense` collections
- shard + BM25 fusion over the same flow is an in-process `HybridSearchManager`
  path today
- auth, TLS termination, ingress policy, and secret management stay outside the
  reference server

## Benchmark Posture

The benchmark story is intentionally split in two:

- `benchmarks/oss_reference_benchmark.py` is the reproducible smoke benchmark
  for package and API sanity
- the 100k comparison is the product benchmark and should be published with
  fixed methodology and raw reports

Rules for published comparisons:

- same corpus and embeddings across all systems
- same `top_k` and recall target
- warmup runs are excluded from measured latency
- streamed and GPU-corpus numbers are reported separately
- QPS claims are paired with recall, never shown alone

### 100k Comparison Placeholder

Pending fresh measurement on the same corpus and hardware:

| System | Mode | Corpus placement | Recall@10 | p50 | p95 | QPS | Status |
|---|---|---|---|---|---|---|---|
| Plaid | pending | pending | pending | pending | pending | pending | pending measurement |
| FastPlaid | GPU corpus | pending | pending | pending | pending | pending | pending measurement |
| Qdrant | pending | pending | pending | pending | pending | pending | pending measurement |
| Voyager | shard streamed | CPU/disk -> GPU | pending | pending | pending | pending | pending measurement |
| Voyager | shard GPU corpus | GPU resident | pending | pending | pending | pending | pending measurement |

## Documentation

- [Quickstart](docs/getting-started/quickstart.md)
- [Installation](docs/getting-started/installation.md)
- [Python API Reference](docs/api/python.md)
- [Reference API Tutorial](docs/reference_api_tutorial.md)
- [Shard Engine Guide](docs/guides/shard-engine.md)
- [Max-Performance Reference API Guide](docs/guides/max-performance-reference-api.md)
- [Scaling Guide](docs/guides/scaling.md)
- [Benchmarks And Methodology](docs/benchmarks.md)
- [Production Notes](PRODUCTION.md)

## Install From Source

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

Supported native add-on story:

- `latence_solver`: optional solver wheel for `tabu` refinement and
  `/reference/optimize`

## Docker

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Public Surface

- `Index` and `IndexBuilder` for local shard collections
- `SearchPipeline` for dense + sparse fusion in process
- `ColbertIndex` for late-interaction text workflows
- `ColPaliEngine` and `MultiModalEngine` for multimodal retrieval
- `encode_vector_payload()`, `encode_roq_payload()`, and `decode_payload()` for
  shared base64 transport
- `voyager-index-server` for the reference HTTP API

## License

Apache-2.0. See `LICENSE`.
