# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Lightweight, high-performance multi-vector retrieval engine for on-prem deployment.**

voyager-index delivers ColBERT / ColPali-grade retrieval quality at production
throughput — no graph indexing, no distributed control plane, no cluster to
operate.  Just shards, Rust cores, GPU kernels, and a FastAPI server you can
`pip install` and run on a single machine.

```
pip install "voyager-index[server,shard,gpu]"
voyager-index-server                          # OpenAPI at :8080/docs
```

## Why

Most vector databases treat late interaction as an afterthought — bolt-on
reranking over an ANN shortlist.  voyager-index is built the other way around:
**MaxSim is the truth scorer**, and the entire system exists to make it fast.

- **No graph construction in the hot path.**  A learned LEMUR MLP collapses
  multi-vector candidate generation to single-vector MIPS over FAISS — routing,
  not indexing.
- **Rust fused exact scoring on CPU.**  The native `latence_shard_engine`
  extension runs MaxSim over memory-mapped shards with SIMD acceleration,
  releasing the Python GIL for true multi-threaded throughput.
- **Triton MaxSim on GPU.**  Custom CUDA kernels for FP16, INT8, FP8, and ROQ-4
  scoring with variable-length document scheduling.
- **Full database semantics.**  CRUD, WAL, checkpoint, crash recovery, payload
  metadata, scroll/retrieve — everything you need for a real application, not
  just a benchmark demo.
- **API-first.**  FastAPI reference server with base64 vector transport,
  OpenAPI docs, batch search, multi-worker serving, and Docker deployment.

## BEIR Benchmark

Measured on NVIDIA RTX A5000 (24 GB) with `lightonai/GTE-ModernColBERT-v1`,
`top_k=100`, search-only (encoding excluded).  CPU uses 8 native Rust workers.

| Dataset | Documents | MAP@100 | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---------|----------:|--------:|--------:|---------:|----------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana | 8,674 | 0.2737 | 0.3785 | 0.4289 | 0.7400 | 0.9600 | 270.2 | 4.2 | 40.2 | 274.3 |
| fiqa | 57,638 | 0.4616 | 0.5120 | 0.5758 | 0.5626 | 0.7958 | 60.3 | 5.9 | 74.2 | 153.5 |
| nfcorpus | 3,633 | 0.1874 | 0.3805 | 0.3078 | 0.3388 | 0.2820 | 284.3 | 3.8 | 132.3 | 129.5 |
| quora | 15,675 | 0.9842 | 0.9871 | 0.9896 | 0.9929 | 1.0000 | 99.6 | 2.7 | 278.8 | 63.7 |
| scidocs | 25,657 | 0.1165 | 0.1719 | 0.2494 | 0.1795 | 0.4065 | 237.2 | 4.6 | 73.2 | 148.1 |
| scifact | 5,183 | 0.7343 | 0.7705 | 0.7836 | 0.8790 | 0.9300 | 261.4 | 4.0 | 58.7 | 147.0 |

### Comparison with next-plaid

[next-plaid](https://github.com/lightonai/next-plaid) (LightOn) is the current
open-source reference for ColBERT-style late-interaction serving.  Their numbers
below are from their README, measured on NVIDIA H100 80 GB with the same
embedding model.  Their QPS **includes** encoding time; ours excludes it
(search-only).

| Dataset | System | NDCG@10 | MAP@100 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---------|--------|--------:|--------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana | **voyager** | **0.3785** | **0.2737** | **0.9600** | **270.2** | **4.2** | **40.2** | **274.3** |
| | next-plaid | 0.3499 | 0.2457 | 0.9337 | 13.6 | 170.1 | 17.4 | 454.7 |
| fiqa | **voyager** | **0.5120** | **0.4616** | **0.7958** | **60.3** | **5.9** | **74.2** | **153.5** |
| | next-plaid | 0.4506 | 0.3871 | 0.7459 | 18.2 | 170.6 | 17.6 | 259.1 |
| nfcorpus | **voyager** | 0.3805 | **0.1874** | 0.2820 | **284.3** | **3.8** | **132.3** | **129.5** |
| | next-plaid | **0.3828** | 0.1870 | **0.3228** | 6.6 | 262.1 | 16.9 | 219.4 |
| scidocs | **voyager** | 0.1719 | 0.1165 | 0.4065 | **237.2** | **4.6** | **73.2** | **148.1** |
| | next-plaid | **0.1914** | **0.1352** | **0.4418** | 17.5 | 139.3 | 16.5 | 281.7 |
| scifact | **voyager** | **0.7705** | **0.7343** | 0.9300 | **261.4** | **4.0** | **58.7** | **147.0** |
| | next-plaid | 0.7593 | 0.7186 | **0.9633** | 7.9 | 169.5 | 16.9 | 305.4 |

**Summary:** voyager-index matches or exceeds next-plaid on quality across most
datasets while delivering **10–40x higher GPU throughput** and **2–8x higher CPU
throughput** on a smaller GPU (A5000 vs H100).  GPU P95 latency is consistently
under 6 ms versus 130–260 ms.

> **Note on QPS methodology:** next-plaid reports near-constant QPS across
> corpus sizes because their retrieval cost is dominated by a fixed candidate
> budget (`n_full_scores=4096`, limited IVF probes) and their QPS includes
> encoding.  Our QPS is pure search-only wall-clock throughput, which scales
> with corpus size as routing and exact scoring do real work over the full
> candidate set.

## Architecture

```text
query vectors (token / patch embeddings)
  → LEMUR routing MLP
  → FAISS ANN over routing representations
  → candidate document IDs
  → optional centroid-approx or doc-mean proxy pruning
  → optional ColBANDIT query-time pruning
  → exact or quantized MaxSim
       Rust fused exact (CPU, mmap, SIMD, GIL-free)
       Triton FP16 / INT8 / FP8 / ROQ-4 (GPU)
       GPU-corpus gather + rerank
  → top-K results
```

| Layer | What it does | Why it matters |
|-------|-------------|----------------|
| **Routing** | LEMUR MLP, FAISS MIPS, candidate budgets | Makes late interaction tractable without graph construction |
| **Storage** | Safetensors shards, merged mmap, GPU-resident corpus | Honest CPU and GPU layouts for any corpus size |
| **Exact scoring** | Triton MaxSim, Rust fused MaxSim, quantized kernels | MaxSim stays the truth scorer across all deployment shapes |
| **Optimization** | ColBANDIT pruning, centroid approximation, ROQ-4 | Moves the latency/recall frontier without changing the retrieval contract |
| **Durability** | WAL, memtable, checkpoint, crash recovery | A retrieval engine that behaves like a real database |
| **Serving** | FastAPI, base64 transport, multi-worker, OpenAPI | One `pip install`, one server, one API contract |

## What Makes It Different

### No graph indexing

HNSW and friends trade recall for speed.  voyager-index uses a learned router
(LEMUR) to narrow candidates, then runs full MaxSim over the shortlist.  The
result: graph-free retrieval with higher recall at lower latency than
graph-based approaches.

### Rust + Triton, not Python

The CPU hot path is a native Rust extension (`latence_shard_engine`) that
memory-maps shard data, runs fused MaxSim with SIMD, and releases the GIL —
enabling true parallel throughput across workers.  The GPU path uses custom
Triton kernels with variable-length document scheduling.

### SOTA research in the production path

LEMUR routing, ColBANDIT query-time pruning, ROQ rotational quantization, and
context packing via Tabu Search are all wired into the shipped serving path —
not isolated research branches.

### Real database operations

Add, update, delete, and query documents with payload metadata.  WAL-backed
mutations survive crashes.  Checkpoint and scroll APIs let you operate it like a
database, not a throwaway index.

### Multimodal native

The same shard engine serves text token embeddings (ColBERT) and image patch
embeddings (ColPali/ColQwen).  Document preprocessing handles PDF, DOCX, XLSX,
and images into page-level patch vectors.

## Quickstart

### Install

```bash
pip install "voyager-index[shard]"               # CPU only
pip install "voyager-index[server,shard]"         # + FastAPI server
pip install "voyager-index[server,shard,gpu]"     # + Triton GPU kernels
pip install "voyager-index[server,shard,native]"  # + Tabu Search solver
```

### Python SDK

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
    n_shards=32,
    k_candidates=256,
    compression="fp16",
)
idx.add(docs, ids=list(range(len(docs))))
results = idx.search(query, k=5)
print(results[0])
idx.close()
```

### HTTP API

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
# OpenAPI docs at http://127.0.0.1:8080/docs
```

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
print(response.json()["results"][0])
```

### Docker

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Execution Modes

| Mode | Corpus placement | Best for |
|------|-----------------|----------|
| **CPU exact** | Disk/mmap → Rust fused MaxSim | Simplest deployment, no GPU required |
| **GPU streamed** | Disk/CPU → GPU transfer → Triton MaxSim | Large corpora that don't fit in VRAM |
| **GPU corpus** | Fully resident in VRAM | Lowest latency when corpus fits |

All three modes share the same collection format, API contract, and retrieval
semantics.  Start with CPU, add GPU when latency matters.

## Engineering Knobs

### Routing and candidate budgets

- `k_candidates` — LEMUR candidate budget before exact scoring
- `max_docs_exact` — hard ceiling for the exact-stage document set
- `n_full_scores` — proxy shortlist size before full MaxSim
- `use_colbandit` — enable query-time pruning

### Storage and transfer

- `n_shards` — number of storage shards
- `compression` — `fp16`, `int8`, or `roq4`
- `transfer_mode` — `pageable`, `pinned`, or `double_buffered`

### Scoring and hardware

- `quantization_mode` — exact, `int8`, `fp8`, or `roq4`
- `router_device` — where LEMUR executes (`cpu` or `cuda`)
- `gpu_corpus_rerank_topn` — GPU rerank frontier for corpus-resident mode

## Hybrid and Multimodal

- **BM25 + dense fusion** via `rrf` or Tabu Search refinement
- **Multimodal collections** share the same base64 vector contract as text
- **Document preprocessing** handles PDF, DOCX, XLSX, and images via
  `render_documents()` and the `/reference/preprocess/documents` API endpoint
- **Graph-aware** workflows compose through stable document IDs and sidecars,
  without forcing graph construction into the retrieval path

## API Surface

| Endpoint | Purpose |
|----------|---------|
| `POST /collections/{name}` | Create collection |
| `POST /collections/{name}/points` | Add / upsert documents |
| `POST /collections/{name}/search` | Search |
| `POST /collections/{name}/search/batch` | Batch search |
| `GET /collections/{name}/scroll` | Scroll through results |
| `POST /collections/{name}/retrieve` | Retrieve by ID |
| `DELETE /collections/{name}/points` | Delete documents |
| `POST /collections/{name}/checkpoint` | Checkpoint WAL |
| `GET /collections/{name}/wal/status` | WAL status |
| `POST /encode` | Encode text/images to vectors |
| `POST /rerank` | Rerank results |
| `POST /reference/optimize` | Context packing (Tabu Search) |

## Public Python Surface

- `Index` and `IndexBuilder` — local shard collections
- `SearchPipeline` — dense + sparse fusion in-process
- `ColbertIndex` — late-interaction text workflows
- `ColPaliEngine` and `MultiModalEngine` — multimodal retrieval
- `encode_vector_payload()`, `decode_payload()` — base64 transport helpers
- `voyager-index-server` — reference HTTP server

## Documentation

- [Quickstart](docs/getting-started/quickstart.md)
- [Installation](docs/getting-started/installation.md)
- [Python API Reference](docs/api/python.md)
- [Reference API Tutorial](docs/reference_api_tutorial.md)
- [Shard Engine Guide](docs/guides/shard-engine.md)
- [Max-Performance Guide](docs/guides/max-performance-reference-api.md)
- [Scaling Guide](docs/guides/scaling.md)
- [Benchmarks And Methodology](docs/benchmarks.md)
- [Production Notes](PRODUCTION.md)

## Install From Source

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

## License

Apache-2.0.  See [LICENSE](LICENSE).
