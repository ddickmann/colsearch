# voyager-index

`voyager-index` is an open-source local retrieval stack for high-performance
multivector search, hybrid dense+BM25 retrieval, and durable on-disk CRUD.
It combines exact Triton MaxSim, optional quantization profiles, restart-safe
collections, delta-style point upserts/deletes, and an optional Tabu Search
knapsack solver exposed through dense refinement and `/reference/optimize`.
That solver is intentionally a more provocative answer to mainstream
`RRF -> heavyweight reranker` pipelines: keep exact retrieval truthful, then
optimize the final packed LLM context under real token and diversity
constraints.

The OSS product surface is built for local-first deployments:

- multivector late-interaction retrieval for text and multimodal workloads
- durable CRUD with point upserts, point deletes, and restart-safe reloads
- exact Triton MaxSim in FP16 by default, with optional INT8, FP8, and RoQ profiles
- hybrid dense+sparse retrieval through `SearchPipeline`
- a canonical Tabu Search solver path for context packing and candidate refinement, positioned as an innovative alternative to rank-fusion-first pipelines
- a truthful FastAPI reference service plus Docker deployment assets

Disk is the source of truth. RAM and VRAM are accelerators layered on top of
the persisted collection layout, not a separate state model. That gives the OSS
repo a fast local serving path without hiding persistence, recovery, or update
semantics behind an in-memory-only demo.

Deferred layers such as compute-sidecar serving, response scoring, and the
broader Voyager product stack are intentionally outside the OSS foundation cut.
The native HNSW wrapper, native solver, and GEM router remain separate optional
packages inside this repo and are only active after an explicit build/install step.
The premium Voyager stack and deeper research code live in the separate
`latence-voyager` repository and are intentionally outside this OSS tree.

## Why It Is Fast

- exact late-interaction and multimodal scoring use the Triton MaxSim path by default
- dense collections keep disk-backed state while using RAM and optional native kernels for serving speed
- multimodal `strategy="optimized"` keeps exact MaxSim as the default scoring contract, may use a restart-safe lightweight screening index first, and exposes explicit solver orderings only as opt-in knobs
- optional native packages bring in the Qdrant-derived HNSW path and the Tabu Search solver without changing the public API contract

## Feature Highlights

- `ColbertIndex` gives you a local multivector text index with exact late-interaction scoring
- `ColPaliEngine` gives you a local multimodal multivector index for supported visual embedding models
- `SearchPipeline` gives you vector-first dense retrieval, BM25-only retrieval, and dense+BM25 hybrid fusion
- `voyager_index.render_documents(...)` and `POST /reference/preprocess/documents` give you a supported doc-to-image preprocessing stage for PDF, DOCX, XLSX, and image inputs before embedding
- `POST /collections/{name}/points` is the delta-ingestion path: add, replace, or upsert points without inventing a second ingestion API
- `DELETE /collections/{name}/points` and `DELETE /collections/{name}` give you durable cleanup semantics, not just ephemeral in-memory deletes
- `POST /reference/optimize` exposes the canonical OSS context-packing solver surface for dense, dense+BM25, multivector, and multivector+BM25 requests, including heterogeneous candidate pools assembled from BM25, ontology/rules, dense retrieval, or multimodal retrieval
- the solver story is intentionally contrarian to mainstream practice: use exact retrieval for truth, then replace simple fusion heuristics with a constraint-aware final packing layer
- multimodal optimized requests can now explicitly choose `multimodal_optimize_mode`, `multimodal_candidate_budget`, `multimodal_prefilter_k`, and `multimodal_maxsim_frontier_k`; the measured default remains exact MaxSim plus screening
- Docker, Compose, examples, notebooks, and validation bundles are all in-tree so the repo can be evaluated without private infrastructure

## GEM-Inspired Routing and Screening

The index integrates ideas from the GEM paper:

> Yao Tian, Zhoujin Tian, Xi Zhao, Ruiyuan Zhang, Xiaofang Zhou.
> "[GEM: A Native Graph-based Index for Multi-Vector Retrieval](https://arxiv.org/abs/2603.20336)."
> arXiv:2603.20336, March 2026.

GEM constructs a proximity graph directly over vector sets with set-level
clustering, metric-decoupled graph construction (EMD for build, Chamfer for
search), and semantic shortcuts. The full paper and reference C++ implementation
are at https://github.com/sigmod26gem/sigmod26gem.

We adopt GEM's most impactful ideas at two levels: a **Rust-native set-native
router** and a **Python GEM-lite screening path**.

### Rust GEM Router (`latence_gem_router`)

A standalone Rust crate (`src/kernels/gem_router/`) that implements the core
GEM routing algorithms in native code with PyO3 bindings:

- **Two-stage codebook** (`C_quant` → `C_index`): k-means clustering of all
  document token vectors into fine centroids, then coarse cluster assignment
- **Per-document cluster profiles**: centroid IDs, `C_top` coarse cluster
  assignments, and IDF-weighted cluster scoring
- **Cluster-based posting lists**: maps coarse clusters to document sets for
  fast candidate retrieval
- **qCH proxy scoring**: centroid-quantized Chamfer distance for cheap
  candidate ranking during routing (ported from GEM's `L2SqrCluster4Search`)
- **Multi-entry HNSW hints**: provides cluster representative point IDs for
  seeding graph traversal from multiple entry points

**Performance optimizations** (validated with zero recall regression):

- `matrixmultiply::sgemm` with AVX2+FMA micro-kernels for the query-centroid
  score matrix, replacing scalar tiled GEMM (3-5x on this kernel)
- Flat contiguous `u16` codes array (`FlatDocCodes`) for cache-sequential
  proxy scoring, eliminating per-document pointer chasing and halving memory
  bandwidth vs `Vec<u32>`
- AVX2 `_mm256_i32gather_ps` + `_mm256_max_ps` vectorized proxy scoring that
  gathers 8 scattered scores per instruction (with scalar fallback)
- `Mutex<Vec<f32>>` score buffer reuse across queries, eliminating ~32KB
  allocation per query

Measured result on 10K clustered documents: **0.96ms → 0.41ms routing latency
(2.3x speedup), 249x vs brute-force MaxSim, identical recall** (R@10=0.970).

The router serves both retrieval lanes:
- **ColBERT** (`ColbertIndex`): alternative balanced-mode candidate generator
  alongside PLAID via `_search_gem_triton`
- **ColPali** (`ColPaliEngine`): `screening_backend="gem_router"` screening
  backend via `GemScreeningIndex`

### Native HNSW Multi-Entry Traversal

The vendored Qdrant HNSW graph (`graph_layers.rs`) now includes a
`search_multi_entry` method that seeds beam search from multiple entry points
simultaneously with a **shared visited set** and **shared result heap**. This
is exposed through the `latence_hnsw` PyO3 binding as `search_gem()`.

### GEM-Lite Prototype Screening

The multimodal `prototype_hnsw` screening lane includes a Python GEM-lite
path that adds codebook-based reranking on top of HNSW prototype search.

### What is NOT adopted yet

- GEM-native dual graph construction (intra-cluster subgraphs + cross-cluster
  bridge edges, GEM Algorithm 2-3)
- qEMD-routed graph edges using Earth Mover's Distance for graph construction
  with metric decoupling (GEM Section 4.2)
- Semantic shortcut injection from training query-positive pairs (GEM Algorithm 4)
- Adaptive per-document cluster cutoff via decision tree (GEM Section 4.4.2)

See `docs/research/GEM_NATIVE_RESEARCH.md` for the full research branch
specification, including the porting plan from the
[GEM C++ reference implementation](https://github.com/sigmod26gem/sigmod26gem),
exact file-to-module mappings, benchmark gates, and risk analysis.

### Research Outlook: GEM-Native Graph Backend

The current architecture uses GEM-inspired routing on top of the vendored Qdrant
HNSW backend. The HNSW graph itself remains a single-vector L2 graph and does
not reason about vector sets natively. The GEM paper (Tian et al., 2026)
demonstrates that constructing a proximity graph directly over vector sets —
using EMD for graph construction, Chamfer/MaxSim for search, and semantic
shortcuts to bridge the metric gap — yields up to 16x additional speedup over
methods like PLAID and DESSERT while matching or improving accuracy on MS MARCO,
LoTTE, OKVQA, and EVQA benchmarks.

The deferred research branch would replace the Qdrant HNSW backend for
multi-vector workloads with a Rust-native GEM graph. The existing router
(codebook, clustering, qCH scoring, flat codes, AVX2 kernels) is designed to
be reused as the foundation layer for the native graph. This is a gated effort:
it proceeds only when the router passes promotion benchmarks at 100K+ documents
and there is concrete evidence that the HNSW path is the remaining bottleneck.

The public contract remains **exact MaxSim reranking**. All GEM-inspired
features only affect candidate generation and screening, and all existing
trust controls still apply.

## Public Surface And Boundaries

Use `voyager_index.*` as the supported public namespace.

- `voyager_index.__init__` lazy-exports the supported OSS surface
- `voyager_index._internal.*` is packaged implementation detail, not a stable public API
- `src.*` still ships as a deprecated compatibility shim for older imports; new code should not depend on it
- `latence-voyager` is a separate standalone repo, not an OSS runtime dependency
- `SearchPipeline` is the vector-first local dense+sparse retrieval pipeline
- `ColbertIndex` owns late-interaction multivector text retrieval
- `ColPaliEngine` owns multimodal multivector retrieval
- `SUPPORTED_MULTIMODAL_MODELS` and `VllmPoolingProvider` define the public multimodal provider seam

The preferred accelerated provider implementation for multimodal embeddings is
`vllm-factory`, but its plugin internals are not part of the `voyager-index`
package contract.

For the standard OSS vLLM-powered ColPali path, `collfm2` is the default public
model choice via `DEFAULT_MULTIMODAL_MODEL` and `DEFAULT_MULTIMODAL_MODEL_SPEC`.

## Documentation Paths

Start with the path that matches your goal:

- new user: `docs/reference_api_tutorial.md`
- full feature exploration: `docs/full_feature_cookbook.md`
- evaluator: `BENCHMARKS.md`, `SCREENING_PROMOTION_DECISION_MEMO.md`, and `docs/validation/README.md`
- advanced user: `examples/README.md` and `notebooks/README.md`
- contributor: `docs/README.md`, `CONTRIBUTING.md`, and `tools/README.md`

## Install

### pip install (recommended)

```bash
pip install voyager-index                # pure Python — works on Linux, macOS, Windows
pip install voyager-index[native]        # + prebuilt Rust kernels (Linux x86_64, macOS x86_64/arm64)
pip install voyager-index[native,server] # + FastAPI reference server
```

The `[native]` extra installs prebuilt wheels for `latence-hnsw`, `latence-solver`,
and `latence-gem-router` — no Rust toolchain or compiler needed. Prebuilt wheels
are available for Linux x86_64, macOS Intel, and macOS Apple Silicon with
Python 3.10-3.12. The base package works without them (pure Python fallbacks).

Other extras: `[server]`, `[multimodal]`, `[preprocessing]`, `[gpu]` (Triton MaxSim).

### Build from source (development)

For contributors or unsupported platforms. Requires Python >= 3.10 and a
Rust toolchain for the native crates.

**One-command install:**

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

This handles system dependencies, Rust toolchain, Python package, all three
native Rust/PyO3 crates, and import verification. Pass `--skip-system-deps`
if build tools and Rust are already installed.

### Using Make

```bash
make install-cpu    # full install with CPU PyTorch
make install        # full install (uses whatever PyTorch is available)
make build-native   # build only the Rust crates (Python package already installed)
make test           # run all Rust + Python tests
make verify         # check all native modules are importable
make help           # show all available targets
```

**Manual step-by-step:**

Prerequisites — Linux (Debian/Ubuntu):

```bash
sudo apt-get install -y build-essential curl pkg-config libssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

Prerequisites — macOS:

```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

Then from a fresh checkout:

```bash
git clone https://github.com/ddickmann/voyager-index.git && cd voyager-index

# CPU-only PyTorch (smaller download, recommended for non-GPU machines)
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Editable install with all extras
pip install -e ".[server,multimodal,dev,native-build,native-gem-build]"

# Build and install native Rust crates
pip install ./src/kernels/hnsw_indexer ./src/kernels/knapsack_solver ./src/kernels/gem_router

# Verify everything works
python -c "import latence_hnsw, latence_solver, latence_gem_router; print('All native modules OK')"
```

Native crates:

- `latence_hnsw` — Qdrant-derived HNSW segment wrapper with RocksDB storage
- `latence_solver` — Tabu Search quadratic knapsack solver (CPU, optional CUDA)
- `latence_gem_router` — GEM-inspired set-native multi-vector routing core
- All three are optional: the base package has pure Python fallbacks
- `maturin >= 1.5` is installed automatically by the native-build extras

## Optional Local Refinement

`SearchPipeline.search(..., enable_refinement=True)` can invoke a locally
installed `latence_solver` backend for candidate refinement.

This refinement lane is:

- optional and off by default
- local/native only; it requires `latence_solver` to be built and importable
- backed by the same canonical optimizer contract used by `/reference/optimize`
- implemented by the in-tree Tabu Search quadratic knapsack solver in `src/kernels/knapsack_solver/`
- best treated as an advanced local enhancement on top of the baseline retrieval surface

## Quickstart

For the full first-run HTTP walkthrough, use `docs/reference_api_tutorial.md`.

For a runnable advanced walkthrough with logs and a JSON report:

```bash
python examples/reference_api_feature_tour.py --output-json feature-tour-report.json
```

Library import:

```python
from voyager_index import ColbertIndex, IndexConfig, SearchPipeline
from voyager_index import SUPPORTED_MULTIMODAL_MODELS, VllmPoolingProvider, render_documents
```

Start the local reference server:

```bash
voyager-index-server
```

Then open the interactive API docs:

```text
http://127.0.0.1:8080/docs
http://127.0.0.1:8080/redoc
```

By default the server is local-safe:

- binds to `127.0.0.1`
- keeps CORS disabled unless enabled explicitly
- stores data under `VOYAGER_INDEX_PATH` or `/data/voyager-index`
- runs with `WORKERS=1` only until shared-state coordination exists
- exposes `/health`, `/ready`, and `/metrics`

Source-doc ingestion starts with preprocessing:

```bash
curl -X POST http://127.0.0.1:8080/reference/preprocess/documents \
  -H "Content-Type: application/json" \
  -d '{"source_paths": ["/data/source/invoice.pdf"]}'
```

That endpoint renders supported source docs into PageBundle-like page-image assets
so an embedding provider can consume them before the resulting vectors are stored
through `POST /collections/{name}/points`.

Default OSS production path:

- disk-backed local collections are the source of truth; Docker mounts the same on-disk layout
- exact Triton MaxSim in FP16 is the canonical late-interaction and multimodal scoring path
- multimodal `strategy="optimized"` defaults to exact MaxSim plus trust-aware lightweight screening; explicit solver orderings stay opt-in via `multimodal_optimize_mode`
- once chunks come from multiple sources such as BM25, ontology hits, metadata/rules, dense retrieval, or multimodal retrieval, the solver becomes the better last-layer packer/selector than simple fusion heuristics like RRF because it can optimize under token, redundancy, quorum, and diversity constraints
- that is the deliberately provocative part of the design: instead of trusting a mainstream fusion stack to produce the final context implicitly, the repo exposes an explicit optimization layer for the last mile
- screening is trust-aware: bootstrap calibration, persisted health states, risky-query bypass, and exact fallback keep the scoring contract honest
- the current full-corpus ordering benchmark on the rendered `tmp_data` set (`547` pages, `8` real-model queries) kept `maxsim_only` as the winner at about `997 ms` average latency and `0.699` `ndcg`; `maxsim_then_solver` and `solver_prefilter_maxsim` both reduced quality while adding latency
- the current optimized multimodal default remains the prototype/HNSW lightweight screening index; the newer centroid backend is wired in behind `ColPaliConfig(screening_backend="centroid")` but remains experimental until it wins the real-model gate
- INT8 is the main optional speed profile where the fused Triton MaxSim path is already strong
- FP8 remains experimental
- RoQ4 is optional for memory reduction and should be treated as a memory-saver profile rather than the default latency path

Evaluator note:

- the screening lanes are documented honestly as experimental/default-adjacent, not as a hidden latency silver bullet
- for the current real-model evidence and promotion thresholds, see `docs/validation/README.md` and `SCREENING_PROMOTION_DECISION_MEMO.md`

## Reference API

Collection kinds:

- `dense`: hybrid dense+sparse retrieval over single vectors
- `late_interaction`: multivector text retrieval via `ColbertIndex`
- `multimodal`: multivector visual retrieval via `ColPaliEngine`

Search semantics:

- `dense` accepts `vector`, `vectors`, and optional `query_text`
- `dense` with only `query_text` uses the sparse BM25 branch
- `SearchPipeline.search()` expects a single dense query vector or sparse query text
- late-interaction multi-vector text queries should use `ColbertIndex` directly
- `late_interaction` and `multimodal` require embedding inputs, not raw text
- `filter` is a flat payload equality filter applied consistently across the supported collection kinds

Preprocessing semantics:

- `POST /reference/preprocess/documents` renders local PDF, DOCX, XLSX, and image inputs into PageBundle-like outputs with `page_id`, `page_number`, `image_path`, and optional extracted text
- the public Python equivalent is `voyager_index.render_documents(...)`
- collection writes still accept embeddings, not raw docs, so the supported product flow is docs -> page images -> embedding provider -> `POST /collections/{name}/points`

Data residency:

- `dense` collections persist under `hybrid/`
- `late_interaction` collections persist under `colbert/` as HDF5 + metadata
- `multimodal` collections persist under `colpali/` as a manifest + chunk files
- in-memory and VRAM caches are runtime accelerators, not separate persistence backends

CRUD and delta updates:

- `POST /collections/{name}/points` is the add/replace/upsert path for local collection maintenance
- `DELETE /collections/{name}/points` removes selected records while preserving the rest of the collection state
- `GET /collections/{name}/info` exposes persisted collection metadata and storage details
- `DELETE /collections/{name}` removes an entire persisted collection

Create a dense collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/docs \
  -H "Content-Type: application/json" \
  -d '{"dimension": 128, "kind": "dense"}'
```

Add multivector late-interaction points:

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "doc-1",
        "vectors": [[0.1, 0.2], [0.3, 0.4]],
        "payload": {"text": "invoice total due"}
      }
    ]
  }'
```

Search:

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1, 0.2], [0.3, 0.4]], "top_k": 5}'
```

Delete one or more points:

```bash
curl -X DELETE http://127.0.0.1:8080/collections/docs/points \
  -H "Content-Type: application/json" \
  -d '{"ids": ["doc-1"]}'
```

## Precision Profiles

Use these profiles as the public OSS guidance:

| Profile | Default | Intended Use | Notes |
| --- | --- | --- | --- |
| `Exact` | yes | prototypes, local production, truthful baseline | FP16 Triton MaxSim for late-interaction and multimodal retrieval |
| `Fast` | opt-in | candidate reranking where latency matters most | INT8 Triton MaxSim where the fused path is already wired |
| `Experimental` | no | hardware-specific benchmarking | FP8 is available, but not yet a native end-to-end default path |
| `Memory Saver` | no | smaller local footprints | RoQ4 materially reduces storage, but usually increases latency and should be enabled deliberately |

## Multimodal Support

Supported phase-1 models:

- `collfm2` — `VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1` (default OSS vLLM ColPali choice)
- `colqwen3` — `VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1`
- `nemotron_colembed` — `nvidia/nemotron-colembed-vl-4b-v2`

See `MULTIMODAL_FOUNDATION.md` for:

- the multimodal model matrix
- ingestion and storage conventions
- chunked manifest-based multimodal persistence
- vLLM pooling provider usage
- collection kind guidance

Runnable examples:

- `examples/reference_api_happy_path.py`
- `examples/reference_api_feature_tour.py`
- `examples/reference_api_late_interaction.py`
- `examples/reference_api_multimodal.py`
- `examples/vllm_pooling_provider.py`
- `notebooks/01_reference_api_happy_path.ipynb`
- `notebooks/02_tmp_data_full_api_tutorial.ipynb`
- `examples/README.md`
- `notebooks/README.md`

## Docker

Build and run the reference service:

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

The container:

- builds the optional native HNSW, solver, and GEM router wheels in a builder stage
- runs as a non-root `voyager` user
- uses the CPU PyTorch wheel for a smaller default local image
- starts the public CLI entrypoint `voyager-index-server`
- ships `/reference/optimize` and dense optimized refinement with `latence_solver` available in the default image
- includes the supported document-rendering runtime for PDF, DOCX, XLSX, and image preprocessing in the default image
- keeps `WORKERS=1` to preserve the local single-writer guarantee
- persists the same disk-backed collection layout under `/data/voyager-index`

Or with Compose:

```bash
docker compose -f deploy/reference-api/docker-compose.yml up --build
```

## Repo Guide

Documentation hub:

- `docs/README.md`: audience-based documentation index
- `docs/reference_api_tutorial.md`: flagship end-to-end API tutorial
- `docs/full_feature_cookbook.md`: step-by-step advanced cookbook for the full OSS feature surface

Public contracts:

- `OSS_FOUNDATION.md`: supported OSS contract
- `MULTIMODAL_FOUNDATION.md`: multimodal contract and provider matrix
- `ADAPTER_CONTRACTS.md`: first-pass cross-stream seams

Evaluation and evidence:

- `BENCHMARKS.md`: supported OSS smoke benchmark harness
- `SCREENING_PROMOTION_DECISION_MEMO.md`: screening promotion policy
- `docs/validation/README.md`: archived validation bundles and who they are for

Contributor material:

- `examples/README.md`: runnable examples index
- `notebooks/README.md`: notebook guide
- `CONTRIBUTING.md`: contribution workflow
- `CHANGELOG.md`: release history
- `tools/README.md`: maintainer-oriented scripts moved out of the landing zone

Legal and vendor:

- `QDRANT_VENDORING.md`: vendored Qdrant boundary
- `LICENSING.md`: repo-level licensing guide
- `THIRD_PARTY_NOTICES.md`: redistributed third-party notices
- `SECURITY.md`: security and disclosure policy

## License

The OSS foundation is Apache-2.0. See `LICENSE`.

Vendored Qdrant code under `src/kernels/vendor/qdrant/` remains Apache-2.0 under
its upstream terms. See `QDRANT_VENDORING.md` and `THIRD_PARTY_NOTICES.md`.
