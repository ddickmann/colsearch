# Changelog

## 0.1.0 — Production

Initial OSS foundation cut for `voyager-index`.

- created the installable `voyager_index` package and root OSS packaging surface
- exposed exact MaxSim and RoQ kernel exports through the public package
- added CPU-safe MaxSim fallback behavior when Triton is unavailable or rejects the shape
- replaced the placeholder in-memory API with a durable reference FastAPI service
- added dense, late-interaction, and multimodal collection kinds with restart-safe persistence
- shipped durable CRUD semantics with point upserts, point deletes, and restart-safe reloads
- added hybrid dense+BM25 retrieval plus optional local solver-backed dense refinement
- exposed the canonical OSS `/reference/optimize` solver API backed by `latence_solver` when installed
- documented multimodal support for `colqwen3`, `collfm2`, and `nemotron_colembed`
- added Docker packaging, examples, notebooks, validation docs, and OSS foundation documents
- published the Apache-2.0 licensing surface and vendor-boundary docs
