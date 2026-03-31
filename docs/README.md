# Documentation Guide

`voyager-index` ships one OSS product surface, but several different reading
paths.

Current release posture:

- users should `git clone` this repo and install locally from source
- the fastest path is `bash scripts/install_from_source.sh --cpu` which handles
  system dependencies, Rust toolchain, Python package, and all native Rust/PyO3
  crates in a single command
- alternatively, use `make install-cpu` or follow the step-by-step instructions in `README.md`
- the install commands in `README.md`, `docs/reference_api_tutorial.md`, and `docs/full_feature_cookbook.md` assume the repo root
- no PyPI package is required for the current OSS release
- `README.md` is the product homepage; the docs below expand the same public OSS story rather than replacing it

## Start Here

- New users: `README.md`
- End-to-end API tutorial: `docs/reference_api_tutorial.md`
- Full-feature cookbook: `docs/full_feature_cookbook.md`
- Runnable advanced feature tour: `examples/reference_api_feature_tour.py`
- Runnable examples: `examples/README.md`
- Notebooks: `notebooks/README.md`

## Public Contracts

- `OSS_FOUNDATION.md`: supported public Python and API contract
- `MULTIMODAL_FOUNDATION.md`: multimodal model matrix, storage, and scoring guidance
- `ADAPTER_CONTRACTS.md`: documentation-level seams between OSS, providers, and future sidecars

## Evaluation And Evidence

- `BENCHMARKS.md`: reproducible OSS smoke benchmark harness
- `SCREENING_PROMOTION_DECISION_MEMO.md`: current promotion policy for multimodal screening
- `docs/validation/README.md`: archived validation bundles and what each one proves

## Contributors And Release

- `CONTRIBUTING.md`: setup, validation, and contribution workflow
- `CHANGELOG.md`: release history
- `SECURITY.md`: supported surface and disclosure instructions
- `tools/README.md`: contributor-only scripts moved out of the repo landing zone

## Legal And Vendor

- `LICENSING.md`: repo-level licensing guide
- `QDRANT_VENDORING.md`: vendored Qdrant boundary
- `THIRD_PARTY_NOTICES.md`: redistributed third-party notices
