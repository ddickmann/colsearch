.PHONY: install install-dev install-native install-cpu build-native \
       test test-rust test-python lint clean help

SHELL := /bin/bash
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Full installs ────────────────────────────────────────────────────

install: ## Full install from source (editable + all native crates)
	bash scripts/install_from_source.sh --skip-system-deps

install-cpu: ## Full install with CPU-only PyTorch
	bash scripts/install_from_source.sh --skip-system-deps --cpu

install-dev: ## Editable install with dev extras (no native crates)
	python -m pip install -e ".[server,multimodal,preprocessing,dev]"

install-system-deps: ## Install system packages (apt/dnf/brew)
	bash scripts/install_from_source.sh 2>&1 | head -20 || true
	@echo "Run the full script for complete install: bash scripts/install_from_source.sh"

# ── Native crates ────────────────────────────────────────────────────

build-native: ## Build all three Rust/PyO3 native crates
	python -m pip install maturin
	python -m pip install ./src/kernels/hnsw_indexer
	python -m pip install ./src/kernels/knapsack_solver
	python -m pip install ./src/kernels/gem_router

install-native: build-native ## Alias for build-native

build-hnsw: ## Build only the HNSW indexer crate
	python -m pip install ./src/kernels/hnsw_indexer

build-solver: ## Build only the knapsack solver crate
	python -m pip install ./src/kernels/knapsack_solver

build-gem-router: ## Build only the GEM router crate
	python -m pip install ./src/kernels/gem_router

# ── Testing ──────────────────────────────────────────────────────────

test: test-rust test-python ## Run all tests

test-rust: ## Run Rust unit tests for all crates
	cd src/kernels/gem_router && cargo test
	cd src/kernels/hnsw_indexer && cargo test
	cd src/kernels/knapsack_solver && cargo test

test-python: ## Run Python test suite
	python -m pytest tests/ -v

test-gem: ## Run GEM router tests only
	cd src/kernels/gem_router && cargo test
	python -m pytest tests/test_gem_router.py -v

# ── Maintenance ──────────────────────────────────────────────────────

lint: ## Run linters
	python -m pip install ruff 2>/dev/null || true
	ruff check voyager_index/ tests/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf src/kernels/gem_router/target
	rm -rf src/kernels/hnsw_indexer/target
	rm -rf src/kernels/knapsack_solver/target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

verify: ## Verify all native modules are importable
	python -c "import voyager_index; print('voyager_index OK')"
	python -c "import latence_hnsw; print('latence_hnsw OK')"
	python -c "import latence_solver; print('latence_solver OK')"
	python -c "import latence_gem_router; print('latence_gem_router OK')"

benchmark: ## Run GEM router benchmark
	python tools/benchmarks/benchmark_audit.py
