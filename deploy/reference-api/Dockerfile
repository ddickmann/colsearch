FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install a recent stable Rust via rustup. Debian's `rustc` package on
# python:3.11-slim is currently 1.85, which predates the stabilization of
# the `avx512_target_feature` gate (stable in 1.89+) used by the fused
# RROQ158 AVX-512 VPOPCNTDQ tier in src/kernels/shard_engine. Pin to a
# concrete recent stable so this build is reproducible and not at the
# mercy of whatever rustup currently considers `stable`.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.94.1
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal --no-modify-path \
    && rustc --version \
    && cargo --version

COPY pyproject.toml README.md LICENSE LICENSING.md THIRD_PARTY_NOTICES.md /app/
COPY colsearch /app/colsearch
COPY voyager_index /app/voyager_index
COPY src /app/src

RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools maturin \
    && python -m pip wheel --no-deps --wheel-dir /wheels . \
    && python -m pip wheel --no-deps --wheel-dir /wheels ./src/kernels/shard_engine \
    && python -m pip wheel --no-deps --wheel-dir /wheels ./src/kernels/knapsack_solver

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COLSEARCH_INDEX_PATH=/data/colsearch
ENV HOST=0.0.0.0
ENV PORT=8080
ENV WORKERS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system colsearch \
    && useradd --system --gid colsearch --create-home --home-dir /home/colsearch colsearch \
    && mkdir -p /data/colsearch \
    && chown -R colsearch:colsearch /data /home/colsearch

COPY --from=builder /wheels /wheels

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir /wheels/latence_shard_engine-*.whl /wheels/latence_solver-*.whl \
    && COLSEARCH_WHEEL=$(ls /wheels/colsearch-*.whl) \
    && python -m pip install --no-cache-dir "colsearch[full] @ file://${COLSEARCH_WHEEL}"

USER colsearch

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["colsearch-server"]
