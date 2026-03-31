from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from .latence_solver import *  # noqa: F401,F403
from .latence_solver import (
    SolverConfig,
    SolverConstraints,
    TabuSearchSolver as _RustTabuSearchSolver,
    cuda_available as _native_cuda_available,
    gpu_available as _native_gpu_available,
)

_PREMIUM_BACKEND_ENV = "LATENCE_SOLVER_PREMIUM_BACKEND"
_EXPERIMENTAL_BACKEND_ENV = "LATENCE_SOLVER_ENABLE_EXPERIMENTAL_BACKENDS"
_STRICT_GPU_ENV = "LATENCE_SOLVER_STRICT_GPU"
_CONTRACT_VERSION = 1
_CPU_REFERENCE_BACKEND = "cpu_reference"
_END_TO_END_GPU_EXECUTION_MODE = "end_to_end_gpu_search"
_HYBRID_EXECUTION_MODE = "gpu_precompute_cpu_search"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _experimental_backends_enabled() -> bool:
    value = os.environ.get(_EXPERIMENTAL_BACKEND_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _copy_solver_config(config: SolverConfig, *, use_gpu: bool) -> SolverConfig:
    return SolverConfig(
        alpha=float(config.alpha),
        beta=float(config.beta),
        gamma=float(config.gamma),
        delta=float(config.delta),
        epsilon=float(config.epsilon),
        mu=float(getattr(config, "mu", 1.0)),
        support_secondary_discount=float(getattr(config, "support_secondary_discount", 0.35)),
        support_quorum_bonus=float(getattr(config, "support_quorum_bonus", 0.18)),
        support_quorum_threshold=float(getattr(config, "support_quorum_threshold", 0.55)),
        support_quorum_cap=int(getattr(config, "support_quorum_cap", 4)),
        lambda_=float(config.lambda_),
        iterations=int(config.iterations),
        tabu_tenure=int(config.tabu_tenure),
        early_stopping_patience=int(config.early_stopping_patience),
        use_gpu=use_gpu,
        random_seed=config.random_seed,
        enable_gpu_move_evaluation=bool(getattr(config, "enable_gpu_move_evaluation", True)),
        enable_path_relinking=bool(getattr(config, "enable_path_relinking", True)),
        enable_destroy_repair=bool(getattr(config, "enable_destroy_repair", True)),
        enable_reactive_tenure=bool(getattr(config, "enable_reactive_tenure", True)),
        enable_exact_window=bool(getattr(config, "enable_exact_window", True)),
        exact_window_size=int(getattr(config, "exact_window_size", 14)),
        exact_window_time_ms=int(getattr(config, "exact_window_time_ms", 25)),
    )


def _load_premium_backend() -> tuple[Any | None, str | None]:
    module_name = os.environ.get(_PREMIUM_BACKEND_ENV, "").strip()
    if not module_name:
        return None, None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on private plugin
        return None, str(exc)
    return module, None


def premium_backend_contract() -> dict[str, Any]:
    return {
        "version": _CONTRACT_VERSION,
        "env_var": _PREMIUM_BACKEND_ENV,
        "factory": "create_solver(config, reference_solver_cls, solver_config_cls, solver_constraints_cls)",
        "required_methods": ["solve", "solve_numpy", "solve_precomputed_numpy"],
        "required_outputs": [
            "selected_indices",
            "objective_score",
            "num_selected",
            "constraints_satisfied",
            "solve_time_ms",
        ],
        "parity_fields": [
            "selected_indices",
            "objective_score",
            "total_tokens",
            "constraints_satisfied",
            "num_selected",
        ],
    }


def _validate_premium_solver(instance: Any) -> None:
    missing = [
        method_name
        for method_name in ("solve", "solve_numpy", "solve_precomputed_numpy")
        if not callable(getattr(instance, method_name, None))
    ]
    if missing:
        raise TypeError(
            "Premium solver backend does not satisfy the OSS contract; "
            f"missing methods: {', '.join(missing)}"
        )


def premium_backend_available() -> bool:
    module, error = _load_premium_backend()
    if module is None or error is not None:
        return False
    status = getattr(module, "backend_status", None)
    if callable(status):
        try:
            details = status()
        except Exception:  # pragma: no cover - private plugin behavior
            return False
        return bool(details.get("available", True))
    return callable(getattr(module, "create_solver", None))


def backend_status() -> dict[str, Any]:
    module, premium_error = _load_premium_backend()
    premium_configured = bool(os.environ.get(_PREMIUM_BACKEND_ENV, "").strip())
    premium_factory = callable(getattr(module, "create_solver", None)) if module is not None else False
    premium_available = premium_factory and premium_error is None and premium_backend_available()
    experimental_enabled = _experimental_backends_enabled()
    native_cuda = bool(_native_cuda_available()) if experimental_enabled else False
    native_gpu = bool(_native_gpu_available()) if experimental_enabled else False
    experimental_solver_backend = _CPU_REFERENCE_BACKEND
    if experimental_enabled and (native_cuda or native_gpu):
        try:
            experimental_solver = _RustTabuSearchSolver(SolverConfig(iterations=2, use_gpu=True))
            if callable(getattr(experimental_solver, "backend_kind", None)):
                experimental_solver_backend = str(experimental_solver.backend_kind())
        except Exception:
            experimental_solver_backend = _CPU_REFERENCE_BACKEND
    if premium_available:
        production_solver_backend = "premium_plugin"
        production_execution_mode = _END_TO_END_GPU_EXECUTION_MODE
    elif experimental_solver_backend != _CPU_REFERENCE_BACKEND:
        production_solver_backend = experimental_solver_backend
        production_execution_mode = _END_TO_END_GPU_EXECUTION_MODE
    else:
        production_solver_backend = _CPU_REFERENCE_BACKEND
        production_execution_mode = _HYBRID_EXECUTION_MODE
    available_backends = ["cpu_reference"]
    if premium_available:
        available_backends.append("premium_plugin")
    elif experimental_solver_backend != _CPU_REFERENCE_BACKEND:
        available_backends.append(experimental_solver_backend)
    return {
        "contract_version": _CONTRACT_VERSION,
        "cpu_reference_available": True,
        "cpu_reference_backend": _CPU_REFERENCE_BACKEND,
        "production_execution_mode": production_execution_mode,
        "production_solver_backend": production_solver_backend,
        "production_gpu_semantics": "When a premium or experimental GPU backend is available, `/optimize` can run end-to-end GPU search; otherwise the service remains GPU precompute plus CPU reference search.",
        "experimental_backends_production_ready": experimental_solver_backend != _CPU_REFERENCE_BACKEND,
        "strict_gpu_required": _env_flag(_STRICT_GPU_ENV),
        "premium_backend_configured": premium_configured,
        "premium_backend_importable": module is not None,
        "premium_backend_available": premium_available,
        "premium_backend_error": premium_error,
        "experimental_backends_enabled": experimental_enabled,
        "experimental_cuda_available": native_cuda,
        "experimental_gpu_available": native_gpu,
        "experimental_solver_backend": experimental_solver_backend,
        "default_backend": production_solver_backend,
        "available_backends": available_backends,
    }


def is_rust_available() -> bool:
    """Compatibility shim for validators and legacy callers."""
    return True


def gpu_available() -> bool:
    status = backend_status()
    return bool(
        status["premium_backend_available"]
        or status["experimental_cuda_available"]
        or status["experimental_gpu_available"]
    )


def cuda_available() -> bool:
    status = backend_status()
    return bool(status["premium_backend_available"] or status["experimental_cuda_available"])


def solver_startup_self_test(*, use_gpu: bool = False) -> dict[str, Any]:
    status = backend_status()
    if use_gpu and not gpu_available():
        return {"ok": False, "reason": "gpu_backend_unavailable", "status": status}

    try:
        solver = TabuSearchSolver(SolverConfig(iterations=5, use_gpu=use_gpu, random_seed=7))
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        token_costs = np.asarray([32, 32], dtype=np.uint32)
        zeros = np.asarray([0.0, 0.0], dtype=np.float32)
        roles = np.asarray([255, 255], dtype=np.uint8)
        clusters = np.asarray([-1, -1], dtype=np.int32)
        relevance = np.asarray([1.0, 0.1], dtype=np.float32)
        similarity = np.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=np.float32)
        coverage = np.asarray([[1.0, 0.1]], dtype=np.float32)
        weights = np.asarray([1.0], dtype=np.float32)
        output = solver.solve_precomputed_numpy(
            embeddings,
            token_costs,
            zeros,
            zeros,
            zeros,
            zeros,
            roles,
            clusters,
            relevance,
            similarity_matrix=similarity,
            fulfilment_scores=relevance,
            coverage_matrix=coverage,
            query_token_weights=weights,
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            constraints=SolverConstraints(max_tokens=64, max_chunks=1),
        )
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "status": status}

    return {
        "ok": output.num_selected >= 1 and output.constraints_satisfied,
        "requested_use_gpu": use_gpu,
        "backend_kind": solver.backend_kind(),
        "selected_indices": list(output.selected_indices),
        "objective_score": float(output.objective_score),
        "status": status,
    }


def cpu_startup_self_test() -> dict[str, Any]:
    return solver_startup_self_test(use_gpu=False)


def gpu_startup_self_test() -> dict[str, Any]:
    return solver_startup_self_test(use_gpu=True)


class TabuSearchSolver:
    """CPU-reference-first wrapper with an optional premium backend hook."""

    def __init__(self, config: SolverConfig | None = None):
        requested = config if config is not None else SolverConfig()
        self.config = requested
        self._backend_kind = "cpu_reference"
        self._delegate: Any
        strict_gpu = bool(getattr(requested, "use_gpu", False)) and _env_flag(_STRICT_GPU_ENV)

        module, premium_error = _load_premium_backend()
        if bool(getattr(requested, "use_gpu", False)) and module is not None and premium_error is None:
            factory = getattr(module, "create_solver", None)
            if callable(factory):
                delegate = factory(
                    config=requested,
                    reference_solver_cls=_RustTabuSearchSolver,
                    solver_config_cls=SolverConfig,
                    solver_constraints_cls=SolverConstraints,
                )
                _validate_premium_solver(delegate)
                self._delegate = delegate
                backend_kind = getattr(self._delegate, "backend_kind", "premium_plugin")
                self._backend_kind = backend_kind() if callable(backend_kind) else str(backend_kind)
                return

        rust_config = requested
        if bool(getattr(requested, "use_gpu", False)):
            status = backend_status()
            if status["experimental_solver_backend"] != _CPU_REFERENCE_BACKEND:
                self._backend_kind = str(status["experimental_solver_backend"])
            else:
                if strict_gpu:
                    raise RuntimeError(
                        "GPU backend requested but unavailable; strict GPU mode forbids CPU fallback"
                    )
                rust_config = _copy_solver_config(requested, use_gpu=False)
        self._delegate = _RustTabuSearchSolver(rust_config)
        backend_kind = getattr(self._delegate, "backend_kind", None)
        if callable(backend_kind):
            self._backend_kind = str(backend_kind())

    def backend_kind(self) -> str:
        return self._backend_kind

    def solve(self, chunks, query_embedding, constraints: SolverConstraints | None = None):
        return self._delegate.solve(chunks, query_embedding, constraints)

    def solve_numpy(
        self,
        embeddings,
        query_embedding,
        token_costs,
        density_scores,
        centrality_scores,
        recency_scores,
        auxiliary_scores,
        roles,
        cluster_ids,
        constraints: SolverConstraints | None = None,
    ):
        return self._delegate.solve_numpy(
            embeddings,
            query_embedding,
            token_costs,
            density_scores,
            centrality_scores,
            recency_scores,
            auxiliary_scores,
            roles,
            cluster_ids,
            constraints,
        )

    def solve_precomputed_numpy(
        self,
        embeddings,
        token_costs,
        density_scores,
        centrality_scores,
        recency_scores,
        auxiliary_scores,
        roles,
        cluster_ids,
        relevance_scores,
        similarity_matrix=None,
        fulfilment_scores=None,
        coverage_matrix=None,
        query_token_weights=None,
        query_embedding=None,
        constraints: SolverConstraints | None = None,
    ):
        return self._delegate.solve_precomputed_numpy(
            embeddings,
            token_costs,
            density_scores,
            centrality_scores,
            recency_scores,
            auxiliary_scores,
            roles,
            cluster_ids,
            relevance_scores,
            similarity_matrix,
            fulfilment_scores,
            coverage_matrix,
            query_token_weights,
            query_embedding,
            constraints,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def __repr__(self) -> str:
        return f"TabuSearchSolver(backend={self._backend_kind!r})"


__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"importlib", "os", "np", "Any"}
]
