"""
Tests for the Latence Solver Python bindings.

These tests work with both the Rust extension and the pure Python fallback.
"""

import pytest
import numpy as np
import sys
import os
import types

try:
    import latence_solver  # type: ignore
except ImportError:
    # Fall back to the source tree only when the package is not already installed.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from latence_solver import (
    TabuSearchSolver,
    SolverConfig,
    SolverConstraints,
    SolverOutput,
    backend_status,
    cpu_startup_self_test,
    is_rust_available,
    premium_backend_contract,
    gpu_available,
    gpu_startup_self_test,
    version,
)


@pytest.fixture
def sample_chunks():
    """Generate sample chunks for testing."""
    np.random.seed(42)
    n = 20
    dim = 64
    
    return [
        {
            "chunk_id": f"chunk_{i}",
            "content": f"This is the content of chunk {i}.",
            "embedding": np.random.randn(dim).astype(np.float32).tolist(),
            "token_count": 50 + np.random.randint(0, 200),
            "fact_density": np.random.random(),
            "centrality_score": np.random.random(),
            "recency_score": np.random.random(),
            "rhetorical_role": ["definition", "example", "evidence", "risk"][i % 4],
            "cluster_id": i % 5,
        }
        for i in range(n)
    ]


@pytest.fixture
def query_embedding():
    """Generate a query embedding."""
    np.random.seed(123)
    return np.random.randn(64).astype(np.float32)


class TestSolverConfig:
    """Test SolverConfig."""
    
    def test_default_config(self):
        config = SolverConfig()
        assert config.alpha == 1.0
        assert config.mu == pytest.approx(1.0)
        assert config.iterations == 100
    
    def test_custom_config(self):
        config = SolverConfig(
            alpha=0.5,
            beta=0.4,
            iterations=50,
            use_gpu=False,
        )
        assert config.alpha == pytest.approx(0.5)
        assert config.beta == pytest.approx(0.4)
        assert config.iterations == 50
    
    def test_config_repr(self):
        config = SolverConfig()
        assert "SolverConfig" in repr(config)


class TestSolverConstraints:
    """Test SolverConstraints."""
    
    def test_default_constraints(self):
        constraints = SolverConstraints()
        assert constraints.max_tokens == 8192
        assert constraints.max_chunks == 50
    
    def test_custom_constraints(self):
        constraints = SolverConstraints(
            max_tokens=4096,
            max_per_cluster=2,
        )
        assert constraints.max_tokens == 4096
        assert constraints.max_per_cluster == 2
    
    def test_constraints_repr(self):
        constraints = SolverConstraints()
        assert "SolverConstraints" in repr(constraints)


class TestTabuSearchSolver:
    """Test TabuSearchSolver."""
    
    def test_solver_creation(self):
        solver = TabuSearchSolver()
        assert solver is not None
    
    def test_solver_with_config(self):
        config = SolverConfig(iterations=50)
        solver = TabuSearchSolver(config)
        assert solver is not None
        assert solver.backend_kind() == "cpu_reference"
    
    def test_solve_basic(self, sample_chunks, query_embedding):
        config = SolverConfig(iterations=20, use_gpu=False)
        solver = TabuSearchSolver(config)
        constraints = SolverConstraints(max_tokens=1000)
        
        output = solver.solve(sample_chunks, query_embedding, constraints)
        
        assert isinstance(output, SolverOutput)
        assert output.num_selected > 0
        assert output.total_tokens <= 1000
        assert output.objective_score > 0
    
    def test_solve_respects_budget(self, sample_chunks, query_embedding):
        config = SolverConfig(iterations=30, use_gpu=False)
        solver = TabuSearchSolver(config)
        
        for budget in [500, 1000, 2000]:
            constraints = SolverConstraints(max_tokens=budget)
            output = solver.solve(sample_chunks, query_embedding, constraints)
            
            assert output.total_tokens <= budget, f"Budget {budget} exceeded"

    def test_solve_preserves_required_role_under_constraints(self, query_embedding):
        solver = TabuSearchSolver(SolverConfig(iterations=40, use_gpu=False, random_seed=7))
        chunks = [
            {
                "chunk_id": "definition_a",
                "content": "Definition",
                "embedding": np.concatenate(([1.0], np.zeros(63))).astype(np.float32).tolist(),
                "token_count": 100,
                "rhetorical_role": "definition",
                "cluster_id": 0,
            },
            {
                "chunk_id": "example_a",
                "content": "Example",
                "embedding": np.concatenate(([0.95], np.zeros(63))).astype(np.float32).tolist(),
                "token_count": 100,
                "rhetorical_role": "example",
                "cluster_id": 0,
            },
            {
                "chunk_id": "risk_a",
                "content": "Risk",
                "embedding": np.concatenate(([0.9], np.zeros(63))).astype(np.float32).tolist(),
                "token_count": 100,
                "rhetorical_role": "risk",
                "cluster_id": 1,
            },
        ]
        constraints = SolverConstraints(
            max_tokens=220,
            max_chunks=2,
            max_per_cluster=1,
            must_include_roles=["risk"],
        )

        output = solver.solve(chunks, query_embedding, constraints)

        assert output.constraints_satisfied
        selected = [chunks[index] for index in output.selected_indices]
        assert any(chunk["rhetorical_role"] == "risk" for chunk in selected)
        assert len({chunk["cluster_id"] for chunk in selected}) == len(selected)

    def test_solve_preserves_required_chunks(self, query_embedding):
        solver = TabuSearchSolver(SolverConfig(iterations=40, use_gpu=False, random_seed=11))
        chunks = [
            {
                "chunk_id": "must_keep",
                "content": "Critical chunk",
                "embedding": np.concatenate(([1.0], np.zeros(63))).astype(np.float32).tolist(),
                "token_count": 90,
                "rhetorical_role": "definition",
                "cluster_id": 0,
            },
            {
                "chunk_id": "competitor_a",
                "content": "Competitor A",
                "embedding": np.concatenate(([0.99], np.zeros(63))).astype(np.float32).tolist(),
                "token_count": 90,
                "rhetorical_role": "example",
                "cluster_id": 1,
            },
            {
                "chunk_id": "competitor_b",
                "content": "Competitor B",
                "embedding": np.concatenate(([0.98], np.zeros(63))).astype(np.float32).tolist(),
                "token_count": 90,
                "rhetorical_role": "evidence",
                "cluster_id": 2,
            },
        ]
        constraints = SolverConstraints(
            max_tokens=180,
            max_chunks=2,
            required_chunks=["must_keep"],
        )

        output = solver.solve(chunks, query_embedding, constraints)

        assert output.constraints_satisfied
        assert "must_keep" in {chunks[index]["chunk_id"] for index in output.selected_indices}
    
    def test_solve_empty_input(self, query_embedding):
        solver = TabuSearchSolver()
        constraints = SolverConstraints()
        
        output = solver.solve([], query_embedding, constraints)
        
        assert output.num_selected == 0
        assert output.total_tokens == 0
        assert output.constraints_satisfied
    
    def test_solve_single_chunk(self, query_embedding):
        solver = TabuSearchSolver()
        chunks = [{
            "embedding": np.random.randn(64).astype(np.float32).tolist(),
            "token_count": 100,
        }]
        constraints = SolverConstraints(max_tokens=500)
        
        output = solver.solve(chunks, query_embedding, constraints)
        
        assert output.num_selected == 1
        assert output.total_tokens == 100
    
    def test_output_structure(self, sample_chunks, query_embedding):
        solver = TabuSearchSolver(SolverConfig(iterations=10))
        constraints = SolverConstraints(max_tokens=1000)
        
        output = solver.solve(sample_chunks, query_embedding, constraints)
        
        # Check all output fields
        assert hasattr(output, 'selected_indices')
        assert hasattr(output, 'objective_score')
        assert hasattr(output, 'relevance_total')
        assert hasattr(output, 'density_total')
        assert hasattr(output, 'centrality_total')
        assert hasattr(output, 'recency_total')
        assert hasattr(output, 'fulfilment_total')
        assert hasattr(output, 'redundancy_penalty')
        assert hasattr(output, 'total_tokens')
        assert hasattr(output, 'num_selected')
        assert hasattr(output, 'iterations_run')
        assert hasattr(output, 'constraints_satisfied')
        assert hasattr(output, 'solve_time_ms')
    
    def test_output_to_dict(self, sample_chunks, query_embedding):
        solver = TabuSearchSolver(SolverConfig(iterations=10))
        constraints = SolverConstraints(max_tokens=1000)
        
        output = solver.solve(sample_chunks, query_embedding, constraints)
        result = output.to_dict()
        
        assert isinstance(result, dict)
        assert 'selected_indices' in result
        assert 'objective_score' in result


class TestNumPyAPI:
    """Test NumPy array API."""
    
    def test_solve_numpy(self):
        np.random.seed(42)
        n = 50
        dim = 128
        
        embeddings = np.random.randn(n, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)
        token_costs = np.random.randint(50, 200, n).astype(np.uint32)
        density = np.random.random(n).astype(np.float32)
        centrality = np.random.random(n).astype(np.float32)
        recency = np.random.random(n).astype(np.float32)
        auxiliary = np.zeros(n, dtype=np.float32)
        roles = np.random.randint(0, 5, n).astype(np.uint8)
        clusters = np.random.randint(-1, 10, n).astype(np.int32)
        
        solver = TabuSearchSolver(SolverConfig(iterations=20, use_gpu=False))
        constraints = SolverConstraints(max_tokens=2000)
        
        output = solver.solve_numpy(
            embeddings, query, token_costs,
            density, centrality, recency,
            auxiliary, roles, clusters, constraints
        )
        
        assert output.num_selected > 0
        assert output.total_tokens <= 2000

    def test_solve_precomputed_numpy_tracks_fulfilment(self):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )
        token_costs = np.asarray([80, 80, 80], dtype=np.uint32)
        zeros = np.zeros((3,), dtype=np.float32)
        roles = np.asarray([255, 255, 255], dtype=np.uint8)
        clusters = np.asarray([-1, -1, -1], dtype=np.int32)
        relevance = np.asarray([0.2, 0.1, 0.7], dtype=np.float32)
        fulfilment = np.asarray([0.0, 0.0, 0.9], dtype=np.float32)
        similarity = np.asarray(
            [
                [0.0, 0.1, 0.8],
                [0.1, 0.0, 0.2],
                [0.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        coverage = np.asarray(
            [
                [0.2, 0.0, 0.9],
                [0.0, 0.1, 0.8],
            ],
            dtype=np.float32,
        )
        weights = np.asarray([0.6, 0.4], dtype=np.float32)

        solver = TabuSearchSolver(SolverConfig(iterations=20, use_gpu=False, mu=1.0, lambda_=0.05))
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
            fulfilment_scores=fulfilment,
            coverage_matrix=coverage,
            query_token_weights=weights,
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            constraints=SolverConstraints(max_tokens=100, max_chunks=1),
        )

        assert output.selected_indices == [2]
        assert output.fulfilment_total > 0.5


class TestDeterminism:
    """Test solver determinism with random seed."""
    
    def test_reproducible_results(self, sample_chunks, query_embedding):
        config = SolverConfig(iterations=30, random_seed=12345, use_gpu=False)
        constraints = SolverConstraints(max_tokens=1000)
        
        solver1 = TabuSearchSolver(config)
        output1 = solver1.solve(sample_chunks, query_embedding, constraints)
        
        solver2 = TabuSearchSolver(config)
        output2 = solver2.solve(sample_chunks, query_embedding, constraints)
        
        assert output1.selected_indices == output2.selected_indices
        assert abs(output1.objective_score - output2.objective_score) < 1e-6


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_is_rust_available(self):
        result = is_rust_available()
        assert isinstance(result, bool)
    
    def test_backend_status_reports_production_backend(self):
        status = backend_status()
        assert status["cpu_reference_available"] is True
        assert status["default_backend"] == status["production_solver_backend"]
        assert "cpu_reference" in status["available_backends"]
        if status["experimental_backends_production_ready"]:
            assert status["production_execution_mode"] == "end_to_end_gpu_search"
            assert status["production_solver_backend"] != "cpu_reference"
        else:
            assert status["production_execution_mode"] == "gpu_precompute_cpu_search"
            assert status["production_solver_backend"] == "cpu_reference"

    def test_premium_backend_contract_shape(self):
        contract = premium_backend_contract()
        assert contract["version"] >= 1
        assert "create_solver" in contract["factory"]
        assert "selected_indices" in contract["parity_fields"]

    def test_gpu_available(self):
        result = gpu_available()
        assert isinstance(result, bool)

    def test_gpu_startup_self_test_shape(self):
        result = gpu_startup_self_test()
        assert isinstance(result, dict)
        assert "ok" in result

    def test_cpu_startup_self_test_shape(self):
        result = cpu_startup_self_test()
        assert isinstance(result, dict)
        assert result["requested_use_gpu"] is False
        assert "backend_kind" in result
    
    def test_version(self):
        v = version()
        assert isinstance(v, str)
        assert len(v) > 0


class TestPerformance:
    """Performance-related tests."""
    
    @pytest.mark.parametrize("n_chunks", [10, 50, 100])
    def test_scaling(self, n_chunks, query_embedding):
        np.random.seed(42)
        
        chunks = [
            {
                "embedding": np.random.randn(64).astype(np.float32).tolist(),
                "token_count": 100,
            }
            for _ in range(n_chunks)
        ]
        
        solver = TabuSearchSolver(SolverConfig(iterations=20, use_gpu=False))
        constraints = SolverConstraints(max_tokens=n_chunks * 50)
        
        output = solver.solve(chunks, query_embedding, constraints)
        
        assert output.num_selected > 0
        assert output.solve_time_ms < 5000  # Should complete in reasonable time


def test_requested_gpu_uses_production_backend():
    solver = TabuSearchSolver(SolverConfig(iterations=5, use_gpu=True, random_seed=7))
    status = backend_status()
    assert solver.backend_kind() == status["production_solver_backend"]


def test_strict_gpu_mode_raises_without_backend(monkeypatch):
    if gpu_available():
        pytest.skip("Strict GPU fallback behavior is only asserted when no GPU backend is available")
    monkeypatch.setenv("LATENCE_SOLVER_STRICT_GPU", "1")
    with pytest.raises(RuntimeError):
        TabuSearchSolver(SolverConfig(iterations=5, use_gpu=True, random_seed=7))


def test_premium_plugin_contract_keeps_cpu_parity(sample_chunks, query_embedding, monkeypatch):
    def create_solver(*, config, reference_solver_cls, solver_config_cls, solver_constraints_cls):
        class DummyPremiumSolver:
            def __init__(self):
                self._delegate = reference_solver_cls(
                    solver_config_cls(
                        alpha=float(config.alpha),
                        beta=float(config.beta),
                        gamma=float(config.gamma),
                        delta=float(config.delta),
                        epsilon=float(config.epsilon),
                        mu=float(config.mu),
                        lambda_=float(config.lambda_),
                        iterations=int(config.iterations),
                        tabu_tenure=int(config.tabu_tenure),
                        early_stopping_patience=int(config.early_stopping_patience),
                        use_gpu=False,
                        random_seed=config.random_seed,
                    )
                )

            def backend_kind(self):
                return "premium_dummy"

            def solve(self, *args, **kwargs):
                return self._delegate.solve(*args, **kwargs)

            def solve_numpy(self, *args, **kwargs):
                return self._delegate.solve_numpy(*args, **kwargs)

            def solve_precomputed_numpy(self, *args, **kwargs):
                return self._delegate.solve_precomputed_numpy(*args, **kwargs)

        return DummyPremiumSolver()

    module = types.ModuleType("latence_solver_dummy_plugin")
    module.create_solver = create_solver
    module.backend_status = lambda: {"available": True, "name": "dummy"}

    monkeypatch.setenv("LATENCE_SOLVER_PREMIUM_BACKEND", "latence_solver_dummy_plugin")
    monkeypatch.setitem(sys.modules, "latence_solver_dummy_plugin", module)

    config = SolverConfig(iterations=20, random_seed=1234, use_gpu=True)
    constraints = SolverConstraints(max_tokens=1000)
    cpu_output = TabuSearchSolver(SolverConfig(iterations=20, random_seed=1234, use_gpu=False)).solve(
        sample_chunks,
        query_embedding,
        constraints,
    )
    premium_solver = TabuSearchSolver(config)
    premium_output = premium_solver.solve(sample_chunks, query_embedding, constraints)

    assert premium_solver.backend_kind() == "premium_dummy"
    assert premium_output.selected_indices == cpu_output.selected_indices
    assert premium_output.constraints_satisfied == cpu_output.constraints_satisfied
    assert premium_output.objective_score == pytest.approx(cpu_output.objective_score, abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

