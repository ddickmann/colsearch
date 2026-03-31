
import sys
import os
import numpy as np

# Ensure we can import the built module (this path might need adjustment based on where maturin builds it)
# For now, we assume it's installed or in PYTHONPATH
try:
    import latence_solver
    from latence_solver import TabuSearchSolver, SolverConfig, SolverConstraints
except ImportError:
    print("Skipping Knapsack verification (latence_solver not found)")
    sys.exit(0)

def verify_auxiliary_score():
    print("--- Test: Knapsack Auxiliary Score ---")
    
    # 1. Config with only Epsilon (Auxiliary) weight
    # We set alpha (relevance) to 0 to ensure selection is driven by auxiliary
    config = SolverConfig(
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
        epsilon=1.0, # Only auxiliary matters
        lambda_=0.0,
        iterations=100
    )
    
    solver = TabuSearchSolver(config)
    
    # 2. Data
    # Candidate 0: Low Aux (0.1)
    # Candidate 1: High Aux (0.9)
    n = 2
    dim = 2
    embeddings = np.zeros((n, dim), dtype=np.float32)
    query = np.zeros(dim, dtype=np.float32)
    
    token_costs = np.array([10, 10], dtype=np.uint32)
    density_scores = np.zeros(n, dtype=np.float32)
    centrality_scores = np.zeros(n, dtype=np.float32)
    recency_scores = np.zeros(n, dtype=np.float32)
    auxiliary_scores = np.array([0.1, 0.9], dtype=np.float32)
    roles = np.zeros(n, dtype=np.uint8)
    cluster_ids = np.array([-1, -1], dtype=np.int32)
    
    constraints = SolverConstraints(max_tokens=100, max_chunks=1)
    
    # 3. Solve
    result = solver.solve_numpy(
        embeddings,
        query,
        token_costs,
        density_scores,
        centrality_scores,
        recency_scores,
        auxiliary_scores,
        roles,
        cluster_ids,
        constraints
    )
    
    # 4. Assertions
    print(f"DEBUG: Selected Indices: {result.selected_indices}")
    print(f"DEBUG: Objective Score: {result.objective_score}")
    print(f"DEBUG: Auxiliary Total: {result.auxiliary_total}")
    
    # Should select Candidate 1 (Index 1) because it has higher auxiliary score
    assert 1 in result.selected_indices
    assert len(result.selected_indices) == 1
    
    # Expected score = 1.0 * 0.9 = 0.9
    assert np.isclose(result.objective_score, 0.9, atol=1e-4)
    assert np.isclose(result.auxiliary_total, 0.9, atol=1e-4)
    
    print("✅ Auxiliary Score Logic Verified")

if __name__ == "__main__":
    verify_auxiliary_score()
