
import time
import numpy as np
import latence_solver
from latence_solver import TabuSearchSolver, SolverConfig, SolverConstraints

print(f"Latence Solver Version: {latence_solver.version()}")
print(f"GPU Available: {latence_solver.gpu_available()}")

# Configuration
n_candidates = 500
dim = 128
iterations = 20

# Synthetic Data
embeddings = np.random.randn(n_candidates, dim).astype(np.float32)
query_embedding = np.random.randn(dim).astype(np.float32)
token_costs = np.random.randint(10, 100, size=n_candidates).astype(np.uint32)
density_scores = np.random.rand(n_candidates).astype(np.float32)
centrality_scores = np.random.rand(n_candidates).astype(np.float32)
recency_scores = np.random.rand(n_candidates).astype(np.float32)
auxiliary_scores = np.zeros(n_candidates, dtype=np.float32)
roles = np.zeros(n_candidates, dtype=np.uint8) # 0 = UNKNOWN
cluster_ids = np.full(n_candidates, -1, dtype=np.int32)

# Solver Setup
config = SolverConfig(
    iterations=iterations,
    use_gpu=False, # CPU for fairness with previous test, or enable if available
    alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, lambda_=1.0
)
solver = TabuSearchSolver(config)
constraints = SolverConstraints(max_tokens=2000)

print("="*50)
print("Rust Knapsack Verification")
print("="*50)

# Warmup
solver.solve_numpy(
    embeddings, query_embedding, token_costs,
    density_scores, centrality_scores, recency_scores,
    auxiliary_scores, roles, cluster_ids, constraints
)

# Benchmark
start = time.time()
result = solver.solve_numpy(
    embeddings, query_embedding, token_costs,
    density_scores, centrality_scores, recency_scores,
    auxiliary_scores, roles, cluster_ids, constraints
)
latency = (time.time() - start) * 1000

print(f"Latency: {latency:.2f}ms")
print(f"Selected: {result.num_selected}")
print(f"Score: {result.objective_score:.4f}")
print(f"Status: {'✓ PASS' if latency < 50 else '⚠ SLOW'}")
