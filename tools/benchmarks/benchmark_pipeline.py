
import time
import numpy as np
import shutil
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.search_pipeline import SearchPipeline

def benchmark_pipeline():
    logger = logging.getLogger("Benchmark")
    print("--- 🚀 Latence Neural Search: Pipeline Benchmark 🚀 ---")
    
    data_path = "./data/benchmark_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    # Config
    N_DOCS = 5000
    DIM = 128
    QUERY_BATCH = 100
    ROQ_BITS = 4
    
    # Initialize Pipeline
    print(f"Initializing Index (N={N_DOCS}, Dim={DIM}, RoQ={ROQ_BITS}-bit)...")
    start_init = time.time()
    pipeline = SearchPipeline(data_path, dim=DIM, use_roq=True, roq_bits=ROQ_BITS)
    print(f"Init Time: {time.time() - start_init:.4f}s")
    
    # Generate Synthetic Data
    print("Generating synthetic data...")
    vectors = np.random.randn(N_DOCS, DIM).astype(np.float32)
    # Normalize vectors for cosine similarity relevance
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    ids = list(range(N_DOCS))
    corpus = [f"Synthetic Document {i}" for i in range(N_DOCS)]
    
    # Ingest
    print("Ingesting data...")
    start_ingest = time.time()
    # Batch ingestion if needed, but pipeline.index takes all
    pipeline.index(corpus, vectors, ids)
    ingest_time = time.time() - start_ingest
    print(f"Ingest Time: {ingest_time:.4f}s ({N_DOCS/ingest_time:.1f} docs/s)")
    
    # Prepare Queries
    queries = np.random.randn(QUERY_BATCH, DIM).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Benchmark Search
    print(f"\nRunning Search Benchmark ({QUERY_BATCH} queries)...")
    
    latencies = []
    map_times = [] # Note: Pipeline doesn't expose internal split times easily without instrumentation.
    # We will measure total E2E latency.
    
    for i in range(QUERY_BATCH):
        q = queries[i]
        t0 = time.time()
        
        # Search: Map (HNSW) + Compass (RoQ) + Driver (Solver)
        results = pipeline.search(q, top_k_retrieval=100, enable_refinement=True)
        
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
        
        # Verify correctness (sanity)
        if i == 0:
            logger.info(f"Sample Result: {len(results.get('selected_ids', []))} items selected")
            
    # Metrics
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_lat = np.mean(latencies)
    qps = 1000.0 / avg_lat
    
    print("\n--- 📊 Results ---")
    print(f"Total Queries: {QUERY_BATCH}")
    print(f"Latency P50:   {p50:.2f} ms")
    print(f"Latency P95:   {p95:.2f} ms")
    print(f"Latency P99:   {p99:.2f} ms")
    print(f"Average:       {avg_lat:.2f} ms")
    print(f"Throughput:    {qps:.2f} QPS (Sequential)")
    
    # Verify RoQ Influence
    # We check if RoQ codes are actually populated (sanity check from verify script logic)
    # Using internal access
    payload = pipeline.manager.hnsw.retrieve([0])[0]['payload']
    has_roq = 'roq_codes' in payload
    print(f"\nRoQ Active: {'✅ Yes' if has_roq else '❌ No'}")
    
    print("------------------------------------------------")

if __name__ == "__main__":
    benchmark_pipeline()
