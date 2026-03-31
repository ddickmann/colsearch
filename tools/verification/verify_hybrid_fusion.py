
import numpy as np
import shutil
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.index_core.hybrid_manager import HybridSearchManager

def verify_hybrid_fusion():
    print("--- Test: Hybrid Search Fusion ---")
    data_path = "./data/verify_hybrid_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    dim = 4
    manager = HybridSearchManager(data_path, dim=dim, on_disk=True)
    
    # Data
    vectors = np.array([
        [0.9, 0.1, 0.0, 0.0], # A (Dense match for Q1)
        [0.1, 0.9, 0.0, 0.0], # B
        [0.0, 0.0, 0.9, 0.1], # C
        [0.1, 0.1, 0.1, 0.1], # D (Weak dense)
    ], dtype=np.float32)
    
    corpus = [
        "quick brown fox",      # A
        "lazy dog",             # B
        "quick red fox",        # C
        "nothing here"          # D
    ]
    
    ids = [10, 20, 30, 40]
    
    payloads = [
        {"token_count": 50, "fact_density": 0.8, "text": corpus[0]},
        {"token_count": 50, "fact_density": 0.2, "text": corpus[1]},
        {"token_count": 50, "fact_density": 0.9, "text": corpus[2]},
        {"token_count": 10, "fact_density": 0.1, "text": corpus[3]},
    ]
    
    print("Indexing documents...")
    manager.index(corpus, vectors, ids, payloads)
    manager.hnsw.active_segment.flush() # Ensure persisted if needed, though active is RAM
    
    # 1. Search Query (Hybrid)
    # Query Text: "fox" (Should match A and C)
    # Query Vector: [1.0, 0.0, 0.0, 0.0] (Should match A)
    print("\n--- Searching 'fox' + Vector A ---")
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = manager.search("fox", q_vec, k=4)
    
    print("Dense Results:", results['dense'])
    print("Sparse Results:", results['sparse'])
    print("Union IDs:", results['union_ids'])
    
    assert 10 in results['union_ids'], "ID 10 should be found (Dense + Sparse)"
    assert 30 in results['union_ids'], "ID 30 should be found (Sparse)"
    
    # 2. Refine (Solver)
    print("\n--- Refining with Solver ---")
    # We want to select best chunks. ID 30 has high density (0.9). ID 10 has 0.8.
    # Solver should pick them if budget allows.
    
    refine_result = manager.refine(
        q_vec, 
        results['union_ids']
    )
    
    print("Solver Output:", refine_result['solver_output'])
    selected_ids = refine_result['selected_ids']
    print("Selected IDs:", selected_ids)
    
    assert len(selected_ids) > 0, "Solver should select something"
    print("✅ Hybrid Search & Refinement Verified")

if __name__ == "__main__":
    verify_hybrid_fusion()
