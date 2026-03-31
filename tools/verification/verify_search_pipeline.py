
import numpy as np
import shutil
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.search_pipeline import SearchPipeline

def verify_pipeline():
    print("--- Test: Unified Search Pipeline ---")
    data_path = "./data/verify_pipeline_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    dim = 16
    # lean and usable
    pipeline = SearchPipeline(data_path, dim=dim, use_roq=True)
    
    # Create data
    vectors = np.random.randn(20, dim).astype(np.float32)
    ids = list(range(20))
    corpus = [f"doc {i}" for i in range(20)]
    
    print("Indexing...")
    pipeline.index(corpus, vectors, ids)
    
    print("Searching...")
    q = vectors[0] # Match first doc
    results = pipeline.search(q, top_k_retrieval=10)
    
    print("Results:", results)
    
    selected = results.get('selected_ids')
    assert selected is not None
    assert len(selected) > 0
    
    # Check solver output presence
    solver_out = results.get('solver_output')
    assert solver_out is not None
    assert 'objective_score' in solver_out
    
    print("✅ SearchPipeline Verified")

if __name__ == "__main__":
    verify_pipeline()
