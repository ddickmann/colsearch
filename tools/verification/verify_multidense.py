
import numpy as np
import shutil
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.index_core.hnsw_manager import HnswSegmentManager

def verify_multidense():
    print("--- Test: Multi-Vector (ColBERT) HNSW ---")
    data_path = "./data/verify_multidense_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    dim = 4
    # Use 'dot' to verify raw storage (Cosine auto-normalizes)
    manager = HnswSegmentManager(
        data_path, 
        dim=dim, 
        distance_metric="dot",
        multivector_comparator="max_sim", 
        on_disk=True
    )
    
    # Create pseudo-ColBERT Documents
    # Doc 1: 3 vectors
    d1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0]
    ], dtype=np.float32)
    
    # Doc 2: 2 vectors (Distinct)
    d2 = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    docs = [d1, d2]
    ids = [100, 200]
    payloads = [{"text": "Doc 1 Multi"}, {"text": "Doc 2 Multi"}]
    
    print("Indexing Multi-Vectors...")
    manager.add_multidense(docs, ids=ids, payloads=payloads)
    manager.flush()
    
    print("Retrieving back...")
    # Retrieve should return the full matrix
    res = manager.retrieve([100])
    
    assert len(res) == 1
    item = res[0]
    vec = item['vector']
    
    print(f"Retrieved shape: {vec.shape}")
    print("Expected:\n", d1)
    print("Actual:\n", vec)
    assert vec.shape == (3, 4), f"Expected (3, 4), got {vec.shape}"
    assert np.allclose(vec, d1), "Vector content mismatch"
    
    print("✅ Multi-Vector Indexing Verified (No Pooling!)")

if __name__ == "__main__":
    verify_multidense()
