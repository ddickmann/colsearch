import numpy as np
import shutil
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.index_core.hnsw_manager import HnswSegmentManager

def verify_retrieval():
    print("--- Test: Retrieval ---")
    data_path = "./data/test_retrieval_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        
    dim = 4
    manager = HnswSegmentManager(data_path, dim=dim, on_disk=True)
    
    # 1. Add Data
    vectors = np.array([
        [0.1, 0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9, 0.9]
    ], dtype=np.float32)
    ids = [100, 200]
    payloads = [{"name": "A", "val": 1}, {"name": "B", "val": 2}]
    
    manager.add(vectors, ids=ids, payloads=payloads)
    
    # 2. Retrieve
    # Query for existing IDs and one non-existent ID
    query_ids = [100, 999, 200]
    results = manager.retrieve(query_ids)
    
    # Assertions
    assert len(results) == 3
    
    # ID 100
    id1, vec1, pay1 = results[0]
    assert id1 == 100
    assert vec1 is not None
    print(f"DEBUG: vec1={vec1}")
    print(f"DEBUG: expected={vectors[0]}")
    # Qdrant normalizes vectors for Cosine distance.
    expected_norm = vectors[0] / np.linalg.norm(vectors[0])
    assert np.allclose(vec1, expected_norm)
    assert pay1 == payloads[0]
    print("✅ Retrieved ID 100 correctly (normalized)")
    print("✅ Retrieved ID 100 correctly")

    # ID 999 (Missing)
    id2, vec2, pay2 = results[1]
    assert id2 == 999
    assert vec2 is None
    assert pay2 is None
    print("✅ Handled missing ID 999 correctly")

    # ID 200
    id3, vec3, pay3 = results[2]
    assert id3 == 200
    assert vec3 is not None
    expected_norm_2 = vectors[1] / np.linalg.norm(vectors[1])
    assert np.allclose(vec3, expected_norm_2)
    assert pay3 == payloads[1]
    print("✅ Retrieved ID 200 correctly")
    
    print("--- Retrieval Verification PASSED ---")

if __name__ == "__main__":
    verify_retrieval()
