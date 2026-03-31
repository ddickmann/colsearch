
import latence_hnsw
import numpy as np
import shutil
import os

print(f"Module imported: {latence_hnsw.__file__}")

# Cleanup
path = "/tmp/test_hnsw_enterprise"
if os.path.exists(path):
    shutil.rmtree(path)

try:
    print("--- Test 1: RocksDB Storage & Initialization ---")
    # on_disk=True triggers RocksDB/Mmap path
    seg = latence_hnsw.HnswSegment(path, 4, "cosine", 16, 100, True)
    
    print("--- Test 2: Ingestion with Payload ---")
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    ids = [1, 2, 3, 4]
    payloads = [
        {"category": "A", "val": 10},
        {"category": "B", "val": 20},
        {"category": "A", "val": 30},
        {"category": "B", "val": 40}
    ]
    
    seg.add(vectors, ids, payloads)
    print(f"Added {seg.len()} points. Expect 4.")
    assert seg.len() == 4

    print("--- Test 3: Filtered Search ---")
    # Search for vector near [1,0,0,0] but filter for category="B" (should match ID 2)
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = seg.search(query, k=1, filter={"category": "B"})
    print(f"Filtered search results: {results}")
    assert len(results) > 0
    assert results[0][0] == 2  # ID 2 is the best match in category B
    
    print("--- Test 4: Update Payload ---")
    # Update ID 2 to category "A"
    seg.upsert_payload(2, {"category": "A", "new_field": True})
    
    # Search again for 'B', ID 2 should verify gone from results
    results_b = seg.search(query, k=1, filter={"category": "B"})
    print(f"Post-update search (cat B): {results_b}")
    # Should match ID 4 now, or nothing if threshold
    if len(results_b) > 0:
        assert results_b[0][0] != 2
        
    print("--- Test 5: Deletion ---")
    deleted = seg.delete([1])
    print(f"Deleted {deleted} points.")
    assert deleted == 1
    assert seg.len() == 3
    
    # Verify ID 1 is gone
    results_del = seg.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=1)
    print(f"Search deleted ID 1 area: {results_del}")
    if len(results_del) > 0:
        assert results_del[0][0] != 1

    print("SUCCESS: Enterprise features (RocksDB, CRUD, Filter) Verified!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILURE: {e}")
    exit(1)
