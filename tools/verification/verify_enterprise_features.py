import shutil
from pathlib import Path
import numpy as np
from inference.index_core.hnsw_manager import HnswSegmentManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_enterprise_features():
    shard_path = Path("/tmp/verification_shard")
    if shard_path.exists():
        shutil.rmtree(shard_path)
    
    dim = 4
    vectors = np.array([
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3, 0.3]
    ], dtype=np.float32)
    
    ids = [1, 2, 3]
    payloads = [
        {"role": "user", "active": True},
        {"role": "admin", "active": True},
        {"role": "user", "active": False}
    ]

    print("\n--- Test 1: Initialization & Persistence ---")
    # Initialize with Persistence
    manager = HnswSegmentManager(shard_path, dim=dim, on_disk=True)
    manager.add(vectors, ids=ids, payloads=payloads)
    
    # Flush and close (re-initialize simulates restart)
    manager.flush()
    print(f"DEBUG: Listing {shard_path}/active:")
    import os
    for root, dirs, files in os.walk(shard_path / "active"):
        for file in files:
            print(f"  {os.path.join(root, file)} ({os.path.getsize(os.path.join(root, file))} bytes)")
            
    del manager
    
    # Re-open
    manager = HnswSegmentManager(shard_path, dim=dim, on_disk=True)
    
    # Debug retrieval
    try:
        ret_debug = manager.retrieve([1])
        print(f"DEBUG: Retrieved ID 1 after reload: {ret_debug}")
    except Exception as e:
        print(f"DEBUG: Retrieval failed: {e}")

    assert manager.total_vectors() == 3, f"Expected 3 vectors after reload, got {manager.total_vectors()}"
    print("✅ Persistence verified.")

    print("\n--- Test 2: Payload Filtering ---")
    # Search for role="admin"
    results = manager.search(vectors[0], k=10, filters={"role": "admin"})
    assert len(results) == 1, f"Expected 1 admin, got {len(results)}"
    assert results[0][0] == 2, f"Expected ID 2, got {results[0][0]}"
    print("✅ Filtering verified.")

    print("\n--- Test 3: Upsert Payload ---")
    # Change ID 1 role to admin
    # Access inner segment directly for now as Manager doesn't expose upsert_payload yet
    # Or assuming we implemented it in Rust wrapper, let's see. 
    # Wait, HnswSegmentManager doesn't wrap upsert_payload yet. accessing active_segment directly.
    manager.active_segment.upsert_payload(1, {"role": "admin", "active": True})
    
    results = manager.search(vectors[0], k=10, filters={"role": "admin"})
    assert len(results) == 2, f"Expected 2 admins (ID 1 & 2), got {len(results)}"
    print("✅ Upsert Payload verified.")

    print("\n--- Test 4: Deletion ---")
    manager.active_segment.delete([2]) # Delete ID 2
    results = manager.search(vectors[0], k=10, filters={"role": "admin"})
    assert len(results) == 1, f"Expected 1 admin (ID 1), got {len(results)}"
    assert results[0][0] == 1
    print("✅ Deletion verified.")

    print("\n🎉 All Enterprise Features Verified!")

if __name__ == "__main__":
    verify_enterprise_features()
