
import numpy as np
import shutil
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference.index_core.hybrid_manager import HybridSearchManager

def verify_roq():
    print("--- Test: RoQ Integration (4-bit) ---")
    data_path = "./data/verify_roq_shard"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    dim = 16
    # Initialize with RoQ 4-bit
    manager = HybridSearchManager(
        data_path, 
        dim=dim, 
        roq_bits=4,
        on_disk=True
    )
    
    # Create random vectors
    # 2 docs, each with 3 vectors (Multi-Vector) for rigorous test
    vectors = np.random.randn(2, 3, dim).astype(np.float32)
    # HybridSearchManager.index expects list of strings for corpus,
    # and vectors as... well currently index() takes (N, D) or (N, M, D)?
    # Let's check HybridSearchManager.index signature.
    # It takes `vectors: np.ndarray`. If we pass (N, M, D) it might fail if index() expects (TotalVectors, D).
    # Wait, HybridSearchManager.index usually takes flattened vectors for BM25 mapping?
    # Let's check the code.
    # The code says: `vectors: np.ndarray`.
    # HNSW `add_multidense` expects list of arrays.
    # `HybridSearchManager.index` currently likely assumes 1:1 mapping if not updated.
    # Let's test standard (N, D) first (Single Vector) to verify RoQ basic flow.
    # Then we can test Multi-Vector separately or if manager handles it.
    # RoQ integration in HNSW `add` supports (N, D).
    
    vecs_flat = np.random.randn(10, dim).astype(np.float32)
    ids = list(range(10))
    corpus = [f"doc {i}" for i in range(10)]
    
    print("Indexing 10 vectors with RoQ enabled...")
    manager.index(corpus, vecs_flat, ids)
    
    # 1. Verify Storage
    print("Verifying Payload Storage...")
    items = manager.hnsw.retrieve([0])
    assert len(items) > 0
    payload = items[0]['payload']
    
    assert 'roq_codes' in payload, "roq_codes missing from payload"
    assert 'roq_scale' in payload, "roq_scale missing"
    
    codes = np.array(payload['roq_codes'])
    print(f"RoQ Codes Shape: {codes.shape}")
    # 4-bit packing: D=16 -> 8 bytes?
    # RotationalQuantizer pack logic: (N, D/2) -> 16/2 = 8 bytes.
    assert codes.size == 8, f"Expected 8 bytes for 16-dim 4-bit, got {codes.size}"
    
    print("✅ Storage Verified")
    
    # 2. Verify Refine (Decoding)
    print("Verifying Refine (Decoding flow)...")
    q = vecs_flat[0] # Match first document to ensure high score
    
    # This should trigger the `_decode_roq` logic in refine
    results = manager.refine(q, ids)
    
    print("Refine results:", results)
    
    print(f"Refine returned {len(results.get('selected_ids', []))} items")
    assert len(results.get('selected_ids', [])) > 0
    
    print("✅ Refine Verified")

if __name__ == "__main__":
    verify_roq()
