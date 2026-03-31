"""
Benchmark script to find optimal batch sizes for ColBERT encoding on MPS.
Tests queries (256 tokens) and documents (variable length) at different batch sizes.
"""

import time
import torch
from pylate import models

def benchmark_colbert():
    print("=" * 60)
    print("ColBERT Batch Size Benchmark for MPS")
    print("=" * 60)
    
    # Load model
    print("\nLoading SauerkrautLM-Multi-ModernColBERT...")
    model = models.ColBERT(
        model_name_or_path="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT",
        query_length=256,
        document_length=512,  # Reduced from 8192 for speed
        attend_to_expansion_tokens=True,
    )
    model = model.to("mps")
    model.eval()
    print("Model loaded on MPS")
    
    # Test data
    test_query = "What is the capital of France and how does it relate to European economics?"
    test_doc = "Paris is the capital of France. " * 50  # ~200 words
    
    # Benchmark queries
    print("\n--- QUERY ENCODING BENCHMARK ---")
    print("Query length: 256 tokens (with expansion)")
    
    for bs in [1, 2, 4, 8, 16, 32]:
        queries = [test_query] * bs
        
        # Warmup
        try:
            _ = model.encode(queries, is_query=True, convert_to_tensor=True)
            torch.mps.synchronize()
        except RuntimeError as e:
            print(f"  bs={bs:3d}: OOM - {str(e)[:50]}")
            continue
        
        # Time it
        start = time.time()
        for _ in range(3):
            _ = model.encode(queries, is_query=True, convert_to_tensor=True)
            torch.mps.synchronize()
        elapsed = (time.time() - start) / 3
        
        throughput = bs / elapsed
        print(f"  bs={bs:3d}: {elapsed:.2f}s/batch, {throughput:.1f} queries/sec")
    
    # Benchmark documents
    print("\n--- DOCUMENT ENCODING BENCHMARK (512 tokens) ---")
    
    for bs in [1, 2, 4, 8, 16]:
        docs = [test_doc] * bs
        
        # Warmup
        try:
            _ = model.encode(docs, is_query=False, convert_to_tensor=True)
            torch.mps.synchronize()
        except RuntimeError as e:
            print(f"  bs={bs:3d}: OOM - {str(e)[:50]}")
            continue
        
        # Time it
        start = time.time()
        for _ in range(3):
            _ = model.encode(docs, is_query=False, convert_to_tensor=True)
            torch.mps.synchronize()
        elapsed = (time.time() - start) / 3
        
        throughput = bs / elapsed
        print(f"  bs={bs:3d}: {elapsed:.2f}s/batch, {throughput:.1f} docs/sec")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_colbert()
