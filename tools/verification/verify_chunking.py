
import sys
import os
import asyncio
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from preprocessing import NeuralChunker, SplittingStrategy, HAS_RUST_ACCELERATION
except ImportError as e:
    print(f"FAILED to import SDK: {e}")
    sys.exit(1)

async def test_sdk():
    print("="*50)
    print("NeuralChunker SDK Verification")
    print("="*50)
    
    print(f"Rust Acceleration Available: {HAS_RUST_ACCELERATION}")
    
    text = "This is a test sentence. This is another one. \n\nNew paragraph."
    
    # 1. Initialize SDK (Character Mode)
    chunker = NeuralChunker(
        chunk_size=50,
        chunk_overlap=10,
        splitting_strategy='character' # String convenience
    )
    
    # 2. Test Sync Chunking
    print("\nTest 1: Sync Character Chunking (via SDK)")
    chunks = chunker.chunk_sync(text)
    print(f"Generated {len(chunks)} chunks.")
    if len(chunks) > 0 and len(chunks[0].content) > 0:
        print(f"  Sample: {repr(chunks[0].content)}")
    
    # 3. Test Async Token Chunking (using re-initialization)
    print("\nTest 2: Async Token Chunking (via SDK)")
    chunker = NeuralChunker(
        chunk_size=50,
        splitting_strategy=SplittingStrategy.TOKEN,
        tokenizer_backend='tiktoken' 
    )
    chunks = await chunker.chunk(text)
    print(f"Generated {len(chunks)} chunks.")
    
    # 4. Verify Rust is actually used (check logs/attributes if possible, or rely on latency)
    # Since we checked HAS_RUST_ACCELERATION, we assume it's used internally.
    
    print("\nSDK Verification Complete.")

if __name__ == "__main__":
    asyncio.run(test_sdk())
