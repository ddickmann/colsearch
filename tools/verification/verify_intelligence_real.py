
import multiprocessing
import os

# Set env var for vLLM just in case
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError as e:
    print(f"Warning: Failed to set start method: {e}")

print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

import sys
import asyncio
import numpy as np
import logging
from typing import List

# Configure logging to show info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_real")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from intelligence import IntelligenceSDK, SDKConfig, DimensionMode
except ImportError as e:
    print(f"FAILED to import SDK: {e}")
    sys.exit(1)

# Sample Data: Technical snippets about RAG and LLMs
SAMPLE_DOCUMENTS = [
    """User: What is Retrieval-Augmented Generation (RAG)?
    Assistant: Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of generative AI models by fetching facts from external sources. Instead of relying solely on training data, RAG allows the model to access a designated knowledge base to answer questions.""",
    
    """Vector databases are specialized databases designed to store and query high-dimensional vector embeddings. They are crucial for semantic search applications, enabling rapid similarity search across millions of documents using algorithms like HNSW (Hierarchical Navigable Small World).""",
    
    """Quantization in Deep Learning involves reducing the precision of the numbers used to represent a model's parameters, such as weights and biases. Moving from FP16 (16-bit floating point) to INT8 or FP8 can significantly reduce memory usage and increase inference speed with minimal loss in accuracy.""",
    
    """User: What is Retrieval-Augmented Generation (RAG)?
    Assistant: Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of generative AI models by fetching facts from external sources. Instead of relying solely on training data, RAG allows the model to access a designated knowledge base to answer questions.""",  # Exact Duplicate
    
    """The Transformer architecture, introduced in 'Attention Is All You Need', revolutionized NLP. It relies on a self-attention mechanism to weigh the significance of different words in a sentence, allowing for parallel processing of sequences unlike RNNs.""",
    
    """To implement a semantic search system, one typically converts text into vectors using an embedding model like BERT or RoBERTa. These vectors capture the semantic meaning of the text. Queries are also embedded, and the system retrieves the nearest neighbor vectors to find relevant documents.""",
    
    """Low quality text often contains little information. For example: "This is a test. Just a test document. Nothing important here. Blah blah blah." Such documents should be filtered out to improve the quality of the retrieval context.""",
    
    """vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It features PagedAttention, an algorithm inspired by virtual memory paging in operating systems, to manage attention key and value memory more effectively."""
]

async def verify_real_pipeline():
    print("="*60)
    print("REAL DOCUMENT INTELLIGENCE PIPELINE VERIFICATION")
    print("="*60)
    print(f"Python: {sys.version}")
    
    # 1. Initialize SDK in Offline Mode
    # offline_mode=True triggers vLLM loading with FP8
    config = SDKConfig(
        offline_mode=True,
        vllm_model_name="Qwen/Qwen3-Embedding-0.6B", 
        default_dimension_mode=DimensionMode.BALANCED,
    )
    
    # NOTE: We use the default model name from the config if we don't override it. 
    # Config default is "Qwen/Qwen3-Embedding-0.6B" which is good for embeddings.
    # Let's check what the config says:
    print(f"Target Model: {config.vllm_model_name}")
    print(f"Offline Mode: {config.offline_mode}")
    
    print("\nInitializing SDK (this may take a moment to load vLLM)...")
    
    # We do NOT pass a Latence client, ensuring self-contained operation
    try:
        async with IntelligenceSDK(config=config) as sdk:
            print("✓ SDK Context Entered")
            
            # 2. Run Pipeline
            # We skip 'quality' filter strictness to see all features, or use a low threshold
            print(f"\nProcessing {len(SAMPLE_DOCUMENTS)} documents...")
            
            results = await sdk.process_pipeline(
                SAMPLE_DOCUMENTS,
                steps=['quality', 'dedup', 'topics', 'metadata', 'graph'],
                num_topics=3,
                quality_threshold=0.2, # Filter only very bad stuff
                return_quality_scores=True
            )
            
            # 3. Analyze Results
            print("\n" + "="*60)
            print("RESULTS ANALYSIS")
            print("="*60)
            
            # Embeddings
            emb_shape = results.get('embeddings_shape')
            print(f"Embeddings: {emb_shape} (should be {len(SAMPLE_DOCUMENTS)}x{config.dimension_balanced})")
            
            # Quality
            quality_res = results.get('quality', {})
            stats = quality_res.get('statistics', {})
            print(f"\n1. Quality Scores (Mean: {stats.get('overall', {}).get('mean', 0):.3f}):")
            if 'scores' in quality_res:
                for i, score in enumerate(quality_res['scores']):
                     print(f"   - Doc {i} ('{SAMPLE_DOCUMENTS[i][:30]}...'): {score['overall_score']:.3f} (Centrality: {score.get('centrality_score', 0):.2f})")
            
            # Dedup
            dedup_res = results.get('dedup', {})
            print(f"\n2. Deduplication:")
            print(f"   Unique Indices: {dedup_res.get('unique_indices')}")
            print(f"   Duplicate Groups: {dedup_res.get('duplicate_groups')}")
            
            # Topics
            topics_res = results.get('topics', {})
            print(f"\n3. Topics Found ({len(topics_res.get('topics', []))}):")
            for t in topics_res.get('topics', []):
                print(f"   - Topic {t['id']}: {t['label']} (Size: {t['size']})")
                print(f"     Keywords: {', '.join(t['keywords'][:5])}")
                
            # Metadata
            meta_res = results.get('metadata', {})
            print(f"\n4. Metadata Samples:")
            if 'metadata' in meta_res and len(meta_res['metadata']) > 0:
                m = meta_res['metadata'][0]
                print(f"   - Doc 0 Category: {m.get('category')}")
                print(f"   - Doc 0 Summary: {m.get('summary')}")
                
            print("\n" + "="*60)
            print("VERIFICATION COMPLETE")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_real_pipeline())
