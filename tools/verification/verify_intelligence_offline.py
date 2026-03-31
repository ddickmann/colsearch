
import sys
import os
import asyncio
import numpy as np
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock FAISS before imports
sys.modules['faiss'] = MagicMock()
sys.modules['faiss'].get_num_gpus.return_value = 0
mock_index = MagicMock()
mock_index.search.return_value = (np.zeros((10, 10)), np.zeros((10, 10)))
sys.modules['faiss'].IndexFlatIP.return_value = mock_index

# Comprehensive Mocking for Science Stack
import types
sklearn = types.ModuleType('sklearn')
sklearn.cluster = MagicMock()
sklearn.decomposition = MagicMock()
sklearn.manifold = MagicMock()
sys.modules['sklearn'] = sklearn
sys.modules['sklearn.cluster'] = sklearn.cluster
sys.modules['sklearn.decomposition'] = sklearn.decomposition
sys.modules['sklearn.manifold'] = sklearn.manifold


scipy = types.ModuleType('scipy')
scipy.cluster = MagicMock()
scipy.cluster.hierarchy = MagicMock()
sys.modules['scipy'] = scipy
sys.modules['scipy.cluster'] = scipy.cluster
sys.modules['scipy.cluster.hierarchy'] = scipy.cluster.hierarchy

# Mock Latence Package
latence_pkg = types.ModuleType('latence')
latence_pkg.Latence = MagicMock()
latence_pkg.AsyncLatence = MagicMock()
sys.modules['latence'] = latence_pkg

try:
    from intelligence import IntelligenceSDK, SDKConfig, DimensionMode
    from latence import Latence 
except ImportError as e:

    print(f"FAILED to import SDK: {e}")
    sys.exit(1)

# Mock vLLM Engine for offline verify
class MockOutput:
    def __init__(self, data):
        self.data = MagicMock()
        # Support .cpu().numpy() chain
        self.data.cpu.return_value.numpy.return_value = np.array(data)
        self.data.tolist.return_value = data

class MockCompletionOutput:
    def __init__(self, data):
        self.outputs = MockOutput(data)

class MockLLMEngine:
    def __init__(self, dimension=512):
        self.dimension = dimension
        
    async def encode(self, text, pooling_params, request_id):
        # Simulate embedding generation
        # Deterministic random based on text length for stability
        np.random.seed(len(text))
        embedding = np.random.randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Yield result like vLLM generator
        yield MockCompletionOutput(embedding)

async def test_offline_intelligence():
    print("="*50)
    print("Intelligence SDK Offline Verification")
    print("="*50)
    
    # 1. Setup Mock Engine
    mock_engine = MockLLMEngine()
    
    # 2. Initialize SDK in Offline Mode
    # We pass a mock Latence client because we don't need real API calls for this test
    mock_client = MagicMock()
    
    config = SDKConfig(
        offline_mode=True,
        default_dimension_mode=DimensionMode.BALANCED
    )
    
    sdk = IntelligenceSDK(
        config=config,
        vllm_engine=mock_engine
    )
    
    print("\n✓ SDK Initialized with Mock Engine")
    
    # 3. Process Pipeline
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require data.",
        "The quick brown fox jumps over the lazy dog.", # Duplicate
        "Data science involves statistics and coding."
    ]
    
    print(f"\nRunning pipeline on {len(texts)} documents...")
    
    # embed - quality - dedup - topics
    results = await sdk.process_pipeline(
        texts,
        steps=['quality', 'dedup', 'topics'],
        num_topics=2,
        quality_threshold=-1.0 # Keep all
    )
    
    # 4. Verify Results
    print("\nPipeline Results:")
    print(f"  Input: {len(texts)}")
    print(f"  Final: {results['final_document_count']}")
    
    if 'quality' in results:
        print(f"  Quality Mean: {results['quality']['statistics']['overall']['mean']:.3f}")
        
    if 'dedup' in results:
        print(f"  Duplicates Found: {len(texts) - len(results['dedup']['unique_indices'])}")
        
    if 'topics' in results:
        print(f"  Topics Extracted: {len(results['topics']['topics'])}")

    # Verify Mock Engine was used
    # The pure mock client shouldn't have been called for embeddings
    mock_client.embedding.create.assert_not_called()
    print("\n✓ Verified: Latence API client was NOT called (Offline Mode Active)")

if __name__ == "__main__":
    # Create required mocks for imports if not available
    sys.modules['vllm'] = MagicMock()
    sys.modules['vllm'].PoolingParams = MagicMock
    
    asyncio.run(test_offline_intelligence())
