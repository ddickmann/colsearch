
import asyncio
import sys
import os
import logging
import multiprocessing

# Set env var for vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError as e:
    print(f"Warning: Failed to set start method: {e}")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from intelligence import IntelligenceSDK, SDKConfig, DimensionMode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("qualitative_test")

# Qualitative Test Data
TEST_CASES = [
    {
        "id": "technical_reference",
        "text": """
        pub fn solve_knapsack(weights: &[u32], values: &[u32], capacity: u32) -> u32 {
            let n = weights.len();
            let mut dp = vec![0; (capacity + 1) as usize];
            for i in 0..n {
                for w in (weights[i]..=capacity).rev() {
                    dp[w as usize] = dp[w as usize].max(dp[(w - weights[i]) as usize] + values[i]);
                }
            }
            dp[capacity as usize]
        }
        This function implements the standard dynamic programming solution for the 0/1 Knapsack problem.
        The time complexity is O(N*W) where N is the number of items and W is the capacity.
        Space complexity is O(W) due to the optimized 1D DP array.
        """
    },
    {
        "id": "simple_narrative",
        "text": """
        Once upon a time, in a small village, there lived a baker named John. He baked the best bread in the county.
        Every morning, the smell of fresh yeast would waft through the streets, waking up the neighbors.
        "Good morning, John!" they would shout. "Good morning!" he would reply with a smile.
        It was a simple life, but a happy one.
        """
    },
    {
        "id": "dense_academic",
        "text": """
        The Transformer model, based on a self-attention mechanism, dispenses with recurrence and convolutions entirely.
        This architecture allows for significant parallelization and reduces the training time for large datasets.
        The attention function can be described as mapping a query and a set of key-value pairs to an output,
        where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
        of the values, where the weight assigned to each value is computed by a compatibility function of the 
        query with the corresponding key.
        """
    },
    {
        "id": "garbage_low_quality",
        "text": """
        asdf jkl; asdf jkl; test test test example
        blah blah blah nothing here just filling space.
        random text random text random text.
        123 456 789 000
        """
    },
    {
        "id": "tutorial_content",
        "text": """
        Step 1: Install the requests library using pip install requests.
        Step 2: Import the library in your script: import requests.
        Step 3: Make a GET request: response = requests.get('https://api.github.com').
        Finally, print the status code: print(response.status_code).
        This guide helps you make your first HTTP request in Python.
        """
    }
]

async def verify_quality():
    print("\n" + "="*80)
    print("QUALITATIVE ENRICHMENT VERIFICATION")
    print("="*80)
    
    config = SDKConfig(
        offline_mode=True,
        vllm_model_name="intfloat/multilingual-e5-small", 
        default_dimension_mode=DimensionMode.BALANCED,
    )
    
    print("Initializing SDK (Offline Mode)...")
    async with IntelligenceSDK(config=config) as sdk:
        print("✓ SDK Initialized\n")
        
        # Process each document through metadata enrichment only
        # We need embeddings first
        texts = [tc["text"] for tc in TEST_CASES] 
        print(f"Generating embeddings for {len(texts)} test cases...")
        embeddings = await sdk.embed(texts)
        
        print("\nComputing Metadata & Quality Scores...")
        # Manually call components to inspect details
        
        # 1. Quality Filter Codes
        quality_scores = sdk.quality_filter.score_documents(texts, embeddings, compute_uniqueness=False)
        
        # 2. Metadata Enrichment
        metadata = sdk.metadata_enricher.enrich(texts, embeddings, quality_scores=quality_scores)
        
        # Display Results
        for i, tc in enumerate(TEST_CASES):
            m = metadata[i]
            q = quality_scores[i]
            
            print(f"\n--- Test Case: {tc['id'].upper()} ---")
            print(f"Snippet: {tc['text'].strip()[:60]}...")
            
            print(f"\n[Classification]")
            print(f"  Type: {m['document_type']} (Expected: {get_expected_type(tc['id'])})")
            print(f"  Reading Level: {m['reading_level']:.2f} (0=Simple, 1=Complex)")
            
            print(f"\n[Quality Metrics]")
            print(f"  Overall Score: {m['quality_score']:.3f} / {q['overall']:.3f}")
            print(f"  Info Density:  {m['information_density']:.3f} (Calc: {q['information_density']:.3f})")
            print(f"  Coherence:     {q['coherence']:.3f}")
            print(f"  Completeness:  {q['completeness']:.3f}")
            
            print(f"\n[Key Concepts]")
            print(f"  {', '.join(m.get('key_concepts', []))}")
            
            # Validation Logic
            validate_case(tc['id'], m, q)

def get_expected_type(case_id):
    if "technical" in case_id: return "technical/reference"
    if "tutorial" in case_id: return "tutorial"
    if "academic" in case_id: return "academic"
    if "narrative" in case_id: return "narrative"
    return "any"

def validate_case(case_id, m, q):
    warnings = []
    
    # Check Type
    if case_id == "technical_reference" and m['document_type'] not in ["technical", "reference"]:
        warnings.append(f"⚠️ Incorrect Type: Got {m['document_type']}")
    elif case_id == "tutorial_content" and m['document_type'] != "tutorial":
        warnings.append(f"⚠️ Incorrect Type: Got {m['document_type']}")
        
    # Check Complexity
    if case_id == "dense_academic" and m['reading_level'] < 0.6:
        warnings.append(f"⚠️ Reading level too low for academic text: {m['reading_level']:.2f}")
    if case_id == "simple_narrative" and m['reading_level'] > 0.4:
        warnings.append(f"⚠️ Reading level too high for simple text: {m['reading_level']:.2f}")
        
    # Check Quality
    if case_id == "garbage_low_quality" and q['overall'] > 0.4:
        warnings.append(f"⚠️ Garbage text got high quality score: {q['overall']:.2f}")

    if warnings:
        for w in warnings: print(w)
    else:
        print("✅ Qualitative Checks Passed")

if __name__ == "__main__":
    asyncio.run(verify_quality())
