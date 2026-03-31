import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data(path, start=750, end=850):
    logger.info(f"Checking {path} from line {start} to {end}")
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
                
            try:
                data = json.loads(line)
                q = data.get('query_text', None)
                chunks = data.get('chunk_texts', None)
                
                if q is None: 
                    # Try fallback keys
                    q = data.get('query', '')
                if chunks is None:
                    chunks = data.get('chunks', [])

                if i == 0:
                    logger.info(f"Keys in JSON: {list(data.keys())}")
                    if chunks and len(chunks) > 0:
                        logger.info(f"Type of first chunk: {type(chunks[0])}")
                        if isinstance(chunks[0], dict):
                            logger.info(f"Chunk keys: {list(chunks[0].keys())}")
                
                # Check 1: Empty?
                if not q.strip():
                    logger.warning(f"Line {i}: Empty Query!")
                
                for c_idx, chunk in enumerate(chunks):
                    # Handle dict vs string
                    if isinstance(chunk, dict):
                        text = chunk.get('text', '')
                    else:
                        text = chunk
                        
                    if not text.strip():
                        logger.warning(f"Line {i}, Chunk {c_idx}: Empty Chunk!")
                        
                    # Check 2: Length
                    if len(text) > 10000:
                        logger.info(f"Line {i}, Chunk {c_idx}: MASSIVE LENGTH {len(text)} chars")
                        # Preview
                        logger.info(f"  Preview: {text[:50]}...")
                        
                    # Check 3: Nulls
                    if '\x00' in text:
                        logger.warning(f"Line {i}, Chunk {c_idx}: Null byte detected!")
                        
                    # Check 3: ASCII/weirdness (heuristic)
                    # if '\x00' in chunk:
                    #     logger.warning(f"Line {i}, Chunk {c_idx}: Null byte detected!")
                        
            except json.JSONDecodeError:
                logger.error(f"Line {i}: JSON Error")

if __name__ == "__main__":
    check_data("data/sft/sft_raw_v1.jsonl", 0, 1000)
