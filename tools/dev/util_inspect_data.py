
import json
import textwrap

def inspect_chain(filepath):
    with open(filepath, 'r') as f:
        # Read first line
        line = f.readline()
        data = json.loads(line)
        
        print(f"=== QUERY ===\n{data['query']}\n")
        print(f"=== TYPE: {data.get('logic_type')} | HOPS: {data.get('hops')} ===")
        
        for i, chunk in enumerate(data['chunks']):
            print(f"\n--- CHUNK {i} ---")
            # Truncate for readability
            text = chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text']
            print(textwrap.fill(text, width=80))

if __name__ == "__main__":
    inspect_chain("/Users/dennisdickmann/omni-index/data/sft/sft_raw_v1.jsonl")
