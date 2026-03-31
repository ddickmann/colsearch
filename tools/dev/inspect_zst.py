import zstandard as zstd
import json
import io

path = 'data/sft/v0.1/sft_enriched_final.jsonl.zst'

print(f"Inspecting {path}...")
with open(path, 'rb') as fh:
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
    
    for i, line in enumerate(text_stream):
        if i >= 5: break
        data = json.loads(line)
        print(f"Sample {i}: Keys={list(data.keys())}")
        if 'provenance_ids' in data:
            print(f"  Provenance: {data['provenance_ids']} (Type: {type(data['provenance_ids'][0]) if data['provenance_ids'] else 'Empty'})")
        if 'negatives' in data:
            print(f"  Negatives Len: {len(data['negatives'])}")
            if len(data['negatives']) > 0:
                print(f"  Neg 0 Type: {type(data['negatives'][0])}")
                print(f"  Neg 0 Sample: {str(data['negatives'][0])[:50]}...")
        if 'chunks' in data:
             print(f"  Chunks Len: {len(data['chunks'])}")
             # Verify if provenance indexes into chunks?
             # Or if chunks are just strings?
             if len(data['chunks']) > 0:
                 print(f"  Chunk 0 Keys: {list(data['chunks'][0].keys())}")
                 print(f"  Chunk 0 ID: {data['chunks'][0].get('id', 'N/A')}")
                 # print(f"  Chunk 0 Text: {data['chunks'][0].get('text', 'N/A')[:50]}...")
        print("-" * 20)
