import zstandard as zstd
import shutil
import os

input_path = 'data/sft/v0.1/sft_enriched_final.jsonl.zst'
output_path = 'data/sft/v0.1/sft_enriched_final.jsonl'

print(f"Decompressing {input_path} to {output_path}...")
with open(input_path, 'rb') as compressed:
    dctx = zstd.ZstdDecompressor()
    with open(output_path, 'wb') as destination:
        dctx.copy_stream(compressed, destination)
print("Decompression complete.")
