#!/usr/bin/env python3
"""Check pilot data query-gold similarity at each hop."""
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F

# Load pilot data
pilot_path = "data/sft/sft_ready_test.pt"
print(f"Loading pilot data from {pilot_path}...")

data = torch.load(pilot_path, map_location='cpu')
print(f"Loaded {len(data)} samples")

# Analyze first few samples
for sample_idx in range(min(3, len(data))):
    sample = data[sample_idx]
    trajectory = sample['trajectory']
    
    print(f"\n=== Sample {sample_idx} ({len(trajectory)} hops) ===")
    print(f"Query: {sample.get('query_text', 'N/A')[:80]}...")
    
    for hop_idx, hop in enumerate(trajectory):
        input_q = hop['input_query_vec']  # (256, 128)
        target_chunk = hop['target_chunk_vec']  # (M, 128)
        target_delta = hop['target_delta']  # (256, 128) or (1, 128)?
        
        # Compute centroids
        q_centroid = input_q.mean(dim=0)  # (128,)
        gold_centroid = target_chunk.mean(dim=0)  # (128,)
        
        # Norms
        q_norm = q_centroid.norm().item()
        g_norm = gold_centroid.norm().item()
        td_norm = target_delta.norm().item() / max(1, target_delta.numel() // 128)  # Per-vector norm
        
        # Similarity
        sim = F.cosine_similarity(q_centroid.unsqueeze(0), gold_centroid.unsqueeze(0)).item()
        
        print(f"  Hop {hop_idx}: q_norm={q_norm:.3f}, g_norm={g_norm:.3f}, delta_norm={td_norm:.4f}, q-gold_sim={sim:.4f}")

print("\n=== Analysis ===")
print("If hop-0 sim is ~0.94 but hop-1+ sim is lower, then multi-hop is where the learning happens!")
print("If all hops have ~0.94 sim, then the task is too easy for ColBERT.")
