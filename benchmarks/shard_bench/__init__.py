"""Shard Late-Interaction Benchmark

CPU-backed / GPU-routed late-interaction retrieval engine.
Stores multi-vector embeddings in contiguous safetensors shards on disk,
routes queries via LEMUR learned proxy (or centroid fallback), streams
only candidate docs to GPU for exact MaxSim scoring.

Components:
- LEMUR router: reduces multi-vector retrieval to single-vector ANN search
- Token pooling: index-time token compression via hierarchical merging
- Col-Bandit: adaptive query-time pruning for MaxSim reranking
- Retrain manager: shadow-generation retraining for LEMUR
"""
