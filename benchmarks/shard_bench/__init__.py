"""
Shard Late-Interaction Benchmark

CPU-backed / GPU-routed late-interaction retrieval prototype.
Stores multi-vector embeddings in contiguous safetensors shards on disk,
routes queries via GPU centroid scoring, streams survivors to GPU for
exact MaxSim scoring.

This is an isolated benchmark — it does not modify the voyager-index package.
"""
