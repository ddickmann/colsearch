"""
Retrieval quality metrics: Recall@K, MRR@K, NDCG@K.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def recall_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Sequence[int],
    k: int,
) -> float:
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / min(k, len(relevant_set))


def mrr_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Sequence[int],
    k: int,
) -> float:
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Sequence[int],
    k: int,
) -> float:
    """Binary relevance NDCG@K."""
    relevant_set = set(relevant_ids)

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    n_rel = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_rel + 1))
    if idcg == 0.0:
        return 1.0
    return dcg / idcg


def compute_all_metrics(
    all_retrieved: List[List[int]],
    all_relevant: List[List[int]],
    ks: Tuple[int, ...] = (10, 100),
) -> dict:
    """Aggregate metrics over a query set."""
    results = {}
    for k in ks:
        recalls = [recall_at_k(r, g, k) for r, g in zip(all_retrieved, all_relevant)]
        results[f"recall_at_{k}"] = float(np.mean(recalls))

    mrrs = [mrr_at_k(r, g, 10) for r, g in zip(all_retrieved, all_relevant)]
    results["mrr_at_10"] = float(np.mean(mrrs))

    ndcgs = [ndcg_at_k(r, g, 10) for r, g in zip(all_retrieved, all_relevant)]
    results["ndcg_at_10"] = float(np.mean(ndcgs))

    return results
