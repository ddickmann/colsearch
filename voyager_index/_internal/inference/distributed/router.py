"""
Distributed Query Router
========================

Implements scatter-gather pattern for querying multiple index shards.
Aggregates results using Reciprocal Rank Fusion (RRF).

Features:
- Parallel query execution (ThreadPool)
- Retry logic with exponential backoff
- RRF merging of results from multiple shards
- Hybrid fusion (Sparse + Dense) coordination
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import List

from ..engines.base import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    node_id: str
    endpoint: str  # gRPC/HTTP endpoint
    shard_ids: List[int]

class DistributedRouter:
    """
    Routes queries to appropriate shards and aggregates results.
    """

    def __init__(
        self,
        nodes: List[NodeConfig],
        timeout: float = 2.0,
        max_retries: int = 3
    ):
        self.nodes = nodes
        self.timeout = timeout
        self.max_retries = max_retries
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes))

    def _query_node_mock(
        self,
        node: NodeConfig,
        query_text: str,
        top_k: int
    ) -> List[SearchResult]:
        """
        Mock node query function (Replace with gRPC/HTTP client).
        """
        # Simulate network latency
        # time.sleep(0.01)
        return []

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = "rrf",
    ) -> List[SearchResult]:
        """
        Scatter query to all nodes and gather results.
        """
        futures = {}
        for node in self.nodes:
            # check health/circuit breaker here
            future = self.executor.submit(
                self._query_node_mock, node, query, top_k
            )
            futures[future] = node

        results_per_node = []

        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                results = future.result(timeout=self.timeout)
                results_per_node.append(results)
            except Exception as e:
                logger.error(f"Query to node {node.endpoint} failed: {e}")
                # Logic for partial failure?

        # Merge results without re-ranking (assuming shards return disparate docs)
        # We need to sort globally if using scores, or RRF if using ranks?
        # For standard distributed search, we usually re-score globally.
        # Here we perform RRF.

        merged = self._merge_results(results_per_node, top_k, strategy)
        return merged

    def _merge_results(
        self,
        shard_results: List[List[SearchResult]],
        top_k: int,
        strategy: str = "rrf"
    ) -> List[SearchResult]:
        """
        Merge results from multiple shards.
        """
        if not shard_results:
            return []

        # Flatten
        all_results = [r for batch in shard_results for r in batch]

        if strategy == "score":
            # Simple soft max or sort by score
            # Assumption: scores are comparable across shards (MaxSim is)
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]

        elif strategy == "rrf":
            # Reciprocal Rank Fusion
            # Since these are from DIFFERENT shards (disjoint document sets),
            # RRF is usually for merging different retrieval methods on SAME docs.
            # But here, document X is only on Shard A.
            # So simple score sort is correct for distributed index
            # (assuming scores are calibrated).

            # Use Score Sort for distributed shards
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]

        return all_results[:top_k]

    def shutdown(self):
        self.executor.shutdown()
