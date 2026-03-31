
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

from .index_core.hybrid_manager import HybridSearchManager

logger = logging.getLogger(__name__)

class SearchPipeline:
    """
    Vector-first local hybrid retrieval pipeline.

    Public OSS behavior focuses on:
    1. HNSW-backed dense retrieval.
    2. Canonical `bm25s` sparse retrieval via `HybridSearchManager`.
    3. Optional local/native refinement when a compatible solver is available.

    This pipeline does not embed raw text into dense late-interaction queries.
    String input is handled as sparse-only retrieval.
    Remote compute-side productization is intentionally out of scope for this
    class.
    """
    
    def __init__(
        self,
        shard_path: str,
        dim: int = 128,
        use_roq: bool = True,
        roq_bits: int = 4,
        on_disk: bool = True
    ):
        """
        Initialize the search pipeline.
        """
        self.config = {
            "dim": dim,
            "use_roq": use_roq,
            "roq_bits": roq_bits,
            "on_disk": on_disk
        }
        
        self.manager = HybridSearchManager(
            shard_path=Path(shard_path),
            dim=dim,
            on_disk=on_disk,
            # Enable RoQ 4-bit/8-bit if requested
            roq_bits=roq_bits if use_roq else None
        )
        
        logger.info(f"SearchPipeline initialized at {shard_path} (RoQ={use_roq})")

    def index(
        self, 
        corpus: List[str], 
        vectors: Union[np.ndarray, List[np.ndarray]], 
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Ingest documents into the pipeline.
        Supports both Single-Vector and Multi-Vector (ColBERT) inputs.
        """
        if isinstance(vectors, list):
            logger.info(f"Indexing {len(corpus)} documents (Multi-Vector)...")
            self.manager.index_multivector(corpus, vectors, ids, payloads)
        else:
            # Standard single vector
            logger.info(f"Indexing {len(corpus)} documents (Single-Vector)...")
            self.manager.index(corpus, vectors, ids, payloads)

    def search(
        self, 
        query: Union[str, np.ndarray],
        top_k_retrieval: int = 100,
        enable_refinement: bool = False
    ) -> Dict[str, Any]:
        """
        Execute retrieval with optional solver refinement.
        
        Args:
            query: Query vector for dense retrieval, or query text for sparse-only retrieval.
            top_k_retrieval: Number of candidates from HNSW (Map).
            enable_refinement: Whether to run optional local/native solver
                refinement when available.
        """
        if isinstance(query, str):
            search_output = self.manager.search(
                query_text=query,
                query_vector=None,
                k=top_k_retrieval,
            )
            retrieval_ids = search_output.get('union_ids', [])
            ordered_ids = retrieval_ids
            return {
                "retrieval": search_output,
                "retrieval_count": len(retrieval_ids),
                "solver_output": None,
                "selected_ids": ordered_ids[: min(10, len(ordered_ids))],
            }

        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.ndim > 1:
            raise ValueError(
                "SearchPipeline.search expects a single dense query vector. "
                "Late-interaction multi-vector queries should use ColbertIndex directly."
            )
            
        search_output = self.manager.search(
            query_text="", 
            query_vector=query_vector,
            k=top_k_retrieval
        )
        
        retrieval_ids = search_output.get('union_ids', [])
        dense_ids = [doc_id for doc_id, _ in search_output.get("dense", [])]
        sparse_ids = [doc_id for doc_id, _ in search_output.get("sparse", [])]
        ordered_ids = dense_ids + [doc_id for doc_id in sparse_ids if doc_id not in dense_ids]

        # Local refinement is optional and remains distinct from any future
        # remote compute productization.
        if not enable_refinement or not getattr(self.manager, "solver_available", False):
            return {
                "retrieval": search_output,
                "retrieval_count": len(retrieval_ids),
                "solver_output": None,
                "selected_ids": ordered_ids[: min(10, len(ordered_ids))],
            }

        refine_results = self.manager.refine(
            query_vector=query_vector,
            query_text="",
            candidate_ids=retrieval_ids
        )
        
        return {
            "retrieval": search_output,
            "retrieval_count": len(retrieval_ids),
            "solver_output": refine_results.get('solver_output'),
            "selected_ids": refine_results.get('selected_ids'),
            "solver_backend": refine_results.get("backend_kind"),
        }
