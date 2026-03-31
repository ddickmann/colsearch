from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager
from voyager_index._internal.inference.search_pipeline import SearchPipeline


def test_search_pipeline_supports_sparse_only_text_queries(tmp_path: Path) -> None:
    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)
    pipeline.index(
        corpus=["alpha document", "beta document"],
        vectors=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        ids=[1, 2],
        payloads=[{"label": "alpha"}, {"label": "beta"}],
    )

    result = pipeline.search("alpha", top_k_retrieval=2)

    assert result["retrieval_count"] >= 1
    assert 1 in result["selected_ids"]


def test_search_pipeline_rejects_multivector_dense_queries(tmp_path: Path) -> None:
    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)

    with pytest.raises(ValueError, match="single dense query vector"):
        pipeline.search(np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32))


def test_search_pipeline_reports_solver_backend_when_refining(tmp_path: Path) -> None:
    pytest.importorskip("latence_solver")

    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)
    pipeline.index(
        corpus=["alpha voyager", "beta support", "gamma noise"],
        vectors=np.asarray(
            [[1, 0, 0, 0], [0.8, 0.2, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        ),
        ids=[1, 2, 3],
        payloads=[
            {"text": "alpha voyager", "token_count": 80},
            {"text": "beta support", "token_count": 90},
            {"text": "gamma noise", "token_count": 120},
        ],
    )

    result = pipeline.search(np.asarray([1, 0, 0, 0], dtype=np.float32), top_k_retrieval=3, enable_refinement=True)

    assert result["solver_output"] is not None
    assert result["solver_backend"] is not None


def test_hybrid_manager_orders_selected_candidates_by_marginal_gain() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    solver_candidates = [
        {
            "embedding": [0.6, 0.8],
            "fact_density": 0.2,
            "centrality_score": 0.3,
            "recency_score": 0.2,
            "auxiliary_score": 0.1,
        },
        {
            "embedding": [1.0, 0.0],
            "fact_density": 0.9,
            "centrality_score": 0.9,
            "recency_score": 0.8,
            "auxiliary_score": 0.6,
        },
    ]

    ordered = manager._selected_order(query, solver_candidates, [0, 1])

    assert ordered == [1, 0]


def test_hybrid_manager_builds_solver_candidates_from_roq_payload_when_vector_missing() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = 4

    class DummyDecoded:
        def __init__(self, array):
            self._array = array

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class DummyQuantizer:
        def decode(self, codes, scale, offset):
            assert codes.shape[0] == 1
            return DummyDecoded(np.asarray([[1.0, 0.0]], dtype=np.float32))

    manager.hnsw = types.SimpleNamespace(
        quantizer=DummyQuantizer(),
        retrieve=lambda ids: [
            {
                "id": ids[0],
                "vector": None,
                "payload": {
                    "text": "voyager example",
                    "roq_codes": [1, 2, 3, 4],
                    "roq_scale": 1.0,
                    "roq_offset": 0.0,
                },
            }
        ],
    )

    candidates = manager._build_solver_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [7],
        query_text="voyager",
    )

    assert len(candidates) == 1
    assert candidates[0]["chunk_id"] == "7"
    assert candidates[0]["embedding"] == [1.0, 0.0]


def test_hybrid_manager_uses_ontology_query_payload_for_solver_features() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = None
    shared_vector = [1.0, 0.0]
    manager.hnsw = types.SimpleNamespace(
        retrieve=lambda ids: [
            {
                "id": 11,
                "vector": shared_vector,
                "payload": {
                    "text": "neutral content",
                    "ontology_terms": ["invoice total"],
                    "ontology_labels": ["table"],
                    "ontology_confidences": [0.9],
                    "ontology_evidence_counts": [4],
                    "ontology_match_count": 1,
                    "ontology_confidence": 0.9,
                    "ontology_concept_density": 0.6,
                    "ontology_relation_density": 0.5,
                },
            },
            {
                "id": 12,
                "vector": shared_vector,
                "payload": {
                    "text": "neutral content",
                },
            },
        ],
    )

    candidates = manager._build_solver_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [11, 12],
        query_text="invoice total",
        query_payload={"ontology_terms": ["invoice total"], "label": "table"},
    )

    by_id = {candidate["chunk_id"]: candidate for candidate in candidates}
    assert by_id["11"]["ontology_query_match"] > by_id["12"]["ontology_query_match"]
    assert by_id["11"]["ontology_entity_coverage"] > by_id["12"]["ontology_entity_coverage"]
    assert by_id["11"]["centrality_score"] > by_id["12"]["centrality_score"]
    assert by_id["11"]["auxiliary_score"] > by_id["12"]["auxiliary_score"]
