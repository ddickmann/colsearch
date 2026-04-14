from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from voyager_index._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)


def _make_manager(tmp_path: Path) -> ShardSegmentManager:
    return ShardSegmentManager(tmp_path / "shard", config=ShardEngineConfig(dim=4), device="cpu")


def test_score_sealed_candidates_prefers_colbandit_pipeline(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    rust_index = Mock()
    try:
        mgr._pipeline = object()
        mgr._rust_index = rust_index
        mgr._gpu_corpus = object()

        scfg = mgr._config.to_search_config()
        scfg.use_colbandit = True

        def fake_pipeline_fetch(*args, **kwargs):
            assert kwargs["exact_path"] == "colbandit_pipeline_fetch"
            assert kwargs["use_colbandit"] is True
            return [(7, 0.9)], {"num_docs_scored": 1}

        mgr._score_pipeline_fetch = fake_pipeline_fetch  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [7],
            {0: [7]},
            1,
            scfg,
            torch.device("cuda"),
        )
    finally:
        mgr.close()

    assert results == [(7, 0.9)]
    assert exact_ids == [7]
    assert exact_path == "colbandit_pipeline_fetch"
    assert stats["num_docs_scored"] == 1
    rust_index.score_candidates_exact.assert_not_called()


def test_score_sealed_candidates_prefers_quantized_pipeline(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    rust_index = Mock()
    try:
        mgr._pipeline = object()
        mgr._rust_index = rust_index

        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "fp8"

        def fake_pipeline_fetch(*args, **kwargs):
            assert kwargs["exact_path"] == "pipeline_quantized"
            assert kwargs["use_colbandit"] is False
            return [(11, 0.7)], {"num_docs_scored": 1}

        mgr._score_pipeline_fetch = fake_pipeline_fetch  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [11],
            {0: [11]},
            1,
            scfg,
            torch.device("cuda"),
        )
    finally:
        mgr.close()

    assert results == [(11, 0.7)]
    assert exact_ids == [11]
    assert exact_path == "pipeline_quantized"
    assert stats["num_docs_scored"] == 1
    rust_index.score_candidates_exact.assert_not_called()


def test_score_sealed_candidates_prefers_roq4_pipeline(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    rust_index = Mock()
    try:
        mgr._rust_index = rust_index

        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "roq4"

        def fake_roq_score(*args, **kwargs):
            return ([(21, 0.95)], {"num_docs_scored": 1})

        mgr._score_roq4_candidates = fake_roq_score  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [21],
            {0: [21]},
            1,
            scfg,
            torch.device("cuda"),
        )
    finally:
        mgr.close()

    assert results == [(21, 0.95)]
    assert exact_ids == [21]
    assert exact_path == "roq4_pipeline"
    assert stats["num_docs_scored"] == 1
    rust_index.score_candidates_exact.assert_not_called()


def test_inspect_query_pipeline_accepts_runtime_override_kwargs(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    try:
        captured: dict[str, object] = {}
        mgr._is_built = True
        mgr._doc_ids = [1]
        mgr._ensure_warmup = lambda: None  # type: ignore[assignment]
        mgr._resolve_scoring_device = lambda: torch.device("cpu")  # type: ignore[assignment]
        mgr._route_prefetch_cap = lambda scfg: 0  # type: ignore[assignment]
        mgr._router = Mock()
        mgr._router.route.return_value = SimpleNamespace(doc_ids=[1], by_shard={0: [1]})

        def fake_apply_search_overrides(scfg, **kwargs):
            captured.update(kwargs)
            return scfg

        mgr._apply_search_overrides = fake_apply_search_overrides  # type: ignore[method-assign]
        mgr._prune_routed_candidates = lambda q, routed, scfg, dev: ([1], {0: [1]}, "none")  # type: ignore[assignment]
        mgr._score_sealed_candidates = lambda q, candidate_ids, docs_by_shard, internal_k, scfg, dev: (  # type: ignore[assignment]
            [(1, 0.9)],
            [1],
            "pipeline_fetch",
            mgr._empty_exact_stage_stats("pipeline_fetch"),
        )

        trace = mgr.inspect_query_pipeline(
            np.zeros((2, 4), dtype=np.float32),
            k=1,
            quantization_mode="fp8",
            max_docs_exact=8,
            n_full_scores=16,
        )
    finally:
        mgr.close()

    assert captured["quantization_mode"] == "fp8"
    assert captured["max_docs_exact"] == 8
    assert captured["n_full_scores"] == 16
    assert trace["result_ids"] == [1]
    assert trace["exact_path"] == "pipeline_fetch"
