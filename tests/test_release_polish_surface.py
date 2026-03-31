from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from examples.reference_api_feature_tour import run_feature_tour, write_report
from examples.reference_api_happy_path import run_happy_path
from voyager_index.server import create_app


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_curated_top_level_layout_hides_root_clutter() -> None:
    unexpected_root_entries = {
        "benchmark_colbert_mps.py",
        "benchmark_comparison.py",
        "benchmark_hnsw_retrieval.py",
        "benchmark_m1.py",
        "benchmark_pipeline.py",
        "distributed_benchmark.py",
        "check_data.py",
        "decompress_zst.py",
        "deploy.py",
        "diag_pilot.py",
        "inspect_zst.py",
        "test_model.py",
        "util_inspect_data.py",
        "verify_chunking.py",
        "verify_enriched_mode.py",
        "verify_enrichment_qualitative.py",
        "verify_enterprise_features.py",
        "verify_hybrid_fusion.py",
        "verify_intelligence_offline.py",
        "verify_intelligence_real.py",
        "verify_knapsack_auxiliary.py",
        "verify_multidense.py",
        "verify_retrieval.py",
        "verify_roq_integration.py",
        "verify_search_pipeline.py",
        "sandbox",
        "validation-centroid",
        "validation-centroid-targeted",
        "validation-screening-audit",
        "validation-sidecar",
        "validation-sidecar-slice",
    }
    for entry in unexpected_root_entries:
        assert not (REPO_ROOT / entry).exists(), entry

    expected_release_paths = {
        "docs/README.md",
        "docs/full_feature_cookbook.md",
        "docs/reference_api_tutorial.md",
        "docs/validation/README.md",
        "examples/README.md",
        "notebooks/README.md",
        "tools/README.md",
        "tools/benchmarks/README.md",
        "tools/verification/README.md",
        "tools/dev/README.md",
    }
    for entry in expected_release_paths:
        assert (REPO_ROOT / entry).exists(), entry


def test_readme_routes_users_to_polished_release_entrypoints() -> None:
    payload = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/reference_api_tutorial.md" in payload
    assert "docs/full_feature_cookbook.md" in payload
    assert "docs/README.md" in payload
    assert "examples/README.md" in payload
    assert "examples/reference_api_feature_tour.py" in payload
    assert "notebooks/README.md" in payload
    assert "tools/README.md" in payload
    assert "docs/validation/README.md" in payload
    assert "Tabu Search" in payload
    assert "/reference/optimize" in payload
    assert "/reference/preprocess/documents" in payload
    assert "http://127.0.0.1:8080/docs" in payload


def test_public_examples_and_benchmark_do_not_reach_into_internal_modules() -> None:
    public_files = [
        REPO_ROOT / "examples" / "reference_api_feature_tour.py",
        REPO_ROOT / "examples" / "reference_api_happy_path.py",
        REPO_ROOT / "examples" / "reference_api_late_interaction.py",
        REPO_ROOT / "examples" / "reference_api_multimodal.py",
        REPO_ROOT / "examples" / "vllm_pooling_provider.py",
        REPO_ROOT / "benchmarks" / "oss_reference_benchmark.py",
    ]
    for path in public_files:
        payload = path.read_text(encoding="utf-8")
        assert "voyager_index._internal" not in payload, path


def test_docs_index_and_tutorial_route_to_full_feature_cookbook() -> None:
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    tutorial = (REPO_ROOT / "docs" / "reference_api_tutorial.md").read_text(encoding="utf-8")
    assert "docs/full_feature_cookbook.md" in docs_index
    assert "docs/full_feature_cookbook.md" in tutorial
    assert "examples/reference_api_feature_tour.py" in docs_index
    assert "examples/reference_api_feature_tour.py" in tutorial


def test_reference_api_happy_path_example_smoke(tmp_path: Path) -> None:
    with TestClient(create_app(index_path=str(tmp_path))) as client:
        summary = run_happy_path(client, prefix="smoke")

    assert "smoke-dense" in summary["collections"]
    assert summary["dense_search"]["results"][0]["id"] == "invoice"
    assert summary["late_interaction_search"]["results"][0]["id"] == "li-1"
    assert summary["multimodal_search"]["results"][0]["id"] == "page-1"


def test_reference_api_feature_tour_smoke_and_report(tmp_path: Path) -> None:
    with TestClient(create_app(index_path=str(tmp_path / "index"))) as client:
        report = run_feature_tour(client, prefix="tour", base_url="http://testserver")

    assert report["summary"]["failed"] == 0
    assert report["summary"]["status"] in {"passed", "passed_with_skips"}
    assert report["searches"]["dense_vector"]["top_id"] == "doc-1"
    assert report["searches"]["dense_bm25"]["top_id"] in {"doc-1", "doc-2"}
    assert report["searches"]["dense_bm25"]["total"] >= 1
    assert report["searches"]["late_interaction"]["top_id"] == "li-1"
    assert report["searches"]["multimodal_exact"]["top_id"] == "page-1"
    assert report["checks"]["reference_optimize_health"]["available"] in {True, False}
    assert "execution_mode" in report["checks"]["reference_optimize_health"]

    output_path = write_report(report, tmp_path / "feature-tour-report.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["failed"] == 0
    assert any(step["name"] == "dense_optimized_search" for step in payload["steps"])
