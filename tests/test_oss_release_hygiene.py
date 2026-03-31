from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from voyager_index._internal.inference.index_core.feature_bridge import FeatureBridge


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_validation_report",
    REPO_ROOT / "scripts" / "release_validation_report.py",
)
assert SPEC is not None and SPEC.loader is not None
rvr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rvr)


def test_release_report_redacts_external_paths() -> None:
    repo_root = Path("/tmp/example-repo")
    assert rvr.describe_path(Path("/tmp/private/dataset.json"), repo_root) == "dataset.json"


def test_no_committed_validation_reports_bundle_remains() -> None:
    assert not (REPO_ROOT / "validation-reports").exists()


def test_feature_bridge_error_is_portable() -> None:
    with pytest.raises(ImportError) as exc:
        FeatureBridge()

    assert "/workspace/" not in str(exc.value)
    assert "private Voyager extension repo" in str(exc.value)


def test_root_pyproject_no_longer_packages_from_compat_src() -> None:
    payload = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "compat/src" not in payload


def test_reference_api_dockerfile_uses_root_src_tree() -> None:
    payload = (REPO_ROOT / "deploy" / "reference-api" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY src /app/src" in payload
    assert "COPY compat" not in payload
    assert "./src/kernels/knapsack_solver" in payload
    assert "latence_solver-*.whl" in payload
