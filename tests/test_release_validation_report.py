from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "release_validation_report",
    Path(__file__).resolve().parents[1] / "scripts" / "release_validation_report.py",
)
assert SPEC is not None and SPEC.loader is not None
rvr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rvr)


def test_summarize_mapping_captures_nested_status_and_keys() -> None:
    summary = rvr.summarize_mapping(
        {
            "maxsim": {"status": "passed", "elapsed_ms": 1.2},
            "solver": {"status": "failed", "reason": "missing_model"},
            "meta": {"records": 7},
        }
    )

    assert summary["top_level_keys"] == ["maxsim", "meta", "solver"]
    assert summary["nested_status"] == {"maxsim": "passed", "solver": "failed"}
    assert summary["nested_keys"]["meta"] == ["records"]
