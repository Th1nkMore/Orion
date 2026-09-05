import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_qwen_navsim_pair.py"
SPEC = importlib.util.spec_from_file_location("audit_qwen_navsim_pair_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pair_audit_checks_membership_and_reports_trajectory_delta(tmp_path):
    scenes = tmp_path / "scenes.jsonl"
    clean = tmp_path / "clean.jsonl"
    corrupted = tmp_path / "corrupted.jsonl"
    _write_jsonl(scenes, [{"meta_info": {"token": "a"}}])
    _write_jsonl(clean, [{"token": "a", "trajectories": [[[0, 0, 0], [1, 0, 0]]]}])
    _write_jsonl(
        corrupted,
        [{"token": "a", "trajectories": [[[0, 0, 0], [1, 2, 0]]]}],
    )

    result = MODULE.audit(scenes, clean, corrupted)

    assert result["pair_integrity"] == "pass"
    assert result["num_tokens"] == 1
    assert result["tokens"][0]["mean_xy_delta_m"] == pytest.approx(1.0)
    assert result["tokens"][0]["endpoint_xy_delta_m"] == pytest.approx(2.0)


def test_pair_audit_rejects_missing_corrupted_token(tmp_path):
    scenes = tmp_path / "scenes.jsonl"
    clean = tmp_path / "clean.jsonl"
    corrupted = tmp_path / "corrupted.jsonl"
    _write_jsonl(scenes, [{"meta_info": {"token": "a"}}])
    _write_jsonl(clean, [{"token": "a", "trajectories": [[[0, 0, 0]]]}])
    _write_jsonl(corrupted, [])

    with pytest.raises(ValueError, match="pair token mismatch"):
        MODULE.audit(scenes, clean, corrupted)
