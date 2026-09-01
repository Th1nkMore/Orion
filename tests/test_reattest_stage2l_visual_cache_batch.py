import json
from pathlib import Path

import pytest

import scripts.reattest_stage2l_visual_cache_batch as module
from scripts.reattest_stage2l_visual_cache_batch import run_batch
from scripts.scenario_factory_lib import sha256_file


CHECKPOINT = "4" * 64


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _manifest(tmp_path: Path, entries):
    return _write_json(
        tmp_path / "batch.json",
        {
            "schema": "orion.stage2l_visual_cache_reuse_batch.v1",
            "expected_orion_checkpoint_sha256": CHECKPOINT,
            "entries": entries,
        },
    )


def _reference(path: Path):
    return {"path": str(path), "sha256": sha256_file(path)}


def test_runs_hash_bound_batch_and_writes_report(tmp_path, monkeypatch):
    source = _write_json(tmp_path / "source.json", {"source": True})
    target = _write_json(tmp_path / "target.json", {"target": True})
    manifest = _manifest(
        tmp_path,
        [
            {
                "event_id": "route1_step20",
                "source_cache_manifest": _reference(source),
                "target_factory_report": _reference(target),
            }
        ],
    )

    def fake_reattest(**kwargs):
        _write_json(kwargs["output_manifest_path"], {"reattested": True})
        _write_json(kwargs["output_attestation_path"], {"attested": True})
        assert kwargs["expected_orion_checkpoint_sha256"] == CHECKPOINT
        return {"group_ids": ["a", "b", "c"]}

    monkeypatch.setattr(module, "reattest_cache", fake_reattest)
    report = run_batch(manifest_path=manifest, output_root=tmp_path / "output")
    assert report["status"] == "completed_all_cache_reuse_attestations"
    assert report["event_count"] == 1
    assert report["events"][0]["group_count"] == 3
    assert report["formal_training_ready"] is False
    assert report["stage2p_allowed"] is False
    assert (tmp_path / "output" / "batch_report.json").is_file()


def test_rejects_duplicate_event_ids_before_writing(tmp_path):
    source = _write_json(tmp_path / "source.json", {"source": True})
    target = _write_json(tmp_path / "target.json", {"target": True})
    row = {
        "event_id": "route1_step20",
        "source_cache_manifest": _reference(source),
        "target_factory_report": _reference(target),
    }
    manifest = _manifest(tmp_path, [row, dict(row)])
    with pytest.raises(ValueError, match="duplicated"):
        run_batch(manifest_path=manifest, output_root=tmp_path / "output")


def test_rejects_hash_mismatch(tmp_path):
    source = _write_json(tmp_path / "source.json", {"source": True})
    target = _write_json(tmp_path / "target.json", {"target": True})
    manifest = _manifest(
        tmp_path,
        [
            {
                "event_id": "route1_step20",
                "source_cache_manifest": {
                    "path": str(source),
                    "sha256": "0" * 64,
                },
                "target_factory_report": _reference(target),
            }
        ],
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_batch(manifest_path=manifest, output_root=tmp_path / "output")
