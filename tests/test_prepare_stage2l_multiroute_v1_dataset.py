import hashlib
import json

import pytest

from scripts.prepare_stage2l_multiroute_v1_dataset import (
    _materialize_sidecar_paths,
    _validate_bank_and_queue,
)


def _bank_event(index, split):
    return {
        "event_id": "route%d_step10" % index,
        "pilot_split": split,
        "town": "Town%02d" % ((index % 5) + 1),
        "scenario_family": "family%d" % (index % 7),
        "qa_input_ready": True,
        "human_review": {"decision": "accept"},
    }


def test_latest_bank_and_queue_require_event_level_6_2_coverage():
    rows = [
        _bank_event(index, "train" if index < 6 else "dev")
        for index in range(8)
    ]
    bank = {
        "schema": "orion.stage2_l.formal_pilot_event_bank.v1",
        "status": "frozen_bank_training_still_locked",
        "selection_policy": {"reassigns_frozen_splits": False},
        "checks": {"locked_test_untouched": True},
        "events": rows,
    }
    queue = {
        "schema": "orion.stage2_l.qa_geometry_review_queue.v1",
        "status": "pending_human_qa_geometry_review",
        "human_review_count": 8,
        "review_order": [
            {"event_id": row["event_id"], "factory_report": {}}
            for row in rows
        ],
    }
    events, queued = _validate_bank_and_queue(bank, queue)
    assert len(events) == len(queued) == 8
    broken = json.loads(json.dumps(bank))
    broken["events"][6]["pilot_split"] = "train"
    with pytest.raises(ValueError, match="coverage or 6/2 split"):
        _validate_bank_and_queue(broken, queue)


def test_materialize_sidecar_paths_preserves_hash_and_refuses_mismatch(
    tmp_path,
):
    source_dir = tmp_path / "event" / "qa_dataset"
    source_dir.mkdir(parents=True)
    records = source_dir / "records.jsonl"
    records.write_text("", encoding="utf-8")
    sidecar = source_dir / "map_sidecars" / "map.npz"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"sidecar")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    row = {
        "target": {
            "map_sidecar": {
                "path": "map_sidecars/map.npz",
                "sha256": digest,
            }
        }
    }
    materialized = _materialize_sidecar_paths([row], records)
    assert materialized[0]["target"]["map_sidecar"]["path"] == str(
        sidecar.resolve()
    )
    assert row["target"]["map_sidecar"]["path"] == "map_sidecars/map.npz"
    row["target"]["map_sidecar"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _materialize_sidecar_paths([row], records)
