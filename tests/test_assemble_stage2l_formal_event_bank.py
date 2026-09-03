import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assemble_stage2l_formal_event_bank import assemble_formal_event_bank


def _write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _event(route, split):
    return {
        "event_id": "route%d_step1" % route,
        "route_index": route,
        "formal_split": split,
        "town": "Town%02d" % route,
        "scenario_family": "Family%d" % route,
        "human_review": {"decision": "accept"},
        "qa_input_ready": True,
        "runtime_valid": True,
        "actor_grounded_event": True,
    }


def _plan(tmp_path):
    return _write(tmp_path / "plan.json", {
        "schema": "orion.stage2_l.formal_route_plan.v1",
        "events": [
            {
                "event_id": "route1_step1",
                "route_index": 1,
                "formal_split": "train",
                "town": "Town01",
                "scenario_family": "Family1",
                "selection_role": "inherited",
            },
            {
                "route_index": 2,
                "formal_split": "test",
                "town": "Town02",
                "scenario_family": "Family2",
                "selection_role": "locked",
            },
        ],
    })


def test_partial_assembly_reports_exact_missing_routes(tmp_path):
    plan = _plan(tmp_path)
    bank = _write(tmp_path / "bank.json", {
        "schema": "orion.stage2_l.formal_reviewed_wave.v1",
        "events": [_event(1, "train")],
    })
    result = assemble_formal_event_bank(
        formal_plan_path=plan, event_bank_paths=[bank]
    )
    assert result["status"] == "formal_event_bank_incomplete_reviewed_subset"
    assert result["missing_routes_by_split"]["test"] == [2]
    assert result["formal_training_ready"] is False


def test_complete_assembly_preserves_lock_even_when_all_events_exist(tmp_path):
    plan = _plan(tmp_path)
    bank = _write(tmp_path / "bank.json", {
        "schema": "orion.stage2_l.formal_reviewed_wave.v1",
        "events": [_event(1, "train"), _event(2, "test")],
    })
    result = assemble_formal_event_bank(
        formal_plan_path=plan, event_bank_paths=[bank]
    )
    assert result["status"] == "formal_event_bank_complete_reviewed"
    assert result["missing_routes"] == []
    assert result["formal_training_ready"] is False


def test_assembly_rejects_split_drift(tmp_path):
    plan = _plan(tmp_path)
    event = _event(1, "dev")
    bank = _write(tmp_path / "bank.json", {
        "schema": "orion.stage2_l.formal_reviewed_wave.v1",
        "events": [event],
    })
    with pytest.raises(ValueError, match="changed frozen formal split"):
        assemble_formal_event_bank(formal_plan_path=plan, event_bank_paths=[bank])
