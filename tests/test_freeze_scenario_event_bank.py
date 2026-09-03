import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_scenario_event_bank.py"
SPEC = importlib.util.spec_from_file_location("freeze_scenario_event_bank", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_inputs(tmp_path: Path, origins=("development_screen", "development_screen")):
    rows = []
    for index, origin in enumerate(origins, start=1):
        rows.append(
            {
                "event_id": "route%d_step%d" % (index, 100 + index),
                "route_index": index,
                "split_origin": origin,
                "town": "Town%02d" % index,
                "scenario_family": "Family%d" % index,
                "event_package": {
                    "path": str(tmp_path / ("package%d.json" % index)),
                    "sha256": "%064d" % index,
                },
            }
        )
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_event_review_queue.v1",
                "review_order": rows,
            }
        )
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_event_review_decisions.v1",
                "reviewer": "reviewer",
                "reviewed_at": "2026-08-29T12:00:00+08:00",
                "review_queue": {
                    "path": str(queue),
                    "sha256": MODULE.sha256_file(queue),
                },
                "decisions": [
                    {
                        "event_id": row["event_id"],
                        "event_package_sha256": row["event_package"]["sha256"],
                        "decision": "accept",
                        "checks": {name: "pass" for name in MODULE.HUMAN_CHECKS},
                        "rejection_basis": None,
                        "notes": "",
                    }
                    for row in rows
                ],
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "data_gates": {
                    "formal_training_target": {
                        "minimum_independent_events": 2,
                        "minimum_towns": 2,
                        "minimum_scenario_families": 2,
                        "route_event_split": {"train": 1, "dev": 1, "test": 0},
                    }
                }
            }
        )
    )
    return queue, decisions, config


def test_freeze_assigns_route_disjoint_development_splits(tmp_path):
    queue, decisions, config = _write_inputs(tmp_path)

    bank = MODULE.freeze_event_bank(
        queue_path=queue,
        decisions_path=decisions,
        stage2_config_path=config,
        split_seed="test-seed",
    )

    assert bank["formal_training_ready"] is True
    assert bank["counts"]["splits"] == {"train": 1, "dev": 1, "test": 0}
    assert {row["stage2_split"] for row in bank["events"]} == {"train", "dev"}


def test_freeze_accepts_scenario_factory_protocol_gate_layout(tmp_path):
    queue, decisions, config = _write_inputs(tmp_path)
    payload = json.loads(config.read_text())
    target = payload.pop("data_gates")["formal_training_target"]
    payload["event_review_and_freezing"] = {
        "formal_target": {
            "events": target["minimum_independent_events"],
            "towns": target["minimum_towns"],
            "scenario_families": target["minimum_scenario_families"],
            "route_event_split": target["route_event_split"],
        }
    }
    config.write_text(json.dumps(payload))

    bank = MODULE.freeze_event_bank(
        queue_path=queue,
        decisions_path=decisions,
        stage2_config_path=config,
        split_seed="test-seed",
    )

    assert bank["formal_training_ready"] is True
    assert bank["counts"]["accepted_events"] == 2


def test_freeze_rejects_accepted_event_with_failed_integrity_check(tmp_path):
    queue, decisions, config = _write_inputs(tmp_path)
    payload = json.loads(decisions.read_text())
    payload["decisions"][0]["checks"][MODULE.HUMAN_CHECKS[0]] = "fail"
    decisions.write_text(json.dumps(payload))

    try:
        MODULE.freeze_event_bank(
            queue_path=queue,
            decisions_path=decisions,
            stage2_config_path=config,
            split_seed="test-seed",
        )
    except ValueError as error:
        assert "failed integrity" in str(error)
    else:
        raise AssertionError("failed event integrity was accepted")


def test_locked_test_cannot_be_rejected_for_model_outcome(tmp_path):
    queue, decisions, config = _write_inputs(tmp_path, origins=("locked_test",))
    payload = json.loads(decisions.read_text())
    payload["decisions"][0]["decision"] = "reject"
    payload["decisions"][0]["rejection_basis"] = "not_useful_for_development"
    decisions.write_text(json.dumps(payload))

    try:
        MODULE.freeze_event_bank(
            queue_path=queue,
            decisions_path=decisions,
            stage2_config_path=config,
            split_seed="test-seed",
        )
    except ValueError as error:
        assert "technical integrity" in str(error)
    else:
        raise AssertionError("outcome-based locked-test rejection was accepted")
