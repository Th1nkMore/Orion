import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.freeze_stage2l_formal_reused_event import freeze_reused_event
from scripts.scenario_factory_lib import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixtures(tmp_path):
    plan = tmp_path / "plan.json"
    package = tmp_path / "package.json"
    queue = tmp_path / "queue.json"
    decisions = tmp_path / "decisions.json"
    _write(plan, {
        "schema": "orion.stage2_l.formal_route_plan.v1",
        "events": [{
            "route_index": 177,
            "formal_split": "train",
            "town": "Town03",
            "scenario_family": "ParkedObstacleTwoWays",
            "split_origin": "development_screen",
            "selection_role": "published_clean_valid",
            "replay_required": False,
        }],
    })
    _write(package, {
        "schema": "orion.scenario_event_package.v1",
        "route": {"route_index": 177, "run_id": "old_clean_run"},
    })
    package_sha = sha256_file(package)
    event = {
        "event_id": "route177_step276",
        "route_index": 177,
        "town": "Town03",
        "scenario_family": "ParkedObstacleTwoWays",
        "split_origin": "development_screen",
        "event_package": {"path": str(package), "sha256": package_sha},
        "runtime_valid": True,
        "qa_input_ready": True,
        "actor_grounded_event": True,
    }
    _write(queue, {
        "schema": "orion.scenario_event_review_queue.v1",
        "review_order": [event],
    })
    _write(decisions, {
        "schema": "orion.scenario_event_review_decisions.v1",
        "review_queue": {"sha256": sha256_file(queue)},
        "decisions": [{
            "event_id": "route177_step276",
            "event_package_sha256": package_sha,
            "decision": "accept",
            "checks": {
                "visual_stream_integrity": "pass",
                "actor_event_semantics": "pass",
                "front_bev_temporal_alignment": "pass",
                "no_actor_disappearance_or_spawn_artifact": "pass",
            },
        }],
    })
    return plan, queue, decisions, package


def test_freeze_reused_event_binds_original_review_to_formal_split(tmp_path):
    plan, queue, decisions, package = _fixtures(tmp_path)
    result = freeze_reused_event(
        formal_plan_path=plan,
        review_queue_path=queue,
        review_decisions_path=decisions,
        event_package_path=package,
        route_index=177,
    )
    assert result["counts"]["accepted_events"] == 1
    assert result["events"][0]["formal_split"] == "train"
    assert result["provenance"]["reuse_authorized_by_formal_plan"] is True


def test_freeze_reused_event_rejects_nonreusable_plan(tmp_path):
    plan, queue, decisions, package = _fixtures(tmp_path)
    value = json.loads(plan.read_text())
    value["events"][0]["replay_required"] = True
    _write(plan, value)
    with pytest.raises(ValueError, match="does not authorize"):
        freeze_reused_event(
            formal_plan_path=plan,
            review_queue_path=queue,
            review_decisions_path=decisions,
            event_package_path=package,
            route_index=177,
        )
