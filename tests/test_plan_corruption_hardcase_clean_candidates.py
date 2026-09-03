import hashlib
import json

from scripts.plan_corruption_hardcase_clean_candidates import build_plan


def write_package(root, route, *, collision=0, stopped=1.0, ttc=1.0):
    path = root / f"wave/route{route}/event_package.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route": {"slurm_job_id": str(1000 + route)},
        "runtime": {"valid": True},
        "outcome_class": "VALID_NEAR_MISS_OR_CONFLICT",
        "official_endpoint": {
            "status": "Completed",
            "collision_count": collision,
            "serious_infraction_count": 0,
            "scores": {"score_route": 100},
        },
        "continuous_safety": {
            "efficiency": {"stopped_below_0_25_mps_seconds": stopped},
            "safety": {
                "critical_frame": {
                    "actor": {
                        "category": "vehicle",
                        "type_id": "vehicle.test",
                        "closing_speed_mps": 4.0,
                        "obb_collision_ttc_seconds": ttc,
                        "obb_separating_axis_gap_m": 2.0,
                    }
                }
            },
        },
        "source_files": {
            "route_xml": {"path": f"route_{route}.xml", "sha256": "abc"}
        },
    }
    path.write_text(json.dumps(payload) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def event(route, digest, *, split="train", accepted=True):
    return {
        "route_index": route,
        "event_id": f"route{route}_step10",
        "town": "Town01",
        "scenario_family": "DynamicObjectCrossing",
        "screen_role": "dynamic_path_conflict",
        "formal_split": split,
        "runtime_valid": True,
        "human_review": {"decision": "accept" if accepted else "reject"},
        "event_package": {
            "path": f"/remote/event_packages/wave/route{route}/event_package.json",
            "sha256": digest,
        },
    }


def test_planner_uses_current_clean_runtime_and_protects_splits(tmp_path):
    root = tmp_path / "event_packages"
    _, hash10 = write_package(root, 10, stopped=2.0, ttc=0.4)
    _, hash11 = write_package(root, 11, collision=1, ttc=0.2)
    _, hash12 = write_package(root, 12, stopped=9.0, ttc=0.1)
    _, hash13 = write_package(root, 13, stopped=1.0, ttc=0.3)
    bank_path = tmp_path / "bank.json"
    bank = {
        "events": [
            event(10, hash10),
            event(11, hash11),
            event(12, hash12),
            event(13, hash13, split="dev"),
        ]
    }
    bank_path.write_text(json.dumps(bank) + "\n")

    plan = build_plan(
        bank,
        event_bank_path=bank_path,
        event_package_roots=[root],
        excluded_routes=[],
        allowed_splits=["train"],
        maximum_total_stopped_seconds=8.0,
        limit=3,
    )

    assert [row["route_index"] for row in plan["selected_candidates"]] == [10]
    reasons = {
        row["route_index"]: set(row["reasons"])
        for row in plan["excluded_reviewed_events"]
    }
    assert "clean_collision" in reasons[11]
    assert "clean_total_stopped_exposure_exceeds_liveness_bound" in reasons[12]
    assert "protected_or_non_development_split" in reasons[13]
    assert plan["policy"]["corruption_conditioned_outputs_used"] is False
    assert plan["execution_locks"]["clean_submission"] is False


def test_planner_ranks_ttc_then_closing_speed_and_honors_explicit_exclusion(tmp_path):
    root = tmp_path / "event_packages"
    events = []
    for route, ttc in ((20, 1.2), (21, 0.5), (22, 0.8)):
        _, digest = write_package(root, route, ttc=ttc)
        events.append(event(route, digest))
    bank_path = tmp_path / "bank.json"
    bank = {"events": events}
    bank_path.write_text(json.dumps(bank) + "\n")

    plan = build_plan(
        bank,
        event_bank_path=bank_path,
        event_package_roots=[root],
        excluded_routes=[21],
        allowed_splits=["train"],
        maximum_total_stopped_seconds=8.0,
        limit=2,
    )
    assert [row["route_index"] for row in plan["selected_candidates"]] == [22, 20]
    assert plan["counts"] == {
        "eligible_before_limit": 2,
        "selected": 2,
        "excluded": 1,
    }
