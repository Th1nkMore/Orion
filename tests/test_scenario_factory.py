import json
import subprocess
import sys
from pathlib import Path

from scripts.scenario_factory_lib import (
    CAMERA_DIRECTORIES,
    PRE_NATIVE_GLARE_AGENT_SHA256,
    build_event_package,
    select_critical_event,
    validate_clean_runtime_manifest,
)


def _actor(actor_id, ttc, gap):
    return {
        "actor_id": actor_id,
        "category": "walker",
        "type_id": "walker.test",
        "obb_collision_ttc_seconds": ttc,
        "obb_separating_axis_gap_m": gap,
    }


def _row(step, time_seconds, ttc, gap):
    actor = _actor(7, ttc, gap)
    return {
        "step": step,
        "sim_time_seconds": time_seconds,
        "route_progress": 0.1 + step * 0.001,
        "speed": 4.0,
        "closedloop_safety": {
            "available": True,
            "actors": [actor],
            "critical_actor": actor,
            "min_obb_collision_ttc_seconds": ttc,
            "min_obb_separating_axis_gap_m": gap,
            "min_disc_clearance_m": gap,
        },
    }


def _write_run(root: Path):
    root.mkdir()
    manifest = {
        "pilot_run_id": "scenario_factory_wave0",
        "pilot_route_index": "42",
        "pilot_variant": "hazard",
        "pilot_condition": "clean_off",
        "slurm_job_id": "123",
        "orion_closedloop_uq_mode": "none",
        "orion_closedloop_risk_mode": "off",
        "orion_planning_response_mode": "off",
        "orion_enable_legacy_density_uq": "0",
        "orion_closedloop_corruption": None,
        "render_condition": {
            "schema": "orion.closedloop_render_condition.v1",
            "kind": "standard_carla_rgb",
            "native_glare_profile": "none",
            "camera_postprocess_override": False,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    evaluation = {
        "eligible": True,
        "entry_status": "Finished",
        "_checkpoint": {
            "records": [
                {
                    "status": "Completed",
                    "scores": {
                        "score_route": 100.0,
                        "score_penalty": 1.0,
                        "score_composed": 100.0,
                    },
                    "infractions": {},
                }
            ]
        },
    }
    (root / "eval_orion_traj_0.json").write_text(json.dumps(evaluation))
    scenario = root / "records_orion_traj_0" / "RouteScenario_test"
    scenario.mkdir(parents=True)
    rows = [
        _row(10, 0.0, None, None),
        _row(11, 3.5, 1.5, 0.4),
        _row(12, 6.0, None, None),
    ]
    (scenario / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    for directory in CAMERA_DIRECTORIES:
        frame_dir = scenario / directory
        frame_dir.mkdir()
        (frame_dir / "0001.png").write_bytes(b"synthetic-test-frame")
    return rows


def test_select_critical_event_uses_minimum_finite_actor_ttc():
    event = select_critical_event(
        [_row(10, 0.5, 3.0, 2.0), _row(11, 0.55, 0.7, 0.3)]
    )
    assert event["selection_basis"] == "minimum_finite_dynamic_actor_obb_ttc"
    assert event["step"] == 11
    assert event["actor"]["actor_id"] == 7


def test_select_critical_event_can_prefer_scenario_actor_category():
    vehicle = _row(10, 0.50, 1.0, 1.0)
    vehicle["closedloop_safety"]["actors"][0]["category"] = "vehicle"
    walker = _row(11, 0.55, 4.0, 8.0)

    event = select_critical_event(
        [vehicle, walker], preferred_actor_categories=("walker",)
    )

    assert event["step"] == 11
    assert event["actor"]["category"] == "walker"
    assert event["selection_basis"] == "minimum_finite_preferred_actor_obb_ttc"
    assert event["preferred_actor_categories"] == ["walker"]


def test_select_critical_event_excludes_terminal_route_tail():
    driving = _row(10, 0.50, 0.8, 0.4)
    driving["route_progress"] = 0.75
    terminal = _row(11, 0.55, 0.1, 0.1)
    terminal["route_progress"] = 0.999

    event = select_critical_event([driving, terminal])

    assert event["step"] == 10
    assert event["selection_policy"] == {
        "maximum_route_progress_exclusive": 0.98,
        "terminal_route_tail_excluded": True,
        "complete_review_window_required": False,
        "minimum_pre_event_seconds": 0.0,
        "minimum_post_event_seconds": 0.0,
    }


def test_select_critical_event_can_require_a_complete_review_window():
    before = _row(10, 0.0, None, None)
    valid = _row(11, 4.0, 0.8, 0.4)
    truncated = _row(12, 8.5, 0.1, 0.1)
    after = _row(13, 10.0, None, None)

    event = select_critical_event(
        [before, valid, truncated, after], require_complete_review_window=True
    )

    assert event["step"] == 11
    assert event["selection_policy"]["minimum_pre_event_seconds"] == 3.0
    assert event["selection_policy"]["minimum_post_event_seconds"] == 2.0


def test_clean_manifest_rejects_legacy_density_even_when_other_modes_are_off():
    result = validate_clean_runtime_manifest(
        {
            "pilot_condition": "clean_off",
            "orion_closedloop_uq_mode": "none",
            "orion_closedloop_risk_mode": "off",
            "orion_planning_response_mode": "off",
            "orion_enable_legacy_density_uq": "1",
            "orion_closedloop_corruption": None,
        }
    )
    assert not result["valid"]
    assert not result["checks"]["legacy_density_disabled"]


def test_clean_manifest_requires_explicit_or_allowlisted_render_condition():
    base = {
        "pilot_condition": "clean_off",
        "orion_closedloop_uq_mode": "none",
        "orion_closedloop_risk_mode": "off",
        "orion_planning_response_mode": "off",
        "orion_enable_legacy_density_uq": "0",
        "orion_closedloop_corruption": None,
    }
    missing = validate_clean_runtime_manifest(base)
    assert not missing["valid"]
    assert not missing["checks"]["render_condition_clean"]

    legacy = dict(base)
    legacy["source_sha256"] = {
        "team_code/orion_b2d_agent.py": next(iter(PRE_NATIVE_GLARE_AGENT_SHA256))
    }
    accepted = validate_clean_runtime_manifest(legacy)
    assert accepted["valid"]
    assert accepted["render_condition_attestation"].startswith("legacy_")


def test_clean_manifest_rejects_native_glare_as_clean_off():
    result = validate_clean_runtime_manifest(
        {
            "pilot_condition": "clean_off",
            "orion_closedloop_uq_mode": "none",
            "orion_closedloop_risk_mode": "off",
            "orion_planning_response_mode": "off",
            "orion_enable_legacy_density_uq": "0",
            "orion_closedloop_corruption": None,
            "render_condition": {
                "schema": "orion.closedloop_render_condition.v1",
                "kind": "carla_native_low_sun_glare",
                "native_glare_profile": "medium",
                "camera_postprocess_override": True,
            },
        }
    )
    assert not result["valid"]
    assert not result["checks"]["render_condition_clean"]


def test_build_event_package_is_qa_ready_only_with_valid_clean_runtime(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    package = build_event_package(run_dir, split="development_screen")

    assert package["schema"] == "orion.scenario_event_package.v1"
    assert package["runtime"]["valid"]
    assert package["qa_input_ready"]
    assert package["outcome_class"] == "VALID_NEAR_MISS_OR_CONFLICT"
    assert package["critical_event"]["step"] == 11
    assert package["stage1_observation_uq"]["control_influence"] is False
    assert all(
        item["frame_count"] == 1
        for item in package["camera_inventory"].values()
    )


def test_cpu_only_cli_runs_from_project_root(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    output = tmp_path / "package.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_scenario_event_package.py",
            "--run-dir",
            str(run_dir),
            "--split",
            "development_screen",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(output.read_text())["qa_input_ready"] is True
    assert json.loads(completed.stdout)["runtime_valid"] is True
