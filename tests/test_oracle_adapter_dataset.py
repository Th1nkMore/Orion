import copy
import json
from pathlib import Path

import pytest

from uq_estimator.oracle_adapter_dataset import (
    DatasetIntegrityError,
    build_run_samples,
    export_dataset,
    validate_dataset,
    validate_sample,
)


CAMERA_DIRS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
    "bev",
)


def _risk(active, oracle_response=True):
    if not oracle_response:
        return {
            "mode": "off",
            "raw_score": 0.2,
            "applied_score": None,
            "intensity": 0.0,
            "speed_cap": 5.0,
            "base_throttle": 0.7,
            "base_brake": 0.0,
            "throttle": 0.7,
            "brake": 0.0,
        }
    return {
        "mode": "oracle",
        "raw_score": 0.2,
        "applied_score": 1.0 if active else 0.0,
        "intensity": 1.0 if active else 0.0,
        "speed_cap": 0.0 if active else 5.0,
        "base_throttle": 0.7,
        "base_brake": 0.0,
        "throttle": 0.0 if active else 0.7,
        "brake": 0.5 if active else 0.0,
    }


def make_run(tmp_path: Path, *, oracle_response=True) -> Path:
    run = tmp_path / "route146_hazard_front_corrupt_transient_oracle_stop-123"
    record = run / "records_orion_traj_0" / "RouteScenario_test"
    for directory in (*CAMERA_DIRS, "meta"):
        (record / directory).mkdir(parents=True, exist_ok=True)

    manifest = {
        "pilot_run_id": "fixture",
        "pilot_route_index": "146",
        "pilot_variant": "hazard",
        "pilot_condition": "front_corrupt_transient_oracle_stop",
        "slurm_job_id": "123",
        "orion_closedloop_corruption": "camera_dropout",
        "orion_closedloop_corruption_views": "front",
        "orion_closedloop_corruption_severity": "1",
    }
    (run / "manifest.json").write_text(json.dumps(manifest))

    trace = []
    metric = {}
    for step in range(71):
        active = step < 10
        risk = _risk(active, oracle_response=oracle_response)
        trace.append(
            {
                "step": step,
                "sim_time_seconds": step / 20.0,
                "route_progress": step / 100.0,
                "speed": 2.0,
                "steer": 0.1,
                "corruption_active": active,
                "corruption_schedule_mode": "route_triggered_timed",
                "corruption_trigger_time_seconds": 0.0,
                "corruption_elapsed_seconds": step / 20.0,
                "raw_uq_score": 0.2,
                "risk": risk,
            }
        )
        metric[str(step)] = {
            "location": [step / 10.0, 2.0 * step / 10.0, 0.0],
            "forward_vector": [1.0, 0.0, 0.0],
            "right_vector": [0.0, 1.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "acceleration": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }
    (record / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace)
    )
    (record / "metric_info.json").write_text(json.dumps(metric))

    for frame in range(8):
        step = frame * 10
        for directory in CAMERA_DIRS:
            (record / directory / f"{frame:04d}.png").write_bytes(b"fixture")
        meta = {
            "plan": [[0.0, 1.0] for _ in range(6)],
            "command": 3,
            "corruption_active": trace[step]["corruption_active"],
            "route_progress": trace[step]["route_progress"],
            "risk_governor": trace[step]["risk"],
        }
        (record / "meta" / f"{frame:04d}.json").write_text(json.dumps(meta))

    evaluation = {
        "eligible": True,
        "entry_status": "Finished",
        "_checkpoint": {
            "records": [
                {
                    "route_id": "RouteScenario_test",
                    "scenario_name": "DynamicObjectCrossing_1",
                    "town_name": "Town01",
                    "status": "Completed",
                    "scores": {
                        "score_composed": 100.0,
                        "score_route": 100.0,
                        "score_penalty": 1.0,
                    },
                    "meta": {"duration_game": 10.0},
                    "infractions": {
                        "collisions_pedestrian": [],
                        "collisions_vehicle": [],
                        "collisions_layout": [],
                        "red_light": [],
                        "stop_infraction": [],
                        "outside_route_lanes": [],
                        "route_dev": [],
                        "vehicle_blocked": [],
                        "route_timeout": [],
                        "scenario_timeouts": [],
                    },
                }
            ]
        },
    }
    (run / "eval_orion_traj_0.json").write_text(json.dumps(evaluation))
    return run


def test_builds_same_rollout_expert_offsets_and_labels(tmp_path):
    run = make_run(tmp_path)
    samples, summary = build_run_samples(run, relevance="on_path")

    assert summary["candidate_2hz_frames"] == 8
    assert summary["exported_samples"] == 2
    assert summary["excluded_terminal_horizon"] == 6
    assert samples[0]["expert"]["trajectory_displacements_m"] == [
        [2.0, 1.0] for _ in range(6)
    ]
    assert samples[0]["expert"]["trajectory_cumulative_m"][-1] == [12.0, 6.0]
    assert samples[0]["expert"]["base_plan_displacements_m"] == [
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert samples[0]["labels"]["stop_required"] is True
    assert samples[0]["labels"]["intervention"] is True
    assert samples[1]["labels"]["recover"] is True
    assert samples[1]["controls"]["base"] == samples[1]["controls"]["post"]


def test_off_path_keeps_oracle_uq_but_zeroes_path_risk(tmp_path):
    samples, _ = build_run_samples(
        make_run(tmp_path, oracle_response=False), relevance="off_path"
    )
    assert samples[0]["oracle"]["uq_global"] == 1.0
    assert samples[0]["oracle"]["path_risk"] == 0.0
    assert samples[0]["labels"]["stop_required"] is False


def test_rejects_relabeling_an_on_path_oracle_rollout_as_off_path(tmp_path):
    with pytest.raises(DatasetIntegrityError, match="Do not relabel"):
        build_run_samples(make_run(tmp_path), relevance="off_path")


def test_missing_one_camera_frame_is_rejected(tmp_path):
    run = make_run(tmp_path)
    missing = next(run.glob("records_*/*/rgb_front_right/0001.png"))
    missing.unlink()
    with pytest.raises(DatasetIntegrityError, match="frame mismatch"):
        build_run_samples(run, relevance="on_path")


def test_internal_metric_gap_is_rejected(tmp_path):
    run = make_run(tmp_path)
    path = next(run.glob("records_*/*/metric_info.json"))
    metric = json.loads(path.read_text())
    del metric["20"]
    path.write_text(json.dumps(metric))
    with pytest.raises(DatasetIntegrityError, match="metric_info steps must be contiguous"):
        build_run_samples(run, relevance="on_path")


def test_cross_rollout_expert_is_rejected(tmp_path):
    samples, _ = build_run_samples(make_run(tmp_path), relevance="on_path")
    invalid = copy.deepcopy(samples[0])
    invalid["expert"]["source_rollout_id"] = "different/rollout"
    with pytest.raises(DatasetIntegrityError, match="Simulation-fork violation"):
        validate_sample(invalid)


def test_export_then_validate(tmp_path):
    run = make_run(tmp_path)
    output = tmp_path / "dataset"
    manifest = export_dataset([run], output, relevance="on_path")
    assert manifest["sample_count"] == 2
    assert validate_dataset(output) == {
        "valid": True,
        "dataset_version": "orion.oracle_adapter.dataset.v1",
        "sample_schema_version": "orion.oracle_adapter.sample.v1",
        "sample_count": 2,
        "rollout_count": 1,
    }
