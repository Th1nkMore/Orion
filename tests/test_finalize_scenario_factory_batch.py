import importlib.util
import json
from pathlib import Path

from scripts.scenario_factory_lib import CAMERA_DIRECTORIES


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finalize_scenario_factory_batch.py"
SPEC = importlib.util.spec_from_file_location("scenario_finalizer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_run(run_dir: Path, job_id: int, runtime_valid: bool):
    run_dir.mkdir(parents=True)
    manifest = {
        "pilot_run_id": "wave0",
        "pilot_route_index": "12",
        "pilot_variant": "hazard",
        "pilot_condition": "clean_off" if runtime_valid else "bad",
        "slurm_job_id": str(job_id),
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
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    evaluation = {
        "eligible": True,
        "entry_status": "Finished",
        "_checkpoint": {
            "records": [
                {
                    "status": "Completed",
                    "scores": {"score_route": 100.0, "score_penalty": 1.0},
                    "infractions": {},
                }
            ]
        },
    }
    (run_dir / "eval_orion_traj_0.json").write_text(json.dumps(evaluation))
    scenario = run_dir / "records_orion_traj_0" / "RouteScenario_test"
    scenario.mkdir(parents=True)
    actor = {
        "actor_id": 9,
        "category": "vehicle",
        "type_id": "vehicle.test",
        "obb_collision_ttc_seconds": 2.0,
        "obb_separating_axis_gap_m": 1.0,
    }
    rows = [
        {
            "step": 10 + offset,
            "sim_time_seconds": 0.5 + offset * 0.05,
            "route_progress": 0.1 + offset * 0.001,
            "speed": 3.0,
            "closedloop_safety": {
                "available": True,
                "actors": [actor],
                "critical_actor": actor,
                "min_obb_collision_ttc_seconds": 2.0,
                "min_obb_separating_axis_gap_m": 1.0,
                "min_disc_clearance_m": 1.0,
            },
        }
        for offset in range(2)
    ]
    (scenario / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    for directory in CAMERA_DIRECTORIES:
        stream = scenario / directory
        stream.mkdir()
        (stream / "0001.png").write_bytes(b"test")


def test_finalizer_prefers_latest_runtime_valid_attempt(tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch.v1",
                "run_id": "wave0",
                "split": "development_screen",
                "routes": [
                    {
                        "route_index": 12,
                        "town": "Town04",
                        "scenario_type": "ConstructionObstacle",
                        "screen_role": "static_visual_hazard",
                    }
                ],
            }
        )
    )
    results = tmp_path / "results"
    _write_run(results / "route12_hazard_clean_off-100", 100, True)
    _write_run(results / "route12_hazard_clean_off-101", 101, False)
    output = tmp_path / "output"

    report = MODULE.finalize_batch(
        project_root=Path.cwd(),
        batch_manifest_path=batch,
        results_root=results,
        output_root=output,
        render_visuals=False,
    )

    assert report["runtime_valid_count"] == 1
    assert report["routes"][0]["selected_job_id"] == 100
    assert report["routes"][0]["qa_input_ready"] is True
    assert len(report["routes"][0]["attempts"]) == 2
    assert (output / "route12" / "event_package.json").is_file()
    assert (output / "batch_screen_report.json").is_file()


def test_finalizer_preserves_locked_test_split(tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch.v1",
                "run_id": "locked_wave0",
                "split": "locked_test",
                "routes": [
                    {
                        "route_index": 12,
                        "town": "Town04",
                        "scenario_type": "ConstructionObstacle",
                        "screen_role": "static_visual_hazard",
                    }
                ],
            }
        )
    )
    results = tmp_path / "results"
    _write_run(results / "route12_hazard_clean_off-100", 100, True)
    output = tmp_path / "output"

    report = MODULE.finalize_batch(
        project_root=Path.cwd(),
        batch_manifest_path=batch,
        results_root=results,
        output_root=output,
        render_visuals=False,
    )

    package = json.loads((output / "route12" / "event_package.json").read_text())
    assert report["split"] == "locked_test"
    assert package["split"] == "locked_test"


def test_finalizer_maps_train_coverage_repair_to_qa_train_candidate(
    tmp_path, monkeypatch
):
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch.v1",
                "run_id": "coverage_repair",
                "split": "train_coverage_repair",
                "routes": [
                    {
                        "route_index": 12,
                        "town": "Town04",
                        "scenario_type": "ConstructionObstacle",
                        "screen_role": "coverage",
                    }
                ],
            }
        )
    )
    results = tmp_path / "results"
    run = results / "route12_hazard_clean_off-1"
    run.mkdir(parents=True)
    seen = []

    def fake_build(run_dir, *, split, batch_manifest_path, visualization_manifest_path=None):
        seen.append(split)
        return {
            "route": {"route_index": 12},
            "runtime": {"valid": True},
            "outcome_class": "VALID_SAFE_COMPLETION",
            "qa_input_ready": True,
            "critical_event": None,
            "split": split,
        }

    monkeypatch.setattr(MODULE, "build_event_package", fake_build)
    report = MODULE.finalize_batch(
        project_root=tmp_path,
        batch_manifest_path=batch_path,
        results_root=results,
        output_root=tmp_path / "output",
        render_visuals=False,
    )
    package = json.loads(
        (tmp_path / "output" / "route12" / "event_package.json").read_text()
    )
    assert report["split"] == "train_coverage_repair"
    assert seen == ["qa_train_candidate", "qa_train_candidate"]
    assert package["split"] == "qa_train_candidate"


def test_finalizer_keeps_batch_when_one_route_is_unpackageable(tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch.v1",
                "run_id": "wave0",
                "split": "development_screen",
                "routes": [
                    {
                        "route_index": index,
                        "town": "Town04",
                        "scenario_type": "ConstructionObstacle",
                        "screen_role": "static_visual_hazard",
                    }
                    for index in (12, 13)
                ],
            }
        )
    )
    results = tmp_path / "results"
    _write_run(results / "route12_hazard_clean_off-100", 100, True)
    broken = results / "route13_hazard_clean_off-101"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{}")
    output = tmp_path / "output"

    report = MODULE.finalize_batch(
        project_root=Path.cwd(),
        batch_manifest_path=batch,
        results_root=results,
        output_root=output,
        render_visuals=False,
    )

    assert report["route_count"] == 2
    assert report["runtime_valid_count"] == 1
    assert report["unpackageable_route_count"] == 1
    invalid = next(row for row in report["routes"] if row["route_index"] == 13)
    assert invalid["outcome_class"] == "INVALID_RUNTIME_UNPACKAGEABLE"
    assert invalid["attempts"][0]["package_buildable"] is False
    assert len(report["event_packages"]) == 1
