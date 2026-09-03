import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import diagnose_stage2l_v12_partial_coverage_collection as module


def test_partial_diagnostic_never_accepts_incomplete_runtime(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "orion.stage2l_v12_train_coverage_repair_protocol.v1",
                "candidate_gate": {
                    "minimum_positive_saved_frames_in_one_independent_event": 3
                },
            }
        )
    )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch.v1",
                "split": "train_coverage_repair",
                "routes": [
                    {
                        "route_index": 167,
                        "town": "Town03",
                        "scenario_type": "YieldToEmergencyVehicle",
                    }
                ],
            }
        )
    )
    run_dir = tmp_path / "run"
    scenario = run_dir / "records_orion_traj_0" / "RouteScenario_test"
    (scenario / "meta").mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    (scenario / "control_trace.jsonl").write_text(
        json.dumps(
            {
                "step": 10,
                "sim_time_seconds": 0.5,
                "route_progress": 0.1,
            }
        )
        + "\n"
    )
    for frame in range(3):
        (scenario / "meta" / ("%04d.json" % frame)).write_text(
            json.dumps({"plan": {}, "closedloop_safety": {}})
        )
        for directory in module.CAMERA_DIRECTORY_BY_VIEW.values():
            (scenario / directory).mkdir(exist_ok=True)
            (scenario / directory / ("%04d.png" % frame)).write_bytes(b"png")

    def fake_geometry(plan, safety, patch_hw):
        relevance = torch.zeros(6, 40, 40)
        relevance[5, 2, 3] = 1.0
        return SimpleNamespace(
            relevance=relevance,
            provenance={"support_mode": "visible_conflict_actor_only"},
        )

    monkeypatch.setattr(module, "build_task_relevance_map", fake_geometry)
    value = module.diagnose(
        protocol_path=protocol,
        batch_manifest_path=batch,
        run_dir=run_dir,
        slurm_job_id=1,
        slurm_state="FAILED",
        slurm_exit_code="124:0",
    )
    assert value["coverage_gate_passed_on_partial_frames"] is True
    assert value["candidate_accepted"] is False
    assert value["runtime_complete"] is False
    assert value["launch_locks"]["gpu_r_only_smoke"] is False
