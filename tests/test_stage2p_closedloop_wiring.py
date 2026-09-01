from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agent_applies_the_same_stage2p_residual_and_zero_k_fails_closed():
    source = (ROOT / "team_code/orion_b2d_agent.py").read_text(
        encoding="utf-8"
    )
    assert "build_stage2_external_k_map" in source
    assert "input_data_batch['stage2_spatial_uq'] = [[task_risk_k]]" in source
    assert "out_truck = out_truck + residual_numpy" in source
    assert "Zero external K changed native trajectory" in source
    assert "stage2p_controlled_k_engineering_smoke" in source


def test_orion_head_uses_k_only_checkpoint_without_privileged_task_context():
    source = (ROOT / "mmcv/models/dense_heads/orion_head.py").read_text(
        encoding="utf-8"
    )
    assert "self.stage2_input_semantics = 'task_risk_k'" in source
    assert "output = self.stage2_task_adapter(planning_context, selected)" in source
    assert "explicit engineering-smoke config flag" in source


def test_detector_preserves_scalar_k_batch_rank():
    source = (ROOT / "mmcv/models/detectors/orion.py").read_text(
        encoding="utf-8"
    )
    assert "if input_semantics == 'task_risk_k':" in source
    assert "task_risk_k must have [B,V,H,W] shape" in source


def test_runner_records_and_fails_closed_for_controlled_k_smoke():
    source = (ROOT / "scripts/run_closedloop_uq_pilot.sh").read_text(
        encoding="utf-8"
    )
    assert "stage2p_controlled_k_smoke)" in source
    assert "controlled_k_to_stage2p_trajectory_response" in source
    assert "Stage2-P checkpoint is missing or its hash differs" in source
    for name in (
        "ORION_STAGE2_ENGINEERING_SMOKE",
        "ORION_STAGE2_EXTERNAL_K_START_PROGRESS",
        "ORION_STAGE2_EXTERNAL_K_DURATION_SECONDS",
        "ORION_STAGE2_EXTERNAL_K_CAMERA",
        "ORION_STAGE2_EXTERNAL_K_REGION",
        "ORION_STAGE2_EXTERNAL_K_STRENGTH",
        "ORION_STAGE2_EXTERNAL_K_GRID_SIZE",
    ):
        assert f'"{name}"' in source


def test_route147_carla_smoke_is_one_submission_and_nonclaim():
    import json

    path = (
        ROOT
        / "configs/scenario_factory/stage2p_v1_route147_vertical_slice_carla_smoke_v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["route"]["index"] == "147"
    assert value["controlled_k"]["source"] == "external_oracle"
    assert value["controlled_k"]["region"] == [0.58, 0.32, 1.0, 0.95]
    assert value["runtime"]["visual_input"] == "clean"
    assert value["runtime"]["risk_governor"] == "off"
    assert value["runtime"]["privileged_planning_response"] == "off"
    assert value["authorization"]["maximum_submissions"] == 1
    assert value["authorization"]["automatic_retry"] is False
    assert value["authorization"]["formal_stage2p_unlock"] is False
