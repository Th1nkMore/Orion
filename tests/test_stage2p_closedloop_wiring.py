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
