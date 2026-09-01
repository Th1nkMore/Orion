from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "team_code/orion_b2d_agent.py"


def test_failed_waterdrop_alias_is_retired_and_paired_template_is_explicit():
    source = AGENT.read_text()
    assert "from uq_estimator.lens_waterdrop import apply_lens_waterdrop" not in source
    assert "if self.closedloop_corruption == 'lens_waterdrop':" in source
    assert "retired failed v1 visual prototype" in source
    assert "'lens_waterdrop_paired_template'," in source
    assert "ORION_PAIRED_WATERDROP_PROFILE" in source


def test_every_hardcase_orion_runtime_is_guarded_by_visual_approval():
    source = AGENT.read_text()
    assert "ORION_CORRUPTION_VISUAL_APPROVAL_GATE" in source
    assert "verify_visual_approval(" in source
    assert "require_approved=True" in source
    assert "stacked corruptions are not approved" in source
    assert "'front_stale'," in source
    assert "'native_motion_blur'," in source


def test_paired_template_is_applied_at_full_resolution_before_orion_pipeline():
    source = AGENT.read_text()
    application = source.index("paired_result = apply_paired_waterdrop_template(")
    pipeline = source.index("results = self.inference_only_pipeline(results)")
    assert application < pipeline
    assert "require_resolution=True" in source[application:pipeline]
    assert "'application_stage': 'pre_pipeline_1600x900_front_rgb'" in source
    assert "paired_waterdrop_front_bgr" in source
    assert "results['img'].append(" in source


def test_clean_raw_front_and_exact_model_tensor_remain_separate_artifacts():
    source = AGENT.read_text()
    assert "tick_data['imgs']['CAM_FRONT'] = paired_waterdrop_front_bgr" not in source
    assert "tick_data['model_input_front'] = (" in source
    assert "normalized_front_tensor_to_bgr(input_data_batch['img'][0])" in source
    assert "rgb_front_model_tensor" in source
    assert "corruption_visual_approval" in source
