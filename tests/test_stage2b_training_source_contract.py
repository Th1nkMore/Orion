from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR = PROJECT_ROOT / "mmcv/models/detectors/orion.py"
AGENT_CONFIG = (
    PROJECT_ROOT / "adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
)


def test_detector_forbids_stateful_closedloop_runtime_in_training():
    source = DETECTOR.read_text(encoding="utf-8")
    assert "stateful learned_adapter runtime is forbidden" in source
    assert "precomputed_adapter requires an attested Stage-1 SHA256" in source
    assert "stage1_spatial_uq_checkpoint_sha256" in source


def test_closedloop_agent_config_does_not_admit_precomputed_training_source():
    source = AGENT_CONFIG.read_text(encoding="utf-8")
    allowed_line = next(
        line for line in source.splitlines()
        if "stage2_source not in" in line
    )
    assert "precomputed_adapter" not in allowed_line
