import hashlib

import pytest
import torch

from uq_estimator.counterfactual_evidence import ObservationEvidenceHurdleAdapter
from uq_estimator.spatial_uq_runtime import (
    CausalSpatialEvidenceCalibrator,
    FrozenSpatialObservationUQRuntime,
)


def _checkpoint(tmp_path, feature_dim=8):
    model_config = {
        "feature_dim": feature_dim,
        "hidden_dim": 6,
        "max_views": 6,
        "presence_bias": -3.0,
        "magnitude_bias": 0.0,
        "use_view_embedding": False,
    }
    model = ObservationEvidenceHurdleAdapter(**model_config)
    path = tmp_path / "pairwise.pt"
    torch.save(
        {
            "schema_version": "orion.counterfactual-evidence-pairwise-native-checkpoint/v1",
            "student_state": model.state_dict(),
            "model_config": model_config,
            "best_epoch": 1,
            "requires_inference_baseline_calibration": True,
        },
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_spatial_calibrator_is_zero_until_frozen_and_local_afterward():
    calibrator = CausalSpatialEvidenceCalibrator(
        warmup_frames=3,
        z_center=1.0,
        absolute_scale_floor=0.01,
        attack_alpha=1.0,
        release_alpha=1.0,
    )
    baseline = torch.full((1, 2, 3, 4, 3), 0.2)
    for _ in range(3):
        assert torch.count_nonzero(calibrator.update(baseline)) == 0
    assert calibrator.ready
    event = baseline.clone()
    event[0, 1, 2, 3, 0] = 0.5
    calibrated = calibrator.update(event)
    assert calibrated.shape == event.shape
    assert calibrated[0, 1, 2, 3, 0] > calibrated[0, 0, 0, 0, 0]


def test_spatial_calibrator_reset_forbids_cross_route_history():
    calibrator = CausalSpatialEvidenceCalibrator(warmup_frames=2)
    score = torch.rand(1, 2, 3, 4, 3)
    calibrator.update(score)
    assert calibrator.count == 1
    calibrator.reset()
    assert calibrator.count == 0
    assert not calibrator.ready


def test_frozen_runtime_attests_checkpoint_and_stops_gradients(tmp_path):
    path, digest = _checkpoint(tmp_path)
    runtime = FrozenSpatialObservationUQRuntime(
        str(path), expected_sha256=digest, warmup_frames=2
    )
    features = torch.randn(1, 6, 8, 4, 5, requires_grad=True)
    first = runtime(features)
    second = runtime(features + 0.1)
    assert first.raw_score.shape == (1, 6, 4, 5, 3)
    assert torch.count_nonzero(first.calibrated_score) == 0
    assert second.baseline_ready
    assert second.checkpoint_sha256 == digest
    assert not second.raw_score.requires_grad
    assert not second.calibrated_score.requires_grad
    assert all(not parameter.requires_grad for parameter in runtime.parameters())


def test_frozen_runtime_hash_mismatch_fails_closed(tmp_path):
    path, _ = _checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="hash differs"):
        FrozenSpatialObservationUQRuntime(
            str(path), expected_sha256="0" * 64, warmup_frames=2
        )
