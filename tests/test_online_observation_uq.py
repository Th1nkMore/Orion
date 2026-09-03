import hashlib
import json

import pytest
import torch

from uq_estimator.counterfactual_evidence import ObservationEvidenceHurdleAdapter
from uq_estimator.online_observation_uq import (
    PAIRWISE_CHECKPOINT_SCHEMA,
    RobustPreEventCalibrator,
    aggregate_observation_evidence,
    load_frozen_pairwise_adapter,
    summarize_spatial_observation_evidence,
)


def test_aggregate_uses_only_selected_front_view_for_control_scalar():
    score = torch.zeros(1, 3, 2, 2, 3)
    score[:, 0] = 2.0
    score[:, 1] = 7.0
    result = aggregate_observation_evidence(score, front_view_index=0)
    assert result.front_raw_score == pytest.approx(2.0)
    assert result.view_raw_scores == pytest.approx((2.0, 7.0, 0.0))
    assert result.front_component_scores == pytest.approx((2.0, 2.0, 2.0))


def test_spatial_summary_localizes_region_without_task_inputs():
    score = torch.ones(1, 2, 4, 4, 3)
    score[:, 0, 2:, 2:] = 5.0
    result = summarize_spatial_observation_evidence(
        score,
        (0.5, 0.5, 1.0, 1.0),
        front_view_index=0,
        grid_size=4,
    )
    assert result.feature_region == (2, 2, 4, 4)
    assert result.region_mean_score == pytest.approx(5.0)
    assert result.outside_mean_score == pytest.approx(1.0)
    assert result.region_minus_outside == pytest.approx(4.0)
    assert result.region_component_scores == pytest.approx((5.0, 5.0, 5.0))
    assert result.outside_component_scores == pytest.approx((1.0, 1.0, 1.0))
    assert result.pooled_front_grid[3][3] == pytest.approx(5.0)
    json.dumps(result.to_dict())


def test_spatial_summary_rejects_full_frame_region():
    with pytest.raises(ValueError, match="some but not all"):
        summarize_spatial_observation_evidence(
            torch.ones(1, 1, 4, 4, 3),
            (0.0, 0.0, 1.0, 1.0),
        )


def test_calibrator_freezes_prefix_and_detects_persistent_uplift():
    calibrator = RobustPreEventCalibrator(minimum_baseline_frames=4)
    for index, value in enumerate((1.00, 1.02, 0.98, 1.01)):
        output = calibrator.update(value, 1.0 + index * 0.5)
        assert output.filtered_score == 0.0
    frozen = calibrator.update(1.0, 4.0)
    assert frozen.baseline_frozen
    assert frozen.baseline_count == 4
    assert frozen.filtered_score < 0.1
    event = calibrator.update(2.0, 4.05)
    assert event.filtered_score > 0.75
    recovered = event
    for index in range(20):
        recovered = calibrator.update(1.0, 4.1 + index * 0.05)
    assert recovered.filtered_score < 0.5


def test_calibrator_rejects_too_short_prefix():
    calibrator = RobustPreEventCalibrator(minimum_baseline_frames=3)
    calibrator.update(1.0, 1.0)
    with pytest.raises(RuntimeError, match="insufficient pre-event"):
        calibrator.update(1.0, 4.0)


def test_checkpoint_loader_attests_hash_and_contract(tmp_path):
    model_config = {
        "feature_dim": 4,
        "hidden_dim": 3,
        "max_views": 2,
        "presence_bias": -3.0,
        "magnitude_bias": 0.0,
        "use_view_embedding": False,
    }
    model = ObservationEvidenceHurdleAdapter(**model_config)
    path = tmp_path / "pairwise.pt"
    torch.save(
        {
            "schema_version": PAIRWISE_CHECKPOINT_SCHEMA,
            "requires_inference_baseline_calibration": True,
            "model_config": model_config,
            "student_state": model.state_dict(),
            "best_epoch": 1,
        },
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, metadata = load_frozen_pairwise_adapter(
        path, expected_sha256=digest, device="cpu"
    )
    assert isinstance(loaded, ObservationEvidenceHurdleAdapter)
    assert not loaded.training
    assert metadata["sha256"] == digest
    with pytest.raises(RuntimeError, match="hash differs"):
        load_frozen_pairwise_adapter(path, expected_sha256="0" * 64)
