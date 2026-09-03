"""Tests for clean-calibrated observation-UQ candidate signals."""

from dataclasses import replace

import pytest
import torch

from uq_estimator.observation_uq_signal_audit import (
    apply_calibrator,
    evaluate_detailed_score_maps,
    evaluate_score_maps,
    feature_rms,
    fit_clean_position_calibrator,
    spatial_neighbor_residual,
    temporal_cosine_residual,
)
from uq_estimator.observation_uq_v3 import ObservationUQError, make_mock_examples


def test_raw_signals_ignore_corruption_audit_metadata():
    source = next(
        item
        for item in make_mock_examples(
            feature_dim=8, routes=6, frames_per_route=2, height=4, width=4, seed=31
        )
        if item.family != "clean" and item.previous_valid
    )
    relabelled = replace(
        source,
        sample_id=source.sample_id + "/relabelled",
        family="invented",
        severity=999.0,
        corruption_mask=1.0 - source.corruption_mask,
    )
    current = torch.stack((source.current, relabelled.current))
    previous = torch.stack((source.previous, relabelled.previous))
    valid = torch.tensor((True, True))
    for score in (
        temporal_cosine_residual(current, previous, valid),
        spatial_neighbor_residual(current),
        feature_rms(current),
    ):
        torch.testing.assert_close(score[0], score[1])


def test_clean_calibrator_fails_closed_and_standardizes_position_maps():
    examples = make_mock_examples(
        feature_dim=8, routes=6, frames_per_route=2, height=4, width=4, seed=32
    )
    clean = [item for item in examples if item.family == "clean"]
    maps = {item.sample_id: feature_rms(item.current.unsqueeze(0))[0] for item in clean}
    calibrator = fit_clean_position_calibrator(maps, clean, tail="absolute")
    transformed = apply_calibrator(maps, calibrator)
    assert calibrator.example_count == len(clean)
    assert all(bool(torch.isfinite(value).all()) for value in transformed.values())
    corrupt = next(item for item in examples if item.family != "clean")
    with pytest.raises(ObservationUQError, match="corruption"):
        fit_clean_position_calibrator(maps, clean + [corrupt])


def test_score_evaluation_uses_masks_only_after_map_creation():
    examples = make_mock_examples(
        feature_dim=8, routes=6, frames_per_route=2, height=4, width=4, seed=33
    )
    selected = [item for item in examples if item.split == "validation"]
    score_maps = {
        item.sample_id: (
            item.corruption_mask.float()
            if item.family != "clean"
            else torch.zeros(item.current.shape[:-1])
        )
        for item in selected
    }
    result = evaluate_score_maps(selected, score_maps)
    assert result["corruption_mask_patch_auroc_diagnostic_only"] == pytest.approx(1.0)
    assert result["by_family"]["local_glare"]["score_uplift_over_clean"] > 0
    detailed = evaluate_detailed_score_maps(selected, score_maps)
    assert detailed["previous_valid_only"][
        "corruption_mask_patch_auroc_diagnostic_only"
    ] == pytest.approx(1.0)
    assert all(
        row["corruption_mask_patch_auroc_diagnostic_only"] == pytest.approx(1.0)
        for row in detailed["by_route"].values()
    )
    assert all(
        severity["inside_minus_outside"] > 0
        for family in detailed["by_severity"].values()
        for severity in family.values()
    )
