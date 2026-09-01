from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluate_clean_render_artifacts import evaluate_gate, evaluate_image


def test_dense_flat_blocks_score_higher_than_smooth_gradient(tmp_path: Path):
    height, width = 180, 320
    x = np.linspace(20, 220, width, dtype=np.float32)
    smooth = np.repeat(x[None, :], height, axis=0)
    smooth_rgb = np.repeat(smooth[:, :, None], 3, axis=2).astype(np.uint8)
    blocked = smooth_rgb.copy()
    for row in range(12, 160, 24):
        for column in range(10, 300, 32):
            blocked[row : row + 10, column : column + 13] = (5, 5, 5)
    smooth_path = tmp_path / "smooth.png"
    blocked_path = tmp_path / "blocked.png"
    Image.fromarray(smooth_rgb).save(smooth_path)
    Image.fromarray(blocked).save(blocked_path)
    smooth_result = evaluate_image(smooth_path)
    blocked_result = evaluate_image(blocked_path)
    assert blocked_result["rectangular_component_count"] > (
        smooth_result["rectangular_component_count"] + 20
    )
    assert blocked_result["rectangular_component_area_fraction"] > (
        smooth_result["rectangular_component_area_fraction"] + 0.02
    )


def test_frozen_sequence_gate_requires_multiple_bad_frames():
    gate = {
        "per_frame_thresholds": {
            "candidate_pixel_fraction_min": 0.05,
            "rectangular_component_count_min": 320,
            "rectangular_component_area_fraction_min": 0.011,
        },
        "sequence_rule": {
            "minimum_frames": 5,
            "suspicious_frame_fraction_reject_at_or_above": 0.2,
        },
    }
    good = {
        "candidate_pixel_fraction": 0.04,
        "rectangular_component_count": 250,
        "rectangular_component_area_fraction": 0.008,
    }
    bad = {
        "candidate_pixel_fraction": 0.06,
        "rectangular_component_count": 400,
        "rectangular_component_area_fraction": 0.015,
    }
    assert evaluate_gate([good] * 5, gate)["passed"] is True
    result = evaluate_gate([bad, good, good, good, good], gate)
    assert result["passed"] is False
    assert result["suspicious_frame_fraction"] == 0.2
