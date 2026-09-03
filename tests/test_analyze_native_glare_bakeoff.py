from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("cv2")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_native_glare_bakeoff import (
    _frame_metrics,
    _match_rows,
    _select_progress_rows,
    _yaw_error,
)


def _row(x, yaw=180.0):
    return {
        "ego_location": [x, 0.0, 0.0],
        "ego_rotation": [0.0, 0.0, yaw],
    }


def test_pose_matching_is_monotonic_and_auditable():
    references = [_row(0.0), _row(1.0), _row(2.0)]
    candidates = [_row(-0.1), _row(0.9), _row(2.1)]
    matches = _match_rows(references, candidates)
    assert len(matches) == 3
    assert [match["candidate"]["ego_location"][0] for match in matches] == [-0.1, 0.9, 2.1]
    assert max(match["distance_m"] for match in matches) < 0.11
    assert _yaw_error(179.0, -179.0) == 2.0


def test_visual_metrics_respond_to_white_hazard_region():
    clean = np.full((100, 200, 3), 64, dtype=np.uint8)
    changed = clean.copy()
    changed[50:85, 70:180] = 255
    metrics = _frame_metrics(clean, changed, (0.35, 0.5, 0.9, 0.85))
    assert metrics["mean_absolute_pixel_delta"] > 30.0
    assert metrics["saturated_pixel_fraction_changed"] > metrics["saturated_pixel_fraction_clean"]
    assert metrics["rectangular_boundary_artifact_score"] > 1.0


def test_fixed_progress_sampling_stays_precontact():
    rows = []
    for index, progress in enumerate((0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.54)):
        rows.append({
            "route_progress": progress,
            "capture_index": index,
            "nearby_actors": [{"type_id": "walker.pedestrian.0001", "distance_m": 20 - index}],
        })
    selected = _select_progress_rows(rows, (0.375, 0.40, 0.425, 0.45, 0.475))
    assert [row["route_progress"] for row in selected] == [0.375, 0.40, 0.425, 0.45, 0.475]
