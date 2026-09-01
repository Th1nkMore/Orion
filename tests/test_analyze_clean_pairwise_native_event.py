import numpy as np

from scripts.analyze_clean_pairwise_native_event import analyze
from scripts.render_closedloop_observation_uq_heatmap import CAMERA_ORDER


def _row(step):
    time = step * 0.05
    scores = [1.0] * len(CAMERA_ORDER)
    if 4.5 <= time < 6.0:
        scores[1] = 5.0
    grids = []
    for view, score in enumerate(scores):
        grid = np.full((10, 10), score, dtype=np.float64)
        if view == 1 and 4.5 <= time < 6.0:
            grid[4:6, 4:6] = 9.0
        grids.append(grid.tolist())
    return {
        "step": step,
        "sim_time_seconds": time,
        "route_progress": time / 10.0,
        "observation_uq": {
            "camera_order": list(CAMERA_ORDER),
            "aggregate": {"view_raw_scores": scores},
            "pooled_grids": grids,
        },
    }


def test_native_event_analysis_keeps_independent_event_anchor_and_ranks_view():
    report = analyze(
        [_row(step) for step in range(140)],
        event_time_seconds=5.0,
        lead_seconds=1.0,
        approach_seconds=0.5,
        post_seconds=1.0,
    )
    assert report["event_time_seconds"] == 5.0
    assert report["largest_approach_uplift_camera"] == "CAM_FRONT_LEFT"
    camera = report["camera_reports"]["CAM_FRONT_LEFT"]
    assert camera["approach_raw_mean_uplift_from_baseline"] == 4.0
    assert camera["first_pre_event_calibrated_trigger"]["sim_time_seconds"] == 4.5
