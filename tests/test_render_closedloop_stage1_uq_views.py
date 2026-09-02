import numpy as np
import json

from scripts.render_closedloop_stage1_uq_views import (
    CAMERA_ORDER,
    _ground_point,
    _load_trace_rows,
    _raw_pixel_from_grid,
    project_camera_u_to_ground,
)


def test_model_grid_maps_to_frozen_raw_crop() -> None:
    assert _raw_pixel_from_grid(0, 0, 40, 40) == (0.0, 100.0)
    assert _raw_pixel_from_grid(40, 40, 40, 40) == (1600.0, 900.0)


def test_front_lower_center_ray_hits_ground_ahead() -> None:
    point = _ground_point("CAM_FRONT", 800.0, 700.0)
    assert point is not None
    forward, right = point
    assert forward > 0.8
    assert abs(right) < 1e-6


def test_six_view_projection_is_finite_and_spatial() -> None:
    uncertainty = np.zeros((len(CAMERA_ORDER), 40, 40), dtype=np.float32)
    uncertainty[0, 20:, 15:25] = 1.0
    projected, stats = project_camera_u_to_ground(uncertainty)
    assert projected.shape == (512, 512)
    assert np.isfinite(projected).all()
    assert float(projected.max()) == 1.0
    assert np.count_nonzero(projected) > 0
    assert stats["projected_cell_count"] > 0
    assert stats["above_horizon_cell_count"] > 0


def test_trace_alignment_selects_saved_control_step(tmp_path) -> None:
    rows = [
        {"step": step, "sim_time_seconds": step * 0.05}
        for step in range(20, 30)
    ]
    (tmp_path / "control_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    aligned = _load_trace_rows(tmp_path)
    assert aligned[2]["step"] == 20
