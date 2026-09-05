import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_qwen_visibility_belief_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


visibility = _load_module()


def _encode_carla_depth_bgra(depth_m):
    depth = np.asarray(depth_m, dtype=np.float64)
    packed = np.rint(
        np.clip(depth / visibility.CARLA_DEPTH_FAR_PLANE_M, 0.0, 1.0)
        * (256**3 - 1)
    ).astype(np.uint32)
    red = (packed & 255).astype(np.uint8)
    green = ((packed >> 8) & 255).astype(np.uint8)
    blue = ((packed >> 16) & 255).astype(np.uint8)
    alpha = np.full_like(red, 255, dtype=np.uint8)
    return np.stack([blue, green, red, alpha], axis=-1)


def _front_camera(sensor_id="DEPTH_FRONT", depth_size=9):
    return visibility.camera_from_carla_sensor(
        sensor_id,
        {
            "width": depth_size,
            "height": depth_size,
            "fov": 90.0,
            "x": 0.0,
            "y": 0.0,
            "z": 1.5,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    )


def _line_grid():
    return visibility.VisibilityGridSpec(
        x_min_m=-2.0,
        x_max_m=20.0,
        y_min_m=-0.5,
        y_max_m=0.5,
        z_min_m=1.0,
        z_max_m=2.0,
        xy_resolution_m=1.0,
        z_resolution_m=1.0,
        max_range_m=20.0,
        surface_tolerance_m=0.51,
    )


def _row_nearest_x(spec, value):
    x, _, _ = spec.centers()
    return int(np.argmin(np.abs(x - value)))


def _column_nearest_y(spec, value):
    _, y, _ = spec.centers()
    return int(np.argmin(np.abs(y - value)))


def _mask_belief(spec, observed_cells=(), unknown_cells=(), frontier_cells=()):
    shape = spec.shape_bev
    free = np.zeros(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=np.float32)
    unknown = np.zeros(shape, dtype=np.float32)
    outside = np.ones(shape, dtype=np.float32)
    frontier = np.zeros(shape, dtype=bool)
    for cell in observed_cells:
        free[cell] = 1.0
        outside[cell] = 0.0
    for cell in unknown_cells:
        unknown[cell] = 1.0
        outside[cell] = 0.0
    for cell in frontier_cells:
        frontier[cell] = True
    return visibility.VisibilityBelief(
        spec=spec,
        visible_free_ratio=free,
        visible_occupied_ratio=occupied,
        occluded_unknown_ratio=unknown,
        outside_fov_ratio=outside,
        frontier=frontier,
    )


def test_standalone_geometry_import_does_not_load_torch_or_carla():
    code = """
import importlib.util, pathlib, sys
path = pathlib.Path(r'%s')
spec = importlib.util.spec_from_file_location('_visibility_isolation', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'carla' not in sys.modules
print(module.VISIBILITY_SCHEMA)
""" % MODULE_PATH
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == visibility.VISIBILITY_SCHEMA


def test_carla_depth_bgra_round_trip_is_metric():
    expected = np.asarray([[0.0, 1.0, 10.0, 123.456, 999.0]])
    decoded = visibility.decode_carla_depth_bgra(_encode_carla_depth_bgra(expected))
    np.testing.assert_allclose(decoded, expected, atol=0.001)


def test_metric_depth_audit_encoding_is_millimetric_and_bounded():
    encoded = visibility.encode_metric_depth_uint16_mm(
        np.asarray([[0.0, 1.2344, 1.2346, 80.0]], dtype=np.float32), 60.0
    )
    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(encoded, [[0, 1234, 1235, 60000]])


def test_carla_left_camera_is_converted_to_qwen_left_coordinates():
    camera = visibility.camera_from_carla_sensor(
        "DEPTH_FRONT_LEFT",
        {
            "width": 1600,
            "height": 900,
            "fov": 70.0,
            "x": 0.27,
            "y": -0.55,
            "z": 1.6,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": -55.0,
        },
    )
    np.testing.assert_allclose(camera.origin_ego, [0.27, 0.55, 1.6])
    optical_forward = camera.optical_to_ego[:, 2]
    np.testing.assert_allclose(
        optical_forward,
        [math.cos(math.radians(55.0)), math.sin(math.radians(55.0)), 0.0],
        atol=1e-7,
    )


def test_oracle_depth_sensor_specs_are_exactly_colocated_with_rgb():
    rgb = {
        "CAM_FRONT": {
            "type": "sensor.camera.rgb",
            "x": 0.8,
            "y": 0.0,
            "z": 1.6,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "width": 1600,
            "height": 900,
            "fov": 70.0,
        }
    }
    result = visibility.make_colocated_depth_sensor_specs(
        rgb, {"DEPTH_FRONT": "CAM_FRONT"}
    )
    assert result["DEPTH_FRONT"] == {
        **rgb["CAM_FRONT"],
        "type": "sensor.camera.depth",
    }
    assert rgb["CAM_FRONT"]["type"] == "sensor.camera.rgb"


def test_grid_mapping_rejects_unversioned_extra_parameters():
    payload = {
        "x_min_m": -10,
        "x_max_m": 50,
        "y_min_m": -25,
        "y_max_m": 25,
        "z_min_m": 0,
        "z_max_m": 3,
        "xy_resolution_m": 0.5,
        "z_resolution_m": 0.5,
        "max_range_m": 60,
        "surface_tolerance_m": 0.45,
    }
    assert visibility.visibility_grid_spec_from_mapping(payload).shape_3d == (
        6,
        120,
        100,
    )
    payload["unknown_field"] = 1
    try:
        visibility.visibility_grid_spec_from_mapping(payload)
    except ValueError as error:
        assert "keys mismatch" in str(error)
    else:
        raise AssertionError("extra grid parameters must fail closed")


def test_constant_depth_plane_yields_free_surface_unknown_and_outside_fov():
    spec = _line_grid()
    camera = _front_camera()
    depth = np.full((camera.height, camera.width), 10.0, dtype=np.float32)
    belief = visibility.compute_visibility_belief(
        {camera.sensor_id: depth}, [camera], spec
    )

    free_row = _row_nearest_x(spec, 5.5)
    surface_row = _row_nearest_x(spec, 9.5)
    unknown_row = _row_nearest_x(spec, 15.5)
    behind_row = _row_nearest_x(spec, -1.5)
    assert belief.visible_free_ratio[free_row, 0] == 1.0
    assert belief.visible_occupied_ratio[surface_row, 0] == 1.0
    assert belief.occluded_unknown_ratio[unknown_row, 0] == 1.0
    assert belief.outside_fov_ratio[behind_row, 0] == 1.0
    np.testing.assert_allclose(belief.as_channels()[:4].sum(axis=0), 1.0)
    assert belief.frontier.any()


def test_second_camera_can_rescue_space_occluded_in_first_view():
    spec = _line_grid()
    first = _front_camera("DEPTH_A")
    second = _front_camera("DEPTH_B")
    belief = visibility.compute_visibility_belief(
        {
            "DEPTH_A": np.full((9, 9), 5.0, dtype=np.float32),
            "DEPTH_B": np.full((9, 9), 20.0, dtype=np.float32),
        },
        [first, second],
        spec,
    )
    row = _row_nearest_x(spec, 10.5)
    assert belief.visible_free_ratio[row, 0] == 1.0
    assert belief.occluded_unknown_ratio[row, 0] == 0.0


def test_visibility_render_and_metadata_are_auditable():
    spec = _line_grid()
    camera = _front_camera()
    belief = visibility.compute_visibility_belief(
        {camera.sensor_id: np.full((9, 9), 10.0, dtype=np.float32)},
        [camera],
        spec,
    )
    rendered = visibility.render_visibility_belief(belief)
    metadata = visibility.belief_metadata(belief)
    assert rendered.shape == spec.shape_bev + (3,)
    assert rendered.dtype == np.uint8
    assert metadata["schema"] == visibility.VISIBILITY_SCHEMA
    assert metadata["coordinate_frame"] == "qwen_ego_x_forward_y_left_z_up"
    assert metadata["shape_3d"] == list(spec.shape_3d)
    assert metadata["summary"]["frontier_cells"] > 0


def test_observation_memory_distinguishes_current_previous_and_never_seen():
    spec = visibility.VisibilityGridSpec(
        x_min_m=-2.0,
        x_max_m=2.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        z_min_m=0.0,
        z_max_m=1.0,
        xy_resolution_m=1.0,
        z_resolution_m=1.0,
        max_range_m=5.0,
        surface_tolerance_m=0.5,
    )
    first_cell = (_row_nearest_x(spec, 0.5), _column_nearest_y(spec, 0.5))
    memory = visibility.VisibilityObservationMemory(
        spec, max_age_seconds=5.0, observed_ratio_threshold=0.5
    )
    first = memory.update(
        _mask_belief(spec, observed_cells=[first_cell]),
        [0.0, 0.0, 0.0],
        0.0,
    )
    assert first.currently_observed[first_cell]
    assert first.age_seconds[first_cell] == 0.0
    assert first.never_observed.sum() == 15

    # The ego moves one metre forward. The same stationary world cell shifts
    # from x=+0.5 to x=-0.5 in the new ego raster.
    shifted_cell = (_row_nearest_x(spec, -0.5), _column_nearest_y(spec, 0.5))
    second = memory.update(
        _mask_belief(spec),
        [1.0, 0.0, 0.0],
        1.0,
    )
    assert second.previously_observed[shifted_cell]
    assert second.age_seconds[shifted_cell] == 1.0
    assert not second.ever_observed[first_cell]
    np.testing.assert_allclose(second.as_channels()[1:].sum(axis=0), 1.0)


def test_observation_memory_rejects_time_reversal():
    spec = _line_grid()
    memory = visibility.VisibilityObservationMemory(spec, max_age_seconds=5.0)
    belief = _mask_belief(spec)
    memory.update(belief, [0.0, 0.0, 0.0], 2.0)
    try:
        memory.update(belief, [0.0, 0.0, 0.0], 1.0)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("observation memory must reject time reversal")


def test_observation_memory_compensates_carla_yaw_rotation():
    spec = visibility.VisibilityGridSpec(
        x_min_m=-2.0,
        x_max_m=2.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        z_min_m=0.0,
        z_max_m=1.0,
        xy_resolution_m=1.0,
        z_resolution_m=1.0,
        max_range_m=5.0,
        surface_tolerance_m=0.5,
    )
    initial = (_row_nearest_x(spec, 1.5), _column_nearest_y(spec, 0.5))
    rotated = (_row_nearest_x(spec, -0.5), _column_nearest_y(spec, 1.5))
    memory = visibility.VisibilityObservationMemory(spec, max_age_seconds=5.0)
    memory.update(
        _mask_belief(spec, observed_cells=[initial]), [0.0, 0.0, 0.0], 0.0
    )
    state = memory.update(_mask_belief(spec), [0.0, 0.0, 90.0], 0.5)
    assert state.previously_observed[rotated]
    assert state.age_seconds[rotated] == 0.5


def test_carla_world_route_conversion_preserves_qwen_left_axis():
    route = np.asarray([[10.0, 0.0], [10.0, -5.0], [10.0, -10.0]])
    local = visibility.carla_world_route_to_qwen_ego(
        route,
        ego_world_pose_xy_yaw_degrees=[0.0, 0.0, 0.0],
        max_length_m=30.0,
    )
    np.testing.assert_allclose(local[0], [0.0, 0.0])
    np.testing.assert_allclose(local[1], [10.0, 0.0])
    np.testing.assert_allclose(local[2], [10.0, 5.0])


def test_carla_world_route_conversion_projects_and_interpolates_horizon():
    local = visibility.carla_world_route_to_qwen_ego(
        np.asarray([[0.0, 0.0], [100.0, 0.0]]),
        ego_world_pose_xy_yaw_degrees=[10.0, -1.0, 0.0],
        max_length_m=12.0,
    )
    np.testing.assert_allclose(local[0], [0.0, 0.0])
    np.testing.assert_allclose(local[1], [0.0, -1.0])
    assert np.linalg.norm(np.diff(local, axis=0), axis=1).sum() == pytest.approx(
        12.0
    )


def test_visibility_exposure_uses_route_and_stopping_margin_without_mutating_u():
    spec = visibility.VisibilityGridSpec(
        x_min_m=0.0,
        x_max_m=20.0,
        y_min_m=-4.0,
        y_max_m=4.0,
        z_min_m=0.0,
        z_max_m=1.0,
        xy_resolution_m=1.0,
        z_resolution_m=1.0,
        max_range_m=20.0,
        surface_tolerance_m=0.5,
    )
    near = (_row_nearest_x(spec, 5.5), _column_nearest_y(spec, 0.5))
    far = (_row_nearest_x(spec, 15.5), _column_nearest_y(spec, 0.5))
    off_route = (_row_nearest_x(spec, 5.5), _column_nearest_y(spec, 3.5))
    belief = _mask_belief(
        spec,
        unknown_cells=[near, far, off_route],
        frontier_cells=[near, far, off_route],
    )
    original_u = belief.u_vis
    exposure = visibility.compute_visibility_exposure(
        belief,
        route_ego_xy=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
        speed_mps=4.0,
        reaction_time_seconds=1.0,
        safe_deceleration_mps2=4.0,
        route_sigma_m=2.0,
        stopping_transition_m=4.0,
    )
    assert exposure.stopping_distance_m == 6.0
    assert exposure.stopping_margin_m[near] < 0.0
    assert exposure.urgency[near] > exposure.urgency[far]
    assert exposure.urgency[near] > exposure.urgency[off_route]
    np.testing.assert_array_equal(belief.u_vis, original_u)
    assert visibility.visibility_exposure_metadata(exposure)["schema"] == (
        visibility.VISIBILITY_EXPOSURE_SCHEMA
    )


def test_temporal_and_exposure_renders_are_auditable_uint8():
    spec = _line_grid()
    belief = _mask_belief(spec)
    state = visibility.VisibilityObservationMemory(
        spec, max_age_seconds=5.0
    ).update(belief, [0.0, 0.0, 0.0], 0.0)
    exposure = visibility.compute_visibility_exposure(
        belief,
        route_ego_xy=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
        speed_mps=1.0,
        reaction_time_seconds=1.0,
        safe_deceleration_mps2=4.0,
        route_sigma_m=2.0,
        stopping_transition_m=4.0,
    )
    memory_rgb = visibility.render_observation_memory(state)
    exposure_rgb = visibility.render_visibility_exposure(exposure)
    urgency_rgb = visibility.render_visibility_urgency(exposure)
    assert memory_rgb.shape == spec.shape_bev + (3,)
    assert exposure_rgb.shape == spec.shape_bev + (3,)
    assert urgency_rgb.shape == spec.shape_bev + (3,)
    assert memory_rgb.dtype == np.uint8
    assert exposure_rgb.dtype == np.uint8
    assert urgency_rgb.dtype == np.uint8
    assert not urgency_rgb[..., 1:].any()
    assert visibility.observation_memory_metadata(state)["schema"] == (
        visibility.OBSERVATION_MEMORY_SCHEMA
    )
