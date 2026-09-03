import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import cv2
import numpy as np
import pytest
import torch

from uq_estimator.bev_target_rasterizer import (
    BEVTargetRasterizerError,
    BICYCLE_PARITY_CAVEAT,
    GT_RASTERIZER_ID,
    PAIRWISE_BEV_IOU_POLICY_ID,
    SELECTED_MODE_RASTERIZER_ID,
    SELECTED_MODE_YAW_POLICY,
    bev_box_corners_lidar_v1,
    pairwise_bev_iou_v1,
    rasterize_orion_selected_mode_v1,
    rasterize_planningmetric_gt_v1,
    selected_mode_occupancy_callback_v1,
    selected_mode_rasterizer_provenance_v1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_repository_planning_metric(monkeypatch):
    """Load the exact repository file while stubbing unused optional imports."""

    skimage = ModuleType("skimage")
    skimage_draw = ModuleType("skimage.draw")
    skimage_draw.polygon = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("collision-only skimage polygon must not be called")
    )
    nuscenes = ModuleType("nuscenes")
    nuscenes_utils = ModuleType("nuscenes.utils")
    nuscenes_data = ModuleType("nuscenes.utils.data_classes")
    nuscenes_data.Box = object
    monkeypatch.setitem(sys.modules, "skimage", skimage)
    monkeypatch.setitem(sys.modules, "skimage.draw", skimage_draw)
    monkeypatch.setitem(sys.modules, "nuscenes", nuscenes)
    monkeypatch.setitem(sys.modules, "nuscenes.utils", nuscenes_utils)
    monkeypatch.setitem(sys.modules, "nuscenes.utils.data_classes", nuscenes_data)
    path = (
        REPOSITORY_ROOT
        / "mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py"
    )
    spec = importlib.util.spec_from_file_location("_metric_stp3_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PlanningMetric


def _load_repository_box_np_ops(monkeypatch):
    # box_np_ops only needs numba as an optimization decorator.  Keep this
    # unit test dependency-light by replacing those decorators with no-ops.
    numba = ModuleType("numba")

    def no_op_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function

    numba.jit = no_op_decorator
    numba.njit = no_op_decorator
    monkeypatch.setitem(sys.modules, "numba", numba)
    path = REPOSITORY_ROOT / "mmcv/core/bbox/box_np_ops.py"
    spec = importlib.util.spec_from_file_location("_box_np_ops_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gt_fixture():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, -math.pi / 2, 0.0, 0.0],
            [10.0, 2.0, 0.0, 0.8, 1.8, 1.6, 0.2, 0.0, 0.0],
            [-12.0, -3.0, 0.0, 0.7, 1.7, 1.5, -0.4, 0.0, 0.0],
            [3.0, -10.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    features = torch.zeros(1, 4, 34, dtype=torch.float32)
    # Per-step XY offsets; the implementation and reference must cumsum them.
    features[0, 0, :12] = torch.tensor(
        [[1.0, 0.0]] * 6, dtype=torch.float32
    ).reshape(-1)
    features[0, 1, :12] = torch.tensor(
        [[0.0, -0.5]] * 6, dtype=torch.float32
    ).reshape(-1)
    features[0, 2, :12] = torch.tensor(
        [[-0.25, 0.25]] * 6, dtype=torch.float32
    ).reshape(-1)
    features[0, :, 12:18] = 1
    features[0, 1, 15] = 0  # one invalid future frame
    features[0, :, 27] = torch.tensor([0.0, 3.0, 7.0, 4.0])
    features[0, 0, 28:34] = torch.tensor([0.1] * 6)
    return boxes, features


def test_gt_matches_repository_planningmetric_pixelwise_without_mutation(monkeypatch):
    boxes, features = _gt_fixture()
    boxes_before = boxes.clone()
    features_before = features.clone()
    actual = rasterize_planningmetric_gt_v1(boxes, features, include_union=True)

    PlanningMetric = _load_repository_planning_metric(monkeypatch)
    reference = PlanningMetric()
    # The repository function mutates a CPU box tensor, hence the private clone.
    box_wrapper = SimpleNamespace(tensor=boxes.clone())
    vehicle_np, human_np = reference.get_birds_eye_view_label(
        box_wrapper, features.clone()
    )
    vehicle = torch.from_numpy(vehicle_np).float()
    human = torch.from_numpy(human_np).float()

    torch.testing.assert_close(actual.union_vehicle.cpu(), vehicle, rtol=0, atol=0)
    torch.testing.assert_close(actual.union_human.cpu(), human, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.union_all.cpu(), torch.logical_or(vehicle, human).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(boxes, boxes_before, rtol=0, atol=0)
    torch.testing.assert_close(features, features_before, rtol=0, atol=0)
    assert actual.per_object.shape == (4, 6, 200, 200)
    assert actual.rasterizer_id == GT_RASTERIZER_ID


def test_gt_pixel_parity_at_grid_edges_and_rotated_overlaps(monkeypatch):
    boxes = torch.tensor(
        [
            [49.5, 0.0, 0.0, 2.0, 5.0, 1.0, 0.73],
            [-49.5, 49.5, 0.0, 1.2, 3.0, 1.0, -1.11],
            [0.2, -0.4, 0.0, 2.5, 4.7, 1.0, 2.2],
        ],
        dtype=torch.float32,
    )
    features = torch.zeros(1, 3, 34, dtype=torch.float32)
    features[0, :, 12:18] = 1
    features[0, :, 27] = torch.tensor([2.0, 3.0, 1.0])
    features[0, 1, :12] = torch.tensor([[0.4, -0.3]] * 6).reshape(-1)
    features[0, 2, 28:34] = torch.linspace(-0.2, 0.3, 6)
    actual = rasterize_planningmetric_gt_v1(boxes, features, include_union=True)

    PlanningMetric = _load_repository_planning_metric(monkeypatch)
    vehicle_np, human_np = PlanningMetric().get_birds_eye_view_label(
        SimpleNamespace(tensor=boxes.clone()), features.clone()
    )
    torch.testing.assert_close(
        actual.union_vehicle.cpu(), torch.from_numpy(vehicle_np).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.union_human.cpu(), torch.from_numpy(human_np).float(), rtol=0, atol=0
    )


def test_bicycle_human_layer_caveat_and_ignored_class_are_explicit():
    boxes, features = _gt_fixture()
    result = rasterize_planningmetric_gt_v1(boxes, features, include_union=True)
    assert result.category_groups == ("vehicle", "human", "human", "ignored")
    assert result.per_object[1].sum() > 0  # class 3 bicycle
    assert result.union_human is not None
    assert torch.all(result.per_object[3] == 0)
    assert not result.valid_mask[3].any()
    assert result.bicycle_parity_caveat == BICYCLE_PARITY_CAVEAT
    assert "bicycle" in result.bicycle_parity_caveat


def test_gt_empty_eligible_object_frame_returns_typed_empty_raster():
    boxes = torch.empty((0, 9), dtype=torch.float32)
    features = torch.empty((1, 0, 34), dtype=torch.float32)

    result = rasterize_planningmetric_gt_v1(boxes, features, include_union=True)

    assert result.per_object.shape == (0, 6, 200, 200)
    assert result.valid_mask.shape == (0, 6)
    assert result.valid_mask.dtype == torch.bool
    assert result.category_groups == ()
    assert result.union_all is not None
    assert result.union_vehicle is not None
    assert result.union_human is not None
    assert not result.union_all.any()
    assert not result.union_vehicle.any()
    assert not result.union_human.any()


def test_selected_mode_cumsum_and_frozen_yaw_match_zero_yaw_delta_gt():
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, -math.pi / 2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    deltas = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]]]
    )
    boxes_before = boxes.clone()
    deltas_before = deltas.clone()
    predicted = rasterize_orion_selected_mode_v1(
        boxes, deltas, include_union=True
    )

    features = torch.zeros(1, 1, 34)
    features[0, 0, :12] = deltas.reshape(-1)
    features[0, 0, 12:18] = 1
    features[0, 0, 27] = 0
    gt = rasterize_planningmetric_gt_v1(boxes, features)
    torch.testing.assert_close(predicted.per_object, gt.per_object, rtol=0, atol=0)
    torch.testing.assert_close(predicted.union_all, gt.per_object[0], rtol=0, atol=0)
    torch.testing.assert_close(boxes, boxes_before, rtol=0, atol=0)
    torch.testing.assert_close(deltas, deltas_before, rtol=0, atol=0)
    assert predicted.yaw_policy == SELECTED_MODE_YAW_POLICY
    assert predicted.occupancy_origin.startswith("derived_")


def test_adapter_callback_is_structural_and_rejects_non_delta_trajectories():
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, -math.pi / 2]],
        dtype=torch.float32,
    )
    deltas = torch.zeros(1, 3, 2)
    value = SimpleNamespace(
        decoded_boxes_lidar=boxes,
        selected_deltas=deltas,
        trajectories_are_step_deltas=True,
    )
    occupancy = selected_mode_occupancy_callback_v1(value)
    assert occupancy.shape == (1, 3, 200, 200)
    value.trajectories_are_step_deltas = False
    with pytest.raises(BEVTargetRasterizerError, match="step_deltas=True"):
        selected_mode_occupancy_callback_v1(value)


def test_callback_accepts_the_real_decode_adapter_value_object():
    from uq_estimator.orion_decode_adapter import SelectedMotionRasterInputV1

    value = SelectedMotionRasterInputV1(
        decoded_boxes_lidar=torch.tensor(
            [[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, -math.pi / 2]],
            dtype=torch.float32,
        ),
        selected_deltas=torch.zeros(1, 2, 2),
        source_query_index=torch.tensor([4]),
        selected_mode_index=torch.tensor([1]),
        batch_index=0,
    )
    occupancy = selected_mode_occupancy_callback_v1(value)
    assert occupancy.shape == (1, 2, 200, 200)
    assert occupancy.device == value.decoded_boxes_lidar.device


def test_selected_mode_provenance_never_claims_native_occupancy():
    provenance = selected_mode_rasterizer_provenance_v1()
    assert provenance.rasterizer_id == SELECTED_MODE_RASTERIZER_ID
    assert provenance.native_orion_occupancy is False
    assert provenance.yaw_policy == SELECTED_MODE_YAW_POLICY
    assert provenance.height == provenance.width == 200
    assert provenance.resolution_m == pytest.approx(0.5)


def test_bev_corners_match_repository_box_fixture_and_inputs_are_unchanged(monkeypatch):
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.0, 0.0],
            [1.0, -2.0, 0.0, 1.5, 3.0, 1.0, 0.37],
            [-5.0, 3.0, 0.0, 4.0, 1.0, 1.0, -1.2],
        ],
        dtype=torch.float32,
    )
    before = boxes.clone()
    actual = bev_box_corners_lidar_v1(boxes).cpu().numpy()
    reference_module = _load_repository_box_np_ops(monkeypatch)
    xywhr = boxes[:, [0, 1, 3, 4, 6]].cpu().numpy()
    expected = reference_module.box2d_to_corner_jit(xywhr)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(boxes, before, rtol=0, atol=0)


def test_pairwise_continuous_bev_iou_matches_analytic_rotated_box_fixtures():
    boxes_a = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    boxes_b = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.0, math.pi / 2],
            [20.0, 20.0, 0.0, 1.0, 1.0, 1.0, 0.3],
        ],
        dtype=torch.float32,
    )
    before_a = boxes_a.clone()
    before_b = boxes_b.clone()
    iou = pairwise_bev_iou_v1(boxes_a, boxes_b)
    assert iou.shape == (2, 3)
    assert iou[0, 0].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert iou[1, 1].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert torch.all(iou[:, 2] == 0)
    torch.testing.assert_close(boxes_a, before_a, rtol=0, atol=0)
    torch.testing.assert_close(boxes_b, before_b, rtol=0, atol=0)
    assert PAIRWISE_BEV_IOU_POLICY_ID.endswith("/v1")


def test_pairwise_iou_agrees_with_repository_corner_polygons_on_fixture(monkeypatch):
    reference_module = _load_repository_box_np_ops(monkeypatch)
    boxes_a = torch.tensor(
        [[0.3, -0.7, 0.0, 2.2, 4.3, 1.0, 0.41]], dtype=torch.float32
    )
    boxes_b = torch.tensor(
        [[1.1, 0.2, 0.0, 1.7, 3.1, 1.0, -0.28]], dtype=torch.float32
    )
    polygons_a = reference_module.box2d_to_corner_jit(
        boxes_a[:, [0, 1, 3, 4, 6]].numpy()
    ).astype(np.float32)
    polygons_b = reference_module.box2d_to_corner_jit(
        boxes_b[:, [0, 1, 3, 4, 6]].numpy()
    ).astype(np.float32)
    intersection, _ = cv2.intersectConvexConvex(polygons_a[0], polygons_b[0])
    area_a = abs(cv2.contourArea(polygons_a[0]))
    area_b = abs(cv2.contourArea(polygons_b[0]))
    expected = intersection / (area_a + area_b - intersection)
    actual = pairwise_bev_iou_v1(boxes_a, boxes_b).item()
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "boxes,deltas,message",
    [
        (torch.zeros(1, 6), torch.zeros(1, 6, 2), "D>=7"),
        (
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0]]),
            torch.zeros(1, 6, 2),
            "positive",
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
            torch.zeros(1, 6, 3),
            r"\[N,positive_T,2\]",
        ),
    ],
)
def test_selected_rasterizer_fails_closed_on_ambiguous_geometry(
    boxes, deltas, message
):
    with pytest.raises(BEVTargetRasterizerError, match=message):
        rasterize_orion_selected_mode_v1(boxes, deltas)
