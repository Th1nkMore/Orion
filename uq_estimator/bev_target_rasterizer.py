"""Side-effect-free BEV geometry for frozen-ORION actual targets.

The GT path reproduces ``PlanningMetric.get_birds_eye_view_label`` without
importing MMCV, CARLA, or the ORION model.  The predicted path is deliberately
named *derived occupancy*: ORION does not emit a native occupancy tensor, so
we rasterize the decoded current box along the model-selected trajectory.

All public functions detach and clone tensor inputs before converting them to
NumPy.  This is important on CPU, where ``Tensor.numpy()`` may otherwise share
storage with its source.  The repository PlanningMetric edits the converted
box yaw in place and can consequently mutate a caller-owned CPU tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np
import torch


GT_RASTERIZER_ID = "planningmetric-gt-side-effect-free-parity/v1"
SELECTED_MODE_RASTERIZER_ID = "orion-selected-mode-derived-occupancy/v1"
PAIRWISE_BEV_IOU_POLICY_ID = "orion-side-effect-free-continuous-rotated-bev-iou/v1"
SELECTED_MODE_YAW_POLICY = "planningmetric-convert-current-yaw-then-freeze/v1"
TRAJECTORY_POLICY = "orion-selected-step-deltas-cumulative-sum/v1"
BICYCLE_PARITY_CAVEAT = (
    "PlanningMetric category_index 3 is bicycle but is rasterized in its "
    "human/pedestrian layer; v1 preserves that behavior for parity only."
)


class BEVTargetRasterizerError(ValueError):
    """Raised when BEV target geometry is ambiguous or invalid."""


@dataclass(frozen=True)
class PlanningMetricGridV1:
    """Frozen grid convention used by the repository PlanningMetric."""

    x_bound: Tuple[float, float, float] = (-50.0, 50.0, 0.5)
    y_bound: Tuple[float, float, float] = (-50.0, 50.0, 0.5)
    timesteps: int = 6

    @property
    def height(self) -> int:
        return int(round((self.x_bound[1] - self.x_bound[0]) / self.x_bound[2]))

    @property
    def width(self) -> int:
        return int(round((self.y_bound[1] - self.y_bound[0]) / self.y_bound[2]))

    @property
    def resolution_xy(self) -> Tuple[float, float]:
        return self.x_bound[2], self.y_bound[2]

    @property
    def start_xy(self) -> Tuple[float, float]:
        return (
            self.x_bound[0] + self.x_bound[2] / 2.0,
            self.y_bound[0] + self.y_bound[2] / 2.0,
        )


PLANNING_METRIC_GRID_V1 = PlanningMetricGridV1()


@dataclass(frozen=True)
class OccupancyRasterV1:
    """Per-object occupancy and optional audit unions.

    ``per_object`` is always ``[N,T,H,W]`` float32 and ``valid_mask`` is
    ``[N,T]`` bool.  The unions remain ``None`` unless requested so callers
    producing training records do not accidentally store redundant grids.
    """

    per_object: torch.Tensor
    valid_mask: torch.Tensor
    union_all: Optional[torch.Tensor]
    union_vehicle: Optional[torch.Tensor]
    union_human: Optional[torch.Tensor]
    category_groups: Tuple[str, ...]
    rasterizer_id: str
    occupancy_origin: str
    yaw_policy: str
    trajectory_policy: str
    bicycle_parity_caveat: str


@dataclass(frozen=True)
class RasterizerProvenanceV1:
    """Immutable provenance suitable for an exporter manifest."""

    rasterizer_id: str
    occupancy_origin: str
    grid_bounds_xyxy_m: Tuple[float, float, float, float]
    resolution_m: float
    height: int
    width: int
    yaw_policy: str
    trajectory_policy: str
    native_orion_occupancy: bool
    bicycle_parity_caveat: str


class SelectedMotionRasterInputLike(Protocol):
    """Structural interface implemented by ``SelectedMotionRasterInputV1``."""

    decoded_boxes_lidar: torch.Tensor
    selected_deltas: torch.Tensor
    trajectories_are_step_deltas: bool


def selected_mode_rasterizer_provenance_v1() -> RasterizerProvenanceV1:
    """Return the exact policy recorded for predicted derived occupancy."""

    grid = PLANNING_METRIC_GRID_V1
    return RasterizerProvenanceV1(
        rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        occupancy_origin="derived_from_decoded_box_and_orion_selected_mode",
        grid_bounds_xyxy_m=(
            grid.x_bound[0],
            grid.y_bound[0],
            grid.x_bound[1],
            grid.y_bound[1],
        ),
        resolution_m=grid.x_bound[2],
        height=grid.height,
        width=grid.width,
        yaw_policy=SELECTED_MODE_YAW_POLICY,
        trajectory_policy=TRAJECTORY_POLICY,
        native_orion_occupancy=False,
        bicycle_parity_caveat=BICYCLE_PARITY_CAVEAT,
    )


def _require_float_matrix(
    value: torch.Tensor,
    name: str,
    *,
    minimum_columns: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise BEVTargetRasterizerError(f"{name} must be a tensor")
    if value.ndim != 2 or value.shape[1] < minimum_columns:
        raise BEVTargetRasterizerError(
            f"{name} must have shape [N,D] with D>={minimum_columns}"
        )
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise BEVTargetRasterizerError(f"{name} must be finite floating point")


def _require_boxes(boxes_lidar: torch.Tensor, name: str) -> None:
    _require_float_matrix(boxes_lidar, name, minimum_columns=7)
    if torch.any(boxes_lidar[:, 3:5] <= 0):
        raise BEVTargetRasterizerError(f"{name} width and length must be positive")


def _copied_numpy(value: torch.Tensor) -> np.ndarray:
    # clone() is not optional: CPU Tensor.numpy() may share storage.
    return value.detach().cpu().clone().numpy()


def _planningmetric_polygon_pixels(
    x: float,
    y: float,
    converted_yaw: float,
    length: float,
    width: float,
    grid: PlanningMetricGridV1,
) -> np.ndarray:
    """Mirror ``PlanningMetric._get_poly_region_in_image`` exactly."""

    trans = np.array([[x, y]]).T
    rotation = np.array(
        [
            [np.cos(converted_yaw), -np.sin(converted_yaw)],
            [np.sin(converted_yaw), np.cos(converted_yaw)],
        ]
    )
    corners = np.array(
        [
            [length / 2.0, -length / 2.0, -length / 2.0, length / 2.0],
            [width / 2.0, width / 2.0, -width / 2.0, -width / 2.0],
        ]
    )
    corners_lidar = rotation @ corners + trans
    lidar_to_cv = np.array([[1, 0], [0, -1]])
    start = np.asarray(grid.start_xy)[:, None]
    resolution = np.asarray(grid.resolution_xy)[:, None]
    corners_cv = (
        lidar_to_cv @ corners_lidar - start + resolution / 2.0
    ).T / np.asarray(grid.resolution_xy)
    return np.round(corners_cv).astype(np.int32)


def _rasterize_polygons(
    positions_xy: np.ndarray,
    converted_yaw: np.ndarray,
    widths: np.ndarray,
    lengths: np.ndarray,
    valid_mask: np.ndarray,
    grid: PlanningMetricGridV1,
) -> np.ndarray:
    object_count, timesteps, _ = positions_xy.shape
    occupancy = np.zeros(
        (object_count, timesteps, grid.height, grid.width), dtype=np.uint8
    )
    for object_index in range(object_count):
        for timestep in range(timesteps):
            if not bool(valid_mask[object_index, timestep]):
                continue
            polygon = _planningmetric_polygon_pixels(
                float(positions_xy[object_index, timestep, 0]),
                float(positions_xy[object_index, timestep, 1]),
                float(converted_yaw[object_index, timestep]),
                float(lengths[object_index]),
                float(widths[object_index]),
                grid,
            )
            cv2.fillPoly(occupancy[object_index, timestep], [polygon], 1)
    return occupancy


def _optional_unions(
    occupancy: torch.Tensor,
    groups: Tuple[str, ...],
    include_union: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not include_union:
        return None, None, None
    shape = occupancy.shape[1:]
    zero = torch.zeros(shape, dtype=occupancy.dtype, device=occupancy.device)

    def union_for(name: str) -> torch.Tensor:
        indices = [index for index, group in enumerate(groups) if group == name]
        if not indices:
            return zero.clone()
        return occupancy[indices].amax(dim=0)

    if occupancy.shape[0]:
        union_all = occupancy.amax(dim=0)
    else:
        union_all = zero.clone()
    return union_all, union_for("vehicle"), union_for("human")


def rasterize_planningmetric_gt_v1(
    gt_boxes_lidar: torch.Tensor,
    gt_agent_features: torch.Tensor,
    *,
    include_union: bool = False,
) -> OccupancyRasterV1:
    """Rasterize B2D GT with pixel parity to the repository PlanningMetric.

    Expected box fields are ``(x,y,z,w,l,h,yaw,...)``.  Feature layout is the
    exact 34-field layout consumed by the repository implementation: six XY
    step deltas, six future-valid flags, and six future-yaw deltas, with the
    class/category at field 27.
    """

    _require_boxes(gt_boxes_lidar, "gt_boxes_lidar")
    if not isinstance(gt_agent_features, torch.Tensor):
        raise BEVTargetRasterizerError("gt_agent_features must be a tensor")
    if (
        gt_agent_features.ndim != 3
        or gt_agent_features.shape[0] != 1
        or gt_agent_features.shape[1] != gt_boxes_lidar.shape[0]
        or gt_agent_features.shape[2] < 34
    ):
        raise BEVTargetRasterizerError(
            "gt_agent_features must have PlanningMetric shape [1,N,D] with D>=34"
        )
    if not gt_agent_features.is_floating_point() or not torch.isfinite(
        gt_agent_features
    ).all():
        raise BEVTargetRasterizerError(
            "gt_agent_features must be finite floating point"
        )

    grid = PLANNING_METRIC_GRID_V1
    boxes = _copied_numpy(gt_boxes_lidar)
    features = _copied_numpy(gt_agent_features)
    object_count = boxes.shape[0]

    step_deltas = features[..., : grid.timesteps * 2].reshape(
        object_count, grid.timesteps, 2
    )
    valid = features[
        ..., grid.timesteps * 2 : grid.timesteps * 3
    ].reshape(object_count, grid.timesteps) == 1
    future_yaw_deltas = features[
        ...,
        grid.timesteps * 3 + 10 : grid.timesteps * 4 + 10,
    ].reshape(object_count, grid.timesteps)

    cumulative_positions = np.cumsum(step_deltas, axis=1)
    cumulative_yaw = np.cumsum(future_yaw_deltas, axis=1)
    # Mirror the reference conversion, but only on our private copied array.
    converted_current_yaw = -1.0 * (boxes[:, 6] + np.pi / 2.0)
    positions = cumulative_positions + boxes[:, None, 0:2]
    yaw = cumulative_yaw + converted_current_yaw[:, None]

    raw_categories = features[0, :, 27].astype(np.int64)
    groups = tuple(
        "vehicle"
        if int(category) in (0, 1, 2)
        else "human"
        if int(category) in (3, 7)
        else "ignored"
        for category in raw_categories
    )
    # ``np.asarray([])`` defaults to float64.  An empty eligible-object frame
    # is valid in B2D, so pin the dtype before combining it with the boolean
    # future-valid mask.
    included = np.asarray(
        [group != "ignored" for group in groups], dtype=np.bool_
    )
    raster_valid = valid & included[:, None]
    occupancy_np = _rasterize_polygons(
        positions,
        yaw,
        widths=boxes[:, 3],
        lengths=boxes[:, 4],
        valid_mask=raster_valid,
        grid=grid,
    )
    occupancy = torch.from_numpy(occupancy_np.astype(np.float32)).to(
        device=gt_boxes_lidar.device
    )
    valid_tensor = torch.from_numpy(raster_valid).to(device=gt_boxes_lidar.device)
    union_all, union_vehicle, union_human = _optional_unions(
        occupancy, groups, include_union
    )
    return OccupancyRasterV1(
        per_object=occupancy,
        valid_mask=valid_tensor,
        union_all=union_all,
        union_vehicle=union_vehicle,
        union_human=union_human,
        category_groups=groups,
        rasterizer_id=GT_RASTERIZER_ID,
        occupancy_origin="privileged_b2d_boxes_and_future_state",
        yaw_policy="planningmetric-cumulative-future-yaw/v1",
        trajectory_policy="b2d-step-deltas-cumulative-sum/v1",
        bicycle_parity_caveat=BICYCLE_PARITY_CAVEAT,
    )


def rasterize_orion_selected_mode_v1(
    decoded_boxes_lidar: torch.Tensor,
    selected_step_deltas: torch.Tensor,
    *,
    future_valid_mask: Optional[torch.Tensor] = None,
    include_union: bool = False,
) -> OccupancyRasterV1:
    """Rasterize the ORION-selected motion mode as derived actor occupancy.

    ORION selected trajectories are step deltas, so positions are accumulated
    before adding the decoded current center.  ORION supplies no future actor
    yaw in this path; v1 freezes the converted current yaw for every future
    timestep.  This is an explicit approximation, not native occupancy.
    """

    _require_boxes(decoded_boxes_lidar, "decoded_boxes_lidar")
    if not isinstance(selected_step_deltas, torch.Tensor):
        raise BEVTargetRasterizerError("selected_step_deltas must be a tensor")
    if (
        selected_step_deltas.ndim != 3
        or selected_step_deltas.shape[0] != decoded_boxes_lidar.shape[0]
        or selected_step_deltas.shape[2] != 2
        or selected_step_deltas.shape[1] <= 0
    ):
        raise BEVTargetRasterizerError(
            "selected_step_deltas must have shape [N,positive_T,2]"
        )
    if (
        not selected_step_deltas.is_floating_point()
        or not torch.isfinite(selected_step_deltas).all()
    ):
        raise BEVTargetRasterizerError(
            "selected_step_deltas must be finite floating point"
        )
    if selected_step_deltas.device != decoded_boxes_lidar.device:
        raise BEVTargetRasterizerError(
            "decoded boxes and selected deltas must share a device"
        )
    object_count, timesteps, _ = selected_step_deltas.shape
    if future_valid_mask is None:
        valid = np.ones((object_count, timesteps), dtype=bool)
    else:
        if (
            not isinstance(future_valid_mask, torch.Tensor)
            or future_valid_mask.dtype != torch.bool
            or tuple(future_valid_mask.shape) != (object_count, timesteps)
            or future_valid_mask.device != decoded_boxes_lidar.device
        ):
            raise BEVTargetRasterizerError(
                "future_valid_mask must be bool [N,T] on the boxes device"
            )
        valid = _copied_numpy(future_valid_mask).astype(bool)

    boxes = _copied_numpy(decoded_boxes_lidar)
    deltas = _copied_numpy(selected_step_deltas)
    positions = np.cumsum(deltas, axis=1) + boxes[:, None, 0:2]
    converted_current_yaw = -1.0 * (boxes[:, 6] + np.pi / 2.0)
    frozen_yaw = np.repeat(converted_current_yaw[:, None], timesteps, axis=1)
    occupancy_np = _rasterize_polygons(
        positions,
        frozen_yaw,
        widths=boxes[:, 3],
        lengths=boxes[:, 4],
        valid_mask=valid,
        grid=PLANNING_METRIC_GRID_V1,
    )
    occupancy = torch.from_numpy(occupancy_np.astype(np.float32)).to(
        device=decoded_boxes_lidar.device
    )
    valid_tensor = torch.from_numpy(valid).to(device=decoded_boxes_lidar.device)
    groups = tuple("predicted" for _ in range(object_count))
    union_all = (
        occupancy.amax(dim=0)
        if include_union and object_count
        else torch.zeros(
            occupancy.shape[1:], dtype=occupancy.dtype, device=occupancy.device
        )
        if include_union
        else None
    )
    return OccupancyRasterV1(
        per_object=occupancy,
        valid_mask=valid_tensor,
        union_all=union_all,
        union_vehicle=None,
        union_human=None,
        category_groups=groups,
        rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        occupancy_origin="derived_from_decoded_box_and_orion_selected_mode",
        yaw_policy=SELECTED_MODE_YAW_POLICY,
        trajectory_policy=TRAJECTORY_POLICY,
        bicycle_parity_caveat=BICYCLE_PARITY_CAVEAT,
    )


def selected_mode_occupancy_callback_v1(
    value: SelectedMotionRasterInputLike,
) -> torch.Tensor:
    """Adapter callback accepted by ``adapt_orion_head_outputs_v1``."""

    if getattr(value, "trajectories_are_step_deltas", None) is not True:
        raise BEVTargetRasterizerError(
            "selected-mode adapter requires trajectories_are_step_deltas=True"
        )
    return rasterize_orion_selected_mode_v1(
        value.decoded_boxes_lidar,
        value.selected_deltas,
        include_union=False,
    ).per_object


def bev_box_corners_lidar_v1(boxes_lidar: torch.Tensor) -> torch.Tensor:
    """Return repository-convention rotated BEV corners without mutation.

    Input fields are ``(x,y,z,w,l,h,yaw,...)`` and output is ``[N,4,2]``.
    The corner order and row-vector rotation mirror
    ``mmcv/core/bbox/box_np_ops.py::box2d_to_corner_jit``.
    """

    _require_boxes(boxes_lidar, "boxes_lidar")
    boxes = _copied_numpy(boxes_lidar).astype(np.float64, copy=False)
    object_count = boxes.shape[0]
    normalized = np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=np.float64,
    )
    corners = np.empty((object_count, 4, 2), dtype=np.float64)
    for index in range(object_count):
        local = normalized * boxes[index, [3, 4]]
        yaw = boxes[index, 6]
        rotation_transpose = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        corners[index] = local @ rotation_transpose + boxes[index, None, :2]
    return torch.from_numpy(corners).to(
        device=boxes_lidar.device, dtype=boxes_lidar.dtype
    )


def pairwise_bev_iou_v1(
    boxes_a_lidar: torch.Tensor,
    boxes_b_lidar: torch.Tensor,
) -> torch.Tensor:
    """Continuous rotated BEV IoU for decoded/GT boxes on CPU or CUDA.

    Geometry is computed on private CPU copies with OpenCV's convex-polygon
    intersection, then returned on the input device.  No source tensor or box
    wrapper is modified.  This is continuous box IoU, not raster IoU.
    """

    _require_boxes(boxes_a_lidar, "boxes_a_lidar")
    _require_boxes(boxes_b_lidar, "boxes_b_lidar")
    if boxes_a_lidar.device != boxes_b_lidar.device:
        raise BEVTargetRasterizerError("both box sets must share a device")
    corners_a = _copied_numpy(bev_box_corners_lidar_v1(boxes_a_lidar)).astype(
        np.float32, copy=False
    )
    corners_b = _copied_numpy(bev_box_corners_lidar_v1(boxes_b_lidar)).astype(
        np.float32, copy=False
    )
    result = np.zeros((corners_a.shape[0], corners_b.shape[0]), dtype=np.float32)
    areas_a = np.asarray(
        [abs(cv2.contourArea(polygon)) for polygon in corners_a], dtype=np.float64
    )
    areas_b = np.asarray(
        [abs(cv2.contourArea(polygon)) for polygon in corners_b], dtype=np.float64
    )
    for row, polygon_a in enumerate(corners_a):
        for column, polygon_b in enumerate(corners_b):
            intersection, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
            intersection = max(float(intersection), 0.0)
            union = areas_a[row] + areas_b[column] - intersection
            if union <= 0:
                raise BEVTargetRasterizerError(
                    "positive box dimensions produced non-positive BEV union"
                )
            result[row, column] = min(max(intersection / union, 0.0), 1.0)
    return torch.from_numpy(result).to(
        device=boxes_a_lidar.device, dtype=boxes_a_lidar.dtype
    )


__all__ = [
    "BEVTargetRasterizerError",
    "BICYCLE_PARITY_CAVEAT",
    "GT_RASTERIZER_ID",
    "OccupancyRasterV1",
    "PAIRWISE_BEV_IOU_POLICY_ID",
    "PLANNING_METRIC_GRID_V1",
    "PlanningMetricGridV1",
    "RasterizerProvenanceV1",
    "SELECTED_MODE_RASTERIZER_ID",
    "SELECTED_MODE_YAW_POLICY",
    "TRAJECTORY_POLICY",
    "bev_box_corners_lidar_v1",
    "pairwise_bev_iou_v1",
    "rasterize_orion_selected_mode_v1",
    "rasterize_planningmetric_gt_v1",
    "selected_mode_occupancy_callback_v1",
    "selected_mode_rasterizer_provenance_v1",
]
