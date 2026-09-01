"""Six-view projected visible-support attribution for Stage-1 targets.

This module is deliberately dependency-light and CPU-only.  It projects full
3D lidar boxes with the *post-augmentation* ``lidar2img`` matrices, clips the
box edges against the camera near plane and processed image, and area-pools the
resulting convex polygon onto the exact visual patch grid.

The output is an attribution proxy, not a causal explanation: a projected box
only identifies image patches in which a failed object may be visible.  It
does not prove that those pixels caused the frozen model's failure.

Semantic/depth refinement is an explicit optional interface.  The base
projector never claims that either refinement happened when the corresponding
data and refiner were not provided.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch

from uq_estimator.object_failure_targets import (
    PATCH_ATTRIBUTION_CLAIM,
    POST_AUGMENTATION_PROJECTION,
    PatchSupportProvenanceV1,
)


VISIBLE_SUPPORT_PROJECTION_VERSION = (
    "orion.six-view-projected-visible-support/v1"
)
BOX_GEOMETRY_VERSION = "lidar-box-xyz-dx-dy-dz-yaw/v1"
SILHOUETTE_METHOD = "near-clipped-3d-box-edge-convex-hull"
PATCH_POOLING_METHOD = "polygon-patch-intersection-area-fraction-row-major"

ORION_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


class VisibleSupportProjectionError(ValueError):
    """Raised when support projection would require guessing geometry."""


@dataclass(frozen=True)
class SupportRefinementRequestV1:
    """One projected object passed to an optional semantic/depth refiner."""

    camera_index: int
    camera_name: str
    object_index: int
    image_hw: tuple[int, int]
    patch_hw: tuple[int, int]
    polygon_xy: tuple[tuple[float, float], ...]
    depth_range: tuple[float, float]
    base_patch_support: torch.Tensor


@dataclass(frozen=True)
class SupportRefinementResultV1:
    """Replacement support returned by an explicitly configured refiner."""

    patch_support: torch.Tensor
    patch_valid_mask: torch.Tensor
    audit_note: str


class ProjectedSupportRefinerV1(Protocol):
    """Optional interface for audited semantic and/or depth refinement.

    Implementations must declare ``refinement_id`` and
    ``required_modalities``.  V1 recognizes only ``"semantic"`` and
    ``"depth"``.  ``project_boxes_to_visible_patch_support`` refuses to run a
    refiner unless a non-``None`` payload exists for every declared modality.
    """

    refinement_id: str
    required_modalities: tuple[str, ...]

    def refine(
        self,
        request: SupportRefinementRequestV1,
        modality_payloads: Mapping[str, Any],
    ) -> SupportRefinementResultV1:
        ...


@dataclass(frozen=True)
class VisibleSupportProjectionProvenanceV1:
    """Complete geometry/refinement claim boundary for one projection call."""

    camera_order: tuple[str, ...]
    matrix_camera_order: tuple[str, ...]
    image_shape_camera_order: tuple[str, ...]
    processed_image_hw: tuple[tuple[int, int], ...]
    patch_hw: tuple[int, int]
    image_transform_id: str
    box_z_origin: str
    near_plane_depth: float
    refinement_id: Optional[str]
    refinement_modalities: tuple[str, ...]
    refinement_call_count: int
    refinement_applied: bool
    projection_matrix_kind: str = POST_AUGMENTATION_PROJECTION
    box_geometry_version: str = BOX_GEOMETRY_VERSION
    silhouette_method: str = SILHOUETTE_METHOD
    patch_pooling_method: str = PATCH_POOLING_METHOD
    attribution: str = PATCH_ATTRIBUTION_CLAIM
    attribution_is_causal: bool = False
    schema_version: str = VISIBLE_SUPPORT_PROJECTION_VERSION


@dataclass(frozen=True)
class ProjectedVisibleSupportV1:
    """Fractional object support aligned to ORION view/patch axes.

    ``support`` has shape ``[V, P, J]`` and can be passed directly as
    ``gt_projected_support`` or ``pred_projected_support`` to the decoded
    actual-target exporter.  Invisible objects have exactly zero support and
    ``object_visible_mask=False``; they are never fabricated as visible.
    """

    support: torch.Tensor
    valid_patch_mask: torch.Tensor
    object_visible_mask: torch.Tensor
    projected_depth_range: torch.Tensor
    projected_polygons_xy: tuple[
        tuple[tuple[tuple[float, float], ...], ...], ...
    ]
    support_provenance: PatchSupportProvenanceV1
    projection_provenance: VisibleSupportProjectionProvenanceV1


def _require_cpu_float_tensor(
    value: torch.Tensor,
    name: str,
    *,
    ndim: Optional[int] = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise VisibleSupportProjectionError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise VisibleSupportProjectionError(
            f"{name} must have {ndim} dimensions"
        )
    if not value.is_floating_point():
        raise VisibleSupportProjectionError(f"{name} must be floating point")
    if value.device.type != "cpu":
        raise VisibleSupportProjectionError(
            f"{name} must be on CPU for dependency-light projection"
        )
    if not bool(torch.isfinite(value).all()):
        raise VisibleSupportProjectionError(f"{name} must contain finite values")


def _normalize_order(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise VisibleSupportProjectionError(f"{name} must be a sequence of names")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise VisibleSupportProjectionError(
            f"{name} must contain non-empty camera names"
        )
    if len(set(result)) != len(result):
        raise VisibleSupportProjectionError(f"{name} contains duplicate cameras")
    return result


def _normalize_hw_rows(
    value: Sequence[Sequence[int]] | torch.Tensor,
    views: int,
    name: str,
) -> tuple[tuple[int, int], ...]:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 2 or value.shape != (views, 2):
            raise VisibleSupportProjectionError(
                f"{name} tensor must have CPU shape [{views}, 2]"
            )
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise VisibleSupportProjectionError(
                f"{name} tensor must use an integer dtype"
            )
        rows = value.tolist()
    else:
        if isinstance(value, (str, bytes)) or len(value) != views:
            raise VisibleSupportProjectionError(
                f"{name} must contain one [height, width] row per camera"
            )
        rows = value
    result: list[tuple[int, int]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or len(row) != 2:
            raise VisibleSupportProjectionError(
                f"{name} rows must be [height, width]"
            )
        h_raw, w_raw = row
        if isinstance(h_raw, bool) or isinstance(w_raw, bool):
            raise VisibleSupportProjectionError(f"{name} entries must be integers")
        h, w = int(h_raw), int(w_raw)
        if h != h_raw or w != w_raw or min(h, w) <= 0:
            raise VisibleSupportProjectionError(
                f"{name} entries must be positive integers"
            )
        result.append((h, w))
    return tuple(result)


def _normalize_hw(value: Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise VisibleSupportProjectionError(f"{name} must be [height, width]")
    raw_h, raw_w = value
    if isinstance(raw_h, bool) or isinstance(raw_w, bool):
        raise VisibleSupportProjectionError(f"{name} entries must be integers")
    h, w = int(raw_h), int(raw_w)
    if h != raw_h or w != raw_w or min(h, w) <= 0:
        raise VisibleSupportProjectionError(
            f"{name} entries must be positive integers"
        )
    return h, w


def lidar_boxes_to_corners(
    boxes_lidar: torch.Tensor,
    *,
    box_z_origin: str,
) -> torch.Tensor:
    """Convert ``[J,D>=7]`` lidar boxes to ordered ``[J,8,3]`` corners.

    Fields are exactly ``x, y, z, dx, dy, dz, yaw, ...``.  ``yaw`` is a
    right-handed rotation around +z.  ``box_z_origin`` is mandatory and must
    be either ``"center"`` or ``"bottom"`` so callers cannot silently mix
    MMDetection bottom-center boxes with center-origin boxes.
    """

    _require_cpu_float_tensor(boxes_lidar, "boxes_lidar", ndim=2)
    if boxes_lidar.shape[1] < 7:
        raise VisibleSupportProjectionError(
            "boxes_lidar must have shape [objects, D] with D>=7"
        )
    if box_z_origin not in ("center", "bottom"):
        raise VisibleSupportProjectionError(
            "box_z_origin must be explicitly 'center' or 'bottom'"
        )
    if boxes_lidar.shape[0] and bool(torch.any(boxes_lidar[:, 3:6] <= 0)):
        raise VisibleSupportProjectionError("box dx/dy/dz must be positive")

    count = boxes_lidar.shape[0]
    dtype = boxes_lidar.dtype
    signs_xy = torch.tensor(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
        dtype=dtype,
    )
    half_xy = boxes_lidar[:, None, 3:5] * 0.5
    local_xy = signs_xy[None, :, :] * half_xy
    yaw = boxes_lidar[:, 6]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    rotated_x = (
        local_xy[..., 0] * cos_yaw[:, None]
        - local_xy[..., 1] * sin_yaw[:, None]
    )
    rotated_y = (
        local_xy[..., 0] * sin_yaw[:, None]
        + local_xy[..., 1] * cos_yaw[:, None]
    )
    xy = torch.stack((rotated_x, rotated_y), dim=-1)
    xy = xy + boxes_lidar[:, None, :2]

    if box_z_origin == "center":
        z_bottom = boxes_lidar[:, 2] - boxes_lidar[:, 5] * 0.5
    else:
        z_bottom = boxes_lidar[:, 2]
    z_top = z_bottom + boxes_lidar[:, 5]
    bottom = torch.cat((xy, z_bottom[:, None, None].expand(count, 4, 1)), dim=-1)
    top = torch.cat((xy, z_top[:, None, None].expand(count, 4, 1)), dim=-1)
    return torch.cat((bottom, top), dim=1)


def _cross(
    origin: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (
        a[1] - origin[1]
    ) * (b[0] - origin[0])


def _convex_hull(
    points: Sequence[tuple[float, float]],
    *,
    epsilon: float = 1e-9,
) -> list[tuple[float, float]]:
    unique = sorted(
        {
            (round(float(point[0]), 12), round(float(point[1]), 12))
            for point in points
            if math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
        }
    )
    if len(unique) < 3:
        return []
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= epsilon:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= epsilon:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return hull if len(hull) >= 3 else []


def _clip_polygon_half_plane(
    polygon: Sequence[tuple[float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float]) -> bool:
        if keep_greater:
            return point[axis] >= boundary - 1e-9
        return point[axis] <= boundary + 1e-9

    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) > 1e-15:
                t = (boundary - previous[axis]) / denominator
                intersection = (
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                )
                output.append(intersection)
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def _clip_polygon_to_rectangle(
    polygon: Sequence[tuple[float, float]],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> list[tuple[float, float]]:
    result = list(polygon)
    result = _clip_polygon_half_plane(
        result, axis=0, boundary=xmin, keep_greater=True
    )
    result = _clip_polygon_half_plane(
        result, axis=0, boundary=xmax, keep_greater=False
    )
    result = _clip_polygon_half_plane(
        result, axis=1, boundary=ymin, keep_greater=True
    )
    return _clip_polygon_half_plane(
        result, axis=1, boundary=ymax, keep_greater=False
    )


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area_twice = 0.0
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        area_twice += current[0] * following[1] - following[0] * current[1]
    return abs(area_twice) * 0.5


def _near_clipped_projected_polygon(
    corners_xyz: torch.Tensor,
    lidar2img: torch.Tensor,
    *,
    near_plane_depth: float,
    image_hw: tuple[int, int],
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    homogeneous = torch.cat(
        (corners_xyz, torch.ones(8, 1, dtype=corners_xyz.dtype)), dim=-1
    )
    projected = homogeneous @ lidar2img.transpose(0, 1)
    depths = projected[:, 2]
    clipped_homogeneous: list[torch.Tensor] = []
    clipped_depths: list[float] = []
    for first, second in _BOX_EDGES:
        endpoint_a = projected[first]
        endpoint_b = projected[second]
        depth_a = float(endpoint_a[2].item())
        depth_b = float(endpoint_b[2].item())
        front_a = depth_a >= near_plane_depth
        front_b = depth_b >= near_plane_depth
        if not front_a and not front_b:
            continue
        if front_a:
            clipped_homogeneous.append(endpoint_a)
            clipped_depths.append(depth_a)
        if front_b:
            clipped_homogeneous.append(endpoint_b)
            clipped_depths.append(depth_b)
        if front_a != front_b:
            denominator = depth_b - depth_a
            if abs(denominator) <= 1e-15:
                continue
            t = (near_plane_depth - depth_a) / denominator
            intersection = endpoint_a + t * (endpoint_b - endpoint_a)
            clipped_homogeneous.append(intersection)
            clipped_depths.append(near_plane_depth)
    if len(clipped_homogeneous) < 3:
        return [], (math.nan, math.nan)

    pixels: list[tuple[float, float]] = []
    for point in clipped_homogeneous:
        depth = float(point[2].item())
        if depth < near_plane_depth:
            continue
        pixels.append(
            (float(point[0].item()) / depth, float(point[1].item()) / depth)
        )
    hull = _convex_hull(pixels)
    image_h, image_w = image_hw
    clipped = _clip_polygon_to_rectangle(
        hull, 0.0, 0.0, float(image_w), float(image_h)
    )
    if len(clipped) < 3 or _polygon_area(clipped) <= 1e-9:
        return [], (math.nan, math.nan)
    front_depths = [
        float(value.item())
        for value in depths
        if float(value.item()) >= near_plane_depth
    ]
    front_depths.extend(clipped_depths)
    return clipped, (min(front_depths), max(front_depths))


def _polygon_to_patch_support(
    polygon: Sequence[tuple[float, float]],
    *,
    image_hw: tuple[int, int],
    patch_hw: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    image_h, image_w = image_hw
    patch_h, patch_w = patch_hw
    support = torch.zeros(patch_h * patch_w, dtype=dtype)
    if not polygon:
        return support
    patch_height = float(image_h) / patch_h
    patch_width = float(image_w) / patch_w
    min_x = max(0.0, min(point[0] for point in polygon))
    max_x = min(float(image_w), max(point[0] for point in polygon))
    min_y = max(0.0, min(point[1] for point in polygon))
    max_y = min(float(image_h), max(point[1] for point in polygon))
    first_x = max(0, min(patch_w - 1, int(math.floor(min_x / patch_width))))
    last_x = max(
        0,
        min(patch_w - 1, int(math.ceil(max_x / patch_width) - 1)),
    )
    first_y = max(0, min(patch_h - 1, int(math.floor(min_y / patch_height))))
    last_y = max(
        0,
        min(patch_h - 1, int(math.ceil(max_y / patch_height) - 1)),
    )
    patch_area = patch_height * patch_width
    for patch_y in range(first_y, last_y + 1):
        ymin, ymax = patch_y * patch_height, (patch_y + 1) * patch_height
        for patch_x in range(first_x, last_x + 1):
            xmin, xmax = patch_x * patch_width, (patch_x + 1) * patch_width
            intersection = _clip_polygon_to_rectangle(
                polygon, xmin, ymin, xmax, ymax
            )
            fraction = _polygon_area(intersection) / patch_area
            if fraction > 0:
                support[patch_y * patch_w + patch_x] = min(1.0, fraction)
    return support


def _validate_refiner(
    refiner: Optional[ProjectedSupportRefinerV1],
    refinement_payloads: Optional[Mapping[str, Any]],
) -> tuple[Optional[str], tuple[str, ...], Mapping[str, Any]]:
    if refiner is None:
        if refinement_payloads:
            raise VisibleSupportProjectionError(
                "refinement_payloads were provided without an explicit refiner"
            )
        return None, (), {}
    refinement_id = str(getattr(refiner, "refinement_id", "")).strip()
    if not refinement_id:
        raise VisibleSupportProjectionError(
            "refiner.refinement_id must be non-empty"
        )
    raw_modalities = getattr(refiner, "required_modalities", None)
    if not isinstance(raw_modalities, tuple):
        raise VisibleSupportProjectionError(
            "refiner.required_modalities must be an explicit tuple"
        )
    modalities = tuple(str(value).strip() for value in raw_modalities)
    if (
        any(value not in ("semantic", "depth") for value in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise VisibleSupportProjectionError(
            "refiner modalities must be unique semantic/depth entries"
        )
    payloads = {} if refinement_payloads is None else dict(refinement_payloads)
    if set(payloads) != set(modalities):
        raise VisibleSupportProjectionError(
            "refinement payload keys must exactly match declared modalities"
        )
    if any(payloads[name] is None for name in modalities):
        raise VisibleSupportProjectionError(
            "semantic/depth refinement payloads must not be None"
        )
    if not callable(getattr(refiner, "refine", None)):
        raise VisibleSupportProjectionError("refiner must implement refine()")
    return refinement_id, modalities, payloads


def project_boxes_to_visible_patch_support(
    boxes_lidar: torch.Tensor,
    post_augmentation_lidar2img: torch.Tensor,
    processed_image_hw: Sequence[Sequence[int]] | torch.Tensor,
    *,
    camera_order: Sequence[str],
    matrix_camera_order: Sequence[str],
    image_shape_camera_order: Sequence[str],
    image_transform_id: str,
    box_z_origin: str,
    patch_hw: Sequence[int] = (40, 40),
    expected_camera_order: Sequence[str] = ORION_CAMERA_ORDER,
    expected_patch_hw: Sequence[int] = (40, 40),
    near_plane_depth: float = 1e-3,
    input_patch_valid_mask: Optional[torch.Tensor] = None,
    refiner: Optional[ProjectedSupportRefinerV1] = None,
    refinement_payloads: Optional[Mapping[str, Any]] = None,
) -> ProjectedVisibleSupportV1:
    """Project full lidar boxes into fractional six-view patch support.

    Camera/matrix/image-shape orders are deliberately separate inputs and
    must exactly match the frozen expected order.  Production defaults also
    require the exact 40 x 40 EVAViT grid.  Tests or a future explicitly
    versioned backbone may override ``expected_patch_hw``, but the requested
    and expected grids must still match exactly.
    """

    _require_cpu_float_tensor(
        post_augmentation_lidar2img,
        "post_augmentation_lidar2img",
        ndim=3,
    )
    if tuple(post_augmentation_lidar2img.shape[1:]) != (4, 4):
        raise VisibleSupportProjectionError(
            "post_augmentation_lidar2img must have shape [views, 4, 4]"
        )
    cameras = _normalize_order(camera_order, "camera_order")
    matrix_order = _normalize_order(matrix_camera_order, "matrix_camera_order")
    shape_order = _normalize_order(
        image_shape_camera_order, "image_shape_camera_order"
    )
    expected_order = _normalize_order(
        expected_camera_order, "expected_camera_order"
    )
    views = len(cameras)
    if cameras != expected_order:
        raise VisibleSupportProjectionError(
            "camera_order does not exactly match the frozen expected camera order"
        )
    if matrix_order != cameras:
        raise VisibleSupportProjectionError(
            "matrix_camera_order does not exactly match camera_order"
        )
    if shape_order != cameras:
        raise VisibleSupportProjectionError(
            "image_shape_camera_order does not exactly match camera_order"
        )
    if post_augmentation_lidar2img.shape[0] != views:
        raise VisibleSupportProjectionError(
            "lidar2img view count does not match camera_order"
        )
    matrix_rank = torch.linalg.matrix_rank(post_augmentation_lidar2img)
    if not bool(torch.all(matrix_rank == 4)):
        raise VisibleSupportProjectionError(
            "every post-augmentation lidar2img matrix must have rank 4"
        )

    image_rows = _normalize_hw_rows(
        processed_image_hw, views, "processed_image_hw"
    )
    if len(set(image_rows)) != 1:
        raise VisibleSupportProjectionError(
            "v1 requires one uniform processed image shape across all cameras"
        )
    patch_shape = _normalize_hw(patch_hw, "patch_hw")
    expected_patch_shape = _normalize_hw(expected_patch_hw, "expected_patch_hw")
    if patch_shape != expected_patch_shape:
        raise VisibleSupportProjectionError(
            "patch_hw does not exactly match expected_patch_hw; 40x40 alignment fails closed"
        )
    if not str(image_transform_id).strip():
        raise VisibleSupportProjectionError("image_transform_id must be non-empty")
    if (
        isinstance(near_plane_depth, bool)
        or not math.isfinite(float(near_plane_depth))
        or float(near_plane_depth) <= 0
    ):
        raise VisibleSupportProjectionError(
            "near_plane_depth must be a finite positive number"
        )

    corners = lidar_boxes_to_corners(
        boxes_lidar, box_z_origin=box_z_origin
    )
    if corners.dtype != post_augmentation_lidar2img.dtype:
        raise VisibleSupportProjectionError(
            "boxes_lidar and lidar2img must use the same floating dtype"
        )
    objects = boxes_lidar.shape[0]
    patches = patch_shape[0] * patch_shape[1]
    if input_patch_valid_mask is None:
        valid_patch_mask = torch.ones(views, patches, dtype=torch.bool)
    else:
        if (
            not isinstance(input_patch_valid_mask, torch.Tensor)
            or input_patch_valid_mask.device.type != "cpu"
            or input_patch_valid_mask.dtype != torch.bool
            or tuple(input_patch_valid_mask.shape) != (views, patches)
        ):
            raise VisibleSupportProjectionError(
                f"input_patch_valid_mask must be CPU bool [{views}, {patches}]"
            )
        valid_patch_mask = input_patch_valid_mask.clone()

    refinement_id, modalities, payloads = _validate_refiner(
        refiner, refinement_payloads
    )
    support = torch.zeros(
        views, patches, objects, dtype=boxes_lidar.dtype
    )
    visible = torch.zeros(views, objects, dtype=torch.bool)
    depth_range = torch.full(
        (views, objects, 2), float("nan"), dtype=boxes_lidar.dtype
    )
    polygons: list[list[tuple[tuple[float, float], ...]]] = [
        [tuple() for _ in range(objects)] for _ in range(views)
    ]
    refinement_calls = 0
    for view_index, camera_name in enumerate(cameras):
        for object_index in range(objects):
            polygon, object_depth_range = _near_clipped_projected_polygon(
                corners[object_index],
                post_augmentation_lidar2img[view_index],
                near_plane_depth=float(near_plane_depth),
                image_hw=image_rows[view_index],
            )
            if not polygon:
                continue
            base_support = _polygon_to_patch_support(
                polygon,
                image_hw=image_rows[view_index],
                patch_hw=patch_shape,
                dtype=boxes_lidar.dtype,
            )
            if not bool(torch.any(base_support > 0)):
                continue
            object_support = base_support
            if refiner is not None:
                request = SupportRefinementRequestV1(
                    camera_index=view_index,
                    camera_name=camera_name,
                    object_index=object_index,
                    image_hw=image_rows[view_index],
                    patch_hw=patch_shape,
                    polygon_xy=tuple(polygon),
                    depth_range=object_depth_range,
                    base_patch_support=base_support.clone(),
                )
                refined = refiner.refine(request, payloads)
                refinement_calls += 1
                if not isinstance(refined, SupportRefinementResultV1):
                    raise VisibleSupportProjectionError(
                        "refiner must return SupportRefinementResultV1"
                    )
                _require_cpu_float_tensor(
                    refined.patch_support, "refined patch_support", ndim=1
                )
                if refined.patch_support.shape != (patches,):
                    raise VisibleSupportProjectionError(
                        f"refined patch_support must have shape [{patches}]"
                    )
                if bool(
                    torch.any(refined.patch_support < 0)
                    or torch.any(refined.patch_support > 1)
                ):
                    raise VisibleSupportProjectionError(
                        "refined patch_support must lie in [0, 1]"
                    )
                if bool(
                    torch.any(refined.patch_support > base_support + 1e-6)
                ):
                    raise VisibleSupportProjectionError(
                        "semantic/depth refinement may remove projected-box "
                        "support but must not expand beyond it"
                    )
                if (
                    not isinstance(refined.patch_valid_mask, torch.Tensor)
                    or refined.patch_valid_mask.device.type != "cpu"
                    or refined.patch_valid_mask.dtype != torch.bool
                    or refined.patch_valid_mask.shape != (patches,)
                ):
                    raise VisibleSupportProjectionError(
                        f"refined patch_valid_mask must be CPU bool [{patches}]"
                    )
                if not str(refined.audit_note).strip():
                    raise VisibleSupportProjectionError(
                        "refinement audit_note must be non-empty"
                    )
                valid_patch_mask[view_index] &= refined.patch_valid_mask
                object_support = refined.patch_support
            support[view_index, :, object_index] = object_support
            visible[view_index, object_index] = True
            depth_range[view_index, object_index] = torch.tensor(
                object_depth_range, dtype=boxes_lidar.dtype
            )
            polygons[view_index][object_index] = tuple(polygon)

    support = torch.where(
        valid_patch_mask[:, :, None], support, torch.zeros_like(support)
    ).clamp(0.0, 1.0)
    # A refiner is considered applied only if it actually processed at least
    # one projected object.  Merely configuring it is not reported as success.
    refinement_applied = refinement_calls > 0
    uniform_image_hw = image_rows[0]
    support_provenance = PatchSupportProvenanceV1(
        camera_order=cameras,
        image_hw=uniform_image_hw,
        patch_hw=patch_shape,
        image_transform_id=str(image_transform_id).strip(),
    )
    projection_provenance = VisibleSupportProjectionProvenanceV1(
        camera_order=cameras,
        matrix_camera_order=matrix_order,
        image_shape_camera_order=shape_order,
        processed_image_hw=image_rows,
        patch_hw=patch_shape,
        image_transform_id=str(image_transform_id).strip(),
        box_z_origin=box_z_origin,
        near_plane_depth=float(near_plane_depth),
        refinement_id=refinement_id,
        refinement_modalities=modalities,
        refinement_call_count=refinement_calls,
        refinement_applied=refinement_applied,
    )
    return ProjectedVisibleSupportV1(
        support=support,
        valid_patch_mask=valid_patch_mask,
        object_visible_mask=visible,
        projected_depth_range=depth_range,
        projected_polygons_xy=tuple(tuple(row) for row in polygons),
        support_provenance=support_provenance,
        projection_provenance=projection_provenance,
    )


def make_projection_overlay_data(
    result: ProjectedVisibleSupportV1,
    view_index: int,
    *,
    minimum_support: float = 0.0,
) -> dict[str, Any]:
    """Build JSON-serializable polygon/patch overlay data for one camera."""

    if isinstance(view_index, bool) or not isinstance(view_index, int):
        raise VisibleSupportProjectionError("view_index must be an integer")
    views, patches, objects = result.support.shape
    if not 0 <= view_index < views:
        raise VisibleSupportProjectionError("view_index is out of range")
    if (
        isinstance(minimum_support, bool)
        or not math.isfinite(float(minimum_support))
        or not 0 <= float(minimum_support) <= 1
    ):
        raise VisibleSupportProjectionError(
            "minimum_support must lie in [0, 1]"
        )
    patch_h, patch_w = result.projection_provenance.patch_hw
    camera_name = result.projection_provenance.camera_order[view_index]
    object_rows: list[dict[str, Any]] = []
    for object_index in range(objects):
        nonzero = torch.nonzero(
            result.support[view_index, :, object_index]
            > float(minimum_support),
            as_tuple=False,
        ).flatten()
        patch_rows = [
            {
                "patch_index": int(index),
                "patch_y": int(index) // patch_w,
                "patch_x": int(index) % patch_w,
                "fractional_support": float(
                    result.support[view_index, index, object_index].item()
                ),
            }
            for index in nonzero.tolist()
        ]
        depth = result.projected_depth_range[view_index, object_index]
        object_rows.append(
            {
                "object_index": object_index,
                "visible": bool(
                    result.object_visible_mask[view_index, object_index].item()
                ),
                "polygon_xy": [
                    [float(x), float(y)]
                    for x, y in result.projected_polygons_xy[view_index][object_index]
                ],
                "depth_range": (
                    [float(depth[0].item()), float(depth[1].item())]
                    if bool(torch.isfinite(depth).all())
                    else None
                ),
                "patches": patch_rows,
            }
        )
    return {
        "schema_version": VISIBLE_SUPPORT_PROJECTION_VERSION,
        "camera_index": view_index,
        "camera_name": camera_name,
        "processed_image_hw": list(
            result.projection_provenance.processed_image_hw[view_index]
        ),
        "patch_hw": [patch_h, patch_w],
        "valid_patch_mask": result.valid_patch_mask[view_index].tolist(),
        "objects": object_rows,
        "claim_boundary": {
            "attribution": result.projection_provenance.attribution,
            "attribution_is_causal": False,
        },
    }


def render_projection_overlay_image(
    result: ProjectedVisibleSupportV1,
    view_index: int,
    *,
    image: Optional[torch.Tensor] = None,
    alpha: float = 0.35,
) -> torch.Tensor:
    """Render patch heat and projected polygons as a CPU uint8 RGB tensor.

    Pillow and NumPy are imported lazily so target construction itself remains
    torch-only.  ``image=None`` creates a neutral canvas; no real data is
    required to exercise or inspect the overlay interface.
    """

    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "overlay image rendering requires optional Pillow and NumPy"
        ) from exc
    data = make_projection_overlay_data(result, view_index)
    image_h, image_w = data["processed_image_hw"]
    if image is None:
        background = torch.full((image_h, image_w, 3), 32, dtype=torch.uint8)
    else:
        if (
            not isinstance(image, torch.Tensor)
            or image.device.type != "cpu"
            or image.dtype != torch.uint8
            or tuple(image.shape) != (image_h, image_w, 3)
        ):
            raise VisibleSupportProjectionError(
                f"image must be CPU uint8 [{image_h}, {image_w}, 3]"
            )
        background = image
    if (
        isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or not 0 <= float(alpha) <= 1
    ):
        raise VisibleSupportProjectionError("alpha must lie in [0, 1]")

    base = Image.fromarray(background.numpy(), mode="RGB")
    heat = Image.new("RGBA", (image_w, image_h), (0, 0, 0, 0))
    heat_draw = ImageDraw.Draw(heat)
    patch_h, patch_w = data["patch_hw"]
    support_max = result.support[view_index].amax(dim=-1)
    for patch_y in range(patch_h):
        y0 = round(patch_y * image_h / patch_h)
        y1 = round((patch_y + 1) * image_h / patch_h)
        for patch_x in range(patch_w):
            patch_index = patch_y * patch_w + patch_x
            if not bool(result.valid_patch_mask[view_index, patch_index]):
                continue
            strength = float(support_max[patch_index].item())
            if strength <= 0:
                continue
            x0 = round(patch_x * image_w / patch_w)
            x1 = round((patch_x + 1) * image_w / patch_w)
            opacity = round(255 * float(alpha) * strength)
            heat_draw.rectangle((x0, y0, x1, y1), fill=(255, 96, 0, opacity))
    composed = Image.alpha_composite(base.convert("RGBA"), heat)
    draw = ImageDraw.Draw(composed)
    colors = ((0, 255, 255, 255), (255, 255, 0, 255), (255, 0, 255, 255))
    for row in data["objects"]:
        polygon = row["polygon_xy"]
        if len(polygon) >= 3:
            draw.line(
                [tuple(point) for point in polygon] + [tuple(polygon[0])],
                fill=colors[row["object_index"] % len(colors)],
                width=2,
            )
    array = np.asarray(composed.convert("RGB")).copy()
    return torch.from_numpy(array)


__all__ = [
    "BOX_GEOMETRY_VERSION",
    "ORION_CAMERA_ORDER",
    "PATCH_POOLING_METHOD",
    "ProjectedSupportRefinerV1",
    "ProjectedVisibleSupportV1",
    "SILHOUETTE_METHOD",
    "SupportRefinementRequestV1",
    "SupportRefinementResultV1",
    "VISIBLE_SUPPORT_PROJECTION_VERSION",
    "VisibleSupportProjectionError",
    "VisibleSupportProjectionProvenanceV1",
    "lidar_boxes_to_corners",
    "make_projection_overlay_data",
    "project_boxes_to_visible_patch_support",
    "render_projection_overlay_image",
]
