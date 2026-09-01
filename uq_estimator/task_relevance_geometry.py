"""Geometry-only weak supervision for a VLM task-relevance map.

This module does not consume observation-UQ values, corruption metadata, TTC,
collision outcomes, or scenario names.  It projects the unmodified ORION path
and currently visible actors whose future occupancy intersects that path.  The
result is privileged *Stage2-L supervision only*, never a Stage-1 target or a
closed-loop controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from uq_estimator.privileged_yield_labels import (
    TrajectoryConflictConfig,
    evaluate_trajectory_conflicts,
)


TASK_RELEVANCE_GEOMETRY_SCHEMA = "orion.task_relevance_geometry.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
IMAGE_HW = (900, 1600)
LIDAR_HEIGHT_ABOVE_EGO_ORIGIN_M = 1.84

# Exact raw-camera calibration used by team_code/orion_b2d_agent.py.  Keeping
# this immutable sidecar calibration makes offline targets reproducible and
# lets overlay review catch any future agent-calibration change.
LIDAR2IMG = np.asarray(
    [
        [[1142.51841, 800.0, 0.0, -952.0], [0.0, 450.0, -1142.51841, -809.704417], [0.0, 1.0, 0.0, -1.19], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 1394.75744, 0.0, -920.539908], [-368.61842, 258.109396, -1142.51841, -647.29675], [-0.819152044, 0.573576436, 0.0, -0.829094072], [0.0, 0.0, 0.0, 1.0]],
        [[1310.64327, -477.035138, 0.0, -406.010608], [368.61842, 258.109396, -1142.51841, -647.29675], [0.819152044, 0.573576436, 0.0, -0.829094072], [0.0, 0.0, 0.0, 1.0]],
        [[-560.166031, -800.0, 0.0, -1288.0], [0.0, -450.0, -560.166031, -858.939847], [0.0, -1.0, 0.0, -1.61], [0.0, 0.0, 0.0, 1.0]],
        [[-1142.51841, 800.0, 0.0, -684.385123], [-422.861679, -153.909064, -1142.51841, -496.004706], [-0.939692621, -0.342020143, 0.0, -0.492889531], [0.0, 0.0, 0.0, 1.0]],
        [[360.989788, -1347.23223, 0.0, -104.238127], [422.861679, -153.909064, -1142.51841, -496.004706], [0.939692621, -0.342020143, 0.0, -0.492889531], [0.0, 0.0, 0.0, 1.0]],
    ],
    dtype=np.float64,
)


class TaskRelevanceGeometryError(ValueError):
    """Raised when relevance geometry is absent or coordinate-ambiguous."""


@dataclass(frozen=True)
class TaskRelevanceGeometryResult:
    relevance: np.ndarray
    route_corridor: np.ndarray
    relevant_actor_support: np.ndarray
    relevant_actor_ids: Tuple[int, ...]
    route_point_coverage: float
    provenance: Dict[str, Any]


def _plan_array(plan: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(plan, dtype=np.float64)
    if value.ndim != 2 or value.shape != (6, 2):
        raise TaskRelevanceGeometryError(
            "ORION plan must have exact [6,2] [right,forward] shape"
        )
    if not np.all(np.isfinite(value)):
        raise TaskRelevanceGeometryError("ORION plan must contain finite values")
    return value


def _densify_plan(plan: np.ndarray, step_m: float = 0.5) -> np.ndarray:
    if not math.isfinite(step_m) or step_m <= 0.0:
        raise TaskRelevanceGeometryError("route interpolation step must be positive")
    points = np.concatenate((np.zeros((1, 2)), plan), axis=0)
    dense = []
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(distance / step_m)))
        dense.extend(
            start + (end - start) * (index / count)
            for index in range(1, count + 1)
        )
    return np.asarray(dense, dtype=np.float64)


def project_local_points(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project local ``[right, forward, up]`` points into all raw cameras."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise TaskRelevanceGeometryError("points must have shape [N,3]")
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    projected = np.einsum("vij,nj->vni", LIDAR2IMG, homogeneous)
    depth = projected[..., 2]
    safe = np.where(np.abs(depth) > 1e-6, depth, 1e-6)
    pixels = np.stack(
        (projected[..., 0] / safe, projected[..., 1] / safe), axis=-1
    )
    height, width = IMAGE_HW
    visible = (
        (depth > 1e-4)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= width - 1)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= height - 1)
    )
    return pixels, visible


def _gaussian_points_to_grid(
    pixels: np.ndarray,
    visible: np.ndarray,
    patch_hw: Tuple[int, int],
    radius_patches: float,
) -> np.ndarray:
    patch_h, patch_w = patch_hw
    grid_y, grid_x = np.meshgrid(
        np.arange(patch_h, dtype=np.float64),
        np.arange(patch_w, dtype=np.float64),
        indexing="ij",
    )
    output = np.zeros((len(CAMERA_ORDER), patch_h, patch_w), dtype=np.float32)
    scale_x = (patch_w - 1) / (IMAGE_HW[1] - 1)
    scale_y = (patch_h - 1) / (IMAGE_HW[0] - 1)
    for view in range(len(CAMERA_ORDER)):
        for point_index in np.flatnonzero(visible[view]):
            center_x = pixels[view, point_index, 0] * scale_x
            center_y = pixels[view, point_index, 1] * scale_y
            distance_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
            weight = np.exp(
                -0.5 * distance_sq / float(radius_patches ** 2)
            )
            weight[distance_sq > 9.0 * radius_patches ** 2] = 0.0
            output[view] = np.maximum(output[view], weight.astype(np.float32))
    return output


def _actor_box_local_corners(
    actor: Mapping[str, Any], ego: Mapping[str, Any]
) -> np.ndarray:
    lateral = float(actor["relative_lateral_m"])
    longitudinal = float(actor["relative_longitudinal_m"])
    ego_origin_z = float(ego["position_z"]) - float(ego["extent_z_m"])
    center_z = (
        float(actor["position_z"])
        - ego_origin_z
        - LIDAR_HEIGHT_ABOVE_EGO_ORIGIN_M
    )
    extent_forward, extent_right = map(float, actor["extent_xy_m"])
    extent_up = float(actor["extent_z_m"])
    relative_yaw = math.radians(
        float(actor["yaw_degrees"]) - float(ego["yaw_degrees"])
    )
    actor_forward = np.asarray(
        [math.sin(relative_yaw), math.cos(relative_yaw)], dtype=np.float64
    )
    actor_right = np.asarray(
        [math.cos(relative_yaw), -math.sin(relative_yaw)], dtype=np.float64
    )
    center = np.asarray([lateral, longitudinal], dtype=np.float64)
    corners = []
    for forward_sign in (-1.0, 1.0):
        for right_sign in (-1.0, 1.0):
            horizontal = (
                center
                + forward_sign * extent_forward * actor_forward
                + right_sign * extent_right * actor_right
            )
            for up_sign in (-1.0, 1.0):
                corners.append(
                    [horizontal[0], horizontal[1], center_z + up_sign * extent_up]
                )
    return np.asarray(corners, dtype=np.float64)


def _rasterize_actor_boxes(
    actors: Sequence[Mapping[str, Any]],
    ego: Mapping[str, Any],
    actor_ids: Sequence[int],
    patch_hw: Tuple[int, int],
) -> np.ndarray:
    patch_h, patch_w = patch_hw
    output = np.zeros((len(CAMERA_ORDER), patch_h, patch_w), dtype=np.float32)
    selected = {int(value) for value in actor_ids}
    for actor in actors:
        if int(actor["actor_id"]) not in selected:
            continue
        pixels, visible = project_local_points(_actor_box_local_corners(actor, ego))
        for view in range(len(CAMERA_ORDER)):
            points = pixels[view, visible[view]]
            if points.shape[0] < 2:
                continue
            x0 = max(0, int(math.floor(points[:, 0].min() * patch_w / IMAGE_HW[1])))
            x1 = min(patch_w - 1, int(math.floor(points[:, 0].max() * patch_w / IMAGE_HW[1])))
            y0 = max(0, int(math.floor(points[:, 1].min() * patch_h / IMAGE_HW[0])))
            y1 = min(patch_h - 1, int(math.floor(points[:, 1].max() * patch_h / IMAGE_HW[0])))
            if x1 >= x0 and y1 >= y0:
                output[view, y0:y1 + 1, x0:x1 + 1] = 1.0
    return output


def build_task_relevance_map(
    plan: Sequence[Sequence[float]],
    closedloop_safety: Mapping[str, Any],
    *,
    patch_hw: Tuple[int, int],
    route_weight: float = 0.75,
    route_radius_patches: float = 1.5,
    conflict_config: TrajectoryConflictConfig = None,
) -> TaskRelevanceGeometryResult:
    """Build dense route/actor relevance independent of observation UQ."""

    if len(patch_hw) != 2 or min(map(int, patch_hw)) <= 0:
        raise TaskRelevanceGeometryError("patch_hw must contain two positive values")
    if not 0.0 < float(route_weight) <= 1.0:
        raise TaskRelevanceGeometryError("route_weight must lie in (0,1]")
    plan_array = _plan_array(plan)
    if closedloop_safety.get("available") is not True:
        raise TaskRelevanceGeometryError("closed-loop actor geometry is unavailable")
    dense = _densify_plan(plan_array)
    route_xyz = np.concatenate(
        (
            dense,
            np.full(
                (dense.shape[0], 1),
                -LIDAR_HEIGHT_ABOVE_EGO_ORIGIN_M,
                dtype=np.float64,
            ),
        ),
        axis=1,
    )
    route_pixels, route_visible = project_local_points(route_xyz)
    coverage = float(np.mean(np.any(route_visible, axis=0)))
    if coverage > 0.0:
        corridor = _gaussian_points_to_grid(
            route_pixels,
            route_visible,
            tuple(map(int, patch_hw)),
            route_radius_patches,
        )
        corridor *= float(route_weight)
    else:
        corridor = np.zeros(
            (len(CAMERA_ORDER), *tuple(map(int, patch_hw))), dtype=np.float32
        )

    config = conflict_config or TrajectoryConflictConfig()
    conflict = evaluate_trajectory_conflicts(
        plan_array.tolist(), closedloop_safety, config=config
    )
    actor_ids = tuple(sorted({
        int(actor_id)
        for horizon in conflict.per_horizon_actor_ids
        for actor_id in horizon
    }))
    actor_support = _rasterize_actor_boxes(
        closedloop_safety.get("actors", []),
        closedloop_safety["ego"],
        actor_ids,
        tuple(map(int, patch_hw)),
    )
    has_route_support = bool(np.any(corridor > 0.0))
    has_actor_support = bool(np.any(actor_support > 0.0))
    if not has_route_support and not has_actor_support:
        raise TaskRelevanceGeometryError(
            "task relevance has no visible route or conflict-actor support"
        )
    if has_route_support and has_actor_support:
        support_mode = "visible_route_and_conflict_actor"
    elif has_route_support:
        support_mode = "visible_route_only"
    else:
        support_mode = "visible_conflict_actor_only"
    relevance = np.maximum(corridor, actor_support).astype(np.float32)
    return TaskRelevanceGeometryResult(
        relevance=relevance,
        route_corridor=corridor.astype(np.float32),
        relevant_actor_support=actor_support,
        relevant_actor_ids=actor_ids,
        route_point_coverage=coverage,
        provenance={
            "schema": TASK_RELEVANCE_GEOMETRY_SCHEMA,
            "source": "projected_actor_route_corridor_geometry_v1",
            "uses_observation_uq": False,
            "uses_corruption_label": False,
            "uses_recorded_ttc": False,
            "uses_collision_outcome": False,
            "uses_scenario_name": False,
            "plan_coordinate_order": ["local_right", "local_forward"],
            "route_ground_z_lidar_m": -LIDAR_HEIGHT_ABOVE_EGO_ORIGIN_M,
            "camera_order": list(CAMERA_ORDER),
            "image_hw": list(IMAGE_HW),
            "patch_hw": list(map(int, patch_hw)),
            "route_weight": float(route_weight),
            "route_radius_patches": float(route_radius_patches),
            "support_mode": support_mode,
            "visible_route_support": has_route_support,
            "visible_conflict_actor_support": has_actor_support,
            "actor_only_fallback_used": (
                has_actor_support and not has_route_support
            ),
            "future_occupancy_horizons_seconds": list(config.horizons_seconds),
            "future_occupancy_safety_margin_m": float(config.safety_margin_m),
            "projected_actor_support_is_causal_attribution": False,
        },
    )


__all__ = [
    "CAMERA_ORDER",
    "IMAGE_HW",
    "LIDAR2IMG",
    "TASK_RELEVANCE_GEOMETRY_SCHEMA",
    "TaskRelevanceGeometryError",
    "TaskRelevanceGeometryResult",
    "build_task_relevance_map",
    "project_local_points",
]
