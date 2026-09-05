"""Physical visibility belief from co-located CARLA depth cameras.

This module is deliberately NumPy-only.  It is imported by the CARLA Python
process, where importing Torch, Qwen, or Orion would violate the sidecar
boundary.  The implementation uses Qwen's ego convention throughout:

``x`` points forward, ``y`` points left, and ``z`` points up.

The generated map is not semantic occupancy and does not predict hidden
actors.  It classifies a bounded 3D volume by whether camera rays establish
free space, hit a surface, terminate before the volume, or never cover it.
Those states are collapsed over height into an inspectable 2.5D BEV.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


VISIBILITY_SCHEMA = "orion.qwen-visibility-belief/v1"
OBSERVATION_MEMORY_SCHEMA = "orion.qwen-observation-memory/v1"
VISIBILITY_EXPOSURE_SCHEMA = "orion.qwen-visibility-exposure/v1"
CARLA_DEPTH_FAR_PLANE_M = 1000.0
_UINT24_MAX = float(256**3 - 1)


@dataclass(frozen=True)
class PinholeCamera:
    """One pinhole camera expressed in Qwen's ego coordinate frame."""

    sensor_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    origin_ego: np.ndarray
    optical_to_ego: np.ndarray

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("camera sensor_id must be non-empty")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("camera image dimensions must be positive")
        if min(float(self.fx), float(self.fy)) <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        origin = np.asarray(self.origin_ego, dtype=np.float64)
        rotation = np.asarray(self.optical_to_ego, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("camera origin_ego must be finite with shape [3]")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("camera optical_to_ego must be finite with shape [3,3]")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("camera optical_to_ego must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("camera optical_to_ego must be a proper rotation")
        object.__setattr__(self, "origin_ego", origin.copy())
        object.__setattr__(self, "optical_to_ego", rotation.copy())


@dataclass(frozen=True)
class VisibilityGridSpec:
    """Metric bounds for the oracle 3D grid and its 2.5D projection.

    Raster row zero is the far-forward edge.  Raster column zero is the
    left-most edge.  This makes the returned arrays directly displayable with
    ego forward pointing up and ego left appearing on image left.
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float
    xy_resolution_m: float
    z_resolution_m: float
    max_range_m: float
    surface_tolerance_m: float

    def __post_init__(self) -> None:
        bounds = (
            self.x_min_m,
            self.x_max_m,
            self.y_min_m,
            self.y_max_m,
            self.z_min_m,
            self.z_max_m,
        )
        if not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("visibility grid bounds must be finite")
        if not (
            self.x_max_m > self.x_min_m
            and self.y_max_m > self.y_min_m
            and self.z_max_m > self.z_min_m
        ):
            raise ValueError("visibility grid upper bounds must exceed lower bounds")
        for name in (
            "xy_resolution_m",
            "z_resolution_m",
            "max_range_m",
            "surface_tolerance_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("%s must be positive" % name)
        self._axis_count(self.x_max_m - self.x_min_m, self.xy_resolution_m, "x")
        self._axis_count(self.y_max_m - self.y_min_m, self.xy_resolution_m, "y")
        self._axis_count(self.z_max_m - self.z_min_m, self.z_resolution_m, "z")

    @staticmethod
    def _axis_count(span: float, resolution: float, name: str) -> int:
        count = float(span) / float(resolution)
        rounded = int(round(count))
        if rounded <= 0 or not math.isclose(count, rounded, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("%s span must be divisible by its resolution" % name)
        return rounded

    @property
    def shape_3d(self) -> Tuple[int, int, int]:
        return (
            self._axis_count(self.z_max_m - self.z_min_m, self.z_resolution_m, "z"),
            self._axis_count(self.x_max_m - self.x_min_m, self.xy_resolution_m, "x"),
            self._axis_count(self.y_max_m - self.y_min_m, self.xy_resolution_m, "y"),
        )

    @property
    def shape_bev(self) -> Tuple[int, int]:
        return self.shape_3d[1:]

    def centers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nz, nx, ny = self.shape_3d
        x = self.x_max_m - (np.arange(nx, dtype=np.float64) + 0.5) * self.xy_resolution_m
        y = self.y_max_m - (np.arange(ny, dtype=np.float64) + 0.5) * self.xy_resolution_m
        z = self.z_min_m + (np.arange(nz, dtype=np.float64) + 0.5) * self.z_resolution_m
        return x, y, z


@dataclass(frozen=True)
class VisibilityBelief:
    """Height-collapsed, mutually exclusive physical visibility states."""

    spec: VisibilityGridSpec
    visible_free_ratio: np.ndarray
    visible_occupied_ratio: np.ndarray
    occluded_unknown_ratio: np.ndarray
    outside_fov_ratio: np.ndarray
    frontier: np.ndarray

    def __post_init__(self) -> None:
        shape = self.spec.shape_bev
        names = (
            "visible_free_ratio",
            "visible_occupied_ratio",
            "occluded_unknown_ratio",
            "outside_fov_ratio",
        )
        arrays = []
        for name in names:
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("%s must be finite with shape %r" % (name, shape))
            if np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6):
                raise ValueError("%s must lie in [0,1]" % name)
            arrays.append(value.copy())
            object.__setattr__(self, name, value.copy())
        total = np.sum(np.stack(arrays, axis=0), axis=0)
        if not np.allclose(total, 1.0, atol=1e-5):
            raise ValueError("visibility state ratios must sum to one per BEV cell")
        frontier = np.asarray(self.frontier, dtype=bool)
        if frontier.shape != shape:
            raise ValueError("frontier must have shape %r" % (shape,))
        object.__setattr__(self, "frontier", frontier.copy())

    @property
    def u_vis(self) -> np.ndarray:
        """Task-agnostic lack-of-observation map."""

        return self.occluded_unknown_ratio.copy()

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return (
            "visible_free_ratio",
            "visible_occupied_ratio",
            "occluded_unknown_ratio",
            "outside_fov_ratio",
            "frontier",
        )

    def as_channels(self) -> np.ndarray:
        """Return the versioned BEV tensor as ``[C,H,W]`` float32."""

        return np.stack(
            [
                self.visible_free_ratio,
                self.visible_occupied_ratio,
                self.occluded_unknown_ratio,
                self.outside_fov_ratio,
                self.frontier.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)


def decode_carla_depth_bgra(
    image: np.ndarray, far_plane_m: float = CARLA_DEPTH_FAR_PLANE_M
) -> np.ndarray:
    """Decode CARLA's 24-bit BGRA depth buffer into metric camera range.

    CARLA stores the least-significant byte in R, then G and B.  Leaderboard
    camera arrays expose those bytes in BGRA memory order, so ``R=image[...,2]``.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("CARLA depth image must have shape [H,W,C>=3]")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("CARLA depth image must contain integer bytes")
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("CARLA depth bytes must lie in [0,255]")
        array = array.astype(np.uint8)
    if not math.isfinite(float(far_plane_m)) or float(far_plane_m) <= 0.0:
        raise ValueError("far_plane_m must be finite and positive")
    blue = array[..., 0].astype(np.float64)
    green = array[..., 1].astype(np.float64)
    red = array[..., 2].astype(np.float64)
    packed = red + green * 256.0 + blue * float(256**2)
    return (packed / _UINT24_MAX * float(far_plane_m)).astype(np.float32)


def encode_metric_depth_uint16_mm(
    depth_m: np.ndarray, clip_depth_m: float
) -> np.ndarray:
    """Encode metric depth as an auditable lossless-within-1mm uint16 PNG plane."""

    clip_depth_m = float(clip_depth_m)
    if not math.isfinite(clip_depth_m) or not 0.0 < clip_depth_m <= 65.535:
        raise ValueError("clip_depth_m must lie in (0,65.535]")
    depth = np.asarray(depth_m, dtype=np.float64)
    if not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError("metric depth must be finite and non-negative")
    return np.rint(np.clip(depth, 0.0, clip_depth_m) * 1000.0).astype(np.uint16)


def camera_from_carla_sensor(
    sensor_id: str, sensor: Mapping[str, object]
) -> PinholeCamera:
    """Build Qwen-frame calibration from one CARLA camera sensor config."""

    width = int(sensor["width"])
    height = int(sensor["height"])
    fov = float(sensor["fov"])
    if not 0.0 < fov < 180.0:
        raise ValueError("camera horizontal fov must lie in (0,180)")
    focal = (width / 2.0) / math.tan(math.radians(fov) / 2.0)

    # CARLA ego coordinates use x forward, y right, z up.  Qwen uses y left.
    carla_to_qwen = np.diag([1.0, -1.0, 1.0])
    origin_carla = np.asarray(
        [float(sensor["x"]), float(sensor["y"]), float(sensor["z"])],
        dtype=np.float64,
    )
    origin_qwen = carla_to_qwen @ origin_carla

    roll = math.radians(float(sensor.get("roll", 0.0)))
    pitch = math.radians(float(sensor.get("pitch", 0.0)))
    yaw = math.radians(float(sensor.get("yaw", 0.0)))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # CARLA/Unreal sensor-local (forward, right, up) to parent CARLA ego.
    sensor_to_ego_carla = np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    sensor_to_ego_qwen = (
        carla_to_qwen @ sensor_to_ego_carla @ carla_to_qwen
    )

    # Optical/OpenCV (right, down, forward) to sensor-local Qwen
    # (forward, left, up).
    optical_to_sensor_qwen = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    optical_to_ego = sensor_to_ego_qwen @ optical_to_sensor_qwen
    return PinholeCamera(
        sensor_id=str(sensor_id),
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=width / 2.0,
        cy=height / 2.0,
        origin_ego=origin_qwen,
        optical_to_ego=optical_to_ego,
    )


def visibility_grid_spec_from_mapping(
    payload: Mapping[str, object]
) -> VisibilityGridSpec:
    """Construct a strict grid spec from a JSON-compatible mapping."""

    names = (
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "z_min_m",
        "z_max_m",
        "xy_resolution_m",
        "z_resolution_m",
        "max_range_m",
        "surface_tolerance_m",
    )
    missing = [name for name in names if name not in payload]
    extra = sorted(set(payload) - set(names))
    if missing or extra:
        raise ValueError(
            "visibility grid keys mismatch: missing=%s extra=%s"
            % (missing, extra)
        )
    return VisibilityGridSpec(**{name: float(payload[name]) for name in names})


def make_colocated_depth_sensor_specs(
    rgb_sensors: Mapping[str, Mapping[str, object]],
    depth_sensor_by_rgb: Mapping[str, str],
) -> Dict[str, Dict[str, object]]:
    """Clone RGB geometry into uniquely named CARLA depth sensor specs."""

    if not depth_sensor_by_rgb:
        raise ValueError("depth_sensor_by_rgb must be non-empty")
    if len(set(depth_sensor_by_rgb.values())) != len(depth_sensor_by_rgb):
        raise ValueError("each RGB camera may have only one oracle depth sensor")
    unknown_rgb = set(depth_sensor_by_rgb.values()) - set(rgb_sensors)
    if unknown_rgb:
        raise ValueError("oracle depth mapping references unknown RGB sensors")
    collisions = set(depth_sensor_by_rgb) & set(rgb_sensors)
    if collisions:
        raise ValueError("oracle depth ids collide with RGB sensor ids")
    result: Dict[str, Dict[str, object]] = {}
    for depth_id, rgb_id in depth_sensor_by_rgb.items():
        source = dict(rgb_sensors[rgb_id])
        if source.get("type") != "sensor.camera.rgb":
            raise ValueError("oracle source %s is not an RGB camera" % rgb_id)
        source["type"] = "sensor.camera.depth"
        result[str(depth_id)] = source
    return result


def _voxel_centers(spec: VisibilityGridSpec) -> np.ndarray:
    x, y, z = spec.centers()
    zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def _adjacent_observed(observed: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(observed, dtype=bool), 1, mode="constant")
    result = np.zeros_like(observed, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            result |= padded[
                row_offset : row_offset + observed.shape[0],
                column_offset : column_offset + observed.shape[1],
            ]
    return result


def compute_visibility_belief(
    depth_by_sensor: Mapping[str, np.ndarray],
    cameras: Sequence[PinholeCamera],
    spec: VisibilityGridSpec,
    frontier_unknown_threshold: float = 0.5,
) -> VisibilityBelief:
    """Fuse metric camera ranges into one task-agnostic 2.5D visibility map."""

    if not cameras:
        raise ValueError("at least one camera is required")
    if not 0.0 <= float(frontier_unknown_threshold) <= 1.0:
        raise ValueError("frontier_unknown_threshold must lie in [0,1]")
    sensor_ids = [camera.sensor_id for camera in cameras]
    if len(sensor_ids) != len(set(sensor_ids)):
        raise ValueError("camera sensor ids must be unique")
    if set(depth_by_sensor) != set(sensor_ids):
        raise ValueError("depth images must match camera sensor ids exactly")

    points_ego = _voxel_centers(spec)
    count = len(points_ego)
    covered_any = np.zeros(count, dtype=bool)
    free_any = np.zeros(count, dtype=bool)
    occupied_any = np.zeros(count, dtype=bool)
    occluded_any = np.zeros(count, dtype=bool)

    for camera in cameras:
        depth = np.asarray(depth_by_sensor[camera.sensor_id], dtype=np.float32)
        if depth.shape != (camera.height, camera.width):
            raise ValueError(
                "depth for %s must have shape (%d,%d)"
                % (camera.sensor_id, camera.height, camera.width)
            )
        delta_ego = points_ego - camera.origin_ego[None, :]
        # Row-vector form of optical = R.T @ ego.
        points_optical = delta_ego @ camera.optical_to_ego
        optical_z = points_optical[:, 2]
        positive = optical_z > 1e-6
        safe_z = np.where(positive, optical_z, 1.0)
        pixel_u = camera.fx * points_optical[:, 0] / safe_z + camera.cx
        pixel_v = camera.fy * points_optical[:, 1] / safe_z + camera.cy
        point_range = np.linalg.norm(points_optical, axis=1)
        in_fov = (
            positive
            & (pixel_u >= 0.0)
            & (pixel_u < camera.width)
            & (pixel_v >= 0.0)
            & (pixel_v < camera.height)
            & (point_range <= spec.max_range_m)
        )
        indices = np.flatnonzero(in_fov)
        if indices.size == 0:
            continue
        columns = np.clip(np.rint(pixel_u[indices]).astype(np.int64), 0, camera.width - 1)
        rows = np.clip(np.rint(pixel_v[indices]).astype(np.int64), 0, camera.height - 1)
        measured = depth[rows, columns]
        valid_depth = np.isfinite(measured) & (measured > 0.0)
        indices = indices[valid_depth]
        measured = measured[valid_depth].astype(np.float64, copy=False)
        if indices.size == 0:
            continue
        ranges = point_range[indices]
        tolerance = spec.surface_tolerance_m
        surface_inside_range = measured <= spec.max_range_m
        before_surface = ranges < (measured - tolerance)
        at_surface = surface_inside_range & (np.abs(ranges - measured) <= tolerance)
        behind_surface = surface_inside_range & (ranges > measured + tolerance)

        covered_any[indices] = True
        free_any[indices[before_surface]] = True
        occupied_any[indices[at_surface]] = True
        occluded_any[indices[behind_surface]] = True

    # A surface observation takes precedence in a coarse voxel.  A free ray
    # from another camera rescues a point that was occluded in one view.
    visible_occupied = occupied_any
    visible_free = free_any & ~visible_occupied
    occluded_unknown = occluded_any & ~visible_occupied & ~visible_free
    outside_fov = ~covered_any

    # Every covered point must fall into exactly one state.  The tolerance
    # bands above are exhaustive for finite positive depth.
    unclassified = covered_any & ~(
        visible_free | visible_occupied | occluded_unknown
    )
    if np.any(unclassified):
        raise RuntimeError("covered visibility voxels were left unclassified")

    nz, nx, ny = spec.shape_3d
    reshape = lambda value: value.reshape(nz, nx, ny).mean(axis=0).astype(np.float32)
    free_ratio = reshape(visible_free)
    occupied_ratio = reshape(visible_occupied)
    unknown_ratio = reshape(occluded_unknown)
    outside_ratio = reshape(outside_fov)
    observed = (free_ratio + occupied_ratio) > 0.0
    frontier = (
        unknown_ratio >= float(frontier_unknown_threshold)
    ) & _adjacent_observed(observed)
    return VisibilityBelief(
        spec=spec,
        visible_free_ratio=free_ratio,
        visible_occupied_ratio=occupied_ratio,
        occluded_unknown_ratio=unknown_ratio,
        outside_fov_ratio=outside_ratio,
        frontier=frontier,
    )


def render_visibility_belief(belief: VisibilityBelief) -> np.ndarray:
    """Render the 2.5D state into an RGB uint8 image for audit artifacts."""

    shape = belief.spec.shape_bev
    rgb = np.zeros(shape + (3,), dtype=np.float32)
    # Outside FOV: charcoal; visible free: green; occupied: red;
    # occluded unknown: amber.  Ratios blend over the collapsed height band.
    rgb += belief.outside_fov_ratio[..., None] * np.asarray([24, 24, 28])
    rgb += belief.visible_free_ratio[..., None] * np.asarray([50, 150, 80])
    rgb += belief.visible_occupied_ratio[..., None] * np.asarray([220, 55, 55])
    rgb += belief.occluded_unknown_ratio[..., None] * np.asarray([240, 165, 35])
    result = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    result[belief.frontier] = np.asarray([255, 255, 255], dtype=np.uint8)
    return result


def belief_metadata(belief: VisibilityBelief) -> Dict[str, object]:
    """Return JSON-safe geometry metadata without embedding the dense arrays."""

    spec = belief.spec
    return {
        "schema": VISIBILITY_SCHEMA,
        "coordinate_frame": "qwen_ego_x_forward_y_left_z_up",
        "raster_orientation": "row0_far_forward_col0_left",
        "shape_bev": list(spec.shape_bev),
        "shape_3d": list(spec.shape_3d),
        "bounds_m": {
            "x": [float(spec.x_min_m), float(spec.x_max_m)],
            "y": [float(spec.y_min_m), float(spec.y_max_m)],
            "z": [float(spec.z_min_m), float(spec.z_max_m)],
        },
        "xy_resolution_m": float(spec.xy_resolution_m),
        "z_resolution_m": float(spec.z_resolution_m),
        "max_range_m": float(spec.max_range_m),
        "surface_tolerance_m": float(spec.surface_tolerance_m),
        "channels": list(belief.channel_names),
        "summary": {
            "visible_free_mean": float(belief.visible_free_ratio.mean()),
            "visible_occupied_mean": float(belief.visible_occupied_ratio.mean()),
            "occluded_unknown_mean": float(belief.occluded_unknown_ratio.mean()),
            "outside_fov_mean": float(belief.outside_fov_ratio.mean()),
            "frontier_cells": int(belief.frontier.sum()),
        },
    }


@dataclass(frozen=True)
class ObservationMemoryState:
    """Ego-warped observation age kept separate from current visibility U."""

    spec: VisibilityGridSpec
    age_seconds: np.ndarray
    currently_observed: np.ndarray
    ever_observed: np.ndarray
    max_age_seconds: float

    def __post_init__(self) -> None:
        shape = self.spec.shape_bev
        max_age = float(self.max_age_seconds)
        if not math.isfinite(max_age) or max_age <= 0.0:
            raise ValueError("max_age_seconds must be finite and positive")
        age = np.asarray(self.age_seconds, dtype=np.float32)
        current = np.asarray(self.currently_observed, dtype=bool)
        ever = np.asarray(self.ever_observed, dtype=bool)
        if age.shape != shape or not np.isfinite(age).all():
            raise ValueError("age_seconds must be finite with shape %r" % (shape,))
        if np.any(age < 0.0) or np.any(age > max_age + 1e-5):
            raise ValueError("age_seconds must lie in [0,max_age_seconds]")
        if current.shape != shape or ever.shape != shape:
            raise ValueError("observation masks must have shape %r" % (shape,))
        if np.any(current & ~ever):
            raise ValueError("currently observed cells must also be ever observed")
        if np.any(np.abs(age[current]) > 1e-6):
            raise ValueError("currently observed cells must have zero age")
        if np.any(np.abs(age[~ever] - max_age) > 1e-5):
            raise ValueError("never-observed cells must use the capped age value")
        object.__setattr__(self, "age_seconds", age.copy())
        object.__setattr__(self, "currently_observed", current.copy())
        object.__setattr__(self, "ever_observed", ever.copy())
        object.__setattr__(self, "max_age_seconds", max_age)

    @property
    def previously_observed(self) -> np.ndarray:
        return self.ever_observed & ~self.currently_observed

    @property
    def never_observed(self) -> np.ndarray:
        return ~self.ever_observed

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return (
            "observation_age_normalized",
            "currently_observed",
            "previously_observed",
            "never_observed",
        )

    def as_channels(self) -> np.ndarray:
        return np.stack(
            [
                self.age_seconds / self.max_age_seconds,
                self.currently_observed.astype(np.float32),
                self.previously_observed.astype(np.float32),
                self.never_observed.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)


class VisibilityObservationMemory:
    """Track when BEV cells were last observed while compensating ego motion."""

    def __init__(
        self,
        spec: VisibilityGridSpec,
        max_age_seconds: float,
        observed_ratio_threshold: float = 0.5,
    ) -> None:
        max_age = float(max_age_seconds)
        threshold = float(observed_ratio_threshold)
        if not math.isfinite(max_age) or max_age <= 0.0:
            raise ValueError("max_age_seconds must be finite and positive")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("observed_ratio_threshold must lie in (0,1]")
        self.spec = spec
        self.max_age_seconds = max_age
        self.observed_ratio_threshold = threshold
        self._state: Optional[ObservationMemoryState] = None
        self._pose: Optional[np.ndarray] = None
        self._timestamp_seconds: Optional[float] = None

    def reset(self) -> None:
        self._state = None
        self._pose = None
        self._timestamp_seconds = None

    def _warp_previous(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._state is None or self._pose is None:
            raise RuntimeError("observation memory has no previous state")
        x_centers, y_centers, _ = self.spec.centers()
        current_x, current_y = np.meshgrid(x_centers, y_centers, indexing="ij")

        current_yaw = math.radians(float(pose[2]))
        current_cos, current_sin = math.cos(current_yaw), math.sin(current_yaw)
        world_x = pose[0] + current_cos * current_x + current_sin * current_y
        world_y = pose[1] + current_sin * current_x - current_cos * current_y

        previous_yaw = math.radians(float(self._pose[2]))
        previous_cos = math.cos(previous_yaw)
        previous_sin = math.sin(previous_yaw)
        delta_x = world_x - self._pose[0]
        delta_y = world_y - self._pose[1]
        previous_x = previous_cos * delta_x + previous_sin * delta_y
        previous_y = previous_sin * delta_x - previous_cos * delta_y

        resolution = float(self.spec.xy_resolution_m)
        rows = np.floor((self.spec.x_max_m - previous_x) / resolution).astype(
            np.int64
        )
        columns = np.floor((self.spec.y_max_m - previous_y) / resolution).astype(
            np.int64
        )
        valid = (
            (rows >= 0)
            & (rows < self.spec.shape_bev[0])
            & (columns >= 0)
            & (columns < self.spec.shape_bev[1])
        )
        warped_age = np.full(
            self.spec.shape_bev, self.max_age_seconds, dtype=np.float32
        )
        warped_ever = np.zeros(self.spec.shape_bev, dtype=bool)
        warped_age[valid] = self._state.age_seconds[rows[valid], columns[valid]]
        warped_ever[valid] = self._state.ever_observed[rows[valid], columns[valid]]
        return warped_age, warped_ever

    def update(
        self,
        belief: VisibilityBelief,
        ego_world_pose_xy_yaw_degrees: Sequence[float],
        timestamp_seconds: float,
    ) -> ObservationMemoryState:
        if belief.spec != self.spec:
            raise ValueError("belief grid does not match observation memory grid")
        pose = np.asarray(ego_world_pose_xy_yaw_degrees, dtype=np.float64)
        timestamp = float(timestamp_seconds)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("ego pose must be finite [world_x,world_y,yaw_degrees]")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_seconds must be finite")
        if self._timestamp_seconds is not None and timestamp < self._timestamp_seconds:
            raise ValueError("observation timestamps must be monotonic")

        currently_observed = (
            belief.visible_free_ratio + belief.visible_occupied_ratio
        ) >= self.observed_ratio_threshold
        if self._state is None:
            age = np.full(
                self.spec.shape_bev, self.max_age_seconds, dtype=np.float32
            )
            ever = currently_observed.copy()
        else:
            warped_age, ever = self._warp_previous(pose)
            delta_seconds = timestamp - float(self._timestamp_seconds)
            age = np.minimum(
                warped_age.astype(np.float64) + delta_seconds,
                self.max_age_seconds,
            ).astype(np.float32)
        ever |= currently_observed
        age[currently_observed] = 0.0
        age[~ever] = self.max_age_seconds
        state = ObservationMemoryState(
            spec=self.spec,
            age_seconds=age,
            currently_observed=currently_observed,
            ever_observed=ever,
            max_age_seconds=self.max_age_seconds,
        )
        self._state = state
        self._pose = pose.copy()
        self._timestamp_seconds = timestamp
        return state


@dataclass(frozen=True)
class VisibilityExposure:
    """Deterministic route/stopping exposure derived without issuing an action."""

    spec: VisibilityGridSpec
    route_distance_m: np.ndarray
    route_progress_m: np.ndarray
    stopping_margin_m: np.ndarray
    route_weight: np.ndarray
    stopping_weight: np.ndarray
    urgency: np.ndarray
    stopping_distance_m: float

    def __post_init__(self) -> None:
        shape = self.spec.shape_bev
        bounded = ("route_weight", "stopping_weight", "urgency")
        for name in (
            "route_distance_m",
            "route_progress_m",
            "stopping_margin_m",
        ) + bounded:
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("%s must be finite with shape %r" % (name, shape))
            if name in bounded and (np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6)):
                raise ValueError("%s must lie in [0,1]" % name)
            object.__setattr__(self, name, value.copy())
        stopping_distance = float(self.stopping_distance_m)
        if not math.isfinite(stopping_distance) or stopping_distance < 0.0:
            raise ValueError("stopping_distance_m must be finite and non-negative")
        object.__setattr__(self, "stopping_distance_m", stopping_distance)

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return (
            "route_distance_m",
            "route_progress_m",
            "stopping_margin_m",
            "route_weight",
            "stopping_weight",
            "urgency",
        )

    def as_channels(self) -> np.ndarray:
        return np.stack(
            [
                self.route_distance_m,
                self.route_progress_m,
                self.stopping_margin_m,
                self.route_weight,
                self.stopping_weight,
                self.urgency,
            ],
            axis=0,
        ).astype(np.float32, copy=False)


def carla_world_route_to_qwen_ego(
    route_world_xy: np.ndarray,
    ego_world_pose_xy_yaw_degrees: Sequence[float],
    max_length_m: float,
) -> np.ndarray:
    """Select the upcoming world route and express it as Qwen ``(forward,left)``."""

    route = np.asarray(route_world_xy, dtype=np.float64)
    pose = np.asarray(ego_world_pose_xy_yaw_degrees, dtype=np.float64)
    max_length = float(max_length_m)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise ValueError("route_world_xy must be finite with shape [N>=2,2]")
    if not np.isfinite(route).all():
        raise ValueError("route_world_xy must be finite with shape [N>=2,2]")
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("ego pose must be finite [world_x,world_y,yaw_degrees]")
    if not math.isfinite(max_length) or max_length <= 0.0:
        raise ValueError("max_length_m must be finite and positive")

    world_segments = route[1:] - route[:-1]
    world_length_squared = np.sum(world_segments * world_segments, axis=1)
    nonzero = world_length_squared > 1e-12
    if not np.any(nonzero):
        raise ValueError("route_world_xy must contain a non-zero segment")
    offset = pose[None, :2] - route[:-1]
    fraction = np.zeros(len(world_segments), dtype=np.float64)
    fraction[nonzero] = np.clip(
        np.sum(offset[nonzero] * world_segments[nonzero], axis=1)
        / world_length_squared[nonzero],
        0.0,
        1.0,
    )
    projections = route[:-1] + fraction[:, None] * world_segments
    distance_squared = np.sum((projections - pose[None, :2]) ** 2, axis=1)
    distance_squared[~nonzero] = np.inf
    nearest_segment = int(np.argmin(distance_squared))
    upcoming = np.concatenate(
        [projections[nearest_segment][None, :], route[nearest_segment + 1 :]],
        axis=0,
    )
    yaw = math.radians(float(pose[2]))
    delta = upcoming - pose[None, :2]
    local = np.stack(
        [
            math.cos(yaw) * delta[:, 0] + math.sin(yaw) * delta[:, 1],
            math.sin(yaw) * delta[:, 0] - math.cos(yaw) * delta[:, 1],
        ],
        axis=-1,
    )
    local = np.concatenate([np.zeros((1, 2), dtype=np.float64), local], axis=0)
    consecutive = np.concatenate(
        [np.asarray([True]), np.linalg.norm(np.diff(local, axis=0), axis=1) > 1e-6]
    )
    local = local[consecutive]
    if len(local) < 2:
        world_direction = world_segments[nearest_segment] / math.sqrt(
            world_length_squared[nearest_segment]
        )
        local_direction = np.asarray(
            [
                math.cos(yaw) * world_direction[0]
                + math.sin(yaw) * world_direction[1],
                math.sin(yaw) * world_direction[0]
                - math.cos(yaw) * world_direction[1],
            ]
        )
        local = np.asarray([[0.0, 0.0], local_direction * max_length])
    segment_lengths = np.linalg.norm(np.diff(local, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] <= max_length:
        return local.astype(np.float32)
    after = int(np.searchsorted(cumulative, max_length, side="right"))
    before = after - 1
    remaining = max_length - cumulative[before]
    fraction = remaining / segment_lengths[before]
    endpoint = local[before] + fraction * (local[after] - local[before])
    return np.concatenate([local[:after], endpoint[None, :]], axis=0).astype(
        np.float32
    )


def compute_visibility_exposure(
    belief: VisibilityBelief,
    route_ego_xy: np.ndarray,
    speed_mps: float,
    reaction_time_seconds: float,
    safe_deceleration_mps2: float,
    route_sigma_m: float,
    stopping_transition_m: float,
) -> VisibilityExposure:
    """Compute inspectable U urgency without modifying ``belief.u_vis``."""

    route = np.asarray(route_ego_xy, dtype=np.float64)
    speed = float(speed_mps)
    reaction_time = float(reaction_time_seconds)
    deceleration = float(safe_deceleration_mps2)
    route_sigma = float(route_sigma_m)
    transition = float(stopping_transition_m)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise ValueError("route_ego_xy must be finite with shape [N>=2,2]")
    if not np.isfinite(route).all():
        raise ValueError("route_ego_xy must be finite with shape [N>=2,2]")
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    for name, value in (
        ("reaction_time_seconds", reaction_time),
        ("safe_deceleration_mps2", deceleration),
        ("route_sigma_m", route_sigma),
        ("stopping_transition_m", transition),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be finite and positive" % name)

    segments = route[1:] - route[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    valid_segments = lengths > 1e-6
    if not np.any(valid_segments):
        raise ValueError("route_ego_xy must contain a non-zero segment")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])[:-1]
    starts = route[:-1][valid_segments]
    segments = segments[valid_segments]
    lengths = lengths[valid_segments]
    cumulative = cumulative[valid_segments]

    x_centers, y_centers, _ = belief.spec.centers()
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    best_distance_squared = np.full(len(points), np.inf, dtype=np.float64)
    best_progress = np.zeros(len(points), dtype=np.float64)
    for start, segment, length, progress in zip(
        starts, segments, lengths, cumulative
    ):
        offset = points - start[None, :]
        fraction = np.clip(
            np.sum(offset * segment[None, :], axis=1) / (length * length),
            0.0,
            1.0,
        )
        projection = start[None, :] + fraction[:, None] * segment[None, :]
        distance_squared = np.sum((points - projection) ** 2, axis=1)
        replace = distance_squared < best_distance_squared
        best_distance_squared[replace] = distance_squared[replace]
        best_progress[replace] = progress + fraction[replace] * length

    shape = belief.spec.shape_bev
    route_distance = np.sqrt(best_distance_squared).reshape(shape)
    route_progress = best_progress.reshape(shape)
    stopping_distance = speed * reaction_time + speed * speed / (2.0 * deceleration)
    stopping_margin = route_progress - stopping_distance
    route_weight = np.exp(-0.5 * (route_distance / route_sigma) ** 2)
    stopping_weight = np.clip(
        (stopping_distance + transition - route_progress) / transition,
        0.0,
        1.0,
    )
    ahead = route_progress > 1e-6
    stopping_weight *= ahead
    frontier_strength = (
        belief.frontier.astype(np.float64) * belief.occluded_unknown_ratio
    )
    urgency = frontier_strength * route_weight * stopping_weight
    return VisibilityExposure(
        spec=belief.spec,
        route_distance_m=route_distance,
        route_progress_m=route_progress,
        stopping_margin_m=stopping_margin,
        route_weight=route_weight,
        stopping_weight=stopping_weight,
        urgency=urgency,
        stopping_distance_m=stopping_distance,
    )


def render_observation_memory(state: ObservationMemoryState) -> np.ndarray:
    """Render current/previous/never observation states into RGB."""

    normalized_age = state.age_seconds / state.max_age_seconds
    rgb = np.zeros(state.spec.shape_bev + (3,), dtype=np.float32)
    rgb[state.never_observed] = np.asarray([28, 28, 32], dtype=np.float32)
    previous = state.previously_observed
    rgb[..., 0] += previous * normalized_age * 220.0
    rgb[..., 1] += previous * (1.0 - normalized_age) * 120.0
    rgb[..., 2] += previous * (1.0 - normalized_age) * 255.0
    rgb[state.currently_observed] = np.asarray([50, 190, 90], dtype=np.float32)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def render_visibility_exposure(exposure: VisibilityExposure) -> np.ndarray:
    """Render route proximity, stopping envelope, and frontier urgency."""

    rgb = np.zeros(exposure.spec.shape_bev + (3,), dtype=np.float32)
    rgb[..., 1] = exposure.route_weight * 130.0
    rgb[..., 2] = exposure.stopping_weight * exposure.route_weight * 150.0
    rgb[..., 0] = exposure.urgency * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def observation_memory_metadata(state: ObservationMemoryState) -> Dict[str, object]:
    """Return JSON-safe temporal observation metadata."""

    return {
        "schema": OBSERVATION_MEMORY_SCHEMA,
        "channels": list(state.channel_names),
        "max_age_seconds": float(state.max_age_seconds),
        "summary": {
            "currently_observed_cells": int(state.currently_observed.sum()),
            "previously_observed_cells": int(state.previously_observed.sum()),
            "never_observed_cells": int(state.never_observed.sum()),
            "maximum_seen_age_seconds": float(
                state.age_seconds[state.ever_observed].max()
                if np.any(state.ever_observed)
                else 0.0
            ),
        },
    }


def visibility_exposure_metadata(exposure: VisibilityExposure) -> Dict[str, object]:
    """Return JSON-safe route/stopping exposure metadata."""

    return {
        "schema": VISIBILITY_EXPOSURE_SCHEMA,
        "channels": list(exposure.channel_names),
        "stopping_distance_m": float(exposure.stopping_distance_m),
        "summary": {
            "urgent_frontier_cells": int((exposure.urgency > 0.0).sum()),
            "urgency_mean": float(exposure.urgency.mean()),
            "urgency_max": float(exposure.urgency.max()),
            "minimum_frontier_stopping_margin_m": float(
                exposure.stopping_margin_m[exposure.urgency > 0.0].min()
                if np.any(exposure.urgency > 0.0)
                else 0.0
            ),
        },
    }
