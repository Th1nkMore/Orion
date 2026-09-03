"""Dynamics-aware privileged expert for Stage-2 yield supervision.

The rejected v3 path clamp changed only distant waypoints.  ORION's existing
PID estimates desired speed from the first two waypoints, so the ego could keep
accelerating and stop inside a crossing lane.  This module instead combines a
map-derived junction entry with a conservative stopping-distance test and a
time-parameterized longitudinal profile along the original ORION path.

It remains a privileged development expert: CARLA actor state supplies dynamic
conflict and map topology supplies the safe waiting boundary.  Neither is an
observation-UQ target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence

from uq_estimator.privileged_yield_labels import (
    DEFAULT_HORIZONS_SECONDS,
    TrajectoryConflictResult,
    YIELD_STATES,
)


SCHEMA_VERSION = "orion.dynamic_yield_expert.v2"
_EPSILON = 1e-9


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _plan(value) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("base plan must be non-empty")
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("base plan points must be XY pairs")
        points.append((_finite(point[0], "plan"), _finite(point[1], "plan")))
    return tuple(points)


@dataclass(frozen=True)
class DynamicYieldExpertConfig:
    horizons_seconds: tuple[float, ...] = DEFAULT_HORIZONS_SECONDS
    certified_deceleration_mps2: float = 3.0
    reaction_seconds: float = 0.1
    junction_front_clearance_m: float = 0.5
    clearance_seconds: float = 1.0
    release_seconds: float = 0.5
    prepare_creep_speed_mps: float = 1.0
    release_creep_speed_mps: float = 0.5
    release_creep_distance_m: float = 1.0

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons_seconds)
        if not horizons or any(value <= 0 or not math.isfinite(value) for value in horizons):
            raise ValueError("horizons must be finite and positive")
        if any(b <= a for a, b in zip(horizons, horizons[1:])):
            raise ValueError("horizons must be strictly increasing")
        positive = (
            self.certified_deceleration_mps2,
            self.clearance_seconds,
            self.release_seconds,
            self.prepare_creep_speed_mps,
            self.release_creep_speed_mps,
            self.release_creep_distance_m,
        )
        if any(value <= 0 or not math.isfinite(value) for value in positive):
            raise ValueError("expert positive thresholds are invalid")
        nonnegative = (self.reaction_seconds, self.junction_front_clearance_m)
        if any(value < 0 or not math.isfinite(value) for value in nonnegative):
            raise ValueError("expert non-negative thresholds are invalid")
        object.__setattr__(self, "horizons_seconds", horizons)


@dataclass(frozen=True)
class JunctionYieldGeometry:
    junction_entry_path_distance_m: float
    ego_forward_extent_m: float
    safe_center_stop_distance_m: float
    conservative_stopping_distance_m: float
    brake_required: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DynamicExpertLabel:
    state: str
    state_index: int
    reason: str
    conflict_present: bool
    critical_actor_id: int | None
    clearance_elapsed_seconds: float
    release_elapsed_seconds: float
    geometry: JunctionYieldGeometry

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(frozen=True)
class PathJunctionEntry:
    distance_m: float | None
    world_xy: tuple[float, float] | None
    ego_is_junction: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class JunctionConflictResolution:
    junction_scoped_conflict: bool
    effective_conflict: TrajectoryConflictResult
    geometry_entry_distance_m: float


def resolve_junction_scoped_conflict(
    raw_conflict: TrajectoryConflictResult,
    junction_entry: PathJunctionEntry | None,
    *,
    expert_state: str,
) -> JunctionConflictResolution:
    """Bind actor conflict to a map-safe boundary without losing a latched hold."""

    if expert_state not in YIELD_STATES:
        raise ValueError(f"invalid expert state: {expert_state}")
    bounded = bool(
        junction_entry is not None
        and (
            junction_entry.distance_m is not None
            or junction_entry.ego_is_junction
        )
    )
    scoped = bool(
        raw_conflict.has_conflict
        and (expert_state != "go" or bounded)
    )
    effective = (
        raw_conflict if scoped else suppress_unbounded_conflict(raw_conflict)
    )
    if junction_entry is not None and junction_entry.distance_m is not None:
        geometry_entry = junction_entry.distance_m
    elif expert_state != "go":
        geometry_entry = 0.0
    else:
        geometry_entry = 1_000_000.0
    return JunctionConflictResolution(scoped, effective, geometry_entry)


def first_path_junction_entry(
    world_points: Sequence[Sequence[float]],
    is_junction_at_xy: Callable[[tuple[float, float]], bool],
    *,
    resolution_m: float = 0.1,
    refinement_iterations: int = 14,
) -> PathJunctionEntry:
    """Find the first non-junction to junction transition on a world path."""

    points = tuple(
        (_finite(point[0], "world point"), _finite(point[1], "world point"))
        for point in world_points
    )
    if not points:
        raise ValueError("world path must be non-empty")
    resolution = _finite(resolution_m, "resolution_m")
    if resolution <= 0 or refinement_iterations < 0:
        raise ValueError("invalid junction sampling parameters")
    ego_is_junction = bool(is_junction_at_xy(points[0]))
    if ego_is_junction:
        return PathJunctionEntry(0.0, points[0], True)

    arc_before = 0.0
    previous_distance = 0.0
    previous_xy = points[0]
    previous_is_junction = False
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= _EPSILON:
            continue
        count = max(1, int(math.ceil(length / resolution)))
        for sample_index in range(1, count + 1):
            fraction = sample_index / count
            xy = (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
            distance = arc_before + fraction * length
            current_is_junction = bool(is_junction_at_xy(xy))
            if current_is_junction and not previous_is_junction:
                false_distance, false_xy = previous_distance, previous_xy
                true_distance, true_xy = distance, xy
                for _ in range(refinement_iterations):
                    mid_distance = 0.5 * (false_distance + true_distance)
                    mid_xy = (
                        0.5 * (false_xy[0] + true_xy[0]),
                        0.5 * (false_xy[1] + true_xy[1]),
                    )
                    if is_junction_at_xy(mid_xy):
                        true_distance, true_xy = mid_distance, mid_xy
                    else:
                        false_distance, false_xy = mid_distance, mid_xy
                return PathJunctionEntry(true_distance, true_xy, False)
            previous_distance = distance
            previous_xy = xy
            previous_is_junction = current_is_junction
        arc_before += length
    return PathJunctionEntry(None, None, False)


def suppress_unbounded_conflict(
    conflict: TrajectoryConflictResult,
) -> TrajectoryConflictResult:
    """Preserve native planning when a conflict has no safe map boundary."""

    count = len(conflict.per_horizon_conflict)
    return TrajectoryConflictResult(
        per_horizon_conflict=(False,) * count,
        per_horizon_min_gap_m=(None,) * count,
        per_horizon_actor_ids=((),) * count,
        earliest_conflict_seconds=None,
        minimum_gap_m=None,
        critical_actor_id=None,
        conflict_path_distance_m=None,
        base_plan_world_xy=conflict.base_plan_world_xy,
    )


def conservative_stopping_distance(
    speed_mps: float,
    *,
    deceleration_mps2: float,
    reaction_seconds: float,
) -> float:
    speed = max(0.0, _finite(speed_mps, "speed_mps"))
    deceleration = _finite(deceleration_mps2, "deceleration_mps2")
    reaction = _finite(reaction_seconds, "reaction_seconds")
    if deceleration <= 0 or reaction < 0:
        raise ValueError("invalid stopping-distance parameters")
    return speed * reaction + speed * speed / (2.0 * deceleration)


def compute_junction_yield_geometry(
    junction_entry_path_distance_m: float,
    ego_forward_extent_m: float,
    speed_mps: float,
    *,
    config: DynamicYieldExpertConfig | None = None,
) -> JunctionYieldGeometry:
    config = config or DynamicYieldExpertConfig()
    entry = _finite(junction_entry_path_distance_m, "junction entry distance")
    extent = _finite(ego_forward_extent_m, "ego forward extent")
    if entry < 0 or extent < 0:
        raise ValueError("junction distance and ego extent must be non-negative")
    safe_stop = max(
        0.0,
        entry - extent - config.junction_front_clearance_m,
    )
    stopping = conservative_stopping_distance(
        speed_mps,
        deceleration_mps2=config.certified_deceleration_mps2,
        reaction_seconds=config.reaction_seconds,
    )
    return JunctionYieldGeometry(
        junction_entry_path_distance_m=entry,
        ego_forward_extent_m=extent,
        safe_center_stop_distance_m=safe_stop,
        conservative_stopping_distance_m=stopping,
        brake_required=safe_stop <= stopping + _EPSILON,
    )


class BrakingAwareYieldStateMachine:
    """Latch hold based on physical stopping feasibility, not only TTC."""

    def __init__(self, config: DynamicYieldExpertConfig | None = None) -> None:
        self.config = config or DynamicYieldExpertConfig()
        self.state = "go"
        self.previous_timestamp: float | None = None
        self.clearance_elapsed = 0.0
        self.release_elapsed = 0.0

    def update(
        self,
        conflict: TrajectoryConflictResult,
        geometry: JunctionYieldGeometry,
        timestamp_seconds: float,
    ) -> DynamicExpertLabel:
        timestamp = _finite(timestamp_seconds, "timestamp_seconds")
        if self.previous_timestamp is not None and timestamp <= self.previous_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        dt = 0.0 if self.previous_timestamp is None else timestamp - self.previous_timestamp
        self.previous_timestamp = timestamp
        if conflict.has_conflict:
            self.clearance_elapsed = 0.0
            self.release_elapsed = 0.0
            if self.state in {"hold", "release"}:
                self.state = "hold"
                reason = "dynamic_conflict_after_yield"
            elif geometry.brake_required:
                self.state = "hold"
                reason = "braking_boundary_reached"
            else:
                self.state = "prepare_yield"
                reason = "decelerate_toward_safe_wait_line"
        elif self.state == "hold":
            self.clearance_elapsed += dt
            if self.clearance_elapsed + _EPSILON >= self.config.clearance_seconds:
                self.state = "release"
                self.release_elapsed = 0.0
                reason = "conflict_clearance_confirmed_begin_creep"
            else:
                reason = "hold_for_conflict_clearance"
        elif self.state == "release":
            self.release_elapsed += dt
            if self.release_elapsed + _EPSILON >= self.config.release_seconds:
                self.state = "go"
                self.release_elapsed = 0.0
                reason = "release_clear_restore_base_plan"
            else:
                reason = "bounded_release_creep"
        else:
            self.state = "go"
            self.clearance_elapsed = 0.0
            self.release_elapsed = 0.0
            reason = "no_dynamic_conflict"
        return DynamicExpertLabel(
            state=self.state,
            state_index=YIELD_STATES.index(self.state),
            reason=reason,
            conflict_present=conflict.has_conflict,
            critical_actor_id=conflict.critical_actor_id,
            clearance_elapsed_seconds=self.clearance_elapsed,
            release_elapsed_seconds=self.release_elapsed,
            geometry=geometry,
        )


def _arc_lengths(points: Sequence[tuple[float, float]]) -> list[float]:
    result = [0.0]
    for first, second in zip(points, points[1:]):
        result.append(result[-1] + math.hypot(
            second[0] - first[0], second[1] - first[1]
        ))
    return result


def _point_at_distance(points, arc_lengths, distance):
    target = max(0.0, min(float(distance), arc_lengths[-1]))
    for index in range(1, len(points)):
        if target <= arc_lengths[index] + _EPSILON:
            segment = arc_lengths[index] - arc_lengths[index - 1]
            if segment <= _EPSILON:
                return points[index]
            fraction = (target - arc_lengths[index - 1]) / segment
            return (
                points[index - 1][0]
                + fraction * (points[index][0] - points[index - 1][0]),
                points[index - 1][1]
                + fraction * (points[index][1] - points[index - 1][1]),
            )
    return points[-1]


def build_dynamics_aware_yield_trajectory(
    base_plan_cumulative_m,
    label: DynamicExpertLabel,
    speed_mps: float,
    *,
    config: DynamicYieldExpertConfig | None = None,
) -> tuple[tuple[float, float], ...]:
    """Retimestamp the base path so its first two points command deceleration."""

    config = config or DynamicYieldExpertConfig()
    plan = _plan(base_plan_cumulative_m)
    if len(plan) != len(config.horizons_seconds):
        raise ValueError("base plan length must match expert horizons")
    if label.state == "go":
        return plan
    points = ((0.0, 0.0),) + plan
    arcs = _arc_lengths(points)
    if label.state == "release":
        progress = [
            min(
                config.release_creep_speed_mps * horizon,
                config.release_creep_distance_m,
            )
            for horizon in config.horizons_seconds
        ]
    else:
        speed = max(0.0, _finite(speed_mps, "speed_mps"))
        stop_distance = label.geometry.safe_center_stop_distance_m
        if (
            label.state == "prepare_yield"
            and speed <= 0.05
            and stop_distance > _EPSILON
        ):
            progress = [
                min(
                    config.prepare_creep_speed_mps * horizon,
                    stop_distance,
                )
                for horizon in config.horizons_seconds
            ]
        elif speed <= 0.05 or stop_distance <= _EPSILON:
            progress = [0.0] * len(config.horizons_seconds)
        else:
            required_deceleration = speed * speed / max(
                2.0 * stop_distance, _EPSILON
            )
            deceleration = max(
                config.certified_deceleration_mps2,
                required_deceleration,
            )
            progress = [
                min(
                    stop_distance,
                    max(0.0, speed * horizon - 0.5 * deceleration * horizon * horizon),
                )
                for horizon in config.horizons_seconds
            ]
            # Numerical saturation can otherwise make a late sample decrease.
            for index in range(1, len(progress)):
                progress[index] = max(progress[index], progress[index - 1])
    return tuple(
        _point_at_distance(points, arcs, min(distance, arcs[-1]))
        for distance in progress
    )


def pid_desired_speed_proxy(trajectory) -> float:
    """Match the existing ORION PID's speed estimate from its first two points."""

    plan = _plan(trajectory)
    if len(plan) < 2:
        raise ValueError("PID speed proxy requires at least two points")
    first = math.hypot(*plan[0])
    delta = math.hypot(
        plan[1][0] - plan[0][0], plan[1][1] - plan[0][1]
    )
    return 1.5 * first + 0.5 * delta
