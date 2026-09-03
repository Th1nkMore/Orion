"""Braking-aware trajectory retiming for finite crossing conflicts.

The v1 bounded-crossing oracle clamped only distant waypoints.  ORION's PID
derives desired speed from the first two waypoints, so it continued to throttle
for several frames even though the tail already stopped before the walker.
This module preserves the same path and privileged conflict state while
retiming every horizon with a certified longitudinal deceleration profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from uq_estimator.dynamic_yield_expert import pid_desired_speed_proxy
from uq_estimator.privileged_yield_labels import (
    DEFAULT_HORIZONS_SECONDS,
    YieldLabel,
)


SCHEMA_VERSION = "orion.braking_aware_bounded_crossing.v2"
_EPSILON = 1e-9


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _plan(value) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("base plan must be non-empty")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("base plan points must be XY pairs")
        result.append((_finite(point[0], "plan"), _finite(point[1], "plan")))
    return tuple(result)


@dataclass(frozen=True)
class BoundedCrossingExpertConfig:
    horizons_seconds: tuple[float, ...] = DEFAULT_HORIZONS_SECONDS
    certified_deceleration_mps2: float = 3.0
    prepare_creep_speed_mps: float = 1.0
    release_creep_speed_mps: float = 0.5
    release_creep_distance_m: float = 1.0

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons_seconds)
        if not horizons or any(
            not math.isfinite(value) or value <= 0 for value in horizons
        ):
            raise ValueError("horizons must be finite and positive")
        if any(right <= left for left, right in zip(horizons, horizons[1:])):
            raise ValueError("horizons must be strictly increasing")
        positive = (
            self.certified_deceleration_mps2,
            self.prepare_creep_speed_mps,
            self.release_creep_speed_mps,
            self.release_creep_distance_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("expert thresholds must be finite and positive")
        object.__setattr__(self, "horizons_seconds", horizons)


@dataclass(frozen=True)
class BoundedCrossingBrakingProfile:
    state: str
    input_speed_mps: float
    safe_stop_path_distance_m: float | None
    kinematic_stop_distance_m: float | None
    applied_deceleration_mps2: float | None
    stop_time_seconds: float | None
    base_pid_desired_speed_mps: float
    target_pid_desired_speed_mps: float
    immediate_brake_ratio: float | None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


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


def build_braking_aware_crossing_trajectory(
    base_plan_cumulative_m,
    label: YieldLabel,
    speed_mps: float,
    *,
    config: BoundedCrossingExpertConfig | None = None,
) -> tuple[
    tuple[tuple[float, float], ...],
    BoundedCrossingBrakingProfile,
]:
    """Retimestamp an unchanged ORION path to command immediate deceleration."""

    config = config or BoundedCrossingExpertConfig()
    plan = _plan(base_plan_cumulative_m)
    if len(plan) != len(config.horizons_seconds):
        raise ValueError("base plan length must match expert horizons")
    speed = max(0.0, _finite(speed_mps, "speed_mps"))
    base_desired = pid_desired_speed_proxy(plan)
    points = ((0.0, 0.0),) + plan
    arcs = _arc_lengths(points)
    safe_stop = None
    kinematic_stop = None
    applied_deceleration = None
    stop_time = None

    if label.state == "go":
        target = plan
    elif label.state == "release":
        progress = [
            min(
                config.release_creep_speed_mps * horizon,
                config.release_creep_distance_m,
            )
            for horizon in config.horizons_seconds
        ]
        target = tuple(
            _point_at_distance(points, arcs, distance) for distance in progress
        )
    else:
        if label.stop_path_distance_m is None:
            raise ValueError("yielding state requires stop_path_distance_m")
        safe_stop = max(0.0, _finite(
            label.stop_path_distance_m, "stop_path_distance_m"
        ))
        if (
            label.state == "prepare_yield"
            and speed <= 0.05
            and safe_stop > _EPSILON
        ):
            progress = [
                min(config.prepare_creep_speed_mps * horizon, safe_stop)
                for horizon in config.horizons_seconds
            ]
        elif speed <= 0.05 or safe_stop <= _EPSILON:
            progress = [0.0] * len(config.horizons_seconds)
            kinematic_stop = 0.0
            applied_deceleration = config.certified_deceleration_mps2
            stop_time = 0.0
        else:
            required_deceleration = speed * speed / max(
                2.0 * safe_stop, _EPSILON
            )
            applied_deceleration = max(
                config.certified_deceleration_mps2,
                required_deceleration,
            )
            stop_time = speed / applied_deceleration
            kinematic_stop = min(
                safe_stop,
                speed * speed / (2.0 * applied_deceleration),
            )
            progress = []
            for horizon in config.horizons_seconds:
                if horizon >= stop_time:
                    distance = kinematic_stop
                else:
                    distance = (
                        speed * horizon
                        - 0.5 * applied_deceleration * horizon * horizon
                    )
                progress.append(min(safe_stop, max(0.0, distance)))
        target = tuple(
            _point_at_distance(points, arcs, min(distance, arcs[-1]))
            for distance in progress
        )

    target_desired = pid_desired_speed_proxy(target)
    brake_ratio = (
        speed / target_desired
        if label.state not in {"go", "release"} and target_desired > _EPSILON
        else None
    )
    return target, BoundedCrossingBrakingProfile(
        state=label.state,
        input_speed_mps=speed,
        safe_stop_path_distance_m=safe_stop,
        kinematic_stop_distance_m=kinematic_stop,
        applied_deceleration_mps2=applied_deceleration,
        stop_time_seconds=stop_time,
        base_pid_desired_speed_mps=base_desired,
        target_pid_desired_speed_mps=target_desired,
        immediate_brake_ratio=brake_ratio,
    )


__all__ = [
    "BoundedCrossingBrakingProfile",
    "BoundedCrossingExpertConfig",
    "SCHEMA_VERSION",
    "build_braking_aware_crossing_trajectory",
]
