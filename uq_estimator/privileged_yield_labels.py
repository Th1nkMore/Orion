"""Privileged task-risk labels for a planning-layer yield response.

These labels are deliberately independent of observation uncertainty.  They
use simulator actor state and ORION's unmodified candidate trajectory to ask a
counterfactual task question: would following the candidate trajectory occupy
the same space as another actor at the same future time?  The resulting
``go/prepare_yield/hold/release`` state is Stage-2 supervision; it must never be
used as a Stage-1 observation-UQ target.

The implementation has no CARLA dependency.  It consumes the raw ego/actor
geometry already recorded in ``closedloop_safety`` telemetry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

from uq_estimator.closedloop_safety_metrics import obb_separating_axis_gap


SCHEMA_VERSION = "orion.privileged_dynamic_yield.v1"
DEFAULT_HORIZONS_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
YIELD_STATES = ("go", "prepare_yield", "hold", "release")
_EPSILON = 1e-9


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xy(value, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return _finite(value[0], name), _finite(value[1], name)


def _validated_plan(plan) -> tuple[tuple[float, float], ...]:
    if not isinstance(plan, (list, tuple)) or not plan:
        raise ValueError("base_plan_cumulative_m must be a non-empty sequence")
    return tuple(_xy(point, "base_plan_cumulative_m") for point in plan)


def select_actor_categories(
    closedloop_safety: Mapping,
    categories: Sequence[str],
) -> dict:
    """Return a planning-only view of telemetry for frozen actor classes.

    The full telemetry remains available for evaluation.  This selector only
    prevents a scenario-specific privileged oracle from silently reacting to
    unrelated traffic (for example adjacent-lane vehicles in a pedestrian
    crossing experiment).
    """

    if not isinstance(closedloop_safety, Mapping):
        raise ValueError("closedloop_safety must be a mapping")
    normalized = tuple(
        str(category).strip().lower() for category in categories
        if str(category).strip()
    )
    if not normalized:
        raise ValueError("categories must contain at least one actor category")
    if len(set(normalized)) != len(normalized):
        raise ValueError("categories must not contain duplicates")
    actors = closedloop_safety.get("actors", [])
    if not isinstance(actors, list):
        raise ValueError("closedloop_safety.actors must be a list")
    selected = [
        actor for actor in actors
        if isinstance(actor, Mapping)
        and str(actor.get("category", "")).strip().lower() in normalized
    ]
    result = dict(closedloop_safety)
    result["actors"] = selected
    result["actor_count_considered"] = len(selected)
    result["planning_actor_categories"] = list(normalized)
    return result


@dataclass(frozen=True)
class TrajectoryConflictConfig:
    """Geometry and state thresholds frozen before an offline audit."""

    horizons_seconds: tuple[float, ...] = DEFAULT_HORIZONS_SECONDS
    interpolation_step_seconds: float = 0.1
    safety_margin_m: float = 0.75
    imminent_horizon_seconds: float = 1.5
    clearance_seconds: float = 1.0
    release_seconds: float = 0.5
    stop_buffer_m: float = 2.0
    release_creep_distance_m: float = 1.0

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons_seconds)
        if not horizons or any(not math.isfinite(value) for value in horizons):
            raise ValueError("horizons_seconds must be finite and non-empty")
        if any(value <= 0.0 for value in horizons):
            raise ValueError("horizons_seconds must be positive")
        if any(right <= left for left, right in zip(horizons, horizons[1:])):
            raise ValueError("horizons_seconds must be strictly increasing")
        positive = (
            self.interpolation_step_seconds,
            self.imminent_horizon_seconds,
            self.clearance_seconds,
            self.release_seconds,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in positive):
            raise ValueError("time thresholds must be finite and positive")
        if not math.isfinite(self.safety_margin_m) or self.safety_margin_m < 0:
            raise ValueError("safety_margin_m must be finite and non-negative")
        distances = (self.stop_buffer_m, self.release_creep_distance_m)
        if any(not math.isfinite(value) or value < 0 for value in distances):
            raise ValueError("distance thresholds must be finite and non-negative")
        object.__setattr__(self, "horizons_seconds", horizons)


@dataclass(frozen=True)
class TrajectoryConflictResult:
    """Time-aligned occupancy conflict along one unmodified ORION plan."""

    per_horizon_conflict: tuple[bool, ...]
    per_horizon_min_gap_m: tuple[float | None, ...]
    per_horizon_actor_ids: tuple[tuple[int, ...], ...]
    earliest_conflict_seconds: float | None
    minimum_gap_m: float | None
    critical_actor_id: int | None
    conflict_path_distance_m: float | None
    base_plan_world_xy: tuple[tuple[float, float], ...]

    @property
    def has_conflict(self) -> bool:
        return self.earliest_conflict_seconds is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class YieldLabel:
    """State-machine label plus the current safe stopping constraint."""

    state: str
    state_index: int
    conflict_present: bool
    imminent_conflict: bool
    clearance_elapsed_seconds: float
    release_elapsed_seconds: float
    stop_path_distance_m: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def orion_local_plan_to_world(
    base_plan_cumulative_m: Sequence[Sequence[float]],
    ego_state: Mapping,
) -> tuple[tuple[float, float], ...]:
    """Convert ORION ``[right, forward]`` points to CARLA world XY."""

    plan = _validated_plan(base_plan_cumulative_m)
    ego_x, ego_y = _xy(ego_state.get("position_xy"), "ego.position_xy")
    yaw = math.radians(_finite(ego_state.get("yaw_degrees", 0.0), "ego.yaw"))
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    return tuple(
        (
            ego_x + local_right * right[0] + local_forward * forward[0],
            ego_y + local_right * right[1] + local_forward * forward[1],
        )
        for local_right, local_forward in plan
    )


def _polyline_arc_lengths(points: Sequence[tuple[float, float]]) -> list[float]:
    lengths = [0.0]
    for first, second in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(
            second[0] - first[0], second[1] - first[1]
        ))
    return lengths


def evaluate_trajectory_conflicts(
    base_plan_cumulative_m: Sequence[Sequence[float]],
    closedloop_safety: Mapping,
    *,
    config: TrajectoryConflictConfig | None = None,
) -> TrajectoryConflictResult:
    """Evaluate synchronous actor occupancy against a candidate trajectory.

    Actors follow the privileged constant-velocity state recorded by CARLA.
    The ego follows the piecewise-linear ORION candidate.  Fixed-orientation
    OBB gaps are sampled densely and grouped into the decoder's six horizons.
    """

    config = config or TrajectoryConflictConfig()
    plan = _validated_plan(base_plan_cumulative_m)
    if len(plan) != len(config.horizons_seconds):
        raise ValueError("plan length must equal the configured horizon count")
    if not isinstance(closedloop_safety, Mapping):
        raise ValueError("closedloop_safety must be a mapping")
    if closedloop_safety.get("available") is False:
        raise ValueError("closedloop_safety telemetry is unavailable")
    ego = closedloop_safety.get("ego")
    if not isinstance(ego, Mapping):
        raise ValueError("closedloop_safety.ego is required")
    actors = closedloop_safety.get("actors", [])
    if not isinstance(actors, list):
        raise ValueError("closedloop_safety.actors must be a list")

    ego_origin = _xy(ego.get("position_xy"), "ego.position_xy")
    world_plan = orion_local_plan_to_world(plan, ego)
    world_points = (ego_origin,) + world_plan
    arc_lengths = _polyline_arc_lengths(world_points)
    ego_extent = _xy(ego.get("extent_xy_m"), "ego.extent_xy_m")
    fallback_yaw = _finite(ego.get("yaw_degrees", 0.0), "ego.yaw")

    per_conflict: list[bool] = []
    per_min_gap: list[float | None] = []
    per_actor_ids: list[tuple[int, ...]] = []
    earliest: float | None = None
    minimum_gap: float | None = None
    critical_actor: int | None = None
    conflict_path_distance: float | None = None

    previous_time = 0.0
    for horizon_index, horizon_time in enumerate(config.horizons_seconds):
        start = world_points[horizon_index]
        end = world_points[horizon_index + 1]
        segment_duration = horizon_time - previous_time
        sample_count = max(
            1,
            int(math.ceil(segment_duration / config.interpolation_step_seconds)),
        )
        segment_yaw = fallback_yaw
        if math.hypot(end[0] - start[0], end[1] - start[1]) > _EPSILON:
            segment_yaw = math.degrees(math.atan2(
                end[1] - start[1], end[0] - start[0]
            ))
        horizon_min_gap: float | None = None
        horizon_conflict_actors: set[int] = set()

        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            sample_time = previous_time + fraction * segment_duration
            ego_position = (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
            path_distance = (
                arc_lengths[horizon_index]
                + fraction
                * (arc_lengths[horizon_index + 1] - arc_lengths[horizon_index])
            )
            for actor in actors:
                if not isinstance(actor, Mapping):
                    raise ValueError("each actor telemetry record must be a mapping")
                actor_id = int(actor.get("actor_id"))
                actor_position = _xy(actor.get("position_xy"), "actor.position_xy")
                actor_velocity = _xy(actor.get("velocity_xy"), "actor.velocity_xy")
                predicted_actor = (
                    actor_position[0] + actor_velocity[0] * sample_time,
                    actor_position[1] + actor_velocity[1] * sample_time,
                )
                relative = (
                    predicted_actor[0] - ego_position[0],
                    predicted_actor[1] - ego_position[1],
                )
                gap = obb_separating_axis_gap(
                    relative,
                    ego_extent,
                    segment_yaw,
                    _xy(actor.get("extent_xy_m"), "actor.extent_xy_m"),
                    _finite(actor.get("yaw_degrees", 0.0), "actor.yaw"),
                )
                if horizon_min_gap is None or gap < horizon_min_gap:
                    horizon_min_gap = gap
                if gap <= config.safety_margin_m:
                    horizon_conflict_actors.add(actor_id)
                    if earliest is None or sample_time < earliest:
                        earliest = sample_time
                        critical_actor = actor_id
                        conflict_path_distance = path_distance
                    if minimum_gap is None or gap < minimum_gap:
                        minimum_gap = gap
                        if earliest is not None and sample_time <= earliest + _EPSILON:
                            critical_actor = actor_id

        per_conflict.append(bool(horizon_conflict_actors))
        per_min_gap.append(horizon_min_gap)
        per_actor_ids.append(tuple(sorted(horizon_conflict_actors)))
        previous_time = horizon_time

    return TrajectoryConflictResult(
        per_horizon_conflict=tuple(per_conflict),
        per_horizon_min_gap_m=tuple(per_min_gap),
        per_horizon_actor_ids=tuple(per_actor_ids),
        earliest_conflict_seconds=earliest,
        minimum_gap_m=minimum_gap,
        critical_actor_id=critical_actor,
        conflict_path_distance_m=conflict_path_distance,
        base_plan_world_xy=world_plan,
    )


class DynamicYieldLabeler:
    """Stateful release logic driven by current conflict, never wall duration."""

    def __init__(self, config: TrajectoryConflictConfig | None = None) -> None:
        self.config = config or TrajectoryConflictConfig()
        self.state = "go"
        self.previous_timestamp: float | None = None
        self.clearance_elapsed = 0.0
        self.release_elapsed = 0.0
        self.stop_path_distance: float | None = None

    def update(
        self,
        conflict: TrajectoryConflictResult,
        timestamp_seconds: float,
    ) -> YieldLabel:
        timestamp = _finite(timestamp_seconds, "timestamp_seconds")
        if self.previous_timestamp is not None and timestamp <= self.previous_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        dt = 0.0 if self.previous_timestamp is None else timestamp - self.previous_timestamp
        self.previous_timestamp = timestamp
        has_conflict = conflict.has_conflict
        imminent = bool(
            has_conflict
            and conflict.earliest_conflict_seconds
            <= self.config.imminent_horizon_seconds
        )

        if has_conflict:
            self.clearance_elapsed = 0.0
            self.release_elapsed = 0.0
            self.stop_path_distance = max(
                0.0,
                float(conflict.conflict_path_distance_m or 0.0)
                - self.config.stop_buffer_m,
            )
            if self.state in {"hold", "release"}:
                self.state = "hold"
                reason = "conflict_present_after_yield"
            elif imminent:
                self.state = "hold"
                reason = "imminent_trajectory_conflict"
            else:
                self.state = "prepare_yield"
                reason = "future_trajectory_conflict"
        elif self.state == "hold":
            self.clearance_elapsed += dt
            if self.clearance_elapsed + _EPSILON >= self.config.clearance_seconds:
                self.state = "release"
                self.release_elapsed = 0.0
                self.stop_path_distance = self.config.release_creep_distance_m
                reason = "conflict_clearance_confirmed"
            else:
                reason = "hold_for_clearance_hysteresis"
        elif self.state == "release":
            self.release_elapsed += dt
            if self.release_elapsed + _EPSILON >= self.config.release_seconds:
                self.state = "go"
                self.release_elapsed = 0.0
                self.stop_path_distance = None
                reason = "release_completed"
            else:
                reason = "release_in_progress"
        else:
            self.state = "go"
            self.clearance_elapsed = 0.0
            self.release_elapsed = 0.0
            self.stop_path_distance = None
            reason = "no_trajectory_conflict"

        return YieldLabel(
            state=self.state,
            state_index=YIELD_STATES.index(self.state),
            conflict_present=has_conflict,
            imminent_conflict=imminent,
            clearance_elapsed_seconds=self.clearance_elapsed,
            release_elapsed_seconds=self.release_elapsed,
            stop_path_distance_m=self.stop_path_distance,
            reason=reason,
        )


def _point_at_arc_length(
    points: Sequence[tuple[float, float]],
    arc_lengths: Sequence[float],
    target_distance: float,
) -> tuple[float, float]:
    distance = max(0.0, min(float(target_distance), arc_lengths[-1]))
    for index in range(1, len(points)):
        if distance <= arc_lengths[index] + _EPSILON:
            segment = arc_lengths[index] - arc_lengths[index - 1]
            if segment <= _EPSILON:
                return points[index]
            fraction = (distance - arc_lengths[index - 1]) / segment
            return (
                points[index - 1][0]
                + fraction * (points[index][0] - points[index - 1][0]),
                points[index - 1][1]
                + fraction * (points[index][1] - points[index - 1][1]),
            )
    return points[-1]


def build_safe_yield_trajectory(
    base_plan_cumulative_m: Sequence[Sequence[float]],
    label: YieldLabel,
) -> tuple[tuple[float, float], ...]:
    """Clamp path progress before a conflict while preserving path geometry.

    ``go`` retains the untouched ORION candidate.  Preparation and hold stop
    along that same path before the privileged conflict.  ``release`` permits
    only a short creep until a second conflict-free interval has elapsed, so
    sequential cross traffic is re-evaluated before full progress resumes.
    """

    plan = _validated_plan(base_plan_cumulative_m)
    if label.state == "go":
        return plan
    if label.stop_path_distance_m is None:
        raise ValueError("yielding states require stop_path_distance_m")
    points = ((0.0, 0.0),) + plan
    arc_lengths = _polyline_arc_lengths(points)
    return tuple(
        _point_at_arc_length(points, arc_lengths, min(distance, label.stop_path_distance_m))
        for distance in arc_lengths[1:]
    )


def trajectory_residual(
    base_plan_cumulative_m: Sequence[Sequence[float]],
    target_plan_cumulative_m: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    base = _validated_plan(base_plan_cumulative_m)
    target = _validated_plan(target_plan_cumulative_m)
    if len(base) != len(target):
        raise ValueError("base and target trajectories must have equal length")
    return tuple(
        (target_point[0] - base_point[0], target_point[1] - base_point[1])
        for base_point, target_point in zip(base, target)
    )


def export_run_labels(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TrajectoryConflictConfig | None = None,
    save_stride_steps: int = 10,
    audit_time_seconds: float | None = None,
) -> dict:
    """Export 2 Hz privileged labels from one recorded closed-loop run."""

    config = config or TrajectoryConflictConfig()
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    trace_paths = list(run_dir.rglob("control_trace.jsonl"))
    meta_dirs = [path for path in run_dir.rglob("meta") if path.is_dir()]
    if len(trace_paths) != 1 or len(meta_dirs) != 1:
        raise ValueError("run_dir must contain exactly one control trace and meta directory")
    trace_by_step = {}
    with trace_paths[0].open() as handle:
        for line in handle:
            row = json.loads(line)
            trace_by_step[int(row["step"])] = row
    labeler = DynamicYieldLabeler(config)
    records = []
    for meta_path in sorted(meta_dirs[0].glob("*.json")):
        frame = int(meta_path.stem)
        step = frame * int(save_stride_steps)
        if step not in trace_by_step:
            raise ValueError(f"missing trace row for saved step {step}")
        trace = trace_by_step[step]
        meta = json.loads(meta_path.read_text())
        base_plan = meta.get("plan")
        conflict = evaluate_trajectory_conflicts(
            base_plan,
            meta.get("closedloop_safety"),
            config=config,
        )
        timestamp = _finite(trace.get("sim_time_seconds"), "sim_time_seconds")
        label = labeler.update(conflict, timestamp)
        target = build_safe_yield_trajectory(base_plan, label)
        records.append({
            "schema_version": SCHEMA_VERSION,
            "source": {
                "run_dir": str(run_dir),
                "meta_path": str(meta_path),
                "frame_2hz": frame,
                "step": step,
                "sim_time_seconds": timestamp,
                "route_progress": trace.get("route_progress"),
            },
            "supervision_contract": {
                "stage": "stage2_task_risk",
                "uses_observation_uq_target": False,
                "uses_density_uq": False,
                "uses_corruption_label": False,
                "source": "privileged_actor_occupancy_x_unmodified_orion_plan",
            },
            "base_plan_cumulative_m": base_plan,
            "conflict": conflict.to_dict(),
            "yield_label": label.to_dict(),
            "safe_target_cumulative_m": target,
            "trajectory_residual_m": trajectory_residual(base_plan, target),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "privileged_yield_labels.jsonl"
    with labels_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    state_counts = {state: 0 for state in YIELD_STATES}
    for record in records:
        state_counts[record["yield_label"]["state"]] += 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "labels_path": str(labels_path),
        "record_count": len(records),
        "config": asdict(config),
        "state_counts": state_counts,
        "conflict_frames": sum(
            record["yield_label"]["conflict_present"] for record in records
        ),
        "first_prepare_time_seconds": next((
            record["source"]["sim_time_seconds"] for record in records
            if record["yield_label"]["state"] == "prepare_yield"
        ), None),
        "first_hold_time_seconds": next((
            record["source"]["sim_time_seconds"] for record in records
            if record["yield_label"]["state"] == "hold"
        ), None),
    }
    if audit_time_seconds is not None and records:
        audit_time = _finite(audit_time_seconds, "audit_time_seconds")
        nearest = min(
            records,
            key=lambda record: abs(record["source"]["sim_time_seconds"] - audit_time),
        )
        report["audit"] = {
            "requested_time_seconds": audit_time,
            "nearest_time_seconds": nearest["source"]["sim_time_seconds"],
            "state": nearest["yield_label"]["state"],
            "conflict_present": nearest["yield_label"]["conflict_present"],
            "critical_actor_id": nearest["conflict"]["critical_actor_id"],
            "first_hold_lead_seconds": (
                audit_time - report["first_hold_time_seconds"]
                if report["first_hold_time_seconds"] is not None else None
            ),
        }
    report_path = output_dir / "privileged_yield_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
