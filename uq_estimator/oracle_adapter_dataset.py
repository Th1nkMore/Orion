"""Build strict oracle-rollout supervision for uncertainty-aware adapters.

The exporter deliberately joins artifacts *within one CARLA rollout*.  It
never aligns an ``off`` rollout with an ``oracle`` rollout after their world
states have diverged.  The resulting target trajectory is reconstructed from
the ego poses actually executed under the post-governor controls.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "orion.oracle_adapter.sample.v1"
DATASET_VERSION = "orion.oracle_adapter.dataset.v1"
SAVE_INTERVAL_STEPS = 10
SIMULATION_HZ = 20.0
HORIZON_STEPS = (10, 20, 30, 40, 50, 60)
CAMERA_DIRECTORIES = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}
CAMERA_ALIASES = {
    "front": ("CAM_FRONT",),
    "front_group": ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),
    "rear": ("CAM_BACK",),
    "rear_group": ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"),
    "all": tuple(CAMERA_DIRECTORIES),
}


class DatasetIntegrityError(ValueError):
    """Raised when closed-loop artifacts cannot support causal supervision."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIntegrityError(f"Cannot read JSON {path}: {exc}") from exc


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise DatasetIntegrityError(f"{field} must be finite, got {result!r}")
    return result


def _single(paths: Iterable[Path], label: str) -> Path:
    values = sorted(paths)
    if len(values) != 1:
        raise DatasetIntegrityError(
            f"Expected exactly one {label}, found {len(values)}: {values}"
        )
    return values[0]


def _indexed_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    by_step: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        step = int(row.get("step", -1))
        if step in by_step:
            raise DatasetIntegrityError(f"Duplicate trace step {step} in {path}")
        by_step[step] = row
        records.append(row)
    if not records:
        raise DatasetIntegrityError(f"Empty control trace: {path}")
    steps = [int(row["step"]) for row in records]
    if steps != list(range(steps[0], steps[-1] + 1)) or steps[0] != 0:
        raise DatasetIntegrityError(
            f"Trace steps must be contiguous from zero in {path}; "
            f"observed {steps[:3]}...{steps[-3:]}"
        )
    for row in records:
        step = int(row["step"])
        expected_time = step / SIMULATION_HZ
        observed_time = _finite_float(row.get("sim_time_seconds"), "sim_time_seconds")
        if abs(observed_time - expected_time) > 1e-6:
            raise DatasetIntegrityError(
                f"Trace step/time mismatch at {step}: {observed_time} vs {expected_time}"
            )
    return records, by_step


def _indexed_metric(path: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise DatasetIntegrityError(f"metric_info must be a non-empty object: {path}")
    indexed: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        try:
            step = int(key)
        except ValueError as exc:
            raise DatasetIntegrityError(f"Non-integer metric step {key!r}") from exc
        if step in indexed:
            raise DatasetIntegrityError(f"Duplicate metric step {step}")
        indexed[step] = value
    steps = sorted(indexed)
    if steps != list(range(steps[0], steps[-1] + 1)) or steps[0] != 0:
        raise DatasetIntegrityError(
            f"metric_info steps must be contiguous from zero; "
            f"observed {steps[:3]}...{steps[-3:]}"
        )
    return indexed


def _frame_files(record_root: Path) -> dict[int, dict[str, Path]]:
    directories = {**CAMERA_DIRECTORIES, "BEV": "bev", "META": "meta"}
    indexed: dict[str, dict[int, Path]] = {}
    for name, directory in directories.items():
        root = record_root / directory
        if not root.is_dir():
            raise DatasetIntegrityError(f"Missing frame directory: {root}")
        suffix = ".json" if name == "META" else ".png"
        files: dict[int, Path] = {}
        for path in sorted(root.glob(f"*{suffix}")):
            try:
                frame = int(path.stem)
            except ValueError as exc:
                raise DatasetIntegrityError(f"Non-integer frame name: {path}") from exc
            if frame in files:
                raise DatasetIntegrityError(f"Duplicate frame {frame} in {root}")
            files[frame] = path.resolve()
        if not files:
            raise DatasetIntegrityError(f"No {suffix} frames in {root}")
        indexed[name] = files

    expected = set(indexed["META"])
    ordered_frames = sorted(expected)
    if ordered_frames != list(range(ordered_frames[0], ordered_frames[-1] + 1)):
        raise DatasetIntegrityError(
            f"Saved 2 Hz frames must be contiguous; observed "
            f"{ordered_frames[:3]}...{ordered_frames[-3:]}"
        )
    for name, files in indexed.items():
        if set(files) != expected:
            missing = sorted(expected - set(files))
            extra = sorted(set(files) - expected)
            raise DatasetIntegrityError(
                f"2 Hz frame mismatch for {name}: missing={missing[:10]}, "
                f"extra={extra[:10]}"
            )

    return {
        frame: {name: files[frame] for name, files in indexed.items()}
        for frame in sorted(expected)
    }


def _parse_views(spec: str) -> tuple[str, ...]:
    normalized = str(spec or "").strip()
    cameras = CAMERA_ALIASES.get(
        normalized,
        tuple(part.strip() for part in normalized.split(",") if part.strip()),
    )
    if not cameras or any(camera not in CAMERA_DIRECTORIES for camera in cameras):
        raise DatasetIntegrityError(f"Invalid corruption view specification: {spec!r}")
    if len(set(cameras)) != len(cameras):
        raise DatasetIntegrityError(f"Duplicate corruption views: {spec!r}")
    return tuple(cameras)


def _terminal_result(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_path = _single(run_dir.glob("eval_*.json"), "official evaluation JSON")
    evaluation = _load_json(eval_path)
    records = evaluation.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise DatasetIntegrityError(
            f"Expected one official route record in {eval_path}, found {len(records)}"
        )
    record = records[0]
    infractions = record.get("infractions", {})

    def count(key: str) -> int:
        value = infractions.get(key, [])
        if not isinstance(value, list):
            raise DatasetIntegrityError(
                f"Official infraction field {key} must be a list, got {type(value).__name__}"
            )
        return len(value)

    scores = record.get("scores", {})
    meta = record.get("meta", {})
    normalized = {
        "eligible": bool(evaluation.get("eligible", False)),
        "entry_status": evaluation.get("entry_status"),
        "route_status": record.get("status"),
        "route_id": str(record.get("route_id")),
        "scenario_name": record.get("scenario_name"),
        "town": record.get("town_name"),
        "score_composed": _finite_float(scores.get("score_composed"), "score_composed"),
        "route_completion": _finite_float(scores.get("score_route"), "score_route"),
        "score_penalty": _finite_float(scores.get("score_penalty"), "score_penalty"),
        "duration_game_seconds": _finite_float(meta.get("duration_game"), "duration_game"),
        "collisions": {
            "pedestrian": count("collisions_pedestrian"),
            "vehicle": count("collisions_vehicle"),
            "layout": count("collisions_layout"),
        },
        "infractions": {
            key: count(key)
            for key in (
                "red_light",
                "stop_infraction",
                "outside_route_lanes",
                "route_dev",
                "vehicle_blocked",
                "route_timeout",
                "scenario_timeouts",
            )
        },
    }
    return normalized, evaluation


def _pose_axes(metric: dict[str, Any], step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    location = np.asarray(metric.get("location", []), dtype=np.float64)
    forward = np.asarray(metric.get("forward_vector", []), dtype=np.float64)
    right = np.asarray(metric.get("right_vector", []), dtype=np.float64)
    if location.shape != (3,) or forward.shape != (3,) or right.shape != (3,):
        raise DatasetIntegrityError(f"Invalid pose vector shape at metric step {step}")
    if not np.isfinite(location).all() or not np.isfinite(forward).all() or not np.isfinite(right).all():
        raise DatasetIntegrityError(f"Non-finite pose at metric step {step}")
    forward_xy = forward[:2]
    right_xy = right[:2]
    forward_norm = float(np.linalg.norm(forward_xy))
    right_norm = float(np.linalg.norm(right_xy))
    if abs(forward_norm - 1.0) > 0.02 or abs(right_norm - 1.0) > 0.02:
        raise DatasetIntegrityError(
            f"Non-unit ego axes at step {step}: forward={forward_norm}, right={right_norm}"
        )
    if abs(float(np.dot(forward_xy, right_xy))) > 0.02:
        raise DatasetIntegrityError(f"Non-orthogonal ego axes at metric step {step}")
    return location, forward_xy / forward_norm, right_xy / right_norm


def future_expert_trajectory(
    metric: dict[int, dict[str, Any]],
    step: int,
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Return six 0.5 s local offsets in ORION ``[right, forward]`` order."""
    if step not in metric:
        raise DatasetIntegrityError(f"Missing current metric pose at step {step}")
    current, forward, right = _pose_axes(metric[step], step)
    max_metric_step = max(metric)
    cumulative: list[list[float]] = []
    mask: list[int] = []
    previous = np.zeros(2, dtype=np.float64)
    increments: list[list[float]] = []
    missing_started = False

    for offset in HORIZON_STEPS:
        future_step = step + offset
        if future_step not in metric:
            if future_step <= max_metric_step:
                raise DatasetIntegrityError(
                    f"Internal future-pose gap: sample step {step} lacks metric step {future_step}"
                )
            missing_started = True
            mask.append(0)
            increments.append([0.0, 0.0])
            cumulative.append(previous.tolist())
            continue
        if missing_started:
            raise DatasetIntegrityError(
                f"Non-trailing future-pose gap for sample step {step}"
            )
        future_location, _, _ = _pose_axes(metric[future_step], future_step)
        delta_world = future_location[:2] - current[:2]
        point = np.asarray(
            [float(np.dot(delta_world, right)), float(np.dot(delta_world, forward))],
            dtype=np.float64,
        )
        increment = point - previous
        mask.append(1)
        increments.append(increment.tolist())
        cumulative.append(point.tolist())
        previous = point
    return increments, cumulative, mask


def _assert_close(left: Any, right: Any, field: str, tolerance: float = 1e-5) -> None:
    left_value = _finite_float(left, field)
    right_value = _finite_float(right, field)
    if abs(left_value - right_value) > tolerance:
        raise DatasetIntegrityError(
            f"Trace/meta mismatch for {field}: {left_value} vs {right_value}"
        )


def _validate_trace_meta(trace: dict[str, Any], meta: dict[str, Any], step: int) -> None:
    if bool(trace.get("corruption_active")) != bool(meta.get("corruption_active")):
        raise DatasetIntegrityError(f"Trace/meta corruption mismatch at step {step}")
    _assert_close(trace.get("route_progress"), meta.get("route_progress"), "route_progress")
    trace_risk = trace.get("risk", {})
    meta_risk = meta.get("risk_governor", {})
    if trace_risk.get("mode") != meta_risk.get("mode"):
        raise DatasetIntegrityError(f"Trace/meta risk mode mismatch at step {step}")
    for key in (
        "base_throttle",
        "base_brake",
        "throttle",
        "brake",
        "speed_cap",
        "intensity",
    ):
        _assert_close(trace_risk.get(key), meta_risk.get(key), f"risk.{key}")


def validate_sample(sample: dict[str, Any], *, check_files: bool = True) -> None:
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise DatasetIntegrityError(
            f"Unsupported sample schema {sample.get('schema_version')!r}"
        )
    source = sample.get("source", {})
    rollout_id = source.get("rollout_id")
    if not rollout_id:
        raise DatasetIntegrityError("Sample has no rollout_id")
    expert = sample.get("expert", {})
    controls = sample.get("controls", {})
    if expert.get("source_rollout_id") != rollout_id:
        raise DatasetIntegrityError(
            "Simulation-fork violation: expert trajectory comes from a different rollout"
        )
    if controls.get("source_rollout_id") != rollout_id:
        raise DatasetIntegrityError(
            "Simulation-fork violation: controls come from a different rollout"
        )
    condition = sample.get("condition", {})
    variant = condition.get("variant")
    if variant not in {"hazard", "nohazard"}:
        raise DatasetIntegrityError(f"Invalid scenario variant {variant!r}")
    if bool(condition.get("hazard_present")) != (variant == "hazard"):
        raise DatasetIntegrityError("hazard_present contradicts scenario variant")
    relevance = condition.get("relevance")
    if relevance not in {"on_path", "off_path"}:
        raise DatasetIntegrityError(f"Invalid relevance {relevance!r}")
    oracle = sample.get("oracle", {})
    oracle_uq = _finite_float(oracle.get("uq_global"), "oracle.uq_global")
    path_risk = _finite_float(oracle.get("path_risk"), "oracle.path_risk")
    if not 0.0 <= path_risk <= oracle_uq <= 1.0:
        raise DatasetIntegrityError("Require 0 <= oracle path risk <= oracle UQ <= 1")
    expected_path_risk = oracle_uq if relevance == "on_path" else 0.0
    if abs(path_risk - expected_path_risk) > 1e-6:
        raise DatasetIntegrityError(
            "Path risk contradicts the explicitly declared relevance"
        )
    if expert.get("coordinate_convention") != "orion_local_right_forward_increment_m":
        raise DatasetIntegrityError("Unknown expert trajectory coordinate convention")
    trajectory = expert.get("trajectory_displacements_m", [])
    cumulative = expert.get("trajectory_cumulative_m", [])
    base_trajectory = expert.get("base_plan_displacements_m", [])
    base_cumulative = expert.get("base_plan_cumulative_m", [])
    mask = expert.get("trajectory_mask", [])
    if len(trajectory) != 6 or len(cumulative) != 6 or len(mask) != 6:
        raise DatasetIntegrityError("Expert trajectory and mask must contain six horizons")
    if any(value not in (0, 1) for value in mask):
        raise DatasetIntegrityError("Expert trajectory mask must be binary")
    if any(mask[index] < mask[index + 1] for index in range(5)):
        raise DatasetIntegrityError("Expert trajectory mask may only have trailing zeros")
    for name, values in (
        ("trajectory", trajectory),
        ("cumulative", cumulative),
        ("base_trajectory", base_trajectory),
        ("base_cumulative", base_cumulative),
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6, 2) or not np.isfinite(array).all():
            raise DatasetIntegrityError(f"{name} must be finite with shape [6, 2]")
    trajectory_array = np.asarray(trajectory, dtype=np.float64)
    cumulative_array = np.asarray(cumulative, dtype=np.float64)
    reconstructed = np.cumsum(trajectory_array, axis=0)
    if not np.allclose(reconstructed, cumulative_array, atol=1e-6):
        raise DatasetIntegrityError(
            "Cumulative expert positions do not equal the displacement cumsum"
        )
    if not np.allclose(
        np.cumsum(np.asarray(base_trajectory, dtype=np.float64), axis=0),
        np.asarray(base_cumulative, dtype=np.float64),
        atol=1e-6,
    ):
        raise DatasetIntegrityError(
            "Cumulative base plan does not equal the base displacement cumsum"
        )
    source_step = int(source.get("step", -1))
    frame_2hz = int(source.get("frame_2hz", -1))
    if source_step != frame_2hz * SAVE_INTERVAL_STEPS:
        raise DatasetIntegrityError("2 Hz frame id does not match trace step")
    if abs(
        _finite_float(source.get("sim_time_seconds"), "source.sim_time_seconds")
        - source_step / SIMULATION_HZ
    ) > 1e-6:
        raise DatasetIntegrityError("Sample step and simulation time disagree")
    base = controls.get("base", {})
    post = controls.get("post", {})
    intervention_l1 = abs(
        _finite_float(post.get("throttle"), "controls.post.throttle")
        - _finite_float(base.get("throttle"), "controls.base.throttle")
    ) + abs(
        _finite_float(post.get("brake"), "controls.post.brake")
        - _finite_float(base.get("brake"), "controls.base.brake")
    )
    labels = sample.get("labels", {})
    if abs(
        _finite_float(labels.get("intervention_l1"), "labels.intervention_l1")
        - intervention_l1
    ) > 1e-6:
        raise DatasetIntegrityError("Intervention label disagrees with controls")
    if bool(labels.get("intervention")) != (intervention_l1 > 1e-6):
        raise DatasetIntegrityError("Binary intervention label disagrees with controls")
    if bool(labels.get("recover")) and bool(condition.get("corruption_active")):
        raise DatasetIntegrityError("A recovery sample cannot still have active corruption")
    if check_files:
        frame_paths = source.get("frame_paths", {})
        expected = set(CAMERA_DIRECTORIES) | {"BEV", "META"}
        if set(frame_paths) != expected:
            raise DatasetIntegrityError(
                f"Frame path keys mismatch: expected {sorted(expected)}, got {sorted(frame_paths)}"
            )
        missing = [path for path in frame_paths.values() if not Path(path).is_file()]
        if missing:
            raise DatasetIntegrityError(f"Sample references missing files: {missing[:3]}")


def _run_context(run_dir: Path, relevance: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if relevance not in {"on_path", "off_path"}:
        raise DatasetIntegrityError("relevance must be on_path or off_path")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetIntegrityError(f"Missing run manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    variant = manifest.get("pilot_variant")
    if variant not in {"hazard", "nohazard"}:
        raise DatasetIntegrityError(f"Invalid pilot_variant in {manifest_path}: {variant!r}")
    condition = manifest.get("pilot_condition")
    if not condition:
        raise DatasetIntegrityError(f"Missing pilot_condition in {manifest_path}")
    rollout_id = f"{manifest.get('pilot_run_id', 'unknown')}/{run_dir.name}"
    record_root = _single(run_dir.glob("records_*/*"), "recorded scenario directory")
    trace_records, trace = _indexed_jsonl(record_root / "control_trace.jsonl")
    metric = _indexed_metric(record_root / "metric_info.json")
    frames = _frame_files(record_root)
    terminal, _ = _terminal_result(run_dir)
    corruption_views = _parse_views(manifest.get("orion_closedloop_corruption_views", "front"))
    if terminal["route_id"] in {"None", ""}:
        raise DatasetIntegrityError(f"Official result has no route id: {run_dir}")
    return {
        "run_dir": run_dir,
        "record_root": record_root.resolve(),
        "manifest": manifest,
        "variant": variant,
        "condition": str(condition),
        "rollout_id": rollout_id,
        "relevance": relevance,
        "trace_records": trace_records,
        "trace": trace,
        "metric": metric,
        "frames": frames,
        "terminal": terminal,
        "corruption_views": corruption_views,
    }


def build_run_samples(
    run_dir: str | Path,
    *,
    relevance: str,
    require_full_horizon: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = _run_context(Path(run_dir), relevance)
    trace = context["trace"]
    metric = context["metric"]
    manifest = context["manifest"]
    samples: list[dict[str, Any]] = []
    excluded_terminal_horizon = 0
    previous_path_risk = 0.0

    for frame, paths in context["frames"].items():
        step = frame * SAVE_INTERVAL_STEPS
        if step not in trace:
            raise DatasetIntegrityError(f"2 Hz frame {frame} has no trace step {step}")
        if step not in metric:
            raise DatasetIntegrityError(f"2 Hz frame {frame} has no metric step {step}")
        trace_row = trace[step]
        meta = _load_json(paths["META"])
        _validate_trace_meta(trace_row, meta, step)

        increments, cumulative, future_mask = future_expert_trajectory(metric, step)
        if require_full_horizon and not all(future_mask):
            excluded_terminal_horizon += 1
            continue

        risk = trace_row.get("risk", {})
        base_throttle = _finite_float(risk.get("base_throttle"), "base_throttle")
        base_brake = _finite_float(risk.get("base_brake"), "base_brake")
        post_throttle = _finite_float(risk.get("throttle"), "throttle")
        post_brake = _finite_float(risk.get("brake"), "brake")
        steer = _finite_float(trace_row.get("steer"), "steer")
        corruption_active = bool(trace_row.get("corruption_active"))
        oracle_uq = 1.0 if corruption_active else 0.0
        path_risk = oracle_uq if relevance == "on_path" else 0.0
        if risk.get("mode") == "oracle":
            applied_score = _finite_float(
                risk.get("applied_score"), "risk.applied_score"
            )
            if abs(applied_score - path_risk) > 1e-6:
                raise DatasetIntegrityError(
                    f"Oracle response at step {step} contradicts declared "
                    f"{relevance} relevance: applied_score={applied_score}, "
                    f"path_risk={path_risk}. Do not relabel an on-path rollout "
                    "as off-path supervision."
                )
        throttle_delta = post_throttle - base_throttle
        brake_delta = post_brake - base_brake
        intervention_l1 = abs(throttle_delta) + abs(brake_delta)
        speed_cap = _finite_float(risk.get("speed_cap"), "speed_cap")
        stop_required = bool(path_risk > 0.0 and speed_cap <= 0.25)
        recover = bool(previous_path_risk > 0.0 and path_risk == 0.0)
        previous_path_risk = path_risk
        current_metric = metric[step]
        location, forward, right = _pose_axes(current_metric, step)
        rotation = current_metric.get("rotation")
        if not isinstance(rotation, list) or len(rotation) != 3:
            raise DatasetIntegrityError(f"Invalid rotation at metric step {step}")
        base_plan_cumulative = np.asarray(meta.get("plan", []), dtype=np.float64)
        if base_plan_cumulative.shape != (6, 2) or not np.isfinite(
            base_plan_cumulative
        ).all():
            raise DatasetIntegrityError(
                f"meta.plan must be a finite cumulative [6, 2] trajectory at step {step}"
            )
        base_plan_displacements = np.diff(
            np.concatenate(
                (np.zeros((1, 2), dtype=np.float64), base_plan_cumulative),
                axis=0,
            ),
            axis=0,
        )

        frame_paths = {name: str(path) for name, path in paths.items()}
        sample = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"{context['rollout_id']}#step={step}",
            "source": {
                "rollout_id": context["rollout_id"],
                "run_dir": str(context["run_dir"]),
                "record_root": str(context["record_root"]),
                "step": step,
                "frame_2hz": frame,
                "sim_time_seconds": _finite_float(
                    trace_row.get("sim_time_seconds"), "sim_time_seconds"
                ),
                "frame_paths": frame_paths,
            },
            "condition": {
                "name": context["condition"],
                "variant": context["variant"],
                "hazard_present": context["variant"] == "hazard",
                "relevance": relevance,
                "relevance_source": "explicit_export_argument",
                "corruption": manifest.get("orion_closedloop_corruption") or "clean",
                "corruption_views": list(context["corruption_views"]),
                "corruption_severity": int(
                    manifest.get("orion_closedloop_corruption_severity") or 0
                ),
                "corruption_active": corruption_active,
            },
            "state": {
                "route_progress": _finite_float(
                    trace_row.get("route_progress"), "route_progress"
                ),
                "speed_mps": _finite_float(trace_row.get("speed"), "speed"),
                "command": int(meta.get("command")),
                "location_world_m": [float(value) for value in location],
                "rotation_world_deg": [
                    _finite_float(value, "rotation") for value in rotation
                ],
                "forward_world_xy": [float(value) for value in forward],
                "right_world_xy": [float(value) for value in right],
            },
            "oracle": {
                "uq_global": oracle_uq,
                "path_risk": path_risk,
                "uq_source": "known_corruption_window",
                "path_risk_source": "declared_relevance_x_oracle_uq",
                "spatial_mask_spec": {
                    "kind": "full_selected_views" if corruption_active else "inactive",
                    "views": list(context["corruption_views"]),
                    "value": oracle_uq,
                    "mask_path": None,
                },
            },
            "controls": {
                "source_rollout_id": context["rollout_id"],
                "base": {
                    "steer": steer,
                    "throttle": base_throttle,
                    "brake": base_brake,
                },
                "post": {
                    "steer": steer,
                    "throttle": post_throttle,
                    "brake": post_brake,
                },
                "risk_mode": risk.get("mode"),
                "speed_cap_mps": speed_cap,
                "risk_intensity": _finite_float(risk.get("intensity"), "intensity"),
            },
            "expert": {
                "source_rollout_id": context["rollout_id"],
                "source": "same_rollout_executed_post_governor_ego_poses",
                "coordinate_convention": "orion_local_right_forward_increment_m",
                "horizon_seconds": [offset / SIMULATION_HZ for offset in HORIZON_STEPS],
                "trajectory_displacements_m": increments,
                "trajectory_cumulative_m": cumulative,
                "trajectory_mask": future_mask,
                "base_plan_displacements_m": base_plan_displacements.tolist(),
                "base_plan_cumulative_m": base_plan_cumulative.tolist(),
            },
            "labels": {
                "stop_required": stop_required,
                "recover": recover,
                "intervention": intervention_l1 > 1e-6,
                "intervention_l1": intervention_l1,
                "throttle_delta": throttle_delta,
                "brake_delta": brake_delta,
            },
            "terminal_result": context["terminal"],
        }
        validate_sample(sample, check_files=True)
        samples.append(sample)

    summary = {
        "rollout_id": context["rollout_id"],
        "run_dir": str(context["run_dir"]),
        "condition": context["condition"],
        "variant": context["variant"],
        "relevance": relevance,
        "trace_steps": len(context["trace_records"]),
        "metric_steps": len(context["metric"]),
        "candidate_2hz_frames": len(context["frames"]),
        "exported_samples": len(samples),
        "excluded_terminal_horizon": excluded_terminal_horizon,
        "active_samples": sum(sample["condition"]["corruption_active"] for sample in samples),
        "stop_samples": sum(sample["labels"]["stop_required"] for sample in samples),
        "recover_samples": sum(sample["labels"]["recover"] for sample in samples),
        "terminal_result": context["terminal"],
    }
    return samples, summary


def export_dataset(
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    relevance: str,
    require_full_horizon: bool = True,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    all_samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    rollout_ids: set[str] = set()
    sample_ids: set[str] = set()
    for run_dir in run_dirs:
        samples, summary = build_run_samples(
            run_dir,
            relevance=relevance,
            require_full_horizon=require_full_horizon,
        )
        if summary["rollout_id"] in rollout_ids:
            raise DatasetIntegrityError(f"Duplicate rollout: {summary['rollout_id']}")
        rollout_ids.add(summary["rollout_id"])
        for sample in samples:
            if sample["sample_id"] in sample_ids:
                raise DatasetIntegrityError(f"Duplicate sample id: {sample['sample_id']}")
            sample_ids.add(sample["sample_id"])
        all_samples.extend(samples)
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as outfile:
        for sample in all_samples:
            outfile.write(json.dumps(sample, sort_keys=True) + "\n")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "sample_schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_count": len(all_samples),
        "samples_file": "samples.jsonl",
        "schema_path": str(Path(schema_path).resolve()) if schema_path else None,
        "sampling": {
            "simulation_hz": SIMULATION_HZ,
            "save_interval_steps": SAVE_INTERVAL_STEPS,
            "sample_hz": SIMULATION_HZ / SAVE_INTERVAL_STEPS,
            "horizon_steps": list(HORIZON_STEPS),
            "require_full_horizon": require_full_horizon,
        },
        "coordinate_convention": {
            "trajectory": "[right_m, forward_m] increments in the current ego frame",
            "cumsum": "cumulative local future positions at 0.5 through 3.0 seconds",
        },
        "causal_integrity": (
            "Every sample's images, base/post controls, and executed expert poses "
            "come from the same rollout. Cross-condition time alignment is prohibited."
        ),
        "runs": summaries,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_dataset(output_dir, check_files=True)
    return manifest


def validate_dataset(dataset_dir: str | Path, *, check_files: bool = True) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest = _load_json(dataset_dir / "dataset_manifest.json")
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise DatasetIntegrityError(
            f"Unsupported dataset version {manifest.get('dataset_version')!r}"
        )
    samples_path = dataset_dir / manifest.get("samples_file", "samples.jsonl")
    records, _ = _indexed_samples(samples_path, check_files=check_files)
    if int(manifest.get("sample_count", -1)) != len(records):
        raise DatasetIntegrityError(
            f"Manifest sample_count={manifest.get('sample_count')} but found {len(records)}"
        )
    return {
        "valid": True,
        "dataset_version": DATASET_VERSION,
        "sample_schema_version": SCHEMA_VERSION,
        "sample_count": len(records),
        "rollout_count": len({sample["source"]["rollout_id"] for sample in records}),
    }


def _indexed_samples(
    path: Path,
    *,
    check_files: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise DatasetIntegrityError(f"Missing samples file: {path}")
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(f"Invalid sample JSON at line {line_number}") from exc
        validate_sample(sample, check_files=check_files)
        sample_id = sample.get("sample_id")
        if not sample_id or sample_id in by_id:
            raise DatasetIntegrityError(f"Missing or duplicate sample id {sample_id!r}")
        by_id[sample_id] = sample
        records.append(sample)
    return records, by_id
