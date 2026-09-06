#!/usr/bin/env python3
"""Measure Qwen visibility-token stability under offline depth quantization.

This is a no-training preflight.  It reads the frozen calibration split,
constructs U in memory for four preregistered depth/tolerance variants, and
writes aggregate stability metrics.  It never imports or loads Qwen and does
not serialize per-frame U tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "orion.qwen-visibility-offline-depth-preflight/v1"
DEPTH_DIRECTORY_BY_CAMERA = {
    "CAM_FRONT": "depth_front",
    "CAM_FRONT_LEFT": "depth_front_left",
    "CAM_FRONT_RIGHT": "depth_front_right",
}


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_visibility = _load_local_module(
    "_orion_qwen_visibility_offline_preflight",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py",
)


@dataclass(frozen=True)
class TokenSnapshot:
    """The token values needed by this stability audit."""

    global_tokens: np.ndarray
    frontier_tokens: np.ndarray
    frontier_mask: np.ndarray
    feature_names: tuple[str, ...]
    x_bounds_m: tuple[float, float]
    y_bounds_m: tuple[float, float]

    @property
    def valid_frontier_tokens(self) -> np.ndarray:
        return self.frontier_tokens[self.frontier_mask]

    def centers_m(self) -> np.ndarray:
        tokens = self.valid_frontier_tokens
        if len(tokens) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        ix = self.feature_names.index("center_x_normalized")
        iy = self.feature_names.index("center_y_normalized")
        x0, x1 = self.x_bounds_m
        y0, y1 = self.y_bounds_m
        x = x0 + (tokens[:, ix].astype(np.float64) + 1.0) * 0.5 * (x1 - x0)
        y = y0 + (tokens[:, iy].astype(np.float64) + 1.0) * 0.5 * (y1 - y0)
        return np.stack([x, y], axis=-1)

    def route_labels(self, threshold: float) -> np.ndarray:
        index = self.feature_names.index("route_weight_mean")
        return self.valid_frontier_tokens[:, index] >= float(threshold)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve_input(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _validated_input(reference: Mapping[str, Any]) -> Path:
    path = _resolve_input(str(reference["path"]))
    digest = _sha256(path)
    if digest != reference["sha256"]:
        raise ValueError("input SHA-256 differs: %s" % path)
    return path


def _load_infos(path: Path) -> list[Mapping[str, Any]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - frozen trusted local artifact
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("infos must contain a list of dictionaries")
    return value


def select_centered_stride_window(
    frames: Sequence[int], *, stride: int, length: int
) -> list[int]:
    """Choose the exact-stride window whose midpoint is closest to the route median."""

    ordered = sorted(set(int(value) for value in frames))
    if stride <= 0 or length <= 0:
        raise ValueError("stride and length must be positive")
    if len(ordered) < length:
        raise ValueError("route has fewer indexed frames than the requested window")
    available = set(ordered)
    median = float(ordered[len(ordered) // 2])
    candidates = []
    for start in ordered:
        window = [start + index * stride for index in range(length)]
        if all(frame in available for frame in window):
            midpoint = 0.5 * (window[0] + window[-1])
            candidates.append((abs(midpoint - median), start, window))
    if not candidates:
        raise ValueError("route has no exact-stride window of the requested length")
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def _folder_split(route_manifest: Mapping[str, Any]) -> Dict[str, str]:
    result = {}
    statistics = route_manifest["lineage_audit"]["split_statistics"]
    for split_name, split in statistics.items():
        for folder in split["folder_route_ids"]:
            if folder in result:
                raise ValueError("folder appears in multiple splits")
            result[str(folder)] = str(split_name)
    return result


def _camera_specs(qwen_config: Mapping[str, Any]):
    return tuple(
        _visibility.camera_from_carla_sensor(camera, qwen_config["sensors"][camera])
        for camera in DEPTH_DIRECTORY_BY_CAMERA
    )


def _grid_spec(qwen_config: Mapping[str, Any], tolerance_m: float):
    grid = dict(qwen_config["oracle_visibility"]["grid"])
    grid["surface_tolerance_m"] = float(tolerance_m)
    return _visibility.VisibilityGridSpec(**grid)


def _load_annotation(dataset_root: Path, folder: str, frame: int) -> Dict[str, Any]:
    path = dataset_root / folder / "anno" / (("%05d" % frame) + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("annotation is not an object: %s" % path)
    return value


def _load_stored_depth(dataset_root: Path, folder: str, frame: int):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to read installed depth PNGs") from error
    result = {}
    for camera, directory in DEPTH_DIRECTORY_BY_CAMERA.items():
        path = (
            dataset_root
            / folder
            / "camera"
            / directory
            / (("%05d" % frame) + ".png")
        )
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise FileNotFoundError(path)
        if value.dtype != np.uint8 or value.ndim != 2:
            raise ValueError("stored depth must be a uint8 grayscale plane: %s" % path)
        result[camera] = value
    return result


def depth_hypothesis(
    stored: Mapping[str, np.ndarray],
    *,
    offset_m: float,
    saturated_value: int,
    saturated_replacement_m: float,
) -> Dict[str, np.ndarray]:
    """Decode rounded metric metres under one quantization endpoint hypothesis."""

    result = {}
    for camera, value in stored.items():
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.ndim != 2:
            raise ValueError("stored depth must be uint8 [H,W]")
        saturated = array == int(saturated_value)
        depth = np.maximum(array.astype(np.float32) + float(offset_m), 1e-3)
        depth[saturated] = float(saturated_replacement_m)
        result[camera] = depth
    return result


def _route_and_pose(annotation: Mapping[str, Any]):
    route = np.asarray(
        [
            [annotation["x"], annotation["y"]],
            [annotation["x_command_near"], annotation["y_command_near"]],
            [annotation["x_command_far"], annotation["y_command_far"]],
        ],
        dtype=np.float64,
    )
    keep = np.concatenate(
        [np.asarray([True]), np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-6]
    )
    route = route[keep]
    if len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("navigation route is degenerate or non-finite")
    yaw = float(annotation["sensors"]["CAM_FRONT"]["rotation"][2])
    pose = np.asarray([annotation["x"], annotation["y"], yaw], dtype=np.float64)
    speed = max(0.0, float(annotation["speed"]))
    return route, pose, speed


def _token_snapshot(tokens) -> TokenSnapshot:
    spec = tokens.spec
    return TokenSnapshot(
        global_tokens=tokens.global_tokens.copy(),
        frontier_tokens=tokens.frontier_tokens.copy(),
        frontier_mask=tokens.frontier_mask.copy(),
        feature_names=tuple(tokens.feature_names),
        x_bounds_m=(float(spec.x_min_m), float(spec.x_max_m)),
        y_bounds_m=(float(spec.y_min_m), float(spec.y_max_m)),
    )


def compare_snapshots(
    left: TokenSnapshot,
    right: TokenSnapshot,
    *,
    radius_m: float,
    route_threshold: float,
) -> Dict[str, Any]:
    """Compare row order and nearest physical frontiers separately."""

    left_centers = left.centers_m()
    right_centers = right.centers_m()
    left_labels = left.route_labels(route_threshold)
    right_labels = right.route_labels(route_threshold)
    if len(left_centers) == 0 or len(right_centers) == 0:
        return {
            "bidirectional_nearest_total": len(left_centers) + len(right_centers),
            "bidirectional_nearest_within_radius": 0,
            "nearest_route_label_total": 0,
            "nearest_route_label_agree": 0,
            "same_slot_total": 0,
            "same_slot_within_radius": 0,
            "same_slot_route_label_agree": 0,
            "same_slot_distance_m": [],
            "nearest_distance_m": [],
            "global_content_absolute_error_sum": 0.0,
            "global_content_value_count": 0,
        }
    distances = np.linalg.norm(
        left_centers[:, None, :] - right_centers[None, :, :], axis=-1
    )
    left_nearest = np.argmin(distances, axis=1)
    right_nearest = np.argmin(distances, axis=0)
    left_distance = distances[np.arange(len(left_centers)), left_nearest]
    right_distance = distances[right_nearest, np.arange(len(right_centers))]
    nearest_distances = np.concatenate([left_distance, right_distance])
    nearest_label_agreement = np.concatenate(
        [
            left_labels == right_labels[left_nearest],
            right_labels == left_labels[right_nearest],
        ]
    )

    same_count = min(len(left_centers), len(right_centers))
    same_distances = np.linalg.norm(
        left_centers[:same_count] - right_centers[:same_count], axis=1
    )
    same_labels = left_labels[:same_count] == right_labels[:same_count]
    content_start = left.feature_names.index("visible_free_height_ratio")
    if left.feature_names != right.feature_names:
        raise ValueError("token feature schemas differ")
    global_error = np.abs(
        left.global_tokens[:, content_start:].astype(np.float64)
        - right.global_tokens[:, content_start:].astype(np.float64)
    )
    return {
        "bidirectional_nearest_total": int(len(nearest_distances)),
        "bidirectional_nearest_within_radius": int(
            np.sum(nearest_distances <= float(radius_m))
        ),
        "nearest_route_label_total": int(len(nearest_label_agreement)),
        "nearest_route_label_agree": int(np.sum(nearest_label_agreement)),
        "same_slot_total": int(same_count),
        "same_slot_within_radius": int(
            np.sum(same_distances <= float(radius_m))
        ),
        "same_slot_route_label_agree": int(np.sum(same_labels)),
        "same_slot_distance_m": same_distances.tolist(),
        "nearest_distance_m": nearest_distances.tolist(),
        "global_content_absolute_error_sum": float(global_error.sum()),
        "global_content_value_count": int(global_error.size),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _aggregate_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sums = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "bidirectional_nearest_total",
            "bidirectional_nearest_within_radius",
            "nearest_route_label_total",
            "nearest_route_label_agree",
            "same_slot_total",
            "same_slot_within_radius",
            "same_slot_route_label_agree",
            "global_content_value_count",
        )
    }
    global_error = sum(float(row["global_content_absolute_error_sum"]) for row in rows)
    nearest = [value for row in rows for value in row["nearest_distance_m"]]
    same = [value for row in rows for value in row["same_slot_distance_m"]]
    return {
        "frame_count": len(rows),
        "bidirectional_nearest_match_fraction_within_radius": _ratio(
            sums["bidirectional_nearest_within_radius"],
            sums["bidirectional_nearest_total"],
        ),
        "nearest_matched_route_label_agreement": _ratio(
            sums["nearest_route_label_agree"], sums["nearest_route_label_total"]
        ),
        "same_slot_match_fraction_within_radius": _ratio(
            sums["same_slot_within_radius"], sums["same_slot_total"]
        ),
        "same_slot_route_label_agreement": _ratio(
            sums["same_slot_route_label_agree"], sums["same_slot_total"]
        ),
        "nearest_distance_m_median": _percentile(nearest, 50.0),
        "nearest_distance_m_p95": _percentile(nearest, 95.0),
        "same_slot_distance_m_median": _percentile(same, 50.0),
        "same_slot_distance_m_p95": _percentile(same, 95.0),
        "global_content_mean_absolute_error": (
            global_error / sums["global_content_value_count"]
            if sums["global_content_value_count"]
            else 0.0
        ),
    }


def run_preflight(protocol_path: Path) -> Dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != "orion.qwen-visibility-offline-depth-preflight-protocol/v1":
        raise ValueError("unexpected protocol schema")
    inputs = protocol["inputs"]
    dataset_root = _resolve_input(inputs["dataset_root"])
    infos_path = _validated_input(inputs["infos"])
    route_manifest_path = _validated_input(inputs["route_manifest"])
    qwen_config_path = _validated_input(inputs["qwen_visibility_config"])
    source_inventory_path = _validated_input(inputs["source_inventory"])
    infos = _load_infos(infos_path)
    route_manifest = _read_json(route_manifest_path)
    qwen_config = _read_json(qwen_config_path)
    source_inventory = _read_json(source_inventory_path)
    if source_inventory["feasibility"]["same_frame_answer_equality_penalty_allowed"]:
        raise ValueError("source inventory permits the rejected equality penalty")

    frames_by_folder: Dict[str, list[int]] = {}
    temporary: Dict[str, list[int]] = {}
    for row in infos:
        temporary.setdefault(str(row["folder"]), []).append(int(row["frame_idx"]))
    for folder, frames in temporary.items():
        frames_by_folder[folder] = sorted(set(frames))
    folder_split = _folder_split(route_manifest)
    sampling = protocol["sampling"]
    split_name = str(sampling["split"])
    folders = sorted(folder for folder, split in folder_split.items() if split == split_name)
    if len(folders) != int(sampling["route_count"]):
        raise ValueError("selected route count differs from protocol")
    windows = {
        folder: select_centered_stride_window(
            frames_by_folder[folder],
            stride=int(sampling["raw_frame_stride"]),
            length=int(sampling["window_frames"]),
        )
        for folder in folders
    }

    cameras = _camera_specs(qwen_config)
    depth_config = protocol["depth"]
    variants = {str(row["name"]): row for row in depth_config["variants"]}
    variant_specs = {
        name: _grid_spec(qwen_config, float(row["surface_tolerance_m"]))
        for name, row in variants.items()
    }
    memory_config = qwen_config["oracle_visibility"]["temporal_memory"]
    memories = {
        folder: {
            name: _visibility.VisibilityObservationMemory(
                variant_specs[name],
                max_age_seconds=float(memory_config["max_age_seconds"]),
                observed_ratio_threshold=float(memory_config["observed_ratio_threshold"]),
            )
            for name in variants
        }
        for folder in folders
    }
    exposure_config = qwen_config["oracle_visibility"]["exposure"]
    tokenizer_config = qwen_config["oracle_visibility"]["tokenizer"]
    frontier_threshold = float(
        qwen_config["oracle_visibility"]["frontier_unknown_threshold"]
    )
    raw_hz = float(sampling["raw_data_hz"])
    snapshots: Dict[tuple[str, int, str], TokenSnapshot] = {}
    frame_records = []
    started = time.perf_counter()
    for folder in folders:
        for frame in windows[folder]:
            annotation = _load_annotation(dataset_root, folder, frame)
            route_world, pose, speed = _route_and_pose(annotation)
            stored_depth = _load_stored_depth(dataset_root, folder, frame)
            variant_records = {}
            for name, variant in variants.items():
                depth = depth_hypothesis(
                    stored_depth,
                    offset_m=float(variant["depth_offset_m"]),
                    saturated_value=int(depth_config["saturated_value"]),
                    saturated_replacement_m=float(
                        depth_config["saturated_replacement_m"]
                    ),
                )
                spec = variant_specs[name]
                belief = _visibility.compute_visibility_belief(
                    depth,
                    cameras,
                    spec,
                    frontier_unknown_threshold=frontier_threshold,
                )
                memory = memories[folder][name].update(
                    belief, pose, timestamp_seconds=float(frame) / raw_hz
                )
                route_ego = _visibility.carla_world_route_to_qwen_ego(
                    route_world,
                    pose,
                    float(exposure_config["route_max_length_m"]),
                )
                exposure = _visibility.compute_visibility_exposure(
                    belief,
                    route_ego,
                    speed_mps=speed,
                    reaction_time_seconds=float(
                        exposure_config["reaction_time_seconds"]
                    ),
                    safe_deceleration_mps2=float(
                        exposure_config["safe_deceleration_mps2"]
                    ),
                    route_sigma_m=float(exposure_config["route_sigma_m"]),
                    stopping_transition_m=float(
                        exposure_config["stopping_transition_m"]
                    ),
                )
                tokens = _visibility.tokenize_visibility_belief(
                    belief,
                    memory,
                    exposure,
                    global_grid_shape=tokenizer_config["global_grid_shape"],
                    max_frontier_tokens=int(
                        tokenizer_config["max_frontier_tokens"]
                    ),
                    frontier_patch_radius_m=float(
                        tokenizer_config["frontier_patch_radius_m"]
                    ),
                    frontier_nms_radius_m=float(
                        tokenizer_config["frontier_nms_radius_m"]
                    ),
                    frontier_selection_floor=float(
                        tokenizer_config["frontier_selection_floor"]
                    ),
                    depth_confidence=float(tokenizer_config["oracle_depth_confidence"]),
                )
                snapshot = _token_snapshot(tokens)
                snapshots[(folder, frame, name)] = snapshot
                variant_records[name] = {
                    "valid_frontier_rows": int(snapshot.frontier_mask.sum()),
                    "frontier_cells": int(belief.frontier.sum()),
                    "occluded_unknown_mean": float(
                        belief.occluded_unknown_ratio.mean()
                    ),
                }
            frame_records.append(
                {
                    "folder": folder,
                    "frame": frame,
                    "speed_mps": speed,
                    "variants": variant_records,
                }
            )
    elapsed = time.perf_counter() - started

    comparisons = protocol["comparisons"]
    radius = float(comparisons["frontier_match_radius_m"])
    threshold_text = str(comparisons["route_label"])
    route_threshold = float(threshold_text.rsplit(" ", 1)[-1])
    aggregate_comparisons = {}
    for left_name, right_name in comparisons["pairs"]:
        rows = [
            compare_snapshots(
                snapshots[(folder, frame, left_name)],
                snapshots[(folder, frame, right_name)],
                radius_m=radius,
                route_threshold=route_threshold,
            )
            for folder in folders
            for frame in windows[folder]
        ]
        aggregate_comparisons["%s__vs__%s" % (left_name, right_name)] = (
            _aggregate_comparison(rows)
        )

    total_frames = sum(len(window) for window in windows.values())
    variant_summary = {}
    for name in variants:
        valid_counts = [
            int(snapshots[(folder, frame, name)].frontier_mask.sum())
            for folder in folders
            for frame in windows[folder]
        ]
        variant_summary[name] = {
            "frame_count": len(valid_counts),
            "minimum_valid_frontier_rows": min(valid_counts),
            "median_valid_frontier_rows": float(np.median(valid_counts)),
            "maximum_valid_frontier_rows": max(valid_counts),
            "frames_with_32_valid_frontier_rows": sum(
                value == 32 for value in valid_counts
            ),
            "fraction_frames_with_32_valid_frontier_rows": _ratio(
                sum(value == 32 for value in valid_counts), len(valid_counts)
            ),
        }

    gates = protocol["engineering_gates"]
    gate_results = []
    for name, summary in variant_summary.items():
        actual = summary["fraction_frames_with_32_valid_frontier_rows"]
        required = float(
            gates["minimum_fraction_frames_with_32_valid_frontiers_per_variant"]
        )
        gate_results.append(
            {
                "name": "32_valid_frontiers:%s" % name,
                "actual": actual,
                "required_minimum": required,
                "passed": actual >= required,
            }
        )
    comparison_gate_fields = (
        (
            "bidirectional_nearest_match_fraction_within_radius",
            "minimum_nearest_physical_frontier_match_fraction_within_2m_per_comparison",
        ),
        (
            "nearest_matched_route_label_agreement",
            "minimum_nearest_matched_route_label_agreement_per_comparison",
        ),
        (
            "same_slot_route_label_agreement",
            "minimum_same_slot_route_label_agreement_per_comparison",
        ),
    )
    for comparison_name, summary in aggregate_comparisons.items():
        for field, gate_name in comparison_gate_fields:
            actual = float(summary[field])
            required = float(gates[gate_name])
            gate_results.append(
                {
                    "name": "%s:%s" % (field, comparison_name),
                    "actual": actual,
                    "required_minimum": required,
                    "passed": actual >= required,
                }
            )
    all_passed = all(row["passed"] for row in gate_results)
    return {
        "schema": SCHEMA,
        "protocol": _reference(protocol_path),
        "inputs": {
            "infos": _reference(infos_path),
            "route_manifest": _reference(route_manifest_path),
            "qwen_visibility_config": _reference(qwen_config_path),
            "source_inventory": _reference(source_inventory_path),
        },
        "claim_boundary": protocol["claim_boundary"],
        "sampling": {
            "split": split_name,
            "route_count": len(folders),
            "frame_count": total_frames,
            "windows": windows,
            "raw_data_hz": raw_hz,
            "effective_u_hz": float(sampling["effective_u_hz"]),
        },
        "variants": variant_summary,
        "comparisons": aggregate_comparisons,
        "engineering_gate_results": gate_results,
        "all_engineering_gates_passed": all_passed,
        "recommendation": (
            "coarse_depth_is_stable_enough_for_bounded_grounding_data_pilot"
            if all_passed
            else "do_not_build_curriculum_audit_lossless_depth_recapture"
        ),
        "runtime": {
            "total_seconds": elapsed,
            "seconds_per_frame_all_variants": elapsed / total_frames,
        },
        "frame_records": frame_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite preflight report: %s" % output)
    report = run_preflight(args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
