#!/usr/bin/env python3
"""Outcome-blind analysis of exact same-tick native-glare RGB triplets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


SCHEMA = "orion.native_glare_same_tick_visual_analysis.v1"
PROFILES = ("clean", "medium", "heavy")


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read %s" % path)
    return image


def _edge_energy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _contrast(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std())


def _project_actor_bbox(actor: dict, world_to_sensor: Sequence[float], width: int, height: int, fov: float) -> Optional[List[int]]:
    matrix = np.asarray(world_to_sensor, dtype=np.float64).reshape(4, 4)
    focal = float(width) / (2.0 * math.tan(math.radians(float(fov)) / 2.0))
    points = []
    for vertex in actor.get("bbox_world_vertices", []):
        sensor = matrix.dot(np.asarray([vertex[0], vertex[1], vertex[2], 1.0], dtype=np.float64))
        depth = float(sensor[0])
        if depth <= 0.1:
            continue
        u = width / 2.0 + focal * float(sensor[1]) / depth
        v = height / 2.0 - focal * float(sensor[2]) / depth
        points.append((u, v))
    if len(points) < 4:
        return None
    left = max(0, int(math.floor(min(point[0] for point in points))))
    top = max(0, int(math.floor(min(point[1] for point in points))))
    right = min(width, int(math.ceil(max(point[0] for point in points))))
    bottom = min(height, int(math.ceil(max(point[1] for point in points))))
    if right - left < 6 or bottom - top < 6:
        return None
    if left >= width or top >= height or right <= 0 or bottom <= 0:
        return None
    return [left, top, right, bottom]


def _select_visible_actor(row: dict, clean_sensor_id: str, width: int, height: int, fov: float):
    transform = row["sensor_readback"][clean_sensor_id]["transform"]
    candidates = []
    for actor in row["nearby_actors"]:
        bbox = _project_actor_bbox(actor, transform["world_to_sensor"], width, height, fov)
        if bbox is None:
            continue
        priority = 0 if actor.get("category") == "walker" else 1
        candidates.append((priority, float(actor["distance_m"]), actor, bbox))
    if not candidates:
        return None, None
    _, _, actor, bbox = min(candidates, key=lambda value: (value[0], value[1]))
    return actor, bbox


def _crop(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    left, top, right, bottom = [int(value) for value in bbox]
    return image[top:bottom, left:right]


def _intersection_area(left: Sequence[int], right: Sequence[int]) -> int:
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return int(width * height)


def _control_roi(actor_bbox: Sequence[int], width: int, height: int) -> List[int]:
    box_width = min(int(actor_bbox[2] - actor_bbox[0]), width)
    box_height = min(int(actor_bbox[3] - actor_bbox[1]), height)
    candidates = [
        [0, 0, box_width, box_height],
        [width - box_width, 0, width, box_height],
        [0, height - box_height, box_width, height],
        [width - box_width, height - box_height, width, height],
    ]
    actor_center = ((actor_bbox[0] + actor_bbox[2]) / 2.0, (actor_bbox[1] + actor_bbox[3]) / 2.0)
    candidates.sort(key=lambda box: -math.hypot(
        (box[0] + box[2]) / 2.0 - actor_center[0],
        (box[1] + box[3]) / 2.0 - actor_center[1],
    ))
    for candidate in candidates:
        if _intersection_area(candidate, actor_bbox) == 0:
            return candidate
    return candidates[0]


def _roi_metrics(clean: np.ndarray, changed: np.ndarray, bbox: Sequence[int]) -> dict:
    clean_roi = _crop(clean, bbox)
    changed_roi = _crop(changed, bbox)
    return {
        "contrast_ratio": _contrast(changed_roi) / max(_contrast(clean_roi), 1e-6),
        "edge_visibility_ratio": _edge_energy(changed_roi) / max(_edge_energy(clean_roi), 1e-6),
    }


def _tile(images: Sequence[np.ndarray], labels: Sequence[str], actor_bbox=None, control_bbox=None, cell=(640, 360)) -> np.ndarray:
    rendered = []
    for image, label in zip(images, labels):
        annotated = image.copy()
        if actor_bbox is not None:
            cv2.rectangle(annotated, tuple(actor_bbox[:2]), tuple(actor_bbox[2:]), (0, 0, 255), 3)
        if control_bbox is not None:
            cv2.rectangle(annotated, tuple(control_bbox[:2]), tuple(control_bbox[2:]), (255, 255, 0), 3)
        frame = cv2.resize(annotated, cell, interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (cell[0], 38), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(frame)
    return np.concatenate(rendered, axis=1)


def _save_gif(frames: Sequence[np.ndarray], path: Path, duration_ms: int = 100) -> None:
    if not frames:
        raise RuntimeError("cannot save an empty GIF")
    rgb = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    rgb[0].save(str(path), save_all=True, append_images=rgb[1:], duration=duration_ms, loop=0)


def _mean(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def analyze(root: Path, protocol: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    output.mkdir(parents=True)
    spec = json.loads(protocol.read_text(encoding="utf-8"))
    trace_paths = list(root.rglob("capture_trace.jsonl"))
    if len(trace_paths) != 1:
        raise RuntimeError("expected one capture trace, found %d" % len(trace_paths))
    rows = [json.loads(line) for line in trace_paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    profile_specs = spec["camera_profiles"]
    clean_sensor_id = profile_specs["clean"]["sensor_id"]
    fov = 70.0
    per_frame = []
    residuals: Dict[str, List[np.ndarray]] = {"medium": [], "heavy": []}
    global_mads: Dict[str, List[float]] = {"medium": [], "heavy": []}
    actor_edge: Dict[str, List[float]] = {"medium": [], "heavy": []}
    actor_contrast: Dict[str, List[float]] = {"medium": [], "heavy": []}
    control_edge: Dict[str, List[float]] = {"medium": [], "heavy": []}
    saturation: Dict[str, List[float]] = {name: [] for name in PROFILES}
    gif_frames = []
    visible_walker_indices = []
    for row in rows:
        images = {name: _read_image(row["front"][name]) for name in PROFILES}
        shapes = {image.shape for image in images.values()}
        if len(shapes) != 1:
            raise RuntimeError("same-tick RGB images have different shapes")
        height, width = images["clean"].shape[:2]
        actor, actor_bbox = _select_visible_actor(row, clean_sensor_id, width, height, fov)
        control_bbox = _control_roi(actor_bbox, width, height) if actor_bbox is not None else None
        if actor is not None and actor.get("category") == "walker":
            visible_walker_indices.append(int(row["capture_index"]))
        frame_result = {
            "capture_index": int(row["capture_index"]),
            "step": int(row["step"]),
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "route_progress": float(row["route_progress"]),
            "sensor_frame": next(iter(row["sensor_frames"].values())),
            "same_tick": bool(row["same_tick"]),
            "selected_actor": ({
                "actor_id": actor["actor_id"],
                "type_id": actor["type_id"],
                "category": actor["category"],
                "distance_m": actor["distance_m"],
                "bbox_pixels": actor_bbox,
            } if actor is not None else None),
            "equal_area_control_bbox_pixels": control_bbox,
            "profiles": {},
        }
        for name in PROFILES:
            saturation[name].append(float((images[name].max(axis=2) >= 250).mean()))
        for name in ("medium", "heavy"):
            residual = cv2.absdiff(images["clean"], images[name])
            mad = float(residual.mean())
            global_mads[name].append(mad)
            residuals[name].append(cv2.resize(residual, (320, 180), interpolation=cv2.INTER_AREA))
            metrics = {
                "global_mean_absolute_pixel_delta": mad,
                "saturated_pixel_fraction": saturation[name][-1],
            }
            if actor_bbox is not None:
                actor_metrics = _roi_metrics(images["clean"], images[name], actor_bbox)
                control_metrics = _roi_metrics(images["clean"], images[name], control_bbox)
                metrics["actor_bbox"] = actor_metrics
                metrics["equal_area_control_roi"] = control_metrics
                if actor is not None and actor.get("category") == "walker":
                    actor_edge[name].append(actor_metrics["edge_visibility_ratio"])
                    actor_contrast[name].append(actor_metrics["contrast_ratio"])
                    control_edge[name].append(control_metrics["edge_visibility_ratio"])
            frame_result["profiles"][name] = metrics
        per_frame.append(frame_result)
        gif_frames.append(_tile(
            [images[name] for name in PROFILES],
            ["clean", "frozen medium", "heavy stress"],
            actor_bbox=actor_bbox,
            control_bbox=control_bbox,
        ))
    summaries = {}
    for name in ("medium", "heavy"):
        jerks = [
            float(cv2.absdiff(left, right).mean())
            for left, right in zip(residuals[name][:-1], residuals[name][1:])
        ]
        mean_effect = float(np.mean(global_mads[name]))
        summaries[name] = {
            "global_mean_absolute_pixel_delta": mean_effect,
            "saturated_pixel_fraction": float(np.mean(saturation[name])),
            "visible_pedestrian_actor_bbox_edge_visibility_ratio_median": _median(actor_edge[name]),
            "visible_pedestrian_actor_bbox_contrast_ratio_median": _median(actor_contrast[name]),
            "equal_area_control_roi_edge_visibility_ratio_median": _median(control_edge[name]),
            "residual_temporal_jerk_mean": _mean(jerks),
            "residual_temporal_jerk_p95": float(np.percentile(jerks, 95)) if jerks else None,
            "residual_temporal_jerk_normalized_by_mean_effect": (
                float(np.mean(jerks)) / max(mean_effect, 1e-6) if jerks else None
            ),
        }
    gates_spec = spec["frozen_visual_gates"]
    gate_results = {
        "all_saved_triplets_same_tick": all(row["same_tick"] for row in rows),
        "minimum_saved_frames": len(rows) >= int(gates_spec["minimum_saved_frames"]),
        "minimum_visible_pedestrian_frames": len(visible_walker_indices) >= int(gates_spec["minimum_visible_pedestrian_frames"]),
        "medium_global_mad_minimum": summaries["medium"]["global_mean_absolute_pixel_delta"] >= float(gates_spec["medium_global_mad_minimum"]),
        "heavy_global_mad_strictly_above_medium": summaries["heavy"]["global_mean_absolute_pixel_delta"] > summaries["medium"]["global_mean_absolute_pixel_delta"],
        "medium_actor_edge_visibility_ratio_maximum": (
            summaries["medium"]["visible_pedestrian_actor_bbox_edge_visibility_ratio_median"] is not None
            and summaries["medium"]["visible_pedestrian_actor_bbox_edge_visibility_ratio_median"] <= float(gates_spec["medium_actor_edge_visibility_ratio_maximum"])
        ),
        "medium_normalized_residual_temporal_jerk_maximum": (
            summaries["medium"]["residual_temporal_jerk_normalized_by_mean_effect"] is not None
            and summaries["medium"]["residual_temporal_jerk_normalized_by_mean_effect"] <= float(gates_spec["medium_normalized_residual_temporal_jerk_maximum"])
        ),
    }
    gif_path = output / "route203_native_glare_same_tick_front.gif"
    _save_gif(gif_frames, gif_path)
    contact_candidates = visible_walker_indices or list(range(len(rows)))
    selected_positions = np.linspace(0, len(contact_candidates) - 1, min(5, len(contact_candidates))).round().astype(int)
    selected_indices = [contact_candidates[index] for index in selected_positions]
    contact = np.concatenate([gif_frames[index] for index in selected_indices], axis=0)
    contact_path = output / "route203_native_glare_same_tick_contact_sheet.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise RuntimeError("failed to save native glare contact sheet")
    payload = {
        "schema": SCHEMA,
        "route_index": int(spec["route"]["route_index"]),
        "profiles": list(PROFILES),
        "frame_count": len(rows),
        "visible_pedestrian_frame_count": len(visible_walker_indices),
        "visible_pedestrian_capture_indices": visible_walker_indices,
        "summary": summaries,
        "per_frame": per_frame,
        "gates": gate_results,
        "eligible_for_one_preregistered_orion_proxy_read": all(gate_results.values()),
        "visuals": {
            "front_gif": str(gif_path.resolve()),
            "contact_sheet": str(contact_path.resolve()),
        },
        "outcome_fields_read": [],
        "orion_loaded": False,
        "adapter_loaded": False,
        "claim_boundary": "same-tick second-route renderer confirmation only; no UQ, model, closed-loop or safety claim",
    }
    report = output / "native_glare_same_tick_visual_analysis.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "gates": gate_results,
        "eligible_for_one_preregistered_orion_proxy_read": payload["eligible_for_one_preregistered_orion_proxy_read"],
    }, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
