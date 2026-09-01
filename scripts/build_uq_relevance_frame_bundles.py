#!/usr/bin/env python3
"""Build Stage2-L frame bundles and matched UQ-location counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import sha256_file
from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    build_task_relevance_map,
)


SCHEMA = "orion.uq_relevance_frame_bundle_batch.v1"
STAGE1_SEQUENCE_SCHEMA = "orion.stage1_observation_uq_sequence.v1"
CAMERA_DIRECTORIES = {
    "CAM_FRONT": "rgb_front_model_input",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve_reference(
    reference: Mapping[str, Any], base_dir: Path, name: str
) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def _ordered_camera_files(
    event_package: Mapping[str, Any], frame_index: int
) -> Tuple[List[Dict[str, Any]], Path]:
    inventory = event_package["camera_inventory"]
    files = []
    scenario_dir = None
    for view in CAMERA_ORDER:
        directory = CAMERA_DIRECTORIES[view]
        root = Path(inventory[directory]["path"])
        path = root / ("%04d.png" % frame_index)
        if not path.is_file():
            raise FileNotFoundError("missing aligned camera frame: %s" % path)
        files.append({
            "view": view,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        })
        scenario_dir = root.parent
    return files, scenario_dir


def _observation_sha(camera_files: List[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(item["sha256"] for item in camera_files).encode("ascii")
    ).hexdigest()


def _route_context(meta: Mapping[str, Any]) -> Dict[str, Any]:
    speedometer_mps = float(meta["speed"])
    if not math.isfinite(speedometer_mps):
        raise ValueError("current ego speedometer reading must be finite")
    payload = {
        "command": int(meta["command"]),
        "orion_unmodified_plan_right_forward_m": meta["plan"],
        "ego_state": {"speedometer_mps": speedometer_mps},
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema": "orion.route_context.v2",
        "payload": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _route_metadata(
    event_package: Mapping[str, Any],
    route_override: Mapping[str, str] = None,
) -> Dict[str, Any]:
    reference = event_package["source_files"].get("batch_manifest")
    if not reference:
        if route_override and route_override.get("town") and route_override.get("scenario_type"):
            return {
                "route_index": int(event_package["route"]["route_index"]),
                "town": str(route_override["town"]),
                "scenario_type": str(route_override["scenario_type"]),
                "metadata_source": "explicit_development_smoke_override",
            }
        raise ValueError(
            "event package lacks batch-manifest provenance; development smoke requires explicit town/scenario override"
        )
    path = _resolve_reference(reference, Path("/"), "batch manifest")
    batch = _load_json(path)
    route_index = int(event_package["route"]["route_index"])
    matches = [row for row in batch["routes"] if int(row["route_index"]) == route_index]
    if len(matches) != 1:
        raise ValueError("batch manifest does not uniquely identify the route")
    return matches[0]


def _load_stage1_sequence(
    manifest_path: Path,
    event_package_path: Path,
    frame_index: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != STAGE1_SEQUENCE_SCHEMA:
        raise ValueError("unsupported Stage-1 sequence manifest schema")
    if manifest.get("control_influence") is not False:
        raise ValueError("offline Stage-1 sequence must attest no control influence")
    if int(manifest.get("latest_frame_index", -1)) != frame_index:
        raise ValueError("Stage-1 sequence is not aligned to selected frame")
    if manifest.get("camera_order") != list(CAMERA_ORDER):
        raise ValueError("Stage-1 sequence camera order mismatch")
    if manifest.get("event_package_sha256") != sha256_file(event_package_path):
        raise ValueError("Stage-1 sequence event-package provenance mismatch")
    checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
    if len(checkpoint_sha) != 64:
        raise ValueError("Stage-1 sequence checkpoint SHA-256 is absent")
    reference = manifest.get("uncertainty") or {}
    path = _resolve_reference(reference, manifest_path.parent, "Stage-1 UQ sequence")
    with np.load(path, allow_pickle=False) as payload:
        uq = np.asarray(payload[reference.get("key", "uncertainty")], dtype=np.float32)
        components = np.asarray(
            payload[reference.get("component_key", "uncertainty_components")],
            dtype=np.float32,
        )
    if uq.ndim == 3:
        uq = uq[None, ...]
    if uq.ndim != 4 or uq.shape[1] != len(CAMERA_ORDER):
        raise ValueError("Stage-1 UQ sequence must have shape [T,6,H,W]")
    if not np.all(np.isfinite(uq)) or np.any(uq < 0.0) or np.any(uq > 1.0):
        raise ValueError("Stage-1 UQ sequence must be normalized to [0,1]")
    if tuple(reference.get("shape", [])) not in (uq.shape, uq.shape[1:]):
        raise ValueError("Stage-1 sequence shape declaration mismatch")
    if components.ndim != 5 or components.shape[:-1] != uq.shape:
        raise ValueError("Stage-1 component sequence must have shape [T,6,H,W,C]")
    if not np.all(np.isfinite(components)) or np.any(components < 0.0) or np.any(components > 1.0):
        raise ValueError("Stage-1 UQ components must be normalized to [0,1]")
    if tuple(reference.get("component_shape", [])) != components.shape:
        raise ValueError("Stage-1 component shape declaration mismatch")
    names = manifest.get("component_names") or []
    if len(names) != components.shape[-1] or len(set(names)) != len(names):
        raise ValueError("Stage-1 component names are missing or inconsistent")
    if not np.allclose(uq, components.mean(axis=-1), atol=1e-5):
        raise ValueError("Stage-1 scalar UQ is not the mean of its components")
    return uq, components, manifest


def _counterfactual_centers(
    relevance: np.ndarray, radius: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if relevance.ndim != 3:
        raise ValueError("relevance must have shape [V,H,W]")
    _, height, width = relevance.shape
    if height < 2 * radius + 1 or width < 2 * radius + 1:
        raise ValueError("relevance grid is too small for matched spatial support")
    interior = np.zeros_like(relevance, dtype=bool)
    interior[:, radius : height - radius, radius : width - radius] = True
    if not np.any(interior):
        raise ValueError("no interior counterfactual support exists")
    interior_values = np.where(interior, relevance, -np.inf)
    on_index = tuple(int(value) for value in np.unravel_index(
        int(np.argmax(interior_values)), relevance.shape
    ))
    if float(relevance[on_index]) <= 0.0:
        raise ValueError("task relevance has no positive interior support")

    view, on_y, on_x = on_index
    candidates = []
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            distance = float(np.hypot(y - on_y, x - on_x))
            if distance < 2 * radius + 1:
                continue
            patch = relevance[
                view,
                y - radius : y + radius + 1,
                x - radius : x + radius + 1,
            ]
            candidates.append(
                (
                    float(np.max(patch)),
                    float(np.mean(patch)),
                    abs(y - on_y),
                    abs(distance - (3 * radius + 1)),
                    y,
                    x,
                )
            )
    if not candidates:
        raise ValueError("no same-view off-path support can be matched")
    best = min(candidates)
    off_index = (view, int(best[-2]), int(best[-1]))
    if float(best[1]) >= float(relevance[on_index]):
        raise ValueError("matched off-path support is not less relevant than on-path support")
    return on_index, off_index


def _gaussian_region(
    shape: Tuple[int, int, int], center: Tuple[int, int, int], radius: int
) -> np.ndarray:
    support = np.zeros(shape, dtype=np.float32)
    view, center_y, center_x = center
    sigma = max(1.0, radius / 1.5)
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            distance_squared = (y - center_y) ** 2 + (x - center_x) ** 2
            if distance_squared <= radius ** 2:
                support[view, y, x] = np.exp(
                    -distance_squared / (2.0 * sigma * sigma)
                )
    if float(support.max()) <= 0.0:
        raise ValueError("empty counterfactual spatial support")
    support /= float(support.max())
    return support


def _controlled_counterfactual(
    uq: np.ndarray,
    components: np.ndarray,
    relevance: np.ndarray,
    variant: str,
    peak: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if variant == "observed":
        return uq.copy(), components.copy(), {"support_type": "frozen_stage1_observed"}
    if variant == "zero_uq":
        return (
            np.zeros_like(uq),
            np.zeros_like(components),
            {"support_type": "zero_everywhere"},
        )
    if variant == "view_shuffled_uq":
        return (
            np.roll(uq, shift=1, axis=1),
            np.roll(components, shift=1, axis=1),
            {"support_type": "cyclic_view_shift", "view_shift": 1},
        )
    if variant not in ("on_path_uq", "off_path_uq"):
        raise ValueError("unsupported counterfactual variant %s" % variant)
    radius = max(1, min(3, (min(uq.shape[-2:]) - 1) // 4))
    on_index, off_index = _counterfactual_centers(relevance, radius)
    index = on_index if variant == "on_path_uq" else off_index
    spatial_support = _gaussian_region(relevance.shape, index, radius)
    target = np.zeros_like(uq)
    target_components = np.zeros_like(components)
    temporal = np.linspace(max(0.2, peak * 0.4), peak, uq.shape[0])
    for time_index, value in enumerate(temporal):
        target[time_index] = spatial_support * value
        target_components[time_index] = (spatial_support * value)[..., None]
    weighted_relevance = float(
        np.sum(spatial_support * relevance) / np.maximum(np.sum(spatial_support), 1e-8)
    )
    return target, target_components, {
        "support_type": "matched_local_gaussian_region_v1",
        "center_view_y_x": list(index),
        "matched_on_path_center_view_y_x": list(on_index),
        "matched_off_path_center_view_y_x": list(off_index),
        "radius_patches": radius,
        "latest_peak": float(peak),
        "latest_spatial_sum": float(target[-1].sum()),
        "latest_nonzero_patches": int(np.count_nonzero(target[-1])),
        "support_weighted_relevance": weighted_relevance,
        "same_view_matched_pair": on_index[0] == off_index[0],
    }


def build_frame_bundles(
    *,
    event_package_path: Path,
    stage1_manifest_path: Path,
    split: str,
    output_dir: Path,
    variants: Tuple[str, ...],
    counterfactual_peak: float,
    route_override: Mapping[str, str] = None,
    selected_frame_index: int = None,
) -> Dict[str, Any]:
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty frame-bundle output")
    event_package = _load_json(event_package_path)
    if event_package.get("schema") != "orion.scenario_event_package.v1":
        raise ValueError("unsupported event-package schema")
    if not event_package.get("qa_input_ready") or not event_package["runtime"]["valid"]:
        raise ValueError("event package is not runtime-valid and QA-ready")
    critical = event_package.get("critical_event")
    if not critical:
        raise ValueError("event package has no actor-grounded critical event")
    available = sorted(
        int(path.stem)
        for path in Path(event_package["camera_inventory"]["rgb_front"]["path"]).glob("*.png")
    )
    if not available:
        raise ValueError("event package has no saved camera frames")
    if selected_frame_index is None:
        expected = float(critical["step"]) / 10.0
        frame_index = min(available, key=lambda value: (abs(value - expected), value))
    else:
        frame_index = int(selected_frame_index)
        if frame_index not in available:
            raise ValueError("selected saved frame is absent from the event package")
    camera_files, scenario_dir = _ordered_camera_files(event_package, frame_index)
    meta_path = scenario_dir / "meta" / ("%04d.json" % frame_index)
    meta = _load_json(meta_path)
    uq, uq_components, stage1_manifest = _load_stage1_sequence(
        stage1_manifest_path, event_package_path, frame_index
    )
    geometry = build_task_relevance_map(
        meta["plan"], meta["closedloop_safety"], patch_hw=tuple(uq.shape[-2:])
    )
    route = _route_metadata(event_package, route_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "maps"
    map_dir.mkdir()
    relevance_path = map_dir / "task_relevance.npz"
    np.savez_compressed(
        relevance_path,
        relevance=geometry.relevance,
        route_corridor=geometry.route_corridor,
        relevant_actor_support=geometry.relevant_actor_support,
    )
    geometry_path = output_dir / "task_relevance_geometry.json"
    geometry_payload = dict(geometry.provenance)
    geometry_payload.update({
        "relevant_actor_ids": list(geometry.relevant_actor_ids),
        "route_point_coverage": geometry.route_point_coverage,
        "source_meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
    })
    geometry_path.write_text(
        json.dumps(geometry_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    event_id = "route%s_step%s" % (
        event_package["route"]["route_index"], critical["step"]
    )
    frame_id = "saved_%04d" % frame_index
    group_id = event_id + "_" + frame_id
    route_context = _route_context(meta)
    bundles = []
    seen_uq_arrays = set()
    for variant in variants:
        (
            counterfactual_uq,
            counterfactual_components,
            counterfactual_support,
        ) = _controlled_counterfactual(
            uq, uq_components, geometry.relevance, variant, counterfactual_peak
        )
        array_identity = hashlib.sha256(
            counterfactual_uq.tobytes() + counterfactual_components.tobytes()
        ).hexdigest()
        if array_identity in seen_uq_arrays:
            continue
        seen_uq_arrays.add(array_identity)
        uq_path = map_dir / ("uq_%s.npz" % variant)
        np.savez_compressed(
            uq_path,
            uncertainty=counterfactual_uq,
            uncertainty_components=counterfactual_components,
        )
        uq_sha = sha256_file(uq_path)
        source = (
            "frozen_stage1_observation_adapter"
            if variant == "observed"
            else "controlled_stage1_uq_counterfactual"
        )
        bundle = {
            "schema": "orion.uq_relevance_frame_bundle.v1",
            "split": split,
            "event_id": event_id,
            "frame_id": frame_id,
            "town": route["town"],
            "scenario_family": route["scenario_type"],
            "route": {"route_id": str(event_package["route"]["route_index"])},
            "counterfactual": {
                "group_id": group_id,
                "variant": variant,
                "spatial_support": counterfactual_support,
            },
            "model_input": {
                "observation": {
                    "camera_files": camera_files,
                    "observation_sha256": _observation_sha(camera_files),
                },
                "route_context": route_context,
                "stage1_observation_uq": {
                    "path": str(uq_path.resolve()),
                    "sha256": uq_sha,
                    "shape": list(counterfactual_uq.shape),
                    "component_key": "uncertainty_components",
                    "component_shape": list(counterfactual_components.shape),
                    "component_names": stage1_manifest["component_names"],
                    "source": source,
                    "checkpoint_sha256": stage1_manifest["checkpoint_sha256"],
                    "normalization": stage1_manifest["normalization"],
                    "control_influence": False,
                },
            },
            "supervision": {
                "task_relevance": {
                    "path": str(relevance_path.resolve()),
                    "sha256": sha256_file(relevance_path),
                    "shape": list(geometry.relevance.shape),
                    "source": "projected_actor_route_corridor_geometry_v1",
                    "uses_corruption_label": False,
                    "privileged_for_supervision_only": True,
                    "geometry_manifest": {
                        "path": str(geometry_path.resolve()),
                        "sha256": sha256_file(geometry_path),
                    },
                }
            },
            "provenance": {
                "event_package": {
                    "path": str(event_package_path.resolve()),
                    "sha256": sha256_file(event_package_path),
                },
                "stage1_sequence_manifest": {
                    "path": str(stage1_manifest_path.resolve()),
                    "sha256": sha256_file(stage1_manifest_path),
                },
                "selected_saved_frame_index": frame_index,
                "critical_control_step": int(critical["step"]),
                "frame_selection": (
                    "explicit_fixed_temporal_keyframe"
                    if selected_frame_index is not None
                    else "nearest_actor_grounded_critical_step"
                ),
                "counterfactual_is_adapter_validation": False,
            },
        }
        bundle_path = output_dir / ("frame_bundle_%s.json" % variant)
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        bundles.append({
            "variant": variant,
            "path": str(bundle_path.resolve()),
            "sha256": sha256_file(bundle_path),
        })
    report = {
        "schema": SCHEMA,
        "status": "pending_human_geometry_review",
        "event_id": event_id,
        "frame_id": frame_id,
        "split": split,
        "town": route["town"],
        "scenario_family": route["scenario_type"],
        "route_point_coverage": geometry.route_point_coverage,
        "relevant_actor_ids": list(geometry.relevant_actor_ids),
        "task_relevance_support_mode": geometry.provenance.get(
            "support_mode", "unknown"
        ),
        "bundles": bundles,
        "formal_training_eligible": False,
        "claim_boundary": (
            "Counterfactual UQ-location variants supervise VLM semantics only; "
            "they do not validate the Stage-1 adapter or closed-loop safety."
        ),
    }
    report_path = output_dir / "frame_bundle_batch.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--stage1-sequence-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="observed,zero_uq,on_path_uq,off_path_uq,view_shuffled_uq",
    )
    parser.add_argument("--counterfactual-peak", type=float, default=0.9)
    parser.add_argument("--town")
    parser.add_argument("--scenario-family")
    parser.add_argument("--selected-frame-index", type=int)
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    report = build_frame_bundles(
        event_package_path=args.event_package.resolve(),
        stage1_manifest_path=args.stage1_sequence_manifest.resolve(),
        split=args.split,
        output_dir=args.output_dir.resolve(),
        variants=variants,
        counterfactual_peak=args.counterfactual_peak,
        route_override=(
            {"town": args.town, "scenario_type": args.scenario_family}
            if args.town or args.scenario_family else None
        ),
        selected_frame_index=args.selected_frame_index,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
