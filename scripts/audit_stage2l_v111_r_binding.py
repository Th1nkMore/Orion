#!/usr/bin/env python3
"""CPU-only audit of the frozen Stage2-L v11.1 contextual-R interface.

The audit does not train or run ORION.  It binds the terminal v11.1
controlled-U outcome to the frozen v10.1 spatial maps, immutable same-view
feature cache, geometry manifests, and original frame metadata.  Its purpose is
to distinguish deterministic target/projection defects from view/event
coverage and held-out representation failures before another GPU run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TASK_RELEVANCE_GEOMETRY_SCHEMA,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_v11_1_r_binding_audit.v1"
SPATIAL_MAP_SHAPE = (1, 6, 10, 10)
FEATURE_SHAPE = (6, 10, 10, 1024)
RAW_MAP_SHAPE = (6, 40, 40)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _read_records(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(reference: Mapping[str, Any], base: Path, label: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (label, path))
    expected = str(reference.get("sha256", ""))
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError("%s hash differs: %s" % (label, path))
    return path


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _pool_target(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != RAW_MAP_SHAPE:
        raise ValueError("raw relevance map shape differs")
    pooled = F.adaptive_avg_pool2d(torch.from_numpy(value).unsqueeze(0), (10, 10))
    return pooled.numpy()


def _support(value: np.ndarray, fraction: float) -> Tuple[np.ndarray, float]:
    value = np.asarray(value, dtype=np.float32)
    peak = float(value.max())
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("task relevance requires finite positive support")
    threshold = peak * float(fraction)
    return value >= threshold, threshold


def _centroid(mask: np.ndarray) -> Optional[list[float]]:
    points = np.argwhere(mask)
    if not len(points):
        return None
    height, width = mask.shape
    mean_y, mean_x = points.mean(axis=0)
    return [
        float(mean_y / max(height - 1, 1)),
        float(mean_x / max(width - 1, 1)),
    ]


def _feature_centroid(
    feature: np.ndarray, target: np.ndarray, support: np.ndarray
) -> Optional[np.ndarray]:
    if feature.shape != (10, 10, 1024):
        raise ValueError("same-view feature grid shape differs")
    if target.shape != (10, 10) or support.shape != (10, 10):
        raise ValueError("target/support grid shape differs")
    if not bool(support.any()):
        return None
    weights = np.where(support, target, 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        weights = support.astype(np.float64)
        total = float(weights.sum())
    centroid = (feature.astype(np.float64) * weights[..., None]).sum(axis=(0, 1))
    centroid /= total
    norm = float(np.linalg.norm(centroid))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("feature centroid is non-finite or zero")
    return (centroid / norm).astype(np.float32)


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def _median(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.median(values))


def view_support_statistics(
    *,
    target: np.ndarray,
    probability: np.ndarray,
    route: np.ndarray,
    actor: np.ndarray,
    support_fraction: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Return exact per-view target/prediction statistics on the 10x10 grid."""

    target = np.asarray(target, dtype=np.float32)
    probability = np.asarray(probability, dtype=np.float32)
    route = np.asarray(route, dtype=np.float32)
    actor = np.asarray(actor, dtype=np.float32)
    expected = (6, 10, 10)
    if any(value.shape != expected for value in (target, probability, route, actor)):
        raise ValueError("pooled view support inputs must have shape [6,10,10]")
    if not all(np.isfinite(value).all() for value in (target, probability, route, actor)):
        raise ValueError("pooled view support input is non-finite")
    foreground, threshold = _support(target, support_fraction)
    predicted = probability >= threshold
    rows: Dict[str, Any] = {}
    total_foreground = int(foreground.sum())
    total_background = int((~foreground).sum())
    for index, view in enumerate(CAMERA_ORDER):
        fg = foreground[index]
        bg = ~fg
        fg_count = int(fg.sum())
        bg_count = int(bg.sum())
        rows[view] = {
            "target_peak": float(target[index].max()),
            "target_mass": float(target[index].sum()),
            "foreground_cells": fg_count,
            "foreground_brier_weight_share": float(fg_count / total_foreground),
            "background_brier_weight_share": float(bg_count / total_background),
            "support_centroid_yx": _centroid(fg),
            "route_mass": float(route[index].sum()),
            "actor_support_cells": int((actor[index] > 0.0).sum()),
            "foreground_recall": _fraction(int((predicted[index] & fg).sum()), fg_count),
            "background_false_positive_rate": _fraction(
                int((predicted[index] & bg).sum()), bg_count
            ),
            "foreground_mean_probability": (
                float(probability[index][fg].mean()) if fg_count else None
            ),
            "background_mean_probability": float(probability[index][bg].mean()),
            "prediction_mass": float(probability[index].sum()),
        }
    return rows, foreground


def nearest_same_view_training_support(
    *,
    query_feature: np.ndarray,
    query_centroid_yx: Sequence[float],
    training_rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the closest train group in the same camera by cosine similarity."""

    if not training_rows:
        return None
    query = np.asarray(query_feature, dtype=np.float32)
    if query.shape != (1024,) or not np.isfinite(query).all():
        raise ValueError("query feature centroid differs")
    best = None
    for row in training_rows:
        candidate = np.asarray(row["feature"], dtype=np.float32)
        cosine = float(np.dot(query, candidate))
        center = row["centroid_yx"]
        distance = float(
            math.hypot(
                float(query_centroid_yx[0]) - float(center[0]),
                float(query_centroid_yx[1]) - float(center[1]),
            )
        )
        key = (cosine, -distance, str(row["group_id"]))
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "cosine_similarity": cosine,
                    "support_centroid_distance": distance,
                    "nearest_train_group": str(row["group_id"]),
                    "nearest_train_event": str(row["event_id"]),
                },
            )
    return best[1]


def _aggregate_event(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    controlled = [bool(row["controlled_on_over_off_passed"]) for row in rows]
    target_views = Counter(str(row["target_dominant_view"]) for row in rows)
    prediction_views = Counter(str(row["prediction_dominant_view"]) for row in rows)
    support_modes = Counter(str(row["support_mode"]) for row in rows)
    return {
        "split": str(rows[0]["split"]),
        "group_count": len(rows),
        "controlled_on_over_off_pass_count": int(sum(controlled)),
        "controlled_on_over_off_pass_fraction": float(np.mean(controlled)),
        "mean_group_foreground_recall": _mean(
            row["group_foreground_recall"] for row in rows
        ),
        "median_route_point_coverage": _median(
            row["route_point_coverage"] for row in rows
        ),
        "target_dominant_view_counts": dict(sorted(target_views.items())),
        "prediction_dominant_view_counts": dict(sorted(prediction_views.items())),
        "support_mode_counts": dict(sorted(support_modes.items())),
    }


def audit(
    *,
    records_path: Path,
    spatial_maps_path: Path,
    view_feature_cache_path: Path,
    v111_report_path: Path,
    support_fraction: float = 0.1,
    expected_record_count: Optional[int] = 1600,
    expected_group_count: Optional[int] = 80,
) -> Dict[str, Any]:
    records_path = records_path.resolve()
    spatial_maps_path = spatial_maps_path.resolve()
    view_feature_cache_path = view_feature_cache_path.resolve()
    v111_report_path = v111_report_path.resolve()
    records = _read_records(records_path)
    if expected_record_count is not None and len(records) != expected_record_count:
        raise ValueError("record count differs")

    selected: Dict[str, Dict[str, Any]] = {}
    for row in records:
        if (
            row.get("question_family") != "task_relevance"
            or row.get("counterfactual", {}).get("variant") != "observed"
        ):
            continue
        group_id = str(row["counterfactual"]["group_id"])
        if group_id in selected:
            raise ValueError("duplicate observed task-relevance group")
        selected[group_id] = row
    if expected_group_count is not None and len(selected) != expected_group_count:
        raise ValueError("observed group count differs")

    spatial_maps = torch.load(spatial_maps_path, map_location="cpu")
    feature_payload = torch.load(view_feature_cache_path, map_location="cpu")
    report = _read_json(v111_report_path)
    if (
        not isinstance(spatial_maps, dict)
        or set(spatial_maps) != set(selected)
        or feature_payload.get("schema")
        != "orion.stage2l_view_aligned_feature_cache.v1"
        or set(feature_payload.get("contexts", {})) != set(selected)
        or feature_payload.get("metadata", {}).get("camera_order")
        != list(CAMERA_ORDER)
        or report.get("schema") != "orion.stage2l_v11_identifiable_smoke.v1"
        or report.get("status")
        != "stopped_before_language_controlled_u_gate_failed"
        or report.get("optimizer_steps") != 0
    ):
        raise ValueError("frozen R audit input contract differs")

    report_groups: Dict[str, Mapping[str, Any]] = {}
    for split in ("train", "dev"):
        current = report.get("factorization_before", {}).get(split, {})
        for group_id, value in current.get("per_group", {}).items():
            if group_id in report_groups:
                raise ValueError("v11.1 report group appears twice")
            report_groups[str(group_id)] = value
    if set(report_groups) != set(selected):
        raise ValueError("v11.1 report group set differs")

    training_support: Dict[str, list[Dict[str, Any]]] = {
        view: [] for view in CAMERA_ORDER
    }
    group_rows: Dict[str, Dict[str, Any]] = {}
    group_features: Dict[Tuple[str, str], np.ndarray] = {}
    maximum_errors = {
        "rebuilt_relevance": 0.0,
        "rebuilt_route_corridor": 0.0,
        "rebuilt_actor_support": 0.0,
        "sidecar_relevance": 0.0,
        "pooled_spatial_target": 0.0,
    }

    for group_id, row in sorted(selected.items()):
        split = str(row["split"])
        if split not in ("train", "dev"):
            raise ValueError("audit may use only train/dev groups")
        geometry_ref = row["provenance"]["relevance_supervision"][
            "geometry_manifest"
        ]
        raw_ref = row["provenance"]["relevance_supervision"]
        sidecar_ref = row["target"]["map_sidecar"]
        geometry_path = _resolve(
            geometry_ref, records_path.parent, "geometry manifest %s" % group_id
        )
        raw_path = _resolve(
            raw_ref, records_path.parent, "raw relevance %s" % group_id
        )
        sidecar_path = _resolve(
            sidecar_ref, records_path.parent, "QA map sidecar %s" % group_id
        )
        geometry_manifest = _read_json(geometry_path)
        if (
            geometry_manifest.get("schema") != TASK_RELEVANCE_GEOMETRY_SCHEMA
            or geometry_manifest.get("camera_order") != list(CAMERA_ORDER)
            or geometry_manifest.get("patch_hw") != [40, 40]
        ):
            raise ValueError("geometry manifest contract differs: %s" % group_id)
        meta_ref = geometry_manifest["source_meta"]
        meta_path = _resolve(meta_ref, geometry_path.parent, "source meta %s" % group_id)
        meta = _read_json(meta_path)
        rebuilt = build_task_relevance_map(
            meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
        )
        if list(rebuilt.relevant_actor_ids) != list(
            geometry_manifest.get("relevant_actor_ids", [])
        ):
            raise ValueError("rebuilt relevant actor IDs differ: %s" % group_id)
        if not math.isclose(
            float(rebuilt.route_point_coverage),
            float(geometry_manifest["route_point_coverage"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("rebuilt route coverage differs: %s" % group_id)

        with np.load(raw_path, allow_pickle=False) as archive:
            raw_relevance = np.asarray(archive["relevance"], dtype=np.float32)
            raw_route = np.asarray(archive["route_corridor"], dtype=np.float32)
            raw_actor = np.asarray(
                archive["relevant_actor_support"], dtype=np.float32
            )
        with np.load(sidecar_path, allow_pickle=False) as archive:
            sidecar_relevance = np.asarray(
                archive[sidecar_ref["relevance_key"]], dtype=np.float32
            )
        if any(
            value.shape != RAW_MAP_SHAPE
            for value in (raw_relevance, raw_route, raw_actor, sidecar_relevance)
        ):
            raise ValueError("stored raw relevance shape differs: %s" % group_id)

        errors = {
            "rebuilt_relevance": float(
                np.max(np.abs(raw_relevance - rebuilt.relevance))
            ),
            "rebuilt_route_corridor": float(
                np.max(np.abs(raw_route - rebuilt.route_corridor))
            ),
            "rebuilt_actor_support": float(
                np.max(np.abs(raw_actor - rebuilt.relevant_actor_support))
            ),
            "sidecar_relevance": float(
                np.max(np.abs(sidecar_relevance - raw_relevance))
            ),
        }
        for key, value in errors.items():
            maximum_errors[key] = max(maximum_errors[key], value)

        pooled_target = _pool_target(raw_relevance)
        pooled_route = _pool_target(raw_route)[0]
        pooled_actor = _pool_target(raw_actor)[0]
        spatial_value = spatial_maps[group_id]
        probability = _numpy(spatial_value["probability"])
        stored_target = _numpy(spatial_value["target"])
        if probability.shape != SPATIAL_MAP_SHAPE or stored_target.shape != SPATIAL_MAP_SHAPE:
            raise ValueError("frozen spatial map shape differs: %s" % group_id)
        pooled_error = float(np.max(np.abs(stored_target - pooled_target)))
        maximum_errors["pooled_spatial_target"] = max(
            maximum_errors["pooled_spatial_target"], pooled_error
        )

        features = _numpy(feature_payload["contexts"][group_id])
        if features.shape != FEATURE_SHAPE:
            raise ValueError("view feature shape differs: %s" % group_id)
        per_view, foreground = view_support_statistics(
            target=pooled_target[0],
            probability=probability[0],
            route=pooled_route,
            actor=pooled_actor,
            support_fraction=support_fraction,
        )
        for view_index, view in enumerate(CAMERA_ORDER):
            centroid = _feature_centroid(
                features[view_index], pooled_target[0, view_index], foreground[view_index]
            )
            if centroid is None:
                continue
            group_features[(group_id, view)] = centroid
            if split == "train":
                training_support[view].append(
                    {
                        "group_id": group_id,
                        "event_id": str(row["event_id"]),
                        "feature": centroid,
                        "centroid_yx": per_view[view]["support_centroid_yx"],
                    }
                )

        target_mass = pooled_target[0].sum(axis=(1, 2))
        prediction_mass = probability[0].sum(axis=(1, 2))
        threshold = float(pooled_target.max()) * float(support_fraction)
        predicted = probability[0] >= threshold
        group_fg = foreground
        group_bg = ~group_fg
        report_row = report_groups[group_id]
        group_rows[group_id] = {
            "event_id": str(row["event_id"]),
            "split": split,
            "frame_id": str(row["frame_id"]),
            "town": str(row["town"]),
            "scenario_family": str(row["scenario_family"]),
            "support_mode": str(geometry_manifest["support_mode"]),
            "relevant_actor_count": len(rebuilt.relevant_actor_ids),
            "route_point_coverage": float(rebuilt.route_point_coverage),
            "target_dominant_view": CAMERA_ORDER[int(np.argmax(target_mass))],
            "prediction_dominant_view": CAMERA_ORDER[int(np.argmax(prediction_mass))],
            "target_supported_views": [
                CAMERA_ORDER[index]
                for index in range(6)
                if bool(group_fg[index].any())
            ],
            "group_foreground_recall": float(
                (predicted & group_fg).sum() / group_fg.sum()
            ),
            "group_background_false_positive_rate": float(
                (predicted & group_bg).sum() / group_bg.sum()
            ),
            "controlled_on_over_off_passed": bool(
                report_row["gates"]["on_over_off_margin"]
            ),
            "controlled_on_minus_off_risk_peak": float(
                report_row["on_minus_off_risk_peak"]
            ),
            "resulting_on_path_risk_view": str(
                report_row["structured_fields"]["on_path_uq"]["risk_view"]
            ),
            "per_view": per_view,
            "reconstruction_max_abs_error": {
                **errors,
                "pooled_spatial_target": pooled_error,
            },
        }

    # Add train-neighbour evidence only after the complete training index exists.
    for group_id, row in group_rows.items():
        neighbours = {}
        for view in row["target_supported_views"]:
            feature = group_features[(group_id, view)]
            neighbours[view] = nearest_same_view_training_support(
                query_feature=feature,
                query_centroid_yx=row["per_view"][view]["support_centroid_yx"],
                training_rows=training_support[view],
            )
        row["same_view_train_neighbours"] = neighbours

    by_event: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows.values():
        by_event[str(row["event_id"])].append(row)
    per_event = {
        event_id: _aggregate_event(rows)
        for event_id, rows in sorted(by_event.items())
    }

    per_view: Dict[str, Any] = {}
    for view in CAMERA_ORDER:
        train_rows = [row for row in group_rows.values() if row["split"] == "train"]
        dev_rows = [row for row in group_rows.values() if row["split"] == "dev"]
        train_positive = [
            row for row in train_rows if row["per_view"][view]["foreground_cells"] > 0
        ]
        dev_positive = [
            row for row in dev_rows if row["per_view"][view]["foreground_cells"] > 0
        ]
        dev_neighbours = [
            row["same_view_train_neighbours"][view]
            for row in dev_positive
            if row["same_view_train_neighbours"].get(view) is not None
        ]
        per_view[view] = {
            "train_positive_group_count": len(train_positive),
            "train_positive_event_count": len(
                {row["event_id"] for row in train_positive}
            ),
            "dev_positive_group_count": len(dev_positive),
            "dev_positive_event_count": len({row["event_id"] for row in dev_positive}),
            "train_mean_foreground_brier_weight_share": _mean(
                row["per_view"][view]["foreground_brier_weight_share"]
                for row in train_rows
            ),
            "dev_mean_foreground_recall_when_positive": _mean(
                row["per_view"][view]["foreground_recall"]
                for row in dev_positive
                if row["per_view"][view]["foreground_recall"] is not None
            ),
            "dev_controlled_pass_count_by_target_dominant_view": sum(
                int(row["controlled_on_over_off_passed"])
                for row in dev_rows
                if row["target_dominant_view"] == view
            ),
            "dev_group_count_by_target_dominant_view": sum(
                1 for row in dev_rows if row["target_dominant_view"] == view
            ),
            "dev_nearest_train_feature_cosine_median": _median(
                row["cosine_similarity"] for row in dev_neighbours
            ),
            "dev_nearest_train_support_centroid_distance_median": _median(
                row["support_centroid_distance"] for row in dev_neighbours
            ),
        }

    dev_rows = [row for row in group_rows.values() if row["split"] == "dev"]
    neighbour_cosines_by_control = {"passed": [], "failed": []}
    for row in dev_rows:
        values = [
            value["cosine_similarity"]
            for value in row["same_view_train_neighbours"].values()
            if value is not None
        ]
        if values:
            key = "passed" if row["controlled_on_over_off_passed"] else "failed"
            neighbour_cosines_by_control[key].append(float(max(values)))

    all_exact = all(value == 0.0 for value in maximum_errors.values())
    return {
        "schema": SCHEMA,
        "status": (
            "cpu_r_binding_audit_complete_exact_geometry_lineage"
            if all_exact
            else "cpu_r_binding_audit_geometry_lineage_mismatch"
        ),
        "passed": all_exact,
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "support_fraction_of_peak": float(support_fraction),
        "inputs": {
            "records": {"path": str(records_path), "sha256": _sha256(records_path)},
            "spatial_maps": {
                "path": str(spatial_maps_path),
                "sha256": _sha256(spatial_maps_path),
            },
            "view_feature_cache": {
                "path": str(view_feature_cache_path),
                "sha256": _sha256(view_feature_cache_path),
            },
            "v11_1_report": {
                "path": str(v111_report_path),
                "sha256": _sha256(v111_report_path),
            },
        },
        "counts": {
            "record_count": len(records),
            "group_count": len(group_rows),
            "event_count": len(per_event),
            "train_group_count": sum(row["split"] == "train" for row in group_rows.values()),
            "dev_group_count": sum(row["split"] == "dev" for row in group_rows.values()),
        },
        "lineage_and_projection_checks": {
            "camera_order": list(CAMERA_ORDER),
            "all_references_hash_verified": True,
            "all_geometry_rebuilt_from_original_meta": True,
            "all_raw_maps_match_current_geometry_builder_exactly": all_exact,
            "maximum_absolute_errors": maximum_errors,
        },
        "per_view": per_view,
        "per_event": per_event,
        "dev_nearest_train_feature_cosine_by_control_outcome": {
            key: {
                "group_count": len(values),
                "median": _median(values),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
            for key, values in neighbour_cosines_by_control.items()
        },
        "per_group": group_rows,
        "decision_locks": {
            "additional_r_epochs_authorized": False,
            "larger_r_model_authorized": False,
            "language_bridge_training_authorized": False,
            "formal_stage2l_authorized": False,
            "stage2p_authorized": False,
            "learned_u_closed_loop_authorized": False,
        },
        "claim_boundary": (
            "CPU-only frozen-artifact diagnosis. Exact reconstruction proves "
            "implementation/lineage consistency, not semantic correctness of the "
            "weak R target. Feature-neighbour statistics are descriptive and do "
            "not by themselves establish distributional causality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--spatial-maps", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--v11-1-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-fraction", type=float, default=0.1)
    parser.add_argument("--expected-record-count", type=int, default=1600)
    parser.add_argument("--expected-group-count", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite R-binding audit output")
    value = audit(
        records_path=args.records,
        spatial_maps_path=args.spatial_maps,
        view_feature_cache_path=args.view_feature_cache,
        v111_report_path=args.v11_1_report,
        support_fraction=args.support_fraction,
        expected_record_count=args.expected_record_count,
        expected_group_count=args.expected_group_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "passed": value["passed"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
