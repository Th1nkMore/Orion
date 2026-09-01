#!/usr/bin/env python3
"""CPU audit of factorized route/actor R targets on the frozen 80 groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_stage2l_v111_r_binding import (
    RAW_MAP_SHAPE,
    SPATIAL_MAP_SHAPE,
    _numpy,
    _pool_target,
    _read_json,
    _read_records,
    _resolve,
)
from uq_estimator.stage2l_factorized_relevance_v121 import (
    COMPONENT_ORDER,
    factorized_relevance_terms_v121,
)
from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TASK_RELEVANCE_GEOMETRY_SCHEMA,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_v12_1_factorized_r_preflight.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v12_1_factorized_r_cpu_preflight.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(np.mean(values)) if values else None


def component_support_statistics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    support_fraction: float,
) -> Dict[str, Dict[str, Any]]:
    target = np.asarray(target, dtype=np.float32)
    probability = np.asarray(probability, dtype=np.float32)
    if target.shape != (6, 10, 10) or probability.shape != target.shape:
        raise ValueError("component statistics require [6,10,10]")
    if not np.isfinite(target).all() or not np.isfinite(probability).all():
        raise ValueError("component statistics inputs must be finite")
    peak = float(target.max())
    threshold = peak * float(support_fraction) if peak > 0.0 else 0.5
    foreground = target >= threshold if peak > 0.0 else np.zeros_like(target, bool)
    predicted = probability >= threshold
    rows: Dict[str, Dict[str, Any]] = {}
    for view_index, view in enumerate(CAMERA_ORDER):
        fg = foreground[view_index]
        bg = ~fg
        fg_count = int(fg.sum())
        rows[view] = {
            "positive": bool(fg_count),
            "foreground_cells": fg_count,
            "target_peak": float(target[view_index].max()),
            "target_mass": float(target[view_index].sum()),
            "foreground_recall": (
                float((predicted[view_index] & fg).sum() / fg_count)
                if fg_count
                else None
            ),
            "background_false_positive_rate": float(
                (predicted[view_index] & bg).sum() / bg.sum()
            ),
            "foreground_cell_share_within_component": (
                float(fg_count / foreground.sum())
                if bool(foreground.any())
                else 0.0
            ),
        }
    return rows


def identifiable_component_views(
    aggregate: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    minimum_train_events: int,
    minimum_dev_events: int,
) -> Dict[str, Any]:
    supported = []
    unsupported = []
    for component in COMPONENT_ORDER:
        for view in CAMERA_ORDER:
            train_count = int(aggregate["train"][component][view]["positive_event_count"])
            dev_count = int(aggregate["dev"][component][view]["positive_event_count"])
            row = {
                "component": component,
                "view": view,
                "train_positive_event_count": train_count,
                "dev_positive_event_count": dev_count,
            }
            if train_count >= minimum_train_events and dev_count >= minimum_dev_events:
                supported.append(row)
            else:
                row["reason"] = (
                    "insufficient_train_events"
                    if train_count < minimum_train_events
                    else "no_positive_dev_event"
                )
                unsupported.append(row)
    return {"supported": supported, "unsupported": unsupported}


def _aggregate(
    per_group: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    buckets: Dict[tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "positive_groups": 0,
            "positive_events": set(),
            "foreground_cells": 0,
            "foreground_recalls": [],
            "background_fprs": [],
            "current_shares": [],
            "proposed_shares": [],
        }
    )
    for group in per_group.values():
        split = str(group["split"])
        for component in COMPONENT_ORDER:
            active_views = [
                view
                for view in CAMERA_ORDER
                if group["components"][component][view]["positive"]
            ]
            proposed_share = 1.0 / len(active_views) if active_views else 0.0
            for view in CAMERA_ORDER:
                value = group["components"][component][view]
                bucket = buckets[(split, component, view)]
                bucket["background_fprs"].append(
                    float(value["background_false_positive_rate"])
                )
                bucket["current_shares"].append(
                    float(value["foreground_cell_share_within_component"])
                )
                bucket["proposed_shares"].append(
                    proposed_share if value["positive"] else 0.0
                )
                if value["positive"]:
                    bucket["positive_groups"] += 1
                    bucket["positive_events"].add(str(group["event_id"]))
                    bucket["foreground_cells"] += int(value["foreground_cells"])
                    bucket["foreground_recalls"].append(
                        float(value["foreground_recall"])
                    )
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split in ("train", "dev"):
        result[split] = {}
        for component in COMPONENT_ORDER:
            result[split][component] = {}
            for view in CAMERA_ORDER:
                value = buckets[(split, component, view)]
                result[split][component][view] = {
                    "positive_group_count": int(value["positive_groups"]),
                    "positive_event_count": len(value["positive_events"]),
                    "foreground_cell_count": int(value["foreground_cells"]),
                    "current_single_union_r_mean_foreground_recall": _mean(
                        value["foreground_recalls"]
                    ),
                    "current_single_union_r_mean_background_false_positive_rate": _mean(
                        value["background_fprs"]
                    ),
                    "current_mean_foreground_cell_share": _mean(
                        value["current_shares"]
                    ),
                    "factorized_equal_active_view_mean_foreground_share": _mean(
                        value["proposed_shares"]
                    ),
                }
    return result


def audit(
    *,
    protocol_path: Path,
    records_path: Path,
    spatial_maps_path: Path,
) -> Dict[str, Any]:
    protocol = _read_json(protocol_path.resolve())
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("factorized R protocol schema differs")
    for name, path in (
        ("records", records_path),
        ("spatial_maps_step120", spatial_maps_path),
    ):
        expected = protocol["frozen_data"][name]["sha256"]
        if _sha256(path.resolve()) != expected:
            raise ValueError("factorized R frozen input hash differs: %s" % name)
    for value in protocol["predecessors"].values():
        path = (PROJECT_ROOT / value["path"]).resolve()
        if _sha256(path) != value["sha256"]:
            raise ValueError("factorized R predecessor hash differs")

    records = _read_records(records_path.resolve())
    selected: Dict[str, Mapping[str, Any]] = {}
    for row in records:
        if (
            row.get("question_family") != "task_relevance"
            or row.get("counterfactual", {}).get("variant") != "observed"
        ):
            continue
        group_id = str(row["counterfactual"]["group_id"])
        if group_id in selected:
            raise ValueError("factorized R group appears twice")
        selected[group_id] = row
    if len(records) != 1600 or len(selected) != 80:
        raise ValueError("factorized R frozen record/group count differs")

    spatial_maps = torch.load(
        spatial_maps_path.resolve(), map_location="cpu", weights_only=True
    )
    if not isinstance(spatial_maps, dict) or set(spatial_maps) != set(selected):
        raise ValueError("factorized R spatial-map group set differs")

    per_group: Dict[str, Dict[str, Any]] = {}
    target_batches = []
    maximum_errors = {
        "stored_union_equals_component_max": 0.0,
        "rebuilt_union": 0.0,
        "rebuilt_route": 0.0,
        "rebuilt_actor": 0.0,
        "consumer_union": 0.0,
    }
    support_mode_counts: Dict[str, Dict[str, int]] = {
        "train": defaultdict(int),
        "dev": defaultdict(int),
    }
    for group_id, row in sorted(selected.items()):
        split = str(row["split"])
        if split not in ("train", "dev"):
            raise ValueError("factorized R may read only train/dev")
        supervision = row["provenance"]["relevance_supervision"]
        geometry_path = _resolve(
            supervision["geometry_manifest"],
            records_path.parent,
            "factorized geometry %s" % group_id,
        )
        raw_path = _resolve(
            supervision,
            records_path.parent,
            "factorized raw maps %s" % group_id,
        )
        geometry_manifest = _read_json(geometry_path)
        if (
            geometry_manifest.get("schema") != TASK_RELEVANCE_GEOMETRY_SCHEMA
            or geometry_manifest.get("camera_order") != list(CAMERA_ORDER)
            or geometry_manifest.get("patch_hw") != [40, 40]
        ):
            raise ValueError("factorized geometry manifest differs")
        meta_path = _resolve(
            geometry_manifest["source_meta"],
            geometry_path.parent,
            "factorized source meta %s" % group_id,
        )
        meta = _read_json(meta_path)
        rebuilt = build_task_relevance_map(
            meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
        )
        with np.load(raw_path, allow_pickle=False) as archive:
            union = np.asarray(archive["relevance"], dtype=np.float32)
            route = np.asarray(archive["route_corridor"], dtype=np.float32)
            actor = np.asarray(archive["relevant_actor_support"], dtype=np.float32)
        if any(value.shape != RAW_MAP_SHAPE for value in (union, route, actor)):
            raise ValueError("factorized raw map shape differs")
        errors = {
            "stored_union_equals_component_max": float(
                np.max(np.abs(union - np.maximum(route, actor)))
            ),
            "rebuilt_union": float(np.max(np.abs(union - rebuilt.relevance))),
            "rebuilt_route": float(np.max(np.abs(route - rebuilt.route_corridor))),
            "rebuilt_actor": float(
                np.max(np.abs(actor - rebuilt.relevant_actor_support))
            ),
        }
        pooled_union = _pool_target(union)
        pooled_route = _pool_target(route)
        pooled_actor = _pool_target(actor)
        frozen_value = spatial_maps[group_id]
        probability = _numpy(frozen_value["probability"])
        frozen_target = _numpy(frozen_value["target"])
        if probability.shape != SPATIAL_MAP_SHAPE or frozen_target.shape != SPATIAL_MAP_SHAPE:
            raise ValueError("factorized frozen map shape differs")
        errors["consumer_union"] = float(
            np.max(np.abs(frozen_target - pooled_union))
        )
        for name, error in errors.items():
            maximum_errors[name] = max(maximum_errors[name], error)

        component_targets = np.stack(
            (pooled_route[0], pooled_actor[0]), axis=0
        ).astype(np.float32)
        target_batches.append(component_targets)
        components = {
            "route": component_support_statistics(
                pooled_route[0], probability[0], support_fraction=0.1
            ),
            "actor": component_support_statistics(
                pooled_actor[0], probability[0], support_fraction=0.1
            ),
        }
        route_present = bool(np.any(pooled_route > 0.0))
        actor_present = bool(np.any(pooled_actor > 0.0))
        if route_present and actor_present:
            component_mode = "route_and_actor"
        elif route_present:
            component_mode = "route_only"
        elif actor_present:
            component_mode = "actor_only"
        else:
            raise ValueError("factorized group has no component support")
        support_mode_counts[split][component_mode] += 1
        per_group[group_id] = {
            "event_id": str(row["event_id"]),
            "split": split,
            "frame_id": str(row["frame_id"]),
            "town": str(row["town"]),
            "scenario_family": str(row["scenario_family"]),
            "component_mode": component_mode,
            "components": components,
            "reconstruction_max_abs_error": errors,
        }

    target_tensor = torch.from_numpy(np.stack(target_batches, axis=0))
    probe_logits = torch.zeros_like(target_tensor, requires_grad=True)
    terms = factorized_relevance_terms_v121(probe_logits, target_tensor)
    terms.loss.backward()
    objective_finite = bool(
        torch.isfinite(terms.loss)
        and probe_logits.grad is not None
        and torch.isfinite(probe_logits.grad).all()
    )
    aggregate = _aggregate(per_group)
    rule = protocol["cpu_audit"]["identifiable_component_view_rule"]
    identifiable = identifiable_component_views(
        aggregate,
        minimum_train_events=int(
            rule["minimum_independent_train_events_with_positive_support"]
        ),
        minimum_dev_events=int(
            rule["minimum_independent_dev_events_with_positive_support"]
        ),
    )
    exact = all(value == 0.0 for value in maximum_errors.values())
    return {
        "schema": SCHEMA,
        "status": (
            "factorized_r_cpu_preflight_passed"
            if exact and objective_finite and identifiable["supported"]
            else "factorized_r_cpu_preflight_failed"
        ),
        "passed": bool(exact and objective_finite and identifiable["supported"]),
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "inputs": {
            "protocol": {
                "path": str(protocol_path.resolve()),
                "sha256": _sha256(protocol_path.resolve()),
            },
            "records": {
                "path": str(records_path.resolve()),
                "sha256": _sha256(records_path.resolve()),
            },
            "spatial_maps": {
                "path": str(spatial_maps_path.resolve()),
                "sha256": _sha256(spatial_maps_path.resolve()),
            },
        },
        "counts": {
            "records": len(records),
            "groups": len(per_group),
            "train_groups": sum(row["split"] == "train" for row in per_group.values()),
            "dev_groups": sum(row["split"] == "dev" for row in per_group.values()),
            "events": len({row["event_id"] for row in per_group.values()}),
        },
        "lineage": {
            "all_component_and_union_maps_exact": exact,
            "maximum_absolute_errors": maximum_errors,
        },
        "objective_probe": {
            "finite_loss_and_gradients": objective_finite,
            "loss_at_zero_logits": float(terms.loss.item()),
            "route_loss": float(terms.route_loss.item()),
            "actor_loss": float(terms.actor_loss.item()),
            "active_sample_component_count": int(
                terms.active_sample_component_count.item()
            ),
            "empty_sample_component_count": int(
                terms.empty_sample_component_count.item()
            ),
            "maximum_absolute_gradient": float(probe_logits.grad.abs().max().item()),
        },
        "support_mode_counts": {
            split: dict(sorted(values.items()))
            for split, values in support_mode_counts.items()
        },
        "per_split_component_view": aggregate,
        "identifiable_component_views": identifiable,
        "per_group": per_group,
        "locks": protocol["release_locks"],
        "claim_boundary": "CPU target/interface identifiability only. Current single-union-R residuals are diagnostic and no training, language, planning or safety claim is authorized.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--spatial-maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite factorized R preflight")
    value = audit(
        protocol_path=args.protocol,
        records_path=args.records,
        spatial_maps_path=args.spatial_maps,
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
                "supported_component_views": [
                    "%s/%s" % (row["component"], row["view"])
                    for row in value["identifiable_component_views"]["supported"]
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
