#!/usr/bin/env python3
"""Audit dense A1 field supervision and A2 calibration feasibility without training."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


SCHEMA = "orion.qwen-visibility-field-alignment-corpus-audit/v1"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    rounded_unique = int(len(np.unique(np.round(array, decimals=7))))
    quantiles = np.quantile(array, [0.0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 1.0])
    return {
        "count": int(len(array)),
        "finite": bool(np.isfinite(array).all()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "rounded_unique_values": rounded_unique,
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("q00", "q05", "q20", "q40", "q50", "q60", "q80", "q95", "q100"),
                quantiles,
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite field-alignment audit")
    if sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    failures: List[str] = []
    values: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        split: {
            family: {}
            for family in ("all", "global", "frontier")
        }
        for split in ("train", "validation", "held_out")
    }
    split_routes: Dict[str, set] = {key: set() for key in values}
    valid_counts = Counter()
    calibrated = 0
    feature_names = None
    for record in records:
        split = str(record["split"])
        split_routes[split].add(str(record["source_folder"]))
        token_path = Path(record["token_artifact"])
        if sha256(token_path) != record["token_sha256"]:
            failures.append("token hash mismatch: %s" % record["sample_id"])
            continue
        with np.load(token_path, allow_pickle=False) as artifact:
            names = tuple(str(x) for x in artifact["visibility_tokens_feature_names"].tolist())
            global_rows = np.asarray(artifact["visibility_tokens_global"], dtype=np.float64)
            frontier_rows = np.asarray(artifact["visibility_tokens_frontier"], dtype=np.float64)
            global_mask = np.asarray(artifact["visibility_tokens_global_mask"], dtype=bool)
            frontier_mask = np.asarray(artifact["visibility_tokens_frontier_mask"], dtype=bool)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            failures.append("feature order mismatch: %s" % record["sample_id"])
            continue
        families = {
            "global": global_rows[global_mask],
            "frontier": frontier_rows[frontier_mask],
        }
        families["all"] = np.concatenate([families["global"], families["frontier"]])
        for family, rows in families.items():
            valid_counts[(split, family)] += len(rows)
            if not np.isfinite(rows).all():
                failures.append("non-finite token: %s/%s" % (record["sample_id"], family))
            for column, name in enumerate(names):
                values[split][family].setdefault(name, []).extend(rows[:, column].tolist())

        annotation = args.data_root / record["source_folder"] / "anno" / ("%05d.json.gz" % int(record["source_frame"]))
        try:
            with gzip.open(annotation, "rt", encoding="utf-8") as handle:
                sensors = json.load(handle)["sensors"]
            ok = True
            for camera in CAMERAS:
                sensor = sensors[camera]
                intrinsic = np.asarray(sensor["intrinsic"], dtype=np.float64)
                cam2ego = np.asarray(sensor["cam2ego"], dtype=np.float64)
                ok = ok and intrinsic.shape == (3, 3) and cam2ego.shape == (4, 4)
                ok = ok and sensor["image_size_x"] == 1600 and sensor["image_size_y"] == 900
                ok = ok and np.isfinite(intrinsic).all() and np.isfinite(cam2ego).all()
            if ok:
                calibrated += 1
            else:
                failures.append("invalid camera calibration: %s" % record["sample_id"])
        except Exception as exc:
            failures.append("missing camera calibration: %s: %s" % (record["sample_id"], exc))

    overlap = {
        "train_validation": sorted(split_routes["train"] & split_routes["validation"]),
        "train_held_out": sorted(split_routes["train"] & split_routes["held_out"]),
        "validation_held_out": sorted(split_routes["validation"] & split_routes["held_out"]),
    }
    if any(overlap.values()):
        failures.append("route overlap exists across splits")
    stats: Dict[str, Any] = {}
    bucket_contract: Dict[str, Any] = {}
    for family in ("all", "global", "frontier"):
        stats[family] = {}
        for name in feature_names or ():
            stats[family][name] = {
                split: describe(values[split][family].get(name, []))
                for split in values
            }
            train = np.asarray(values["train"][family].get(name, []), dtype=np.float64)
            evaluation = np.asarray(
                values["validation"][family].get(name, [])
                + values["held_out"][family].get(name, []),
                dtype=np.float64,
            )
            raw_edges = np.quantile(train, [0.2, 0.4, 0.6, 0.8]) if len(train) else []
            edges = sorted({float(x) for x in raw_edges})
            bucket_contract.setdefault(family, {})[name] = {
                "training_only_edges": edges,
                "nondegenerate_five_bucket_target": len(edges) == 4,
                "evaluation_below_training_min": int(np.sum(evaluation < train.min())) if len(train) else None,
                "evaluation_above_training_max": int(np.sum(evaluation > train.max())) if len(train) else None,
            }
    result = {
        "schema": SCHEMA,
        "status": "passed" if not failures else "failed",
        "training_performed": False,
        "failures": failures,
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": args.expected_manifest_sha256,
            "data_root": str(args.data_root.resolve()),
        },
        "record_count": len(records),
        "record_counts_by_split": dict(sorted(Counter(row["split"] for row in records).items())),
        "unique_routes_by_split": {key: len(value) for key, value in split_routes.items()},
        "route_overlap": overlap,
        "feature_count": len(feature_names or ()),
        "feature_names": list(feature_names or ()),
        "valid_record_tokens_by_split_and_family": {
            "%s/%s" % key: count for key, count in sorted(valid_counts.items())
        },
        "camera_calibration": {
            "required_cameras": list(CAMERAS),
            "valid_frames": calibrated,
            "total_frames": len(records),
            "projection_targets_generated": False,
        },
        "statistics": stats,
        "training_only_bucket_contract": bucket_contract,
        "claim_boundary": {
            "corpus_preflight_only": True,
            "field_alignment_claim_allowed": False,
            "visual_alignment_claim_allowed": False,
            "planning_or_safety_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "failures", "record_count", "record_counts_by_split", "unique_routes_by_split", "valid_record_tokens_by_split_and_family", "camera_calibration")}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
