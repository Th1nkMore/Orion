#!/usr/bin/env python3
"""Audit observed Stage1-U maps for task-agnostic tokenizer pretraining.

The audit intentionally extracts only frozen observed U-component tensors.
Counterfactual U variants, relevance maps, route context, QA text, task fields,
TTC, collision outcomes and corruption metadata are never training inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np


SCHEMA = "orion.stage1_u_tokenizer_source_audit.v1"
COMPONENT_NAMES = (
    "persistent_direction",
    "persistent_magnitude",
    "transient_inconsistency",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _require_file(reference: Mapping[str, Any], label: str) -> Path:
    path = Path(str(reference.get("path", "")))
    expected = str(reference.get("sha256", ""))
    if not path.is_file() or len(expected) != 64:
        raise ValueError("%s path/hash is invalid" % label)
    if sha256_file(path) != expected:
        raise ValueError("%s hash mismatch" % label)
    return path


def audit_sources(dataset_manifest: Path) -> Dict[str, Any]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    events = manifest.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("dataset manifest has no events")

    all_maps: Dict[str, Dict[str, Any]] = {}
    group_paths: Dict[str, str] = {}
    checkpoint_hashes = set()
    event_rows = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        split = str(event.get("split", ""))
        if not event_id or split not in ("train", "dev"):
            raise ValueError("event identity/split is invalid")
        records_path = _require_file(event.get("source_records", {}), "source records")
        observed_groups = set()
        observed_paths = set()
        record_count = 0
        for row in _read_jsonl(records_path):
            record_count += 1
            counterfactual = row.get("counterfactual", {})
            if counterfactual.get("variant") != "observed":
                continue
            group_id = str(counterfactual.get("group_id", ""))
            uq = row.get("model_input", {}).get("stage1_observation_uq", {})
            if not group_id or uq.get("source") != "frozen_stage1_observation_adapter":
                raise ValueError("observed record lacks frozen Stage1 U provenance")
            if uq.get("control_influence") is not False:
                raise ValueError("Stage1 U had control influence")
            if tuple(uq.get("component_names", ())) != COMPONENT_NAMES:
                raise ValueError("Stage1 U component names differ")
            if uq.get("component_key") != "uncertainty_components":
                raise ValueError("Stage1 U component key differs")
            if list(uq.get("component_shape", ())) != [4, 6, 40, 40, 3]:
                raise ValueError("Stage1 U component shape differs")
            checkpoint_sha256 = str(uq.get("checkpoint_sha256", ""))
            if len(checkpoint_sha256) != 64:
                raise ValueError("Stage1 checkpoint hash is invalid")
            checkpoint_hashes.add(checkpoint_sha256)

            map_path = _require_file(uq, "observed Stage1 U map")
            path_text = str(map_path)
            if group_id in group_paths and group_paths[group_id] != path_text:
                raise ValueError("one observed group references multiple U maps")
            group_paths[group_id] = path_text
            observed_groups.add(group_id)
            observed_paths.add(path_text)
            if path_text in all_maps:
                continue
            with np.load(map_path, allow_pickle=False) as payload:
                if "uncertainty_components" not in payload.files:
                    raise ValueError("observed Stage1 U NPZ lacks components")
                components = np.asarray(payload["uncertainty_components"])
            if components.shape != (4, 6, 40, 40, 3):
                raise ValueError("observed Stage1 U tensor shape differs")
            if not np.isfinite(components).all():
                raise ValueError("observed Stage1 U tensor is non-finite")
            if float(components.min()) < 0.0 or float(components.max()) > 1.0:
                raise ValueError("observed Stage1 U tensor is outside [0,1]")
            all_maps[path_text] = {
                "path": path_text,
                "sha256": sha256_file(map_path),
                "event_id": event_id,
                "split": split,
                "shape": list(components.shape),
                "minimum": float(components.min()),
                "maximum": float(components.max()),
                "mean": float(components.mean()),
            }
        expected_groups = int(event.get("keyframe_count", -1))
        if len(observed_groups) != expected_groups or len(observed_paths) != expected_groups:
            raise ValueError("observed U coverage does not match event keyframes")
        event_rows.append({
            "event_id": event_id,
            "split": split,
            "source_record_count": record_count,
            "unique_observed_group_count": len(observed_groups),
            "unique_observed_u_map_count": len(observed_paths),
        })

    if len(checkpoint_hashes) != 1:
        raise ValueError("observed maps do not share one Stage1 checkpoint")
    train_maps = sum(row["split"] == "train" for row in all_maps.values())
    dev_maps = sum(row["split"] == "dev" for row in all_maps.values())
    checks = {
        "one_frozen_stage1_checkpoint": True,
        "only_observed_stage1_u_selected": True,
        "counterfactual_u_variants_excluded": True,
        "no_route_context_consumed": True,
        "no_task_relevance_consumed": True,
        "no_qa_text_or_fields_consumed": True,
        "no_ttc_collision_or_control_consumed": True,
        "no_corruption_metadata_consumed": True,
        "train_and_dev_maps_present": train_maps > 0 and dev_maps > 0,
    }
    return {
        "schema": SCHEMA,
        "status": "task_agnostic_stage1_u_tokenizer_sources_ready",
        "dataset_manifest": {
            "path": str(dataset_manifest),
            "sha256": sha256_file(dataset_manifest),
        },
        "stage1_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "event_count": len(event_rows),
        "unique_observed_u_map_count": len(all_maps),
        "train_map_count": train_maps,
        "dev_map_count": dev_maps,
        "events": event_rows,
        "maps": sorted(all_maps.values(), key=lambda row: (row["split"], row["event_id"], row["path"])),
        "checks": checks,
        "forbidden_inputs_consumed": {
            "route_context": False,
            "task_relevance": False,
            "qa_text_or_fields": False,
            "ttc_collision_or_control": False,
            "corruption_metadata": False,
        },
        "passed": all(checks.values()),
        "training_input_contract": "Only normalized frozen Stage1 uncertainty_components tensors are permitted.",
        "claim_boundary": "This source audit does not train or validate a tokenizer and makes no task-relevance, QA, trajectory, closed-loop or safety claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite %s" % args.output)
    report = audit_sources(args.dataset_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "event_count", "unique_observed_u_map_count",
        "train_map_count", "dev_map_count", "passed",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
