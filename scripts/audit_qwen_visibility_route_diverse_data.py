#!/usr/bin/env python3
"""Independently audit route-diverse random-row U data before Qwen training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


SCHEMA = "orion.qwen-visibility-route-diverse-data-audit/v1"
MANIFEST_SCHEMA = "orion.qwen-visibility-grounding-manifest/v1"
CURRICULUM_SCHEMA = "orion.qwen-visibility-route-diverse-curriculum/v1"
PREFIXES = {
    "true_u": "visibility_tokens",
    "zero_u": "visibility_tokens_zero_u",
    "spatial_shuffle": "visibility_tokens_spatial_shuffle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def route_label(frontier: np.ndarray, names, row_index: int) -> str:
    index = tuple(str(value) for value in names).index("route_weight_mean")
    return "ON_ROUTE" if float(frontier[row_index, index]) >= 0.2 else "OFF_ROUTE"


def audit(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    curriculum_path = root / "curriculum.json"
    build_report_path = root / "build_report.json"
    manifest = _read_json(manifest_path)
    curriculum = _read_json(curriculum_path)
    build_report = _read_json(build_report_path)
    failures = []

    if manifest.get("schema") != MANIFEST_SCHEMA:
        failures.append("manifest_schema")
    if curriculum.get("schema") != CURRICULUM_SCHEMA:
        failures.append("curriculum_schema")
    if build_report.get("status") != "complete":
        failures.append("build_not_complete")
    if build_report.get("manifest", {}).get("sha256") != _sha256(manifest_path):
        failures.append("build_manifest_hash")
    if build_report.get("curriculum", {}).get("sha256") != _sha256(curriculum_path):
        failures.append("build_curriculum_hash")
    if curriculum.get("base_manifest_sha256") != _sha256(manifest_path):
        failures.append("curriculum_manifest_hash")
    if Path(curriculum.get("base_manifest_path", "")).resolve() != manifest_path:
        failures.append("curriculum_manifest_path")

    required_manifest = {
        "reportable_generalization": True,
        "bounded_route_disjoint_grounding_baseline_only": True,
        "oracle_depth": False,
        "hidden_actor_labels_used": False,
        "controls_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
    }
    for key, value in required_manifest.items():
        if manifest.get(key) is not value:
            failures.append("manifest_boundary:%s" % key)
    required_curriculum = {
        "reportable_generalization": True,
        "bounded_route_disjoint_grounding_baseline_only": True,
        "oracle_depth": False,
        "ordinary_per_example_supervision_only": True,
        "same_frame_answer_equality_penalty": False,
        "pair_flip_loss": False,
        "controls_used_for_optimizer": False,
        "spatial_shuffle_examples_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "natural_row_order_only": True,
        "one_query_per_route": True,
        "route_disjoint_splits": True,
    }
    for key, value in required_curriculum.items():
        if curriculum.get(key) is not value:
            failures.append("curriculum_boundary:%s" % key)

    records = manifest.get("records", [])
    by_id = {row.get("sample_id"): row for row in records}
    if None in by_id or len(by_id) != len(records):
        failures.append("duplicate_sample_id")
    route_keys = [str(row.get("source_folder")) for row in records]
    if len(set(route_keys)) != len(route_keys):
        failures.append("duplicate_physical_route")
    query_keys = [
        (row.get("source_folder"), row.get("source_frame"), row.get("query_frontier"))
        for row in records
    ]
    if len(set(query_keys)) != len(query_keys):
        failures.append("duplicate_route_frame_Fxx")

    split_counts = {}
    valid_counts = []
    control_semantics = Counter()
    for split in ("train", "validation", "held_out"):
        selected = [row for row in records if row.get("split") == split]
        labels = Counter(row.get("target", {}).get("route") for row in selected)
        split_counts[split] = {
            "routes": len(selected),
            **dict(sorted(labels.items())),
        }
    if split_counts != {
        "train": {"routes": 70, "ON_ROUTE": 35, "OFF_ROUTE": 35},
        "validation": {"routes": 10, "ON_ROUTE": 5, "OFF_ROUTE": 5},
        "held_out": {"routes": 10, "ON_ROUTE": 5, "OFF_ROUTE": 5},
    }:
        failures.append("split_or_label_counts")

    for sample_id, record in by_id.items():
        token_path = Path(str(record.get("token_artifact", "")))
        if not token_path.is_file() or _sha256(token_path) != record.get("token_sha256"):
            failures.append("token_hash:%s" % sample_id)
            continue
        images = [Path(str(value)) for value in record.get("camera_images", [])]
        digests = list(record.get("camera_sha256", []))
        if len(images) != 3 or len(digests) != 3:
            failures.append("image_count:%s" % sample_id)
        elif any(not path.is_file() or _sha256(path) != digest for path, digest in zip(images, digests)):
            failures.append("image_hash:%s" % sample_id)
        try:
            with np.load(token_path, allow_pickle=False) as artifact:
                provenance = json.loads(str(artifact["provenance_json"]))
                controls = {}
                masks = {}
                names_by_control = {}
                for control, prefix in PREFIXES.items():
                    controls[control] = np.asarray(
                        artifact[prefix + "_frontier"], dtype=np.float32
                    )
                    masks[control] = np.asarray(
                        artifact[prefix + "_frontier_mask"], dtype=bool
                    )
                    names_by_control[control] = tuple(
                        str(value)
                        for value in artifact[prefix + "_feature_names"].tolist()
                    )
        except Exception as error:
            failures.append("token_read:%s:%s" % (sample_id, type(error).__name__))
            continue
        if provenance.get("source_oracle_depth") is not False:
            failures.append("oracle_provenance:%s" % sample_id)
        if provenance.get("source_visibility_supervision") != "bench2drive_offline_uint8_depth":
            failures.append("source_provenance:%s" % sample_id)
        if float(provenance.get("surface_tolerance_m", -1.0)) != 0.95:
            failures.append("surface_tolerance:%s" % sample_id)
        if provenance.get("future_executed_ego_motion_used") is not False:
            failures.append("future_motion:%s" % sample_id)
        if len(set(names_by_control.values())) != 1:
            failures.append("control_feature_names:%s" % sample_id)
            continue
        true_mask = masks["true_u"]
        valid_count = int(true_mask.sum())
        valid_counts.append(valid_count)
        if (
            true_mask.shape != (32,)
            or not np.all(true_mask[:valid_count])
            or np.any(true_mask[valid_count:])
            or any(not np.array_equal(mask, true_mask) for mask in masks.values())
        ):
            failures.append("frontier_mask:%s" % sample_id)
            continue
        row_index = int(str(record.get("query_frontier", "F99"))[1:])
        if not 0 <= row_index < valid_count:
            failures.append("query_not_valid:%s" % sample_id)
            continue
        if record.get("valid_frontier_rows") != valid_count:
            failures.append("valid_count_record:%s" % sample_id)
        if record.get("frontier_permutation_new_to_old") != list(range(valid_count)):
            failures.append("non_identity_manifest_order:%s" % sample_id)
        for control, frontier in controls.items():
            actual = route_label(frontier, names_by_control[control], row_index)
            expected = record.get("control_expected_answers", {}).get(control)
            if actual != expected:
                failures.append("control_label:%s:%s" % (sample_id, control))
            control_semantics[(control, actual)] += 1
        if record.get("target") != {
            "route": record.get("control_expected_answers", {}).get("true_u")
        }:
            failures.append("true_target:%s" % sample_id)

    examples = curriculum.get("examples", [])
    examples_by_id = {row.get("example_id"): row for row in examples}
    if None in examples_by_id or len(examples_by_id) != len(examples):
        failures.append("duplicate_example_id")
    if {row.get("sample_id") for row in examples} != set(by_id):
        failures.append("example_sample_set")
    split_id_fields = {
        "train": "training_example_ids",
        "validation": "validation_example_ids",
        "held_out": "held_out_example_ids",
    }
    for split, field in split_id_fields.items():
        expected_ids = {
            row["example_id"] for row in examples if row.get("split") == split
        }
        if set(curriculum.get(field, [])) != expected_ids:
            failures.append("curriculum_split:%s" % split)
    evaluation_ids = set(curriculum.get("evaluation_example_ids", []))
    if evaluation_ids != set(curriculum.get("validation_example_ids", [])) | set(
        curriculum.get("held_out_example_ids", [])
    ):
        failures.append("evaluation_split")
    for example in examples:
        record = by_id.get(example.get("sample_id"))
        if record is None:
            continue
        valid_count = int(record["valid_frontier_rows"])
        if example.get("sequence_permutation_new_to_manifest") != list(range(valid_count)):
            failures.append("non_identity_example_order:%s" % example.get("example_id"))
        if (
            example.get("query_frontier") != record.get("query_frontier")
            or example.get("expected_answer") != record.get("target", {}).get("route")
            or example.get("control_expected_answers") != record.get("control_expected_answers")
        ):
            failures.append("example_record_mismatch:%s" % example.get("example_id"))
    schedule = list(curriculum.get("training_schedule", []))
    training_ids = set(curriculum.get("training_example_ids", []))
    if len(schedule) != 240 or any(value not in training_ids for value in schedule):
        failures.append("optimizer_schedule_scope")
    schedule_counts = Counter(
        examples_by_id[value]["expected_answer"]
        for value in schedule
        if value in examples_by_id
    )
    if schedule_counts != Counter({"ON_ROUTE": 120, "OFF_ROUTE": 120}):
        failures.append("optimizer_schedule_balance")

    return {
        "schema": SCHEMA,
        "passed": not failures,
        "failures": failures,
        "inputs": {
            "build_report": _reference(build_report_path),
            "manifest": _reference(manifest_path),
            "curriculum": _reference(curriculum_path),
        },
        "summary": {
            "record_count": len(records),
            "split_counts": split_counts,
            "valid_frontier_rows_minimum": min(valid_counts) if valid_counts else 0,
            "valid_frontier_rows_maximum": max(valid_counts) if valid_counts else 0,
            "frames_with_all_32_rows": sum(value == 32 for value in valid_counts),
            "control_label_counts": {
                control: {
                    label: control_semantics[(control, label)]
                    for label in ("ON_ROUTE", "OFF_ROUTE")
                }
                for control in PREFIXES
            },
            "optimizer_label_counts": dict(sorted(schedule_counts.items())),
        },
        "claim_boundary": {
            "training_started": False,
            "qwen_loaded": False,
            "safety_claim_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite data audit: %s" % output)
    report = audit(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
