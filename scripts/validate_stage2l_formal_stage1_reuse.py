#!/usr/bin/env python3
"""Fail-closed validation for reusing frozen Stage1 event sequences.

This validator permits CPU-only QA reconstruction without rerunning the
adapter.  It verifies the clean source run, frozen route split, event package,
fixed keyframes, adapter checkpoint, and prohibited-input contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts.scenario_factory_lib import sha256_file, validate_clean_runtime_manifest
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file, validate_clean_runtime_manifest


PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
EVENT_SCHEMA = "orion.scenario_event_package.v1"
STAGE1_SCHEMA = "orion.stage1_observation_uq_multiframe.v1"
SEQUENCE_SCHEMA = "orion.stage1_observation_uq_sequence.v1"
FORBIDDEN_INPUTS = (
    "route",
    "actor_geometry",
    "ttc",
    "collision_outcome",
    "corruption_metadata",
)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def validate_reuse(
    *,
    formal_route_plan: Path,
    event_package: Path,
    stage1_multiframe_manifest: Path,
    expected_checkpoint_sha256: str,
) -> Dict[str, Any]:
    formal_route_plan = formal_route_plan.resolve()
    event_package = event_package.resolve()
    stage1_multiframe_manifest = stage1_multiframe_manifest.resolve()
    plan = _load(formal_route_plan)
    event = _load(event_package)
    stage1 = _load(stage1_multiframe_manifest)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported formal route plan")
    if event.get("schema") != EVENT_SCHEMA:
        raise ValueError("unsupported event package")
    if event.get("runtime", {}).get("valid") is not True or event.get("qa_input_ready") is not True:
        raise ValueError("event package is not runtime-valid and QA-ready")
    route_index = int(event.get("route", {}).get("route_index", -1))
    planned = [row for row in plan.get("events", []) if int(row["route_index"]) == route_index]
    if len(planned) != 1:
        raise ValueError("event route is absent or duplicated in the formal plan")
    critical = event.get("critical_event")
    if not isinstance(critical, Mapping):
        raise ValueError("event package lacks an actor-grounded critical event")
    event_id = "route%s_step%s" % (route_index, int(critical["step"]))
    if planned[0].get("event_id") is not None and str(planned[0]["event_id"]) != event_id:
        raise ValueError("event identity differs from the frozen formal plan")

    run_manifest_path = _resolve(
        event.get("source_files", {}).get("run_manifest", {}),
        event_package.parent,
        "clean source run manifest",
    )
    render_validation = validate_clean_runtime_manifest(_load(run_manifest_path))
    if render_validation.get("valid") is not True:
        raise ValueError("source run is not clean_off with a clean render condition")

    if (
        stage1.get("schema") != STAGE1_SCHEMA
        or stage1.get("status") != "offline_frozen_stage1_multiframe_output"
        or stage1.get("control_influence") is not False
        or stage1.get("event_package", {}).get("sha256") != sha256_file(event_package)
    ):
        raise ValueError("Stage1 multiframe manifest lineage is invalid")
    keyframe_ref = stage1.get("keyframe_manifest", {})
    keyframe_path = _resolve(
        keyframe_ref, stage1_multiframe_manifest.parent, "fixed keyframe manifest"
    )
    keyframes = _load(keyframe_path)
    if keyframes.get("event_id") != event_id:
        raise ValueError("Stage1 keyframes identify a different event")
    expected_frames = {
        int(row["selected_saved_frame_index"])
        for row in keyframes.get("keyframes", [])
    }
    sequences = stage1.get("sequences")
    if not isinstance(sequences, list) or not 3 <= len(sequences) <= 5:
        raise ValueError("Stage1 reuse requires three to five sequences")
    sequence_frames = {int(row["selected_saved_frame_index"]) for row in sequences}
    if len(sequence_frames) != len(sequences) or sequence_frames != expected_frames:
        raise ValueError("Stage1 sequences differ from the fixed keyframes")
    sequence_rows = []
    for row in sequences:
        sequence_path = _resolve(
            row.get("manifest", {}),
            stage1_multiframe_manifest.parent,
            "Stage1 keyframe sequence",
        )
        sequence = _load(sequence_path)
        if (
            sequence.get("schema") != SEQUENCE_SCHEMA
            or sequence.get("status") != "offline_frozen_stage1_output"
            or sequence.get("control_influence") is not False
            or sequence.get("event_package_sha256") != sha256_file(event_package)
            or sequence.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            raise ValueError("Stage1 sequence uses invalid lineage or checkpoint")
        forbidden = sequence.get("forbidden_inputs", {})
        if any(forbidden.get(key) is not False for key in FORBIDDEN_INPUTS):
            raise ValueError("Stage1 sequence used prohibited task inputs")
        sequence_rows.append(
            {
                "saved_frame": int(row["selected_saved_frame_index"]),
                "path": str(sequence_path),
                "sha256": sha256_file(sequence_path),
            }
        )
    return {
        "schema": "orion.stage2_l.formal_stage1_reuse_validation.v1",
        "eligible": True,
        "event_id": event_id,
        "route_index": route_index,
        "formal_split": str(planned[0]["formal_split"]),
        "keyframe_count": len(sequence_rows),
        "stage1_checkpoint_sha256": expected_checkpoint_sha256,
        "render_condition_attestation": render_validation["render_condition_attestation"],
        "sequences": sorted(sequence_rows, key=lambda row: row["saved_frame"]),
        "provenance": {
            "formal_route_plan": {"path": str(formal_route_plan), "sha256": sha256_file(formal_route_plan)},
            "event_package": {"path": str(event_package), "sha256": sha256_file(event_package)},
            "stage1_multiframe_manifest": {
                "path": str(stage1_multiframe_manifest),
                "sha256": sha256_file(stage1_multiframe_manifest),
            },
            "keyframe_manifest": {"path": str(keyframe_path), "sha256": sha256_file(keyframe_path)},
            "run_manifest": {"path": str(run_manifest_path), "sha256": sha256_file(run_manifest_path)},
        },
        "claim_boundary": "Stage1 reuse eligibility only; no QA, model, planning or safety result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-route-plan", type=Path, required=True)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--stage1-multiframe-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_reuse(
        formal_route_plan=args.formal_route_plan,
        event_package=args.event_package,
        stage1_multiframe_manifest=args.stage1_multiframe_manifest,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    if args.output:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite Stage1 reuse validation")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
