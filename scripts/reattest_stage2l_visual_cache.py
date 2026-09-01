#!/usr/bin/env python3
"""Re-attest an immutable ORION visual cache against a rebuilt v5 QA event.

The cache tensor is reused only when the old and new event factories identify
the same event, counterfactual groups, saved frames, camera bytes and frame
metadata.  The original cache payload must also prove that it was produced by
the frozen ORION checkpoint without UQ, task targets, answers, trajectory or
privileged safety inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
BATCH_SCHEMA = "orion.uq_relevance_frame_bundle_batch.v1"
BUNDLE_SCHEMA = "orion.uq_relevance_frame_bundle.v1"
ATTESTATION_SCHEMA = "orion.stage2l_visual_cache_reuse_attestation.v1"
FRESH_STATUS = "immutable_multiframe_visual_context_cache"
REATTESTED_STATUS = "immutable_multiframe_visual_context_cache_reattested"
PROHIBITED_FLAGS = (
    "privileged_safety_inputs_used",
    "stage1_uq_inputs_used",
    "task_relevance_targets_used",
    "qa_answers_used",
)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", reference.get("output", ""))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def _observed_bundle(batch_path: Path) -> Tuple[Path, Dict[str, Any]]:
    batch = _load(batch_path)
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported frame-bundle batch schema")
    rows = [row for row in batch.get("bundles", []) if row.get("variant") == "observed"]
    if len(rows) != 1:
        raise ValueError("frame-bundle batch must identify one observed bundle")
    path = _resolve(rows[0], batch_path.parent, "observed frame bundle")
    bundle = _load(path)
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("counterfactual", {}).get("variant") != "observed"
    ):
        raise ValueError("observed frame-bundle contract mismatch")
    return path, bundle


def _bundle_signature(bundle_path: Path, bundle: Mapping[str, Any]) -> Dict[str, Any]:
    group_id = str(bundle.get("counterfactual", {}).get("group_id", ""))
    frame = int(bundle.get("provenance", {}).get("selected_saved_frame_index", -1))
    cameras = bundle.get("model_input", {}).get("observation", {}).get("camera_files")
    if not group_id or frame < 0 or not isinstance(cameras, list) or len(cameras) != 6:
        raise ValueError("observed bundle lacks the six-view frame identity")
    camera_rows = []
    front_image = None
    for row in cameras:
        image = _resolve(row, bundle_path.parent, "observed camera image")
        if str(row.get("view", "")) == "CAM_FRONT":
            front_image = image
        camera_rows.append(
            {
                "view": str(row.get("view", "")),
                "sha256": sha256_file(image),
            }
        )
    if len({row["view"] for row in camera_rows}) != 6:
        raise ValueError("observed bundle camera views are absent or duplicated")
    if front_image is None or int(front_image.stem) != frame:
        raise ValueError("observed CAM_FRONT is not aligned to the selected frame")
    frame_meta = front_image.parent.parent / "meta" / ("%04d.json" % frame)
    if not frame_meta.is_file():
        raise ValueError("observed frame metadata is absent")
    return {
        "group_id": group_id,
        "saved_frame": frame,
        "frame_meta_sha256": sha256_file(frame_meta),
        "camera_sha256_by_view": {
            row["view"]: row["sha256"] for row in sorted(camera_rows, key=lambda row: row["view"])
        },
    }


def _factory_signatures(report_path: Path) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    report = _load(report_path)
    if report.get("schema") != FACTORY_SCHEMA:
        raise ValueError("unsupported event-factory report schema")
    event_id = str(report.get("event_id", ""))
    rows = report.get("frame_reports")
    if not event_id or not isinstance(rows, list) or not 3 <= len(rows) <= 5:
        raise ValueError("event factory must contain three to five keyframes")
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        batch = _resolve(
            row.get("frame_bundle_batch", {}),
            report_path.parent,
            "frame-bundle batch",
        )
        bundle_path, bundle = _observed_bundle(batch)
        signature = _bundle_signature(bundle_path, bundle)
        if signature["saved_frame"] != int(row["selected_saved_frame_index"]):
            raise ValueError("event factory and observed bundle disagree on frame")
        group_id = signature["group_id"]
        if group_id in result:
            raise ValueError("event factory duplicates an observed group")
        result[group_id] = signature
    return event_id, result


def _validate_prohibited_flags(value: Mapping[str, Any], name: str) -> None:
    if any(value.get(key) is not False for key in PROHIBITED_FLAGS):
        raise ValueError("%s used a prohibited input" % name)


def reattest_cache(
    *,
    source_manifest_path: Path,
    target_factory_report_path: Path,
    expected_orion_checkpoint_sha256: str,
    output_manifest_path: Path,
    output_attestation_path: Path,
) -> Dict[str, Any]:
    for path in (output_manifest_path, output_attestation_path):
        if path.exists():
            raise FileExistsError("refusing to overwrite cache re-attestation output")
    source_manifest_path = source_manifest_path.resolve()
    target_factory_report_path = target_factory_report_path.resolve()
    source = _load(source_manifest_path)
    if source.get("schema") != CACHE_SCHEMA or source.get("status") != FRESH_STATUS:
        raise ValueError("source is not an immutable original visual cache")
    _validate_prohibited_flags(source, "source cache manifest")
    source_report_path = _resolve(
        source.get("event_factory_report", {}),
        source_manifest_path.parent,
        "source event-factory report",
    )
    cache_path = _resolve(source, source_manifest_path.parent, "source visual cache")
    source_event, source_signatures = _factory_signatures(source_report_path)
    target_event, target_signatures = _factory_signatures(target_factory_report_path)
    if source_event != target_event:
        raise ValueError("source and target event factories identify different events")
    if source_signatures != target_signatures:
        raise ValueError("source and target event factories use different observation bytes")
    if set(source.get("group_ids", [])) != set(source_signatures):
        raise ValueError("source cache manifest group coverage is inconsistent")

    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CACHE_SCHEMA:
        raise ValueError("source cache payload schema is invalid")
    contexts = payload.get("contexts")
    metadata = payload.get("metadata")
    if not isinstance(contexts, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("source cache payload lacks contexts or metadata")
    if set(contexts) != set(source_signatures):
        raise ValueError("source cache payload group coverage is inconsistent")
    for group_id, tensor in contexts.items():
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, 529, 4096):
            raise ValueError("visual context has an unexpected tensor shape: %s" % group_id)
    _validate_prohibited_flags(metadata, "source cache payload")
    if (
        metadata.get("head_memory_reset_per_keyframe") is not True
        or metadata.get("llm_run_during_cache") is not False
        or metadata.get("trajectory_decoder_run_during_cache") is not False
    ):
        raise ValueError("source cache payload violates inference-isolation constraints")
    payload_report = metadata.get("event_factory_report", {})
    if payload_report.get("sha256") != sha256_file(source_report_path):
        raise ValueError("source cache payload does not bind its source event factory")
    checkpoint = metadata.get("orion_checkpoint", {})
    if checkpoint.get("sha256") != expected_orion_checkpoint_sha256:
        raise ValueError("source cache used a different ORION checkpoint")

    payload_frames = metadata.get("frames")
    if not isinstance(payload_frames, list) or len(payload_frames) != len(source_signatures):
        raise ValueError("source cache payload frame inventory is incomplete")
    payload_frame_hashes = {}
    for row in payload_frames:
        group_id = str(row.get("group_id", ""))
        bundle_path = _resolve(
            row.get("frame_bundle", {}), cache_path.parent, "cached observed frame bundle"
        )
        signature = _bundle_signature(bundle_path, _load(bundle_path))
        meta_path = _resolve(row.get("frame_meta", {}), cache_path.parent, "cached frame metadata")
        if group_id != signature["group_id"] or signature != source_signatures.get(group_id):
            raise ValueError("source cache payload frame inventory changed")
        payload_frame_hashes[group_id] = sha256_file(meta_path)
        if payload_frame_hashes[group_id] != source_signatures[group_id]["frame_meta_sha256"]:
            raise ValueError("source cache payload frame metadata changed")
    if set(payload_frame_hashes) != set(source_signatures):
        raise ValueError("source cache payload metadata coverage is incomplete")

    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "status": "verified_observation_equivalent_cache_reuse",
        "event_id": target_event,
        "source_visual_cache_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
        },
        "source_event_factory_report": {
            "path": str(source_report_path),
            "sha256": sha256_file(source_report_path),
        },
        "target_event_factory_report": {
            "path": str(target_factory_report_path),
            "sha256": sha256_file(target_factory_report_path),
        },
        "visual_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "orion_checkpoint_sha256": expected_orion_checkpoint_sha256,
        "group_ids": sorted(source_signatures),
        "frame_signatures": [
            source_signatures[group_id] for group_id in sorted(source_signatures)
        ],
        "checks": {
            "same_event_id": True,
            "same_group_ids": True,
            "same_saved_frames": True,
            "same_six_view_camera_bytes": True,
            "source_cache_hash_verified": True,
            "source_payload_report_binding_verified": True,
            "frozen_orion_checkpoint_verified": True,
            "head_memory_reset_per_keyframe": True,
            "no_privileged_safety_inputs": True,
            "no_stage1_uq_inputs": True,
            "no_task_relevance_targets": True,
            "no_qa_answers": True,
            "llm_not_run": True,
            "trajectory_decoder_not_run": True,
        },
        "claim_boundary": "Cache observation-equivalence and inference-isolation only; no QA, model, planning or safety result.",
    }
    output_attestation_path.parent.mkdir(parents=True, exist_ok=True)
    output_attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reattested = {
        "schema": CACHE_SCHEMA,
        "status": REATTESTED_STATUS,
        "output": str(cache_path),
        "sha256": sha256_file(cache_path),
        "keyframe_count": len(source_signatures),
        "group_ids": sorted(source_signatures),
        "event_factory_report": attestation["target_event_factory_report"],
        "source_event_factory_report": attestation["source_event_factory_report"],
        "source_visual_cache_manifest": attestation["source_visual_cache_manifest"],
        "reuse_attestation": {
            "path": str(output_attestation_path.resolve()),
            "sha256": sha256_file(output_attestation_path.resolve()),
        },
        "orion_checkpoint_sha256": expected_orion_checkpoint_sha256,
        "head_memory_reset_per_keyframe": True,
        "privileged_safety_inputs_used": False,
        "stage1_uq_inputs_used": False,
        "task_relevance_targets_used": False,
        "qa_answers_used": False,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(reattested, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return reattested


def validate_cache_manifest_for_factory(
    *,
    cache_manifest_path: Path,
    factory_report_path: Path,
    expected_orion_checkpoint_sha256: str,
) -> Dict[str, Any]:
    cache_manifest_path = cache_manifest_path.resolve()
    factory_report_path = factory_report_path.resolve()
    manifest = _load(cache_manifest_path)
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported visual-context cache manifest")
    _validate_prohibited_flags(manifest, "visual cache manifest")
    cache_path = _resolve(manifest, cache_manifest_path.parent, "visual-context cache")
    bound_report = _resolve(
        manifest.get("event_factory_report", {}),
        cache_manifest_path.parent,
        "visual-cache factory report",
    )
    if sha256_file(bound_report) != sha256_file(factory_report_path):
        raise ValueError("visual cache is not bound to the current QA factory report")
    report = _load(factory_report_path)
    if report.get("schema") != FACTORY_SCHEMA or not report.get("event_id"):
        raise ValueError("unsupported current event-factory report")
    event_id = str(report["event_id"])
    group_ids = set(manifest.get("group_ids", []))
    if not group_ids:
        raise ValueError("visual cache has no group coverage")
    status = manifest.get("status")
    result = {
        "event_id": event_id,
        "status": status,
        "visual_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "factory_report_sha256": sha256_file(factory_report_path),
        "group_count": len(group_ids),
        "reuse_attested": False,
    }
    if status == FRESH_STATUS:
        if manifest.get("orion_checkpoint_sha256") != expected_orion_checkpoint_sha256:
            raise ValueError("fresh visual cache lacks the frozen ORION checkpoint binding")
        return result
    if status != REATTESTED_STATUS:
        raise ValueError("visual cache has an unsupported lineage status")
    attestation_path = _resolve(
        manifest.get("reuse_attestation", {}),
        cache_manifest_path.parent,
        "visual-cache reuse attestation",
    )
    attestation = _load(attestation_path)
    if (
        attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("status") != "verified_observation_equivalent_cache_reuse"
        or attestation.get("event_id") != event_id
        or attestation.get("target_event_factory_report", {}).get("sha256")
        != sha256_file(factory_report_path)
        or attestation.get("visual_cache", {}).get("sha256") != sha256_file(cache_path)
        or attestation.get("orion_checkpoint_sha256")
        != expected_orion_checkpoint_sha256
        or set(attestation.get("group_ids", [])) != group_ids
        or not attestation.get("checks")
        or any(value is not True for value in attestation["checks"].values())
    ):
        raise ValueError("visual-cache reuse attestation is invalid")
    source_manifest_path = _resolve(
        attestation.get("source_visual_cache_manifest", {}),
        attestation_path.parent,
        "source visual-cache manifest",
    )
    if manifest.get("source_visual_cache_manifest", {}).get("sha256") != sha256_file(
        source_manifest_path
    ):
        raise ValueError("re-attested cache does not bind its source manifest")
    result["reuse_attested"] = True
    result["reuse_attestation"] = {
        "path": str(attestation_path),
        "sha256": sha256_file(attestation_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--target-factory-report", type=Path, required=True)
    parser.add_argument("--expected-orion-checkpoint-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-attestation", type=Path, required=True)
    args = parser.parse_args()
    result = reattest_cache(
        source_manifest_path=args.source_cache_manifest,
        target_factory_report_path=args.target_factory_report,
        expected_orion_checkpoint_sha256=args.expected_orion_checkpoint_sha256,
        output_manifest_path=args.output_manifest,
        output_attestation_path=args.output_attestation,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
