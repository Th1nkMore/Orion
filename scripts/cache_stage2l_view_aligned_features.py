#!/usr/bin/env python3
"""Cache frozen, explicitly indexed six-view ORION feature grids for Stage2-L.

The cache is built from the exact 80 observed keyframes in the frozen
17-event diagnostic dataset.  It runs only ORION's frozen image preprocessing
and backbone, pools the canonical [6,1024,40,40] feature tensor to the 10x10
R-map resolution, and never reads Stage-1 U, relevance targets, QA answers,
trajectory labels, collision outcomes, or control signals.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from mmcv.parallel.collate import collate as mm_collate_to_batch_form

from scripts.cache_closedloop_orion_visual_context import (
    CAMERA_ORDER,
    _build_pipeline_record,
    _load_json,
    _make_offline_orion_agent,
    _move_agent_batch_to_cuda,
    _setup_offline_orion_agent,
)
from scripts.cache_stage2l_multiframe_visual_contexts import _frame_meta
from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_view_aligned_feature_cache.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v101_view_aligned_cache_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v101_view_aligned_cache_preflight.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
DATASET_SCHEMA = "orion.stage2l_expanded_coverage_dataset.v1"
RECORD_SCHEMA = "orion.uq_relevance_qa_record.v5"
BUNDLE_SCHEMA = "orion.uq_relevance_frame_bundle.v1"
EXPECTED_EVENTS = 17
EXPECTED_GROUPS = 80
EXPECTED_RECORDS = 1600
SOURCE_SHAPE = (1, 6, 1024, 40, 40)
POOLED_SHAPE = (6, 10, 10, 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "cache_builder_sha256": _sha256(Path(__file__).resolve()),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "orion_config_sha256": _sha256(args.orion_config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.orion_checkpoint.resolve()),
    }


def _validate_protocol(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "frozen_view_aligned_cache_protocol_launch_locked"
        or protocol.get("validated_inputs") != _validated_inputs(args)
        or protocol.get("output") != str(args.output.resolve())
        or protocol.get("event_count") != EXPECTED_EVENTS
        or protocol.get("group_count") != EXPECTED_GROUPS
        or protocol.get("camera_order") != list(CAMERA_ORDER)
        or protocol.get("source_feature_shape") != list(SOURCE_SHAPE)
        or protocol.get("pooled_context_shape") != list(POOLED_SHAPE)
    ):
        raise ValueError("view-aligned cache protocol is absent or stale")
    locks = protocol.get("locks", {})
    if any(
        locks.get(key) is not False
        for key in (
            "stage1_uq_input",
            "task_relevance_target_input",
            "qa_answer_input",
            "trajectory_or_control_input",
            "training",
            "phase_b",
            "phase_c",
            "formal_stage2l",
            "stage2p",
            "closed_loop",
            "route203_native_glare_submission",
        )
    ):
        raise ValueError("view-aligned cache protocol expands a locked scope")


def _preflight(args: argparse.Namespace, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(args.dataset_manifest.resolve())
    records_path = _resolve_records(manifest, args.dataset_manifest.resolve())
    groups = _selected_groups(_records(records_path))
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "view_aligned_cache_preflight_pass_launch_locked",
        "passed": True,
        "gpu_used": False,
        "training_started": False,
        "event_count": len({row["event_id"] for row in groups.values()}),
        "group_count": len(groups),
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.protocol.resolve()),
        "output": str(args.output.resolve()),
        "locks": dict(protocol["locks"]),
    }


def _validate_launch(args: argparse.Namespace) -> None:
    preflight = _load_json(args.preflight.resolve())
    amendment = _load_json(args.launch_amendment.resolve())
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("event_count") != EXPECTED_EVENTS
        or preflight.get("group_count") != EXPECTED_GROUPS
        or preflight.get("validated_inputs") != _validated_inputs(args)
        or preflight.get("protocol_sha256") != _sha256(args.protocol.resolve())
        or preflight.get("output") != str(args.output.resolve())
    ):
        raise ValueError("view-aligned cache preflight is absent or stale")
    authorized = amendment.get("authorized_run", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("status") != "immutable_cache_only_authorization"
        or amendment.get("validated_inputs") != _validated_inputs(args)
        or amendment.get("protocol_sha256") != _sha256(args.protocol.resolve())
        or amendment.get("preflight_sha256") != _sha256(args.preflight.resolve())
        or authorized.get("output") != str(args.output.resolve())
        or authorized.get("maximum_submissions") != 1
        or authorized.get("automatic_retry") is not False
        or authorized.get("training") is not False
    ):
        raise ValueError("view-aligned cache launch amendment is absent or stale")


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_records(manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    reference = manifest.get("records", {})
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("dataset records are absent or stale")
    return path


def _selected_groups(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError("view-aligned cache requires exactly 1600 records")
    selected: dict[str, dict[str, Any]] = {}
    event_ids = set()
    for row in rows:
        if row.get("schema") != RECORD_SCHEMA:
            raise ValueError("view-aligned cache encountered a non-V5 record")
        event_ids.add(str(row["event_id"]))
        if not (
            row.get("question_family") == "task_relevance"
            and row.get("counterfactual", {}).get("variant") == "observed"
        ):
            continue
        group_id = str(row["counterfactual"]["group_id"])
        if group_id in selected:
            raise ValueError("duplicate observed task-relevance group")
        cameras = row["model_input"]["observation"]["camera_files"]
        if tuple(value.get("view") for value in cameras) != CAMERA_ORDER:
            raise ValueError("record camera order is not canonical")
        bundle_path = Path(str(row["provenance"]["frame_bundle_path"]))
        if (
            not bundle_path.is_file()
            or sha256_file(bundle_path)
            != row["provenance"]["frame_bundle_sha256"]
        ):
            raise ValueError("observed frame bundle is absent or stale")
        bundle = _load_json(bundle_path)
        if (
            bundle.get("schema") != BUNDLE_SCHEMA
            or bundle.get("counterfactual", {}).get("variant") != "observed"
            or bundle.get("counterfactual", {}).get("group_id") != group_id
        ):
            raise ValueError("observed frame bundle identity differs")
        selected[group_id] = {
            "event_id": str(row["event_id"]),
            "split": str(row["split"]),
            "frame_id": str(row["frame_id"]),
            "bundle_path": bundle_path,
            "bundle": bundle,
            "meta_path": _frame_meta(bundle),
            "observation_sha256": str(
                row["model_input"]["observation"]["observation_sha256"]
            ),
        }
    if len(event_ids) != EXPECTED_EVENTS or len(selected) != EXPECTED_GROUPS:
        raise ValueError("view-aligned cache event/group coverage differs")
    return selected


def cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite view-aligned feature cache")
    manifest = _load_json(args.dataset_manifest.resolve())
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("event_count") != EXPECTED_EVENTS
        or manifest.get("record_count") != EXPECTED_RECORDS
        or manifest.get("formal_stage2l_training_allowed") is not False
        or manifest.get("stage2p_allowed") is not False
    ):
        raise ValueError("dataset scope or locks differ")
    records_path = _resolve_records(manifest, args.dataset_manifest.resolve())
    groups = _selected_groups(_records(records_path))

    agent = _make_offline_orion_agent()
    _setup_offline_orion_agent(
        agent,
        "%s+%s+stage2l_view_aligned_feature_cache"
        % (args.orion_config.resolve(), args.orion_checkpoint.resolve()),
    )
    model = agent.model
    model.eval()
    lm_head = model.lm_head
    model.lm_head = None
    contexts = {}
    metadata = []
    try:
        for index, (group_id, row) in enumerate(sorted(groups.items()), start=1):
            processed = _build_pipeline_record(
                agent, row["bundle"], _load_json(row["meta_path"])
            )
            batch = mm_collate_to_batch_form([processed], samples_per_gpu=1)
            _move_agent_batch_to_cuda(batch)
            with torch.inference_mode():
                feature_grid = model.extract_feat(batch["img"][0])
            if tuple(feature_grid.shape) != SOURCE_SHAPE:
                raise RuntimeError(
                    "unexpected frozen ORION view feature shape: %s"
                    % (tuple(feature_grid.shape),)
                )
            pooled = F.adaptive_avg_pool2d(feature_grid[0].float(), (10, 10))
            pooled = pooled.permute(0, 2, 3, 1).contiguous()
            if tuple(pooled.shape) != POOLED_SHAPE or not bool(
                torch.isfinite(pooled).all()
            ):
                raise RuntimeError("pooled view-aligned feature tensor is malformed")
            contexts[group_id] = pooled.to(dtype=torch.float16, device="cpu")
            metadata.append(
                {
                    "group_id": group_id,
                    "event_id": row["event_id"],
                    "split": row["split"],
                    "frame_id": row["frame_id"],
                    "observation_sha256": row["observation_sha256"],
                    "frame_bundle": {
                        "path": str(row["bundle_path"]),
                        "sha256": sha256_file(row["bundle_path"]),
                    },
                    "frame_meta": {
                        "path": str(row["meta_path"]),
                        "sha256": sha256_file(row["meta_path"]),
                    },
                }
            )
            print(
                "[ViewAlignedCache] %d/%d group=%s"
                % (index, len(groups), group_id),
                flush=True,
            )
    finally:
        model.lm_head = lm_head
        del model, agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema": SCHEMA,
        "contexts": contexts,
        "metadata": {
            "dataset_manifest": {
                "path": str(args.dataset_manifest.resolve()),
                "sha256": sha256_file(args.dataset_manifest.resolve()),
            },
            "records": {
                "path": str(records_path),
                "sha256": sha256_file(records_path),
            },
            "orion_config": {
                "path": str(args.orion_config.resolve()),
                "sha256": sha256_file(args.orion_config.resolve()),
            },
            "orion_checkpoint": {
                "path": str(args.orion_checkpoint.resolve()),
                "sha256": sha256_file(args.orion_checkpoint.resolve()),
            },
            "camera_order": list(CAMERA_ORDER),
            "source_feature_shape": list(SOURCE_SHAPE),
            "pooled_context_shape": list(POOLED_SHAPE),
            "pooling": "adaptive_average_40x40_to_10x10",
            "event_count": EXPECTED_EVENTS,
            "group_count": EXPECTED_GROUPS,
            "frames": metadata,
            "orion_image_backbone_frozen": True,
            "privileged_safety_inputs_used": False,
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
            "trajectory_or_control_inputs_used": False,
            "llm_run_during_cache": False,
            "det_map_heads_run_during_cache": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "schema": SCHEMA,
        "status": "immutable_frozen_view_aligned_feature_cache",
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "event_count": EXPECTED_EVENTS,
        "group_count": EXPECTED_GROUPS,
        "camera_order": list(CAMERA_ORDER),
        "source_feature_shape": list(SOURCE_SHAPE),
        "pooled_context_shape": list(POOLED_SHAPE),
        "orion_image_backbone_frozen": True,
        "privileged_safety_inputs_used": False,
        "stage1_uq_inputs_used": False,
        "task_relevance_targets_used": False,
        "qa_answers_used": False,
        "trajectory_or_control_inputs_used": False,
        "claim_boundary": (
            "Input cache for an engineering Phase-A repair only; not a trained "
            "model, Stage2-L result, planning result, closed-loop result, or "
            "safety claim."
        ),
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["manifest"] = str(manifest_path.resolve())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.dataset_manifest,
        args.orion_config,
        args.orion_checkpoint,
        args.protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("view-aligned cache prerequisite is missing")
    protocol = _load_json(args.protocol.resolve())
    _validate_protocol(args, protocol)
    if args.preflight_only:
        if args.preflight is not None or args.launch_amendment is not None:
            raise ValueError("preflight-only mode cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("preflight-only mode requires a fresh output")
        value = _preflight(args, protocol)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if (
        args.preflight_output is not None
        or args.preflight is None
        or args.launch_amendment is None
    ):
        raise ValueError("real cache run requires preflight and launch amendment")
    _validate_launch(args)
    if not torch.cuda.is_available():
        raise SystemExit("view-aligned feature caching requires CUDA")
    print(json.dumps(cache(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
