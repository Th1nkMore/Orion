#!/usr/bin/env python3
"""Cache ORION det/map visual tokens for one fixed-keyframe QA event.

The full ORION checkpoint is loaded once.  Each of the event's three to five
fixed temporal keyframes is evaluated independently with detection/map memory
reset, while the LLM and trajectory decoder remain disabled.  This produces
the immutable visual context needed by the later Stage2-L pilot without using
UQ values, task-relevance targets, answers, or closed-loop outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from mmcv.parallel.collate import collate as mm_collate_to_batch_form

from scripts.cache_closedloop_orion_visual_context import (
    DET_VISUAL_TOKENS,
    MAP_VISUAL_TOKENS,
    TOTAL_VISUAL_TOKENS,
    _build_pipeline_record,
    _load_json,
    _make_offline_orion_agent,
    _move_agent_batch_to_cuda,
    _setup_offline_orion_agent,
)
from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
BATCH_SCHEMA = "orion.uq_relevance_frame_bundle_batch.v1"
BUNDLE_SCHEMA = "orion.uq_relevance_frame_bundle.v1"


def _resolve_reference(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def _observed_bundle(batch_path: Path) -> Tuple[Path, Dict[str, Any]]:
    batch = _load_json(batch_path)
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported frame-bundle batch schema")
    matches = [row for row in batch.get("bundles", []) if row.get("variant") == "observed"]
    if len(matches) != 1:
        raise ValueError("frame-bundle batch must contain exactly one observed variant")
    path = _resolve_reference(matches[0], batch_path.parent, "observed frame bundle")
    bundle = _load_json(path)
    if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("counterfactual", {}).get("variant") != "observed":
        raise ValueError("observed frame-bundle contract mismatch")
    return path, bundle


def _frame_meta(bundle: Mapping[str, Any]) -> Path:
    cameras = bundle["model_input"]["observation"]["camera_files"]
    front = [row for row in cameras if row.get("view") == "CAM_FRONT"]
    if len(front) != 1:
        raise ValueError("frame bundle does not uniquely identify CAM_FRONT")
    image = Path(str(front[0]["path"]))
    frame_index = int(bundle["provenance"]["selected_saved_frame_index"])
    if int(image.stem) != frame_index:
        raise ValueError("CAM_FRONT filename is not aligned to selected keyframe")
    meta = image.parent.parent / "meta" / ("%04d.json" % frame_index)
    if not meta.is_file():
        raise FileNotFoundError("saved keyframe metadata is missing: %s" % meta)
    return meta


def _factory_inputs(factory_report: Path) -> Dict[str, Dict[str, Any]]:
    report = _load_json(factory_report)
    if report.get("schema") != FACTORY_SCHEMA:
        raise ValueError("unsupported multi-frame event-factory schema")
    if report.get("status") != "pending_multiframe_human_geometry_review":
        raise ValueError("multi-frame event factory is not at the expected review stage")
    count = int(report.get("keyframe_count", 0))
    if count < 3 or count > 5 or len(report.get("frame_reports", [])) != count:
        raise ValueError("multi-frame event factory must contain three to five frames")
    result = {}
    for row in report["frame_reports"]:
        frame = int(row["selected_saved_frame_index"])
        batch_path = _resolve_reference(
            row["frame_bundle_batch"], factory_report.parent, "frame-bundle batch"
        )
        bundle_path, bundle = _observed_bundle(batch_path)
        if int(bundle["provenance"]["selected_saved_frame_index"]) != frame:
            raise ValueError("multi-frame factory and observed bundle frame mismatch")
        group_id = str(bundle["counterfactual"]["group_id"])
        if group_id in result:
            raise ValueError("duplicate observed counterfactual group")
        result[group_id] = {
            "frame": frame,
            "bundle_path": bundle_path,
            "bundle": bundle,
            "meta_path": _frame_meta(bundle),
        }
    return result


def cache_multiframe_contexts(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite multi-frame visual-context cache")
    manifest_path = args.output.with_suffix(".json")
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite visual-context cache manifest")
    inputs = _factory_inputs(args.factory_report.resolve())

    agent = _make_offline_orion_agent()
    _setup_offline_orion_agent(agent, "%s+%s+stage2l_multiframe_visual_cache" % (
        args.orion_config.resolve(), args.orion_checkpoint.resolve()
    ))
    model = agent.model
    model.eval()
    captured: Dict[str, torch.Tensor] = {}

    def capture_det(_module, _inputs, output):
        captured["det"] = output[1].detach()

    def capture_map(_module, _inputs, output):
        captured["map"] = output[1].detach()

    hooks = [
        model.pts_bbox_head.register_forward_hook(capture_det),
        model.map_head.register_forward_hook(capture_map),
    ]
    lm_head = model.lm_head
    model.lm_head = None
    contexts = {}
    metadata = []
    try:
        for group_id, row in sorted(inputs.items(), key=lambda item: item[1]["frame"]):
            captured.clear()
            model.pts_bbox_head.reset_memory()
            model.map_head.reset_memory()
            frame_meta = _load_json(row["meta_path"])
            processed = _build_pipeline_record(agent, row["bundle"], frame_meta)
            batch = mm_collate_to_batch_form([processed], samples_per_gpu=1)
            _move_agent_batch_to_cuda(batch)
            with torch.inference_mode():
                model(batch, return_loss=False)
            if set(captured) != {"det", "map"}:
                raise RuntimeError("ORION det/map hooks did not both fire")
            if tuple(captured["det"].shape) != (1, DET_VISUAL_TOKENS, 4096):
                raise RuntimeError("unexpected ORION detection-token shape: %s" % (captured["det"].shape,))
            if tuple(captured["map"].shape) != (1, MAP_VISUAL_TOKENS, 4096):
                raise RuntimeError("unexpected ORION map-token shape: %s" % (captured["map"].shape,))
            visual = torch.cat((captured["det"], captured["map"]), dim=1)
            if tuple(visual.shape) != (1, TOTAL_VISUAL_TOKENS, 4096):
                raise RuntimeError("unexpected ORION visual context shape: %s" % (visual.shape,))
            contexts[group_id] = visual.detach().to(dtype=torch.float16, device="cpu")
            metadata.append({
                "group_id": group_id,
                "selected_saved_frame_index": row["frame"],
                "frame_bundle": {
                    "path": str(row["bundle_path"]),
                    "sha256": sha256_file(row["bundle_path"]),
                },
                "frame_meta": {
                    "path": str(row["meta_path"]),
                    "sha256": sha256_file(row["meta_path"]),
                },
            })
            print("[Stage2LVisualCache] group=%s frame=%d" % (group_id, row["frame"]), flush=True)
    finally:
        model.lm_head = lm_head
        for hook in hooks:
            hook.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "schema": SCHEMA,
        "contexts": contexts,
        "metadata": {
            "event_factory_report": {
                "path": str(args.factory_report.resolve()),
                "sha256": sha256_file(args.factory_report.resolve()),
            },
            "keyframe_count": len(contexts),
            "frames": metadata,
            "orion_config": str(args.orion_config.resolve()),
            "orion_checkpoint": {
                "path": str(args.orion_checkpoint.resolve()),
                "sha256": sha256_file(args.orion_checkpoint.resolve()),
            },
            "token_layout": {
                "det": DET_VISUAL_TOKENS,
                "map": MAP_VISUAL_TOKENS,
                "total": TOTAL_VISUAL_TOKENS,
                "det_breakdown": {
                    "current_object_queries": 256,
                    "temporal_scene_queries": 16,
                    "can_bus": 1,
                },
            },
            "head_memory_reset_per_keyframe": True,
            "privileged_safety_inputs_used": False,
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
            "llm_run_during_cache": False,
            "trajectory_decoder_run_during_cache": False,
        },
    }
    torch.save(cache_payload, args.output)
    manifest = {
        "schema": SCHEMA,
        "status": "immutable_multiframe_visual_context_cache",
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "keyframe_count": len(contexts),
        "group_ids": sorted(contexts),
        "event_factory_report": cache_payload["metadata"]["event_factory_report"],
        "orion_checkpoint_sha256": cache_payload["metadata"]["orion_checkpoint"][
            "sha256"
        ],
        "head_memory_reset_per_keyframe": True,
        "privileged_safety_inputs_used": False,
        "stage1_uq_inputs_used": False,
        "task_relevance_targets_used": False,
        "qa_answers_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-report", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("multi-frame ORION visual-context caching requires CUDA")
    print(json.dumps(cache_multiframe_contexts(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
