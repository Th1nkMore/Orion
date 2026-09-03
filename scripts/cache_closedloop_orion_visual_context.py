#!/usr/bin/env python3
"""Cache ORION's native det/map visual tokens for one saved closed-loop frame.

This is an engineering utility for the Route196 Stage2-L overfit smoke.  It
replays the saved six-camera observation through the same inference-only image
pipeline, EVAViT backbone, position embedding, detection head and map head used
by the closed-loop agent.  The LLM and trajectory decoder are deliberately not
run.  With ORION temporal memory enabled, the native LLM input contains 273
detection-side tokens (256 current object queries, 16 temporal scene queries,
and one CAN-bus token) plus 256 map tokens. These 529 tokens are immutable
inputs to the later UQ-language smoke; no privileged labels enter this cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import cv2
import numpy as np
import torch

from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type

from scripts.scenario_factory_lib import sha256_file
import team_code.orion_b2d_agent as orion_agent_module
from team_code.orion_b2d_agent import OrionAgent, command2hot, command2nohot


SCHEMA = "orion.closedloop_visual_context_cache.v1"
DET_VISUAL_TOKENS = 273
MAP_VISUAL_TOKENS = 256
TOTAL_VISUAL_TOKENS = DET_VISUAL_TOKENS + MAP_VISUAL_TOKENS
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _make_offline_orion_agent() -> OrionAgent:
    """Construct the agent for saved-frame replay without touching CARLA.

    Bench2Drive's AutonomousAgent constructor immediately resolves the hero
    actor from CarlaDataProvider.  Saved-frame visual caching intentionally has
    no CARLA world, and OrionAgent.setup() does not require that actor: it only
    loads the model, inference pipeline, and fixed sensor calibration used
    below.  Bypass only the online base constructor; never call _init() or
    run_step() on this offline instance.
    """
    return OrionAgent.__new__(OrionAgent)


def _setup_offline_orion_agent(agent: OrionAgent, config: str) -> None:
    """Set up saved-frame replay without CARLA's compiled extension.

    ``OrionAgent.setup`` uses ``VehicleControl`` only to initialize three
    previous-control scalars. Saved-frame replay never connects to CARLA, so
    provide exactly that temporary container when the compiled type is absent,
    then restore the imported module. Online evaluation never calls this.
    """
    carla_module = orion_agent_module.carla
    sentinel = object()
    original = getattr(carla_module, "VehicleControl", sentinel)
    if original is sentinel:
        carla_module.VehicleControl = lambda: SimpleNamespace(
            steer=0.0, throttle=0.0, brake=0.0
        )
    try:
        agent.setup(config)
    finally:
        if original is sentinel:
            delattr(carla_module, "VehicleControl")


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (900, 1600, 3):
        raise ValueError("saved camera must be a 1600x900 RGB PNG: %s" % path)
    return image


def _camera_paths(bundle: Mapping[str, Any]) -> Dict[str, Path]:
    rows = bundle["model_input"]["observation"]["camera_files"]
    paths = {str(row["view"]): Path(str(row["path"])) for row in rows}
    if tuple(paths) != CAMERA_ORDER:
        raise ValueError("frame bundle camera order is not canonical ORION order")
    for view, path in paths.items():
        if not path.is_file() or sha256_file(path) != next(
            row["sha256"] for row in rows if row["view"] == view
        ):
            raise ValueError("camera provenance mismatch: %s" % path)
    return paths


def _build_pipeline_record(
    agent: OrionAgent,
    bundle: Mapping[str, Any],
    frame_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    paths = _camera_paths(bundle)
    images = [_load_bgr(paths[view]) for view in CAMERA_ORDER]
    lidar2img = np.stack([agent.lidar2img[view] for view in CAMERA_ORDER])
    lidar2cam = np.stack([agent.lidar2cam[view] for view in CAMERA_ORDER])
    cam_intrinsic = np.stack([
        np.matmul(agent.lidar2img[view], np.linalg.inv(agent.lidar2cam[view]))
        for view in CAMERA_ORDER
    ])
    command = int(bundle["model_input"]["route_context"]["payload"]["command"])
    ego_pose = np.asarray(agent.lidar2ego, dtype=np.float32)
    ego_pose_inv = np.linalg.inv(ego_pose).astype(np.float32)
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[7] = float(frame_meta.get("speed", 0.0))
    critical_step = int(bundle["provenance"]["critical_control_step"])
    stacked = np.stack(images, axis=-1)
    record = {
        "lidar2img": lidar2img,
        "lidar2cam": lidar2cam,
        "cam_intrinsic": cam_intrinsic,
        "img": images,
        "folder": "route%s" % bundle["route"]["route_id"],
        "scene_token": "stage2l_%s" % bundle["event_id"],
        "frame_idx": critical_step,
        "timestamp": critical_step / 20.0,
        "box_type_3d": get_box_type("LiDAR")[0],
        "can_bus": can_bus,
        "command": command2nohot(command),
        "ego_fut_cmd": command2hot(command),
        "ego_pose": ego_pose,
        "ego_pose_inv": ego_pose_inv,
        "lidar2ego": np.asarray(agent.lidar2ego),
        "l2g_r_mat": ego_pose[:3, :3],
        "l2g_t": ego_pose[:3, 3],
        "img_shape": stacked.shape,
        "ori_shape": stacked.shape,
        "pad_shape": stacked.shape,
    }
    return agent.inference_only_pipeline(record)


def _move_agent_batch_to_cuda(batch: Mapping[str, Any]) -> None:
    for key, data in batch.items():
        if key != "img_metas" and torch.is_tensor(data[0]):
            data[0] = data[0].cuda(non_blocking=True)
        if key == "input_ids":
            for item in data[0]:
                for index in range(len(item)):
                    item[index] = item[index].cuda(non_blocking=True)


def cache_context(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite visual-context cache")
    bundle = _load_json(args.frame_bundle.resolve())
    if bundle.get("schema") != "orion.uq_relevance_frame_bundle.v1":
        raise ValueError("unsupported frame-bundle schema")
    meta_path = Path(str(args.frame_meta)).resolve()
    frame_meta = _load_json(meta_path)

    os.environ.pop("SAVE_PATH", None)
    agent = _make_offline_orion_agent()
    _setup_offline_orion_agent(agent, "%s+%s+stage2l_route196_cache" % (
        args.orion_config.resolve(), args.orion_checkpoint.resolve()
    ))
    model = agent.model
    model.eval()
    model.pts_bbox_head.reset_memory()
    model.map_head.reset_memory()

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
    try:
        processed = _build_pipeline_record(agent, bundle, frame_meta)
        batch = mm_collate_to_batch_form([processed], samples_per_gpu=1)
        _move_agent_batch_to_cuda(batch)
        with torch.inference_mode():
            model(batch, return_loss=False)
    finally:
        model.lm_head = lm_head
        for hook in hooks:
            hook.remove()
    if set(captured) != {"det", "map"}:
        raise RuntimeError("ORION det/map hooks did not both fire")
    expected_det = (1, DET_VISUAL_TOKENS, 4096)
    expected_map = (1, MAP_VISUAL_TOKENS, 4096)
    if tuple(captured["det"].shape) != expected_det:
        raise RuntimeError("unexpected ORION detection-token shape: %s" % (captured["det"].shape,))
    if tuple(captured["map"].shape) != expected_map:
        raise RuntimeError("unexpected ORION map-token shape: %s" % (captured["map"].shape,))
    visual = torch.cat((captured["det"], captured["map"]), dim=1)
    if tuple(visual.shape) != (1, TOTAL_VISUAL_TOKENS, 4096):
        raise RuntimeError("unexpected ORION visual context shape: %s" % (visual.shape,))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "baseline_vision": visual.detach().to(dtype=torch.float16, device="cpu"),
        "metadata": {
            "event_id": bundle["event_id"],
            "route_id": bundle["route"]["route_id"],
            "frame_bundle": {
                "path": str(args.frame_bundle.resolve()),
                "sha256": sha256_file(args.frame_bundle.resolve()),
            },
            "frame_meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
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
            "privileged_safety_inputs_used": False,
            "llm_run_during_cache": False,
            "trajectory_decoder_run_during_cache": False,
        },
    }
    torch.save(payload, args.output)
    return {
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "shape": list(visual.shape),
        "schema": SCHEMA,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-bundle", type=Path, required=True)
    parser.add_argument("--frame-meta", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ORION visual-context caching requires CUDA")
    print(json.dumps(cache_context(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
