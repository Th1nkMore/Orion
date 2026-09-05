#!/usr/bin/env python3
"""Verify continuous visibility-token insertion on the released Qwen 4B VLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_bridge = _load_local_module(
    "_orion_qwen_bridge_vlm_smoke",
    PROJECT_ROOT / "uq_estimator" / "qwen_drive_bridge.py",
)
_adapter = _load_local_module(
    "_orion_qwen_visibility_vlm_smoke",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_vlm.py",
)


def _scene_from_images(front, front_left, front_right):
    from qwen_drive import CAMERA_VIEWS, CameraFrame, DrivingScene

    paths = (front, front_left, front_right)
    views = {
        view: [CameraFrame(path) for _ in range(4)]
        for view, path in zip(CAMERA_VIEWS, paths)
    }
    return DrivingScene(
        views=views,
        history=np.zeros((16, 3), dtype=np.float32),
        history_velocity=np.zeros((16, 2), dtype=np.float32),
        history_acceleration=np.zeros((16, 2), dtype=np.float32),
        ego_velocity=np.zeros(2, dtype=np.float32),
        ego_acceleration=np.zeros(2, dtype=np.float32),
        driving_command=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        nav_command=0,
        token="qwen-visibility-vlm-v0-smoke",
    )


def _load_tokens(path):
    with np.load(path, allow_pickle=False) as artifact:
        global_tokens = np.asarray(artifact["visibility_tokens_global"], dtype=np.float32)
        frontier_tokens = np.asarray(
            artifact["visibility_tokens_frontier"], dtype=np.float32
        )
        global_mask = np.asarray(artifact["visibility_tokens_global_mask"], dtype=bool)
        frontier_mask = np.asarray(
            artifact["visibility_tokens_frontier_mask"], dtype=bool
        )
        feature_names = artifact["visibility_tokens_feature_names"].tolist()
        metadata = json.loads(str(artifact["visibility_tokens_metadata_json"]))
    tokens = np.concatenate([global_tokens, frontier_tokens], axis=0)
    mask = np.concatenate([global_mask, frontier_mask], axis=0)
    if tokens.ndim != 2 or tokens.shape[1] != len(feature_names):
        raise ValueError("physical token feature contract mismatch")
    if metadata["feature_names"] != feature_names:
        raise ValueError("token metadata feature order mismatch")
    return tokens, mask, feature_names, metadata


def _cache_equal(first, second):
    if len(first) != len(second):
        return False
    return all(
        torch.equal(first_key, second_key) and torch.equal(first_value, second_value)
        for (first_key, first_value), (second_key, second_value) in zip(first, second)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-artifact", type=Path, required=True)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--front-left", type=Path, required=True)
    parser.add_argument("--front-right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "reasoning"), default="direct")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite %s" % args.output)

    from qwen_drive import QwenDriveForPlanning

    config = _bridge.load_bridge_config(args.config)
    runtime = config["runtime"]
    planning = config["planning"]
    dtype = getattr(torch, str(runtime["dtype"]))
    load_started = time.monotonic()
    model = QwenDriveForPlanning.from_pretrained(
        runtime["model"],
        planner=runtime["planner"],
        dtype=dtype,
        attn_implementation=runtime["attention_implementation"],
    ).to(runtime["device"]).eval()
    load_seconds = time.monotonic() - load_started

    scene = _scene_from_images(args.front, args.front_left, args.front_right)
    inputs = model.processor(
        scene, with_reasoning=args.mode == "reasoning", device="cpu"
    )
    inputs = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    token_array, token_mask, feature_names, token_metadata = _load_tokens(
        args.token_artifact
    )
    projector = _adapter.VisibilityTokenProjector(
        feature_dim=len(feature_names),
        hidden_dim=512,
        vlm_hidden_dim=int(model.config.vlm_config.text_config.hidden_size),
    ).to(model.device).eval()
    token_tensor = torch.from_numpy(token_array).to(model.device)
    mask_tensor = torch.from_numpy(token_mask).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        official_started = time.monotonic()
        if args.mode == "reasoning":
            official_cache, official_anchor, official_reasoning = (
                model._prefill_with_reasoning(
                    inputs, int(model.config.max_reasoning_tokens)
                )
            )
        else:
            official_cache, official_anchor = model._prefill(inputs)
            official_reasoning = None
        official_seconds = time.monotonic() - official_started
        reference_started = time.monotonic()
        if args.mode == "reasoning":
            reference = _adapter.manual_reasoning_prefill_without_visibility(
                model, inputs, int(model.config.max_reasoning_tokens)
            )
            reference_cache = reference.scene_cache
            reference_anchor = reference.anchor
            reference_reasoning = reference.reasoning
            reference_prompt = reference.prompt_prefill
            reference_final_length = reference.final_sequence_length
        else:
            reference = _adapter.prefill_with_visibility_tokens(
                model,
                inputs,
                token_features=None,
                token_mask=None,
                projector=None,
                enabled=False,
            )
            reference_cache = reference.scene_cache
            reference_anchor = reference.anchor
            reference_reasoning = None
            reference_prompt = reference
            reference_final_length = reference.augmented_sequence_length
        reference_seconds = time.monotonic() - reference_started
        augmented_started = time.monotonic()
        if args.mode == "reasoning":
            augmented = _adapter.prefill_with_visibility_reasoning(
                model,
                inputs,
                token_features=token_tensor,
                token_mask=mask_tensor,
                projector=projector,
                enabled=True,
                max_new_tokens=int(model.config.max_reasoning_tokens),
            )
            augmented_cache = augmented.scene_cache
            augmented_anchor = augmented.anchor
            augmented_reasoning = augmented.reasoning
            augmented_prompt = augmented.prompt_prefill
            augmented_final_length = augmented.final_sequence_length
        else:
            augmented = _adapter.prefill_with_visibility_tokens(
                model,
                inputs,
                token_features=token_tensor,
                token_mask=mask_tensor,
                projector=projector,
                enabled=True,
            )
            augmented_cache = augmented.scene_cache
            augmented_anchor = augmented.anchor
            augmented_reasoning = None
            augmented_prompt = augmented
            augmented_final_length = augmented.augmented_sequence_length
        augmented_seconds = time.monotonic() - augmented_started

        official_trajectory = model._plan_from_cache(
            official_cache,
            official_anchor,
            inputs,
            num_samples=1,
            num_steps=int(model.config.num_inference_steps),
            seed=int(planning["seed"]),
        )
        reference_trajectory = model._plan_from_cache(
            reference_cache,
            reference_anchor,
            inputs,
            num_samples=1,
            num_steps=int(model.config.num_inference_steps),
            seed=int(planning["seed"]),
        )
        augmented_trajectory = model._plan_from_cache(
            augmented_cache,
            augmented_anchor,
            inputs,
            num_samples=1,
            num_steps=int(model.config.num_inference_steps),
            seed=int(planning["seed"]),
        )
        projector_zero_initialized = not bool(
            projector(token_tensor[mask_tensor]).any().item()
        )

    reference_position_contract = _adapter.visibility_position_contract(
        reference_prompt
    )
    augmented_position_contract = _adapter.visibility_position_contract(
        augmented_prompt
    )
    reference_cache_equal = _cache_equal(official_cache, reference_cache)
    reference_anchor_equal = torch.equal(official_anchor, reference_anchor)
    official_np = np.asarray(official_trajectory, dtype=np.float32)
    reference_np = np.asarray(reference_trajectory, dtype=np.float32)
    augmented_np = np.asarray(augmented_trajectory, dtype=np.float32)
    failures = []
    if not reference_cache_equal:
        failures.append("reference_cache_not_official_identity")
    if not reference_anchor_equal:
        failures.append("reference_anchor_not_official_identity")
    if not np.array_equal(official_np, reference_np):
        failures.append("reference_trajectory_not_official_identity")
    if official_reasoning != reference_reasoning:
        failures.append("reference_reasoning_not_official_identity")
    for key in (
        "prefix_positions_equal",
        "suffix_shift_exact",
        "visibility_positions_contiguous",
        "anchor_exact",
        "scene_cache_length_exact",
    ):
        if not augmented_position_contract[key]:
            failures.append("augmented_%s_failed" % key)
    if augmented_prompt.augmented_sequence_length != (
        augmented_prompt.base_sequence_length
        + augmented_prompt.visibility_token_count
        + 2
    ):
        failures.append("augmented_sequence_length_mismatch")
    reference_cache_lengths = [int(key.shape[1]) for key, _ in reference_cache]
    augmented_cache_lengths = [int(key.shape[1]) for key, _ in augmented_cache]
    if not all(length == reference_final_length for length in reference_cache_lengths):
        failures.append("reference_final_cache_length_mismatch")
    if not all(length == augmented_final_length for length in augmented_cache_lengths):
        failures.append("augmented_final_cache_length_mismatch")
    if not projector_zero_initialized:
        failures.append("new_projector_is_not_zero_initialized")

    metrics = {}
    if torch.cuda.is_available():
        megabytes = 1024.0 * 1024.0
        metrics = {
            "cuda_allocated_mb": torch.cuda.memory_allocated() / megabytes,
            "cuda_reserved_mb": torch.cuda.memory_reserved() / megabytes,
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / megabytes,
            "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / megabytes,
        }
    report = {
        "schema": _adapter.VISIBILITY_VLM_SCHEMA,
        "status": "complete" if not failures else "failed",
        "failures": failures,
        "mode": args.mode,
        "config": str(args.config.resolve()),
        "token_artifact": str(args.token_artifact.resolve()),
        "token_schema": token_metadata["schema"],
        "feature_count": len(feature_names),
        "physical_token_rows": int(len(token_array)),
        "valid_physical_tokens": int(token_mask.sum()),
        "projector_parameters": int(sum(p.numel() for p in projector.parameters())),
        "projector_zero_initialized": projector_zero_initialized,
        "insertion_location": "after_last_vision_end_before_instruction",
        "reference_position_contract": reference_position_contract,
        "augmented_position_contract": augmented_position_contract,
        "reference_cache_equal": bool(reference_cache_equal),
        "reference_anchor_equal": bool(reference_anchor_equal),
        "reference_trajectory_equal": bool(np.array_equal(official_np, reference_np)),
        "official_reasoning": official_reasoning,
        "reference_reasoning": reference_reasoning,
        "augmented_reasoning": augmented_reasoning,
        "reference_final_sequence_length": reference_final_length,
        "augmented_final_sequence_length": augmented_final_length,
        "reference_final_cache_lengths": reference_cache_lengths,
        "augmented_final_cache_lengths": augmented_cache_lengths,
        "zero_initialized_augmented_trajectory_max_abs_difference": float(
            np.max(np.abs(official_np - augmented_np))
        ),
        "load_seconds": float(load_seconds),
        "official_prefill_seconds": float(official_seconds),
        "reference_prefill_seconds": float(reference_seconds),
        "augmented_prefill_seconds": float(augmented_seconds),
        "runtime_metrics": metrics,
        "scientific_scope": (
            "interface smoke only; the projector is untrained and no behavior or "
            "safety improvement is claimed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
