#!/usr/bin/env python3
"""Derive fixed-shape physical U tokens from immutable visibility artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


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
    "_orion_qwen_drive_bridge_tokenizer",
    PROJECT_ROOT / "uq_estimator" / "qwen_drive_bridge.py",
)
_visibility = _load_local_module(
    "_orion_qwen_visibility_tokenizer",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py",
)
load_bridge_config = _bridge.load_bridge_config
ObservationMemoryState = _visibility.ObservationMemoryState
VisibilityBelief = _visibility.VisibilityBelief
VisibilityExposure = _visibility.VisibilityExposure
VisibilityGridSpec = _visibility.VisibilityGridSpec
spatially_shuffle_visibility_tokens = _visibility.spatially_shuffle_visibility_tokens
tokenize_visibility_belief = _visibility.tokenize_visibility_belief
visibility_token_metadata = _visibility.visibility_token_metadata
visibility_token_npz_payload = _visibility.visibility_token_npz_payload
zero_visibility_tokens = _visibility.zero_visibility_tokens


def _channel_array(artifact, array_key, names_key, expected_names):
    names = tuple(str(value) for value in artifact[names_key].tolist())
    if names != tuple(expected_names):
        raise ValueError(
            "%s channel contract mismatch: expected=%s actual=%s"
            % (array_key, tuple(expected_names), names)
        )
    array = np.asarray(artifact[array_key], dtype=np.float32)
    if array.shape[0] != len(expected_names) or not np.isfinite(array).all():
        raise ValueError("%s has invalid shape or non-finite values" % array_key)
    return array


def load_dense_visibility_artifact(path):
    """Reconstruct validated physical states from one O2 NPZ artifact."""

    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata_json"]))
        bounds = metadata["bounds_m"]
        spec = VisibilityGridSpec(
            x_min_m=float(bounds["x"][0]),
            x_max_m=float(bounds["x"][1]),
            y_min_m=float(bounds["y"][0]),
            y_max_m=float(bounds["y"][1]),
            z_min_m=float(bounds["z"][0]),
            z_max_m=float(bounds["z"][1]),
            xy_resolution_m=float(metadata["xy_resolution_m"]),
            z_resolution_m=float(metadata["z_resolution_m"]),
            max_range_m=float(metadata["max_range_m"]),
            surface_tolerance_m=float(metadata["surface_tolerance_m"]),
        )
        visibility_names = (
            "visible_free_ratio",
            "visible_occupied_ratio",
            "occluded_unknown_ratio",
            "outside_fov_ratio",
            "frontier",
        )
        visibility = _channel_array(
            artifact, "channels", "channel_names", visibility_names
        )
        belief = VisibilityBelief(
            spec=spec,
            visible_free_ratio=visibility[0],
            visible_occupied_ratio=visibility[1],
            occluded_unknown_ratio=visibility[2],
            outside_fov_ratio=visibility[3],
            frontier=visibility[4] > 0.5,
        )
        memory_names = (
            "observation_age_normalized",
            "currently_observed",
            "previously_observed",
            "never_observed",
        )
        memory_channels = _channel_array(
            artifact,
            "observation_memory_channels",
            "observation_memory_channel_names",
            memory_names,
        )
        max_age_seconds = float(metadata["observation_memory"]["max_age_seconds"])
        memory = ObservationMemoryState(
            spec=spec,
            age_seconds=memory_channels[0] * max_age_seconds,
            currently_observed=memory_channels[1] > 0.5,
            ever_observed=memory_channels[3] <= 0.5,
            max_age_seconds=max_age_seconds,
        )
        exposure_names = (
            "route_distance_m",
            "route_progress_m",
            "stopping_margin_m",
            "route_weight",
            "stopping_weight",
            "urgency",
        )
        exposure_channels = _channel_array(
            artifact,
            "exposure_channels",
            "exposure_channel_names",
            exposure_names,
        )
        exposure = VisibilityExposure(
            spec=spec,
            route_distance_m=exposure_channels[0],
            route_progress_m=exposure_channels[1],
            stopping_margin_m=exposure_channels[2],
            route_weight=exposure_channels[3],
            stopping_weight=exposure_channels[4],
            urgency=exposure_channels[5],
            stopping_distance_m=float(metadata["exposure"]["stopping_distance_m"]),
        )
    if not metadata.get("oracle_depth") or metadata.get("used_by_qwen"):
        raise ValueError("source must be oracle depth not yet consumed by Qwen")
    return belief, memory, exposure, metadata


def tokenize_run(input_root, output_root, config_path):
    """Tokenize every O2 frame into a new, refuse-to-overwrite directory."""

    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    config_path = Path(config_path).resolve()
    if output_root.exists():
        raise FileExistsError("refusing to reuse token output: %s" % output_root)
    source_paths = sorted(input_root.glob("step_*.npz"))
    if not source_paths:
        raise FileNotFoundError("no step_*.npz artifacts under %s" % input_root)
    config = load_bridge_config(config_path)
    tokenizer = config["oracle_visibility"]["tokenizer"]
    if not tokenizer["enabled"]:
        raise ValueError("visibility tokenizer must be enabled")
    output_root.mkdir(parents=True, exist_ok=False)

    records = []
    sha_lines = []
    for source_path in source_paths:
        belief, memory, exposure, source_metadata = load_dense_visibility_artifact(
            source_path
        )
        started = time.perf_counter()
        tokens = tokenize_visibility_belief(
            belief,
            memory,
            exposure,
            global_grid_shape=tokenizer["global_grid_shape"],
            max_frontier_tokens=int(tokenizer["max_frontier_tokens"]),
            frontier_patch_radius_m=float(tokenizer["frontier_patch_radius_m"]),
            frontier_nms_radius_m=float(tokenizer["frontier_nms_radius_m"]),
            frontier_selection_floor=float(tokenizer["frontier_selection_floor"]),
            depth_confidence=float(tokenizer["oracle_depth_confidence"]),
        )
        controls = {}
        if tokenizer["write_controls"]:
            controls = {
                "zero_u": zero_visibility_tokens(tokens),
                "spatial_shuffle": spatially_shuffle_visibility_tokens(
                    tokens, int(tokenizer["spatial_shuffle_seed"])
                ),
            }
        elapsed = time.perf_counter() - started
        payload = visibility_token_npz_payload(tokens, "visibility_tokens")
        for control_name, control_tokens in controls.items():
            payload.update(
                visibility_token_npz_payload(
                    control_tokens, "visibility_tokens_" + control_name
                )
            )
        provenance = {
            "source_npz": str(source_path),
            "source_step": int(source_metadata["step"]),
            "source_oracle_depth": True,
            "source_used_by_qwen": False,
            "tokenize_seconds": float(elapsed),
            "tokens": visibility_token_metadata(tokens),
            "controls": sorted(controls),
        }
        payload["provenance_json"] = np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        )
        output_path = output_root / source_path.name
        np.savez_compressed(output_path, **payload)
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        sha_lines.append("%s  %s" % (digest, output_path.name))
        records.append(
            {
                "source": str(source_path),
                "output": output_path.name,
                "sha256": digest,
                "step": int(source_metadata["step"]),
                "tokenize_seconds": float(elapsed),
                "valid_frontier_tokens": tokens.valid_frontier_count,
                "maximum_frontier_selection_score": float(
                    tokens.frontier_tokens[
                        tokens.frontier_mask,
                        tokens.feature_names.index("frontier_selection_score"),
                    ].max()
                    if tokens.valid_frontier_count
                    else 0.0
                ),
            }
        )

    manifest = {
        "schema": "orion.qwen-visibility-token-artifact-run/v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "config_path": str(config_path),
        "frame_count": len(records),
        "source_artifacts_modified": False,
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "artifact_sha256.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = tokenize_run(args.input_root, args.output_root, args.config)
    elapsed = [record["tokenize_seconds"] for record in manifest["records"]]
    valid = [record["valid_frontier_tokens"] for record in manifest["records"]]
    print("TOKENIZED_FRAMES=%d" % manifest["frame_count"])
    print("TOKENIZE_SECONDS_MEAN=%.6f" % float(np.mean(elapsed)))
    print("TOKENIZE_SECONDS_P95=%.6f" % float(np.percentile(elapsed, 95)))
    print("VALID_FRONTIER_TOKENS_MIN=%d" % min(valid))
    print("VALID_FRONTIER_TOKENS_MAX=%d" % max(valid))
    print("OUTPUT_ROOT=%s" % manifest["output_root"])


if __name__ == "__main__":
    main()
