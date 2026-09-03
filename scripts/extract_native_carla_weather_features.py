#!/usr/bin/env python3
"""Extract frozen ORION EVAViT features from native CARLA weather captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.extract_paired_spatial_features import (  # noqa: E402
    _build_real_backbone,
    _extract_tokens,
)
from uq_estimator.native_weather_audit import (  # noqa: E402
    EXPECTED_CONDITIONS,
    NATIVE_WEATHER_FEATURE_SCHEMA_VERSION,
)


CAMERA_DIRECTORIES = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
)
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preprocess_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (900, 1600, 3):
        raise RuntimeError("native camera image has unexpected shape: %s" % path)
    # Match orion_stage3_agent.py inference: deterministic IDA resize/crop,
    # ResizeMultiview3D to 640x640, BGR->RGB, then ImageNet normalization.
    pil = Image.fromarray(image.astype(np.uint8))
    pil = pil.resize((640, 360)).crop((0, 40, 640, 360))
    image = cv2.resize(np.asarray(pil, dtype=np.float32), (640, 640))
    image = image[..., ::-1].copy()
    image = (image - MEAN) / STD
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def _image_path(capture_root, condition, item, directory):
    route_directory = str(item["route_id"]).replace("/", "_")
    return (
        capture_root
        / condition
        / route_directory
        / directory
        / ("%04d.png" % int(item["sequence_index"]))
    )


def _load_batch(capture_root, condition, items):
    rows = []
    for item in items:
        views = []
        for directory in CAMERA_DIRECTORIES:
            path = _image_path(capture_root, condition, item, directory)
            if not path.is_file():
                raise RuntimeError("missing native camera image: %s" % path)
            views.append(_preprocess_image(path))
        rows.append(torch.stack(views))
    return torch.stack(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if args.batch_size <= 0 or not torch.cuda.is_available():
        raise SystemExit("feature extraction requires a positive batch and CUDA")
    expected_camera_order = (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )
    manifests = []
    item_sources = []
    combined_conditions = None
    seen_samples = set()
    for capture_root in args.capture_root:
        manifest_path = capture_root / "capture_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit("capture manifest is missing: %s" % manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != "orion.native-carla-weather-capture/v1":
            raise SystemExit("unexpected capture manifest schema")
        if manifest.get("paired_world_pose") is not True or manifest.get(
            "pixel_corruption_generator_used"
        ) is not False:
            raise SystemExit("capture is not an exact-pose native intervention")
        if manifest.get("renderer_quality") != "Epic":
            raise SystemExit("native weather feature gate requires Epic rendering")
        if tuple(manifest.get("camera_order", ())) != expected_camera_order:
            raise SystemExit("capture camera order does not match ORION")
        if combined_conditions is None:
            combined_conditions = manifest.get("conditions")
        elif combined_conditions != manifest.get("conditions"):
            raise SystemExit("native weather definitions differ across captures")
        manifest_items = manifest.get("items")
        if not isinstance(manifest_items, list) or not manifest_items:
            raise SystemExit("capture manifest has no items")
        for item in manifest_items:
            sample_id = str(item.get("sample_id", ""))
            if not sample_id or sample_id in seen_samples:
                raise SystemExit("capture sample IDs are empty or duplicated")
            seen_samples.add(sample_id)
            item_sources.append((capture_root, item))
        manifests.append(
            {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
            }
        )
    if len(args.capture_root) < 2:
        raise SystemExit("native gate requires independently captured routes")
    items = [item for _, item in item_sources]

    _, backbone, backbone_metadata = _build_real_backbone(args)
    features_by_condition = {}
    patch_shape = None
    for condition in EXPECTED_CONDITIONS:
        chunks = []
        for start in range(0, len(items), args.batch_size):
            batch_sources = item_sources[start : start + args.batch_size]
            images = torch.stack(
                [
                    _load_batch(capture_root, condition, [item])[0]
                    for capture_root, item in batch_sources
                ]
            ).cuda(non_blocking=True)
            with torch.inference_mode():
                tokens, height, width = _extract_tokens(backbone, images)
            grid = tokens.reshape(
                tokens.shape[0], tokens.shape[1], height, width, tokens.shape[-1]
            ).half().cpu()
            chunks.append(grid)
            if patch_shape is None:
                patch_shape = tuple(grid.shape[1:])
            elif tuple(grid.shape[1:]) != patch_shape:
                raise RuntimeError("EVAViT grid shape changed during extraction")
            print(
                "[NativeWeatherFeature] condition=%s progress=%d/%d"
                % (condition, min(start + args.batch_size, len(items)), len(items)),
                flush=True,
            )
        features_by_condition[condition] = torch.cat(chunks)

    payload = {
        "schema_version": NATIVE_WEATHER_FEATURE_SCHEMA_VERSION,
        "items": items,
        "features_by_condition": features_by_condition,
        "conditions": combined_conditions,
        "camera_order": list(expected_camera_order),
        "paired_world_pose": True,
        "pixel_corruption_generator_used": False,
        "renderer_quality": "Epic",
        "capture_manifests": manifests,
        "backbone": backbone_metadata,
        "preprocessing": {
            "source": "orion_stage3_agent.py deterministic inference transforms",
            "input_shape": [900, 1600],
            "ida_resize": [360, 640],
            "ida_crop_xyxy": [0, 40, 640, 360],
            "final_shape": [640, 640],
            "bgr_to_rgb": True,
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp-%d" % os.getpid())
    torch.save(payload, temporary)
    os.replace(str(temporary), str(args.output))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "item_count": len(items),
                "condition_count": len(features_by_condition),
                "feature_shape": list(features_by_condition["clear"].shape),
                "output_bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )
    print("NATIVE_CARLA_WEATHER_FEATURES_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
