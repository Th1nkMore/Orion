#!/usr/bin/env python3
"""Render a six-camera U/R/K contact sheet for human Stage2-L review."""

from __future__ import annotations

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.uq_relevance_bundle_visualization.v1"


def _heatmap(value: np.ndarray, size: tuple[int, int]) -> Image.Image:
    scalar = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    gray = Image.fromarray(np.uint8(np.round(scalar * 255)), mode="L")
    gray = gray.resize(size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    red = np.clip(2.0 * array, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(array - 0.5)), 0.0, 1.0)
    blue = np.clip(2.0 * (0.5 - array), 0.0, 1.0)
    alpha = np.clip(array * 0.72, 0.0, 0.72)
    rgba = np.stack((red, green, blue, alpha), axis=-1)
    rgba[..., :3] *= 255.0
    rgba[..., 3] *= 255.0
    return Image.fromarray(np.uint8(rgba), mode="RGBA")


def _overlay(image: Image.Image, value: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    return Image.alpha_composite(base, _heatmap(value, base.size)).convert("RGB")


def render_bundle(bundle_path: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty visualization output")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    stage1 = bundle["model_input"]["stage1_observation_uq"]
    with np.load(stage1["path"], allow_pickle=False) as payload:
        uncertainty = np.asarray(payload["uncertainty"], dtype=np.float32)
    if uncertainty.ndim == 3:
        uncertainty = uncertainty[None]
    relevance_ref = bundle["supervision"]["task_relevance"]
    with np.load(relevance_ref["path"], allow_pickle=False) as payload:
        relevance = np.asarray(payload["relevance"], dtype=np.float32)
    latest = uncertainty[-1]
    task_risk = latest * relevance
    tile_size = (320, 180)
    label_height = 22
    rows = ("RGB", "U scalar", "R relevance", "K = U x R")
    cameras = bundle["model_input"]["observation"]["camera_files"]
    sheet = Image.new(
        "RGB",
        (tile_size[0] * len(cameras), (tile_size[1] + label_height) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for view_index, camera in enumerate(cameras):
        with Image.open(camera["path"]) as source:
            rgb = source.convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)
        values = (None, latest[view_index], relevance[view_index], task_risk[view_index])
        for row_index, (row_name, value) in enumerate(zip(rows, values)):
            tile = rgb if value is None else _overlay(rgb, value)
            x = view_index * tile_size[0]
            y = row_index * (tile_size[1] + label_height) + label_height
            sheet.paste(tile, (x, y))
            label = "%s | %s" % (camera["view"], row_name)
            draw.text((x + 4, y - label_height + 4), label, fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "uq_relevance_contact_sheet.png"
    sheet.save(output)
    manifest = {
        "schema": SCHEMA,
        "bundle": {"path": str(bundle_path.resolve()), "sha256": sha256_file(bundle_path)},
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output)},
        "variant": bundle["counterfactual"]["variant"],
        "rows": list(rows),
        "camera_order": [item["view"] for item in cameras],
        "claim_boundary": "Human geometry/map consistency review only; not an uncertainty or safety metric.",
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = render_bundle(args.bundle.resolve(), args.output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
