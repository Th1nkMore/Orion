#!/usr/bin/env python3
"""Render Route151 stale/waterdrop/motion-blur GIFs without loading ORION."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from uq_estimator.corruptions import IMAGENET_MEAN, IMAGENET_STD
from uq_estimator.lens_waterdrop import apply_lens_waterdrop


SCHEMA = "orion.corruption_hardcase_visual_bakeoff_result.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _read_trace(profile_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    paths = tuple(profile_root.glob("records_*/capture_trace.jsonl"))
    if len(paths) != 1:
        raise ValueError("expected one capture trace below %s" % profile_root)
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    if not rows:
        raise ValueError("empty capture trace: %s" % paths[0])
    return paths[0], rows


def stale_source_indices(timestamps: Iterable[float], delay_ms: int) -> list[int]:
    values = [float(value) for value in timestamps]
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError("timestamps must be strictly increasing")
    delay = int(delay_ms) / 1000.0
    return [max(0, bisect.bisect_right(values, value - delay + 1e-9) - 1)
            for value in values]


def nearest_progress_indices(
    source_progress: Iterable[float], target_progress: Iterable[float]
) -> list[int]:
    source = [float(value) for value in source_progress]
    if not source:
        raise ValueError("source progress must not be empty")
    # Independent CARLA runs can exhibit millimetric progress regression at a
    # stop; use an order-agnostic nearest match rather than silently sorting
    # away the original frame index.
    return [
        min(range(len(source)), key=lambda item: abs(source[item] - float(target)))
        for target in target_progress
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame(path: str | Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _label(image: Image.Image, text: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 24), fill=(0, 0, 0))
    draw.text((7, 5), text, fill=(255, 255, 255))
    return result


def _panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    labeled = [_label(image, label) for image, label in zip(images, labels)]
    canvas = Image.new("RGB", (sum(image.width for image in labeled), labeled[0].height))
    x = 0
    for image in labeled:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _save_gif(path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        raise ValueError("refusing to save empty GIF")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )


def _rgb_to_normalized(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (tensor - mean) / std


def _normalized_to_rgb(tensor: torch.Tensor) -> Image.Image:
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    array = (tensor * std + mean)[0, 0].permute(1, 2, 0)
    return Image.fromarray(array.round().clamp(0, 255).byte().cpu().numpy(), "RGB")


def _edge_variance(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _select_evenly(count: int, maximum: int) -> list[int]:
    if count <= maximum:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, maximum).round().astype(int).tolist()))


def render(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite bake-off output")
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != "orion.corruption_hardcase_visual_bakeoff.v1":
        raise ValueError("protocol schema differs")
    output.mkdir(parents=True)
    profiles = protocol["conditions"]["native_motion_blur"]["profiles"]
    traces, rows = {}, {}
    for profile in profiles:
        trace_path, profile_rows = _read_trace(root / "captures" / profile)
        if any(row.get("orion_loaded") is not False for row in profile_rows):
            raise RuntimeError("visual capture loaded ORION")
        if any(row.get("corruption_family") != "native_motion_blur" for row in profile_rows):
            raise RuntimeError("visual capture family differs")
        if any(row.get("profile") != profile for row in profile_rows):
            raise RuntimeError("visual capture profile differs")
        if any((row.get("camera_postprocess_readback") or {}).get("status") != "verified"
               for row in profile_rows):
            raise RuntimeError("motion-blur actor readback is not verified")
        traces[profile], rows[profile] = trace_path, profile_rows

    clean = rows["clean"]
    render_spec = protocol["render"]
    size = (int(render_spec["width"]), int(render_spec["height"]))
    fps = int(render_spec["output_fps"])
    selected = _select_evenly(len(clean), int(render_spec["maximum_frames"]))
    clean_timestamps = [float(row["sim_time_seconds"]) for row in clean]
    clean_progress = [float(row["route_progress"]) for row in clean]
    selected_progress = [clean_progress[index] for index in selected]
    clean_frames = [_frame(clean[index]["front"], size) for index in selected]

    stale_metrics = {}
    stale_columns: dict[int, list[Image.Image]] = {}
    for delay_ms in protocol["conditions"]["front_stale"]["delays_ms"]:
        source_indices = stale_source_indices(clean_timestamps, int(delay_ms))
        stale_columns[int(delay_ms)] = [
            _frame(clean[source_indices[index]]["front"], size) for index in selected
        ]
        effective = [
            1000.0 * (clean_timestamps[index] - clean_timestamps[source_indices[index]])
            for index in selected
            if clean_timestamps[index] - clean_timestamps[0]
            >= int(delay_ms) / 1000.0 - 1e-9
        ]
        stale_metrics[str(delay_ms)] = {
            "effective_delay_ms_min": min(effective),
            "effective_delay_ms_max": max(effective),
            "effective_delay_ms_mean": float(np.mean(effective)),
            "maximum_overshoot_ms": max(value - int(delay_ms) for value in effective),
        }
    stale_panels = [
        _panel(
            [clean_frames[offset]] + [stale_columns[int(delay)][offset]
                                      for delay in protocol["conditions"]["front_stale"]["delays_ms"]],
            ["clean", "stale 100 ms", "stale 200 ms", "stale 400 ms"],
        )
        for offset in range(len(selected))
    ]
    _save_gif(output / "front_stale_bakeoff.gif", stale_panels, fps)

    waterdrop_columns: dict[int, list[Image.Image]] = {}
    waterdrop_metrics = {}
    seed = int(protocol["conditions"]["lens_waterdrop"]["seed"])
    base_time = clean_timestamps[selected[0]]
    for severity in protocol["conditions"]["lens_waterdrop"]["severities"]:
        rendered, fractions = [], []
        for clean_index, image in zip(selected, clean_frames):
            result = apply_lens_waterdrop(
                _rgb_to_normalized(image),
                severity=int(severity),
                view_indices=[0],
                seed=seed,
                elapsed_seconds=clean_timestamps[clean_index] - base_time,
            )
            rendered.append(_normalized_to_rgb(result.images))
            fractions.append(float(result.metadata["mask_fraction"]))
        waterdrop_columns[int(severity)] = rendered
        waterdrop_metrics[str(severity)] = {
            "mask_fraction_mean": float(np.mean(fractions)),
            "mask_fraction_min": min(fractions),
            "mask_fraction_max": max(fractions),
        }
    waterdrop_panels = [
        _panel(
            [clean_frames[offset]] + [waterdrop_columns[int(severity)][offset]
                                      for severity in protocol["conditions"]["lens_waterdrop"]["severities"]],
            ["clean", "waterdrop light", "waterdrop medium", "waterdrop heavy"],
        )
        for offset in range(len(selected))
    ]
    _save_gif(output / "lens_waterdrop_bakeoff.gif", waterdrop_panels, fps)

    motion_columns, motion_metrics = {}, {}
    for profile in profiles:
        profile_progress = [float(row["route_progress"]) for row in rows[profile]]
        indices = nearest_progress_indices(profile_progress, selected_progress)
        images = [_frame(rows[profile][index]["front"], size) for index in indices]
        motion_columns[profile] = images
        edge = [_edge_variance(image) for image in images]
        motion_metrics[profile] = {
            "edge_variance_mean": float(np.mean(edge)),
            "matched_progress_abs_error_max": max(
                abs(profile_progress[index] - target)
                for index, target in zip(indices, selected_progress)
            ),
        }
    clean_edge = motion_metrics["clean"]["edge_variance_mean"]
    for profile in profiles:
        motion_metrics[profile]["edge_variance_ratio_to_clean"] = (
            motion_metrics[profile]["edge_variance_mean"] / clean_edge
            if clean_edge > 0 else None
        )
    motion_panels = [
        _panel([motion_columns[profile][offset] for profile in profiles], profiles)
        for offset in range(len(selected))
    ]
    _save_gif(output / "native_motion_blur_bakeoff.gif", motion_panels, fps)

    bev_frames = [_frame(clean[index]["bev"], size) for index in selected]
    _save_gif(
        output / "route151_clean_bev_context.gif",
        [_label(image, "clean BEV context (corruption is front-camera only)")
         for image in bev_frames],
        fps,
    )
    contact_offset = min(
        range(len(selected)),
        key=lambda offset: abs(
            selected_progress[offset] - float(render_spec["contact_progress"])
        ),
    )
    stale_panels[contact_offset].save(output / "front_stale_contact.png")
    waterdrop_panels[contact_offset].save(output / "lens_waterdrop_contact.png")
    motion_panels[contact_offset].save(output / "native_motion_blur_contact.png")

    result = {
        "schema": SCHEMA,
        "status": "visual_bakeoff_rendered_pending_human_severity_review",
        "orion_loaded": False,
        "route": protocol["route"],
        "selected_frame_count": len(selected),
        "stale_metrics": stale_metrics,
        "waterdrop_metrics": waterdrop_metrics,
        "motion_blur_metrics": motion_metrics,
        "provenance": {
            "protocol": {"path": str(protocol_path.resolve()), "sha256": _sha256(protocol_path)},
            "traces": {profile: {"path": str(traces[profile].resolve()), "sha256": _sha256(traces[profile])}
                       for profile in profiles},
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["artifacts"] = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path != result_path
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "selected_frame_count": result["selected_frame_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
