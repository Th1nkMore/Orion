#!/usr/bin/env python3
"""Overlay captured new Stage-1 spatial UQ on a closed-loop camera GIF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.stage2_artifact_capture import ARTIFACT_INDEX_SCHEMA


CAMERA_FRAME_DIRS = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=tuple(CAMERA_FRAME_DIRS), default="CAM_FRONT")
    parser.add_argument("--center-time-seconds", type=float)
    parser.add_argument("--pre-seconds", type=float, default=5.0)
    parser.add_argument("--post-seconds", type=float, default=6.0)
    parser.add_argument("--full-route", action="store_true")
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _tensor(path: str) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload["observation_uq"]
    return tensor.detach().float().numpy()


def _critical_time(rows: list[dict]) -> tuple[float, str]:
    response_rows = []
    for row in rows:
        response = row.get("planning_response") or {}
        state = (response.get("yield_label") or {}).get("state")
        if state in {"prepare_yield", "hold", "release"}:
            response_rows.append(row)
    if response_rows:
        # Center the rendering window on the first task-relevant response.  It
        # is more specific to Route147 than a global TTC over unrelated cars.
        return (
            float(response_rows[0]["sim_time_seconds"]),
            "first_privileged_yield_response",
        )

    candidates = []
    for row in rows:
        value = (row.get("closedloop_safety") or {}).get(
            "min_obb_collision_ttc_seconds"
        )
        if value is not None:
            candidates.append((float(value), int(row["step"]), float(row["sim_time_seconds"])))
    if candidates:
        return min(candidates)[2], "minimum_finite_obb_ttc"

    for row in rows:
        value = (row.get("closedloop_safety") or {}).get(
            "min_disc_collision_ttc_seconds"
        )
        if value is not None:
            candidates.append((float(value), int(row["step"]), float(row["sim_time_seconds"])))
    if candidates:
        return min(candidates)[2], "minimum_finite_disc_ttc"
    return float(rows[len(rows) // 2]["sim_time_seconds"]), "route_midpoint_fallback"


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or min(args.pre_seconds, args.post_seconds) < 0:
        raise ValueError("invalid GIF timing")
    run_dir = args.run_dir.resolve()
    index_path = run_dir / "stage2_artifacts" / "artifact_index.json"
    index = json.loads(index_path.read_text())
    if index.get("schema_version") != ARTIFACT_INDEX_SCHEMA:
        raise ValueError("unexpected Stage-2 artifact index")
    camera_order = index["camera_order"]
    camera_index = camera_order.index(args.camera)
    trace_paths = list(run_dir.rglob("control_trace.jsonl"))
    if len(trace_paths) != 1:
        raise ValueError("run must contain one control trace")
    trace_path = trace_paths[0]
    scenario_dir = trace_path.parent
    frame_dir = scenario_dir / CAMERA_FRAME_DIRS[args.camera]
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    rows_by_step = {int(row["step"]): row for row in rows}
    if args.full_route:
        start_time = float(rows[0]["sim_time_seconds"])
        end_time = float(rows[-1]["sim_time_seconds"])
        selection = "full_route"
    else:
        if args.center_time_seconds is not None:
            center = float(args.center_time_seconds)
            selection = "explicit_center"
        else:
            center, selection = _critical_time(rows)
        start_time = max(float(rows[0]["sim_time_seconds"]), center - args.pre_seconds)
        end_time = min(float(rows[-1]["sim_time_seconds"]), center + args.post_seconds)
    selected = []
    for record in index["records"]:
        step = int(record["step"])
        row = rows_by_step.get(step)
        frame_path = frame_dir / ("%04d.png" % (step // 10))
        if row is None or not frame_path.is_file():
            continue
        timestamp = float(row["sim_time_seconds"])
        if start_time <= timestamp <= end_time:
            selected.append((record, row, frame_path))
    if not selected:
        raise ValueError("no aligned UQ/camera frames in selected window")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Stage-2 visualization directory")
    output_dir.mkdir(parents=True)
    text_font = _font(18)
    frames = []
    frame_stats = []
    for record, row, frame_path in selected:
        observation = _tensor(record["observation_uq_path"])
        score = observation[camera_index].mean(axis=-1)
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((800, 450), Image.Resampling.LANCZOS)
        rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = np.rint(255.0 * (1.0 - np.clip(score, 0.0, 1.0))).astype(np.uint8)
        rgba[..., 3] = np.rint(210.0 * np.clip(score, 0.0, 1.0)).astype(np.uint8)
        heat = Image.fromarray(rgba, mode="RGBA").resize(image.size, Image.Resampling.BILINEAR)
        canvas = Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")
        q95 = float(np.quantile(score, 0.95))
        caption = (
            f"new Stage-1 {args.camera} | t={float(row['sim_time_seconds']):.2f}s "
            f"p={float(row['route_progress']):.3f} v={float(row['speed']):.2f}m/s "
            f"mean/q95/max={float(score.mean()):.3f}/{q95:.3f}/{float(score.max()):.3f}"
        )
        box = draw.textbbox((0, 0), caption, font=text_font)
        draw.rectangle((0, 0, canvas.width, box[3] + 12), fill=(0, 0, 0, 190))
        draw.text((7, 5), caption, font=text_font, fill=(255, 255, 255, 255))
        frames.append(canvas.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
        frame_stats.append({
            "step": int(record["step"]),
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "mean": float(score.mean()),
            "q95": q95,
            "max": float(score.max()),
        })
    gif_path = output_dir / (args.camera.lower() + "_spatial_uq.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, int(round(1000.0 / args.fps))),
        loop=0,
        disposal=2,
    )
    report = {
        "schema_version": "orion.stage2-spatial-uq-gif/v1",
        "claim_boundary": "Visualization of the new Stage-1 map, not task risk or a safety result.",
        "run_dir": str(run_dir),
        "artifact_index_sha256": _sha256(index_path),
        "trace_sha256": _sha256(trace_path),
        "camera": args.camera,
        "selection": selection,
        "time_range_seconds": [start_time, end_time],
        "frame_count": len(frames),
        "gif_path": str(gif_path),
        "gif_sha256": _sha256(gif_path),
        "frame_stats": frame_stats,
    }
    report_path = output_dir / "visualization_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
