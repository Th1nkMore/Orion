#!/usr/bin/env python3
"""Render front-camera and ground-plane BEV views of frozen Stage-1 U.

The BEV visualization is an explicit geometric projection of camera-grid U
onto a flat road plane.  It is not a learned BEV uncertainty map and does not
infer object depth.  Camera cells above the horizon are therefore omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = "orion.stage1_uq_dual_view_visualization.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
MODEL_SIZE = 640.0
RAW_WIDTH = 1600.0
RAW_HEIGHT = 900.0
RAW_CROP_TOP = 100.0
BEV_SIZE = 512
BEV_FOCAL = 548.993771650447
BEV_CAMERA_HEIGHT = 50.0
BEV_PIXELS_PER_METER = BEV_FOCAL / BEV_CAMERA_HEIGHT

CAMERA_SPECS = {
    "CAM_FRONT": {"x": 0.80, "y": 0.0, "z": 1.60, "yaw": 0.0, "fov": 70.0},
    "CAM_FRONT_LEFT": {"x": 0.27, "y": -0.55, "z": 1.60, "yaw": -55.0, "fov": 70.0},
    "CAM_FRONT_RIGHT": {"x": 0.27, "y": 0.55, "z": 1.60, "yaw": 55.0, "fov": 70.0},
    "CAM_BACK": {"x": -2.0, "y": 0.0, "z": 1.60, "yaw": 180.0, "fov": 110.0},
    "CAM_BACK_LEFT": {"x": -0.32, "y": -0.55, "z": 1.60, "yaw": -110.0, "fov": 70.0},
    "CAM_BACK_RIGHT": {"x": -0.32, "y": 0.55, "z": 1.60, "yaw": 110.0, "fov": 70.0},
}


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


def _scenario_dir(run_dir: Path) -> Path:
    traces = sorted(run_dir.glob("records_*/**/control_trace.jsonl"))
    if len(traces) != 1:
        raise ValueError("run must contain exactly one control trace")
    return traces[0].parent


def _load_trace_rows(scenario_dir: Path) -> dict[int, dict[str, Any]]:
    trace_path = scenario_dir / "control_trace.jsonl"
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["step"])
        frame = step // 10
        previous = by_frame.get(frame)
        # Saved camera file N is written at control step 10*N.  Select that
        # exact row when present, otherwise the nearest row in the bin; never
        # silently attach the bin's final state to its first saved image.
        if previous is None or abs(step - 10 * frame) < abs(
            int(previous["step"]) - 10 * frame
        ):
            by_frame[frame] = row
    return by_frame


def _front_frame_root(scenario_dir: Path) -> Path:
    model_input = scenario_dir / "rgb_front_model_input"
    if model_input.is_dir() and any(model_input.glob("*.png")):
        return model_input
    return scenario_dir / "rgb_front"


def _color_overlay(base: Image.Image, score: np.ndarray, alpha_scale: float = 0.82) -> Image.Image:
    value = np.clip(score, 0.0, 1.0)
    rgba = np.zeros((*value.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.rint(255.0 * (1.0 - value)).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = np.rint(255.0 * alpha_scale * value).astype(np.uint8)
    heat = Image.fromarray(rgba).resize(base.size, Image.Resampling.BILINEAR)
    return Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB")


def _front_overlay(image: Image.Image, score: np.ndarray) -> Image.Image:
    raw = image.convert("RGB")
    crop = raw.crop((0, int(RAW_CROP_TOP), int(RAW_WIDTH), int(RAW_HEIGHT)))
    overlaid = _color_overlay(crop, score)
    canvas = raw.copy()
    canvas.paste(overlaid, (0, int(RAW_CROP_TOP)))
    return canvas


def _raw_pixel_from_grid(grid_x: float, grid_y: float, width: int, height: int) -> tuple[float, float]:
    model_u = MODEL_SIZE * grid_x / float(width)
    model_v = MODEL_SIZE * grid_y / float(height)
    raw_u = RAW_WIDTH * model_u / MODEL_SIZE
    raw_v = RAW_CROP_TOP + (RAW_HEIGHT - RAW_CROP_TOP) * model_v / MODEL_SIZE
    return raw_u, raw_v


def _ground_point(camera: str, raw_u: float, raw_v: float) -> tuple[float, float] | None:
    spec = CAMERA_SPECS[camera]
    focal = RAW_WIDTH / (2.0 * math.tan(math.radians(spec["fov"]) / 2.0))
    ray_y = (raw_u - RAW_WIDTH / 2.0) / focal
    ray_z = -(raw_v - RAW_HEIGHT / 2.0) / focal
    if ray_z >= -1e-6:
        return None
    distance_scale = spec["z"] / -ray_z
    yaw = math.radians(spec["yaw"])
    direction_x = math.cos(yaw) - math.sin(yaw) * ray_y
    direction_y = math.sin(yaw) + math.cos(yaw) * ray_y
    return (
        spec["x"] + distance_scale * direction_x,
        spec["y"] + distance_scale * direction_y,
    )


def _bev_pixel(point: tuple[float, float]) -> tuple[float, float]:
    forward, right = point
    return (
        BEV_SIZE / 2.0 + right * BEV_PIXELS_PER_METER,
        BEV_SIZE / 2.0 - forward * BEV_PIXELS_PER_METER,
    )


def project_camera_u_to_ground(uncertainty: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Project six image grids onto the z=0 plane using frozen sensor geometry."""

    values = np.asarray(uncertainty, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(CAMERA_ORDER):
        raise ValueError("uncertainty must have shape [6,H,W]")
    height, width = values.shape[-2:]
    combined = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    projected_cells = 0
    skipped_above_horizon = 0
    for view_index, camera in enumerate(CAMERA_ORDER):
        view_image = Image.new("F", (BEV_SIZE, BEV_SIZE), 0.0)
        view_draw = ImageDraw.Draw(view_image)
        for row in range(height):
            for column in range(width):
                corners = []
                for grid_x, grid_y in (
                    (column, row),
                    (column + 1, row),
                    (column + 1, row + 1),
                    (column, row + 1),
                ):
                    raw_u, raw_v = _raw_pixel_from_grid(
                        grid_x, grid_y, width, height
                    )
                    point = _ground_point(camera, raw_u, raw_v)
                    if point is None:
                        corners = []
                        break
                    corners.append(_bev_pixel(point))
                if not corners:
                    skipped_above_horizon += 1
                    continue
                polygon = np.clip(
                    np.rint(np.asarray(corners)), -100000, 100000
                ).astype(np.int32)
                view_draw.polygon(
                    [tuple(map(int, point)) for point in polygon],
                    fill=float(values[view_index, row, column]),
                )
                projected_cells += 1
        view_map = np.asarray(view_image, dtype=np.float32)
        combined = np.maximum(combined, view_map)
    stats = {
        "projected_cell_count": projected_cells,
        "above_horizon_cell_count": skipped_above_horizon,
        "nonzero_pixel_fraction": float(np.count_nonzero(combined) / combined.size),
    }
    return combined, stats


def _annotate(
    image: Image.Image,
    lines: list[str],
    *,
    target_width: int,
) -> Image.Image:
    canvas = image.convert("RGB")
    if canvas.width != target_width:
        target_height = int(round(canvas.height * target_width / canvas.width))
        canvas = canvas.resize((target_width, target_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _font(17)
    line_height = 22
    bar_height = 8 + line_height * len(lines)
    draw.rectangle((0, 0, canvas.width, bar_height), fill=(0, 0, 0, 190))
    for index, line in enumerate(lines):
        draw.text((8, 5 + index * line_height), line, font=font, fill=(255, 255, 255, 255))
    # Compact calibrated-U legend.
    legend_width = 130
    legend_height = 10
    left = canvas.width - legend_width - 12
    top = bar_height + 8
    gradient = np.linspace(0.0, 1.0, legend_width, dtype=np.float32)[None]
    gradient = np.repeat(gradient, legend_height, axis=0)
    legend = Image.new("RGB", (legend_width, legend_height), (0, 0, 0))
    legend = _color_overlay(legend, gradient, alpha_scale=1.0)
    canvas.paste(legend, (left, top))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((left, top + 12), "low U             high U", font=_font(12), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return canvas


def _quantize(frames: list[Image.Image]) -> list[Image.Image]:
    return [
        frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
        for frame in frames
    ]


def render(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    replay_dir = args.replay_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    manifest_path = replay_dir / "manifest.json"
    replay_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if replay_manifest.get("schema") != "orion.stage1_uq_visualization_replay.v1":
        raise ValueError("unexpected replay manifest schema")
    payload_path = replay_dir / "stage1_uq_visualization_replay.npz"
    payload = np.load(payload_path)
    frame_indices = payload["frame_indices"].astype(int).tolist()
    uncertainty = np.asarray(payload["uncertainty"], dtype=np.float32)
    if uncertainty.shape[0] != len(frame_indices):
        raise ValueError("replay frame/U count mismatch")

    scenario_dir = _scenario_dir(run_dir)
    front_root = _front_frame_root(scenario_dir)
    bev_root = scenario_dir / "bev"
    trace_rows = _load_trace_rows(scenario_dir)
    selected_positions = [
        index
        for index, frame in enumerate(frame_indices)
        if args.start_frame <= frame <= args.end_frame
        and (front_root / f"{frame:04d}.png").is_file()
        and (bev_root / f"{frame:04d}.png").is_file()
    ]
    if not selected_positions:
        raise ValueError("selected window has no aligned replay/camera/BEV frames")

    output_dir.mkdir(parents=True)
    front_frames = []
    bev_frames = []
    frame_stats = []
    for position in selected_positions:
        frame = frame_indices[position]
        u = uncertainty[position]
        front_u = u[CAMERA_ORDER.index("CAM_FRONT")]
        row = trace_rows.get(frame, {})
        sim_time = float(row.get("sim_time_seconds", frame / 2.0))
        corruption_active = bool(row.get("corruption_active", False))

        front_image = Image.open(front_root / f"{frame:04d}.png").convert("RGB")
        front_canvas = _front_overlay(front_image, front_u)
        front_canvas = _annotate(
            front_canvas,
            [
                f"{args.label} | FRONT spatial U | t={sim_time:.2f}s | corruption={'ON' if corruption_active else 'off'}",
                f"calibrated mean/q95/max={float(front_u.mean()):.3f}/{float(np.quantile(front_u, 0.95)):.3f}/{float(front_u.max()):.3f}",
            ],
            target_width=args.output_width,
        )
        front_frames.append(front_canvas)

        bev_u, projection_stats = project_camera_u_to_ground(u)
        bev_image = Image.open(bev_root / f"{frame:04d}.png").convert("RGB")
        if bev_image.size != (BEV_SIZE, BEV_SIZE):
            bev_image = bev_image.resize((BEV_SIZE, BEV_SIZE), Image.Resampling.LANCZOS)
        bev_canvas = _color_overlay(bev_image, bev_u)
        bev_canvas = _annotate(
            bev_canvas,
            [
                f"{args.label} | 6-view U projected to road plane | t={sim_time:.2f}s",
                f"projected mean/q95/max={float(bev_u.mean()):.3f}/{float(np.quantile(bev_u, 0.95)):.3f}/{float(bev_u.max()):.3f}",
                "diagnostic flat-ground projection; not learned BEV U / not object depth",
            ],
            target_width=args.output_width,
        )
        bev_frames.append(bev_canvas)
        frame_stats.append(
            {
                "saved_frame_index": frame,
                "sim_time_seconds": sim_time,
                "corruption_active": corruption_active,
                "front_mean": float(front_u.mean()),
                "front_q95": float(np.quantile(front_u, 0.95)),
                "front_max": float(front_u.max()),
                "bev_projected_mean": float(bev_u.mean()),
                "bev_projected_q95": float(np.quantile(bev_u, 0.95)),
                "bev_projected_max": float(bev_u.max()),
                **projection_stats,
            }
        )

    duration = max(20, int(round(1000.0 / args.fps)))
    front_path = output_dir / "front_spatial_u_heatmap.gif"
    bev_path = output_dir / "bev_ground_projected_u_heatmap.gif"
    quantized_front = _quantize(front_frames)
    quantized_bev = _quantize(bev_frames)
    quantized_front[0].save(
        front_path,
        save_all=True,
        append_images=quantized_front[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    quantized_bev[0].save(
        bev_path,
        save_all=True,
        append_images=quantized_bev[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    report = {
        "schema": SCHEMA,
        "label": args.label,
        "claim_boundary": (
            "Front heatmap is calibrated task-agnostic observation evidence. "
            "BEV is a six-camera flat-ground geometric projection, not learned "
            "BEV uncertainty, depth, task risk, or a safety result."
        ),
        "run_dir": str(run_dir),
        "replay_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "replay_payload": {"path": str(payload_path), "sha256": _sha256(payload_path)},
        "selection": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "selected_frame_count": len(selected_positions),
            "fps": args.fps,
        },
        "front": {"path": str(front_path), "sha256": _sha256(front_path)},
        "bev": {
            "path": str(bev_path),
            "sha256": _sha256(bev_path),
            "projection": {
                "surface": "ego-frame z=0 flat ground",
                "camera_geometry_source": "team_code/orion_b2d_agent.py frozen sensor specifications",
                "above_horizon_cells_omitted": True,
                "depth_sensor_used": False,
            },
        },
        "frame_stats": frame_stats,
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--output-width", type=int, default=800)
    args = parser.parse_args()
    if args.start_frame > args.end_frame or args.fps <= 0 or args.output_width <= 0:
        raise SystemExit("invalid rendering interval, fps, or output width")
    report = render(args)
    print(
        json.dumps(
            {
                "front": report["front"]["path"],
                "bev": report["bev"]["path"],
                "frame_count": report["selection"]["selected_frame_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
