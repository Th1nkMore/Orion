#!/usr/bin/env python3
"""Render a causally calibrated front-view observation-UQ heatmap GIF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = "orion.closedloop_observation_uq_heatmap.v1"
BASELINE_START_SECONDS = 1.0
BASELINE_END_SECONDS = 4.0
MINIMUM_BASELINE_FRAMES = 40
RELATIVE_SCALE_FLOOR = 0.05
ABSOLUTE_SCALE_FLOOR = 0.001
Z_CENTER = 4.0
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_FRAME_DIRS = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-seconds", type=float, default=6.0)
    parser.add_argument("--post-seconds", type=float, default=8.0)
    parser.add_argument("--full-route", action="store_true")
    parser.add_argument("--center-time-seconds", type=float)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--camera", choices=CAMERA_ORDER, default="CAM_FRONT")
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
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {root}, found {len(paths)}"
        )
    return paths[0]


def _load_rows(trace_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("control trace is empty")
    return rows


def _grid_from_row(
    row: dict[str, Any], camera_name: str
) -> np.ndarray | None:
    observation = row.get("observation_uq")
    if not isinstance(observation, dict):
        return None
    camera_order = tuple(observation.get("camera_order") or ())
    all_grids = observation.get("pooled_grids")
    if all_grids is not None and camera_name in camera_order:
        raw = all_grids[camera_order.index(camera_name)]
    elif camera_name == "CAM_FRONT":
        raw = observation.get("front_pooled_grid")
    else:
        raw = None
    if raw is None:
        return None
    grid = np.asarray(raw, dtype=np.float64)
    if grid.ndim != 2 or min(grid.shape) <= 0 or not np.isfinite(grid).all():
        raise RuntimeError("front_pooled_grid must be one finite 2-D array")
    return grid


def fit_spatial_baseline(
    grids: np.ndarray,
    *,
    relative_scale_floor: float = RELATIVE_SCALE_FLOOR,
    absolute_scale_floor: float = ABSOLUTE_SCALE_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a robust location/scale independently at every grid position."""

    values = np.asarray(grids, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("grids must be finite [frames,height,width]")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None]), axis=0)
    scale = np.maximum.reduce(
        (
            1.4826 * mad,
            relative_scale_floor * np.abs(median),
            np.full_like(median, absolute_scale_floor),
        )
    )
    return median, scale


def calibrate_spatial_grid(
    grid: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    *,
    z_center: float = Z_CENTER,
) -> np.ndarray:
    """Map position-normalized UQ evidence to [0,1] without future frames."""

    value = np.asarray(grid, dtype=np.float64)
    if value.shape != median.shape or value.shape != scale.shape:
        raise ValueError("grid and baseline shapes differ")
    z = (value - median) / scale - float(z_center)
    calibrated = np.empty_like(z)
    positive = z >= 0
    calibrated[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    calibrated[~positive] = exponential / (1.0 + exponential)
    return calibrated


def _critical_time(rows: list[dict[str, Any]]) -> float:
    candidates = []
    for row in rows:
        safety = row.get("closedloop_safety") or {}
        ttc = safety.get("min_obb_collision_ttc_seconds")
        if ttc is not None:
            candidates.append(
                (float(ttc), int(row["step"]), float(row["sim_time_seconds"]))
            )
    if not candidates:
        raise RuntimeError("trace has no finite OBB-TTC row for a critical window")
    return min(candidates)[2]


def _nearest_grid_row(
    rows_with_grids: list[tuple[dict[str, Any], np.ndarray]],
    target_step: int,
) -> tuple[dict[str, Any], np.ndarray]:
    return min(
        rows_with_grids,
        key=lambda item: abs(int(item[0]["step"]) - target_step),
    )


def _heatmap_overlay(image: Image.Image, calibrated: np.ndarray) -> Image.Image:
    score = np.clip(calibrated, 0.0, 1.0)
    rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.rint(255.0 * (1.0 - score)).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = np.rint(210.0 * score).astype(np.uint8)
    heat = Image.fromarray(rgba, mode="RGBA").resize(
        image.size, Image.Resampling.BILINEAR
    )
    return Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")


def _caption(
    row: dict[str, Any], calibrated: np.ndarray, camera_name: str
) -> str:
    safety = row.get("closedloop_safety") or {}
    ttc = safety.get("min_obb_collision_ttc_seconds")
    ttc_text = "--" if ttc is None else f"{float(ttc):.2f}s"
    observation = row.get("observation_uq") or {}
    filtered = (observation.get("calibration") or {}).get("filtered_score")
    filtered_text = "--" if filtered is None else f"{float(filtered):.2f}"
    return (
        f"{camera_name}  t={float(row['sim_time_seconds']):.2f}s  "
        f"p={float(row['route_progress']):.3f}  "
        f"v={float(row['speed']):.2f}m/s  TTC={ttc_text}  "
        f"UQ scalar={filtered_text}  spatial mean/max="
        f"{float(calibrated.mean()):.2f}/{float(calibrated.max()):.2f}"
    )


def resolve_time_window(
    rows: list[dict[str, Any]],
    *,
    first_time: float,
    last_time: float,
    pre_seconds: float,
    post_seconds: float,
    full_route: bool,
    center_time_seconds: float | None,
) -> tuple[float, float, str, float | None]:
    if full_route and center_time_seconds is not None:
        raise ValueError("full_route and center_time_seconds are mutually exclusive")
    if full_route:
        return first_time, last_time, "full_route", None
    if center_time_seconds is None:
        center = _critical_time(rows)
        basis = "minimum_finite_obb_ttc_window"
    else:
        center = float(center_time_seconds)
        if not math.isfinite(center):
            raise ValueError("center_time_seconds must be finite")
        basis = "explicit_preregistered_or_evaluator_event_time"
    return (
        max(first_time, center - float(pre_seconds)),
        min(last_time, center + float(post_seconds)),
        basis,
        center,
    )


def render(run_dir: Path, output_dir: Path, *, pre_seconds: float, post_seconds: float, full_route: bool, fps: float, camera_name: str = "CAM_FRONT", center_time_seconds: float | None = None) -> dict[str, Any]:
    if camera_name not in CAMERA_FRAME_DIRS:
        raise ValueError("unsupported camera name")
    trace_path = _find_one(run_dir, "records_*/**/control_trace.jsonl")
    scenario_dir = trace_path.parent
    frame_dir = scenario_dir / CAMERA_FRAME_DIRS[camera_name]
    if not frame_dir.is_dir():
        raise RuntimeError("run does not contain saved front-view frames")
    rows = _load_rows(trace_path)
    rows_with_grids = [
        (row, grid)
        for row in rows
        if (grid := _grid_from_row(row, camera_name)) is not None
    ]
    if not rows_with_grids:
        raise RuntimeError("trace contains no front_pooled_grid records")
    grid_shapes = {grid.shape for _, grid in rows_with_grids}
    if len(grid_shapes) != 1:
        raise RuntimeError("front_pooled_grid shape changes within the trace")

    baseline = np.stack(
        [
            grid
            for row, grid in rows_with_grids
            if BASELINE_START_SECONDS
            <= float(row["sim_time_seconds"])
            < BASELINE_END_SECONDS
        ]
    )
    if baseline.shape[0] < MINIMUM_BASELINE_FRAMES:
        raise RuntimeError(
            f"insufficient heatmap baseline frames: {baseline.shape[0]} "
            f"< {MINIMUM_BASELINE_FRAMES}"
        )
    median, scale = fit_spatial_baseline(baseline)
    first_time = float(rows_with_grids[0][0]["sim_time_seconds"])
    last_time = float(rows_with_grids[-1][0]["sim_time_seconds"])
    start_time, end_time, basis, center = resolve_time_window(
        rows,
        first_time=first_time,
        last_time=last_time,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        full_route=full_route,
        center_time_seconds=center_time_seconds,
    )

    selected = []
    for path in sorted(frame_dir.glob("*.png"), key=lambda item: int(item.stem)):
        row, grid = _nearest_grid_row(rows_with_grids, int(path.stem) * 10)
        current_time = float(row["sim_time_seconds"])
        if start_time <= current_time <= end_time:
            selected.append((path, row, grid))
    if not selected:
        raise RuntimeError("no saved front frames fall in the requested window")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera_name.lower()}_observation_uq_overlay.gif"
    text_font = _font(18)
    frames = []
    spatial_scores = []
    for path, row, grid in selected:
        image = Image.open(path).convert("RGB")
        image.thumbnail((720, 405), Image.Resampling.LANCZOS)
        calibrated = calibrate_spatial_grid(grid, median, scale)
        spatial_scores.append(
            {"mean": float(calibrated.mean()), "max": float(calibrated.max())}
        )
        rendered = _heatmap_overlay(image, calibrated)
        draw = ImageDraw.Draw(rendered, "RGBA")
        caption = _caption(row, calibrated, camera_name)
        box = draw.textbbox((0, 0), caption, font=text_font)
        draw.rectangle((0, 0, rendered.width, box[3] + 12), fill=(0, 0, 0, 190))
        draw.text((7, 5), caption, font=text_font, fill=(255, 255, 255, 255))
        frames.append(
            rendered.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
        )
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, int(round(1000.0 / float(fps)))),
        loop=0,
        disposal=2,
    )
    report = {
        "schema": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "selection_basis": basis,
        "center_time_seconds": center,
        "time_range_seconds": [start_time, end_time],
        "frame_count": len(frames),
        "fps": float(fps),
        "camera": camera_name,
        "grid_shape": list(median.shape),
        "baseline": {
            "start_seconds": BASELINE_START_SECONDS,
            "end_seconds": BASELINE_END_SECONDS,
            "frame_count": int(baseline.shape[0]),
            "relative_scale_floor": RELATIVE_SCALE_FLOOR,
            "absolute_scale_floor": ABSOLUTE_SCALE_FLOOR,
            "z_center": Z_CENTER,
        },
        "rendered_spatial_score_range": {
            "minimum_frame_mean": min(item["mean"] for item in spatial_scores),
            "maximum_frame_mean": max(item["mean"] for item in spatial_scores),
            "maximum_cell_score": max(item["max"] for item in spatial_scores),
        },
        "claim_boundary": (
            "The overlay visualizes task-agnostic observation-evidence uplift "
            "relative to a causal 1-4 second per-position baseline. It does not "
            "establish that a highlighted region caused a planning error or that "
            "the adapter understands driving risk."
        ),
    }
    manifest_path = output_dir / "observation_uq_heatmap_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    if args.pre_seconds < 0 or args.post_seconds < 0 or args.fps <= 0:
        raise ValueError("pre/post seconds must be non-negative and fps positive")
    report = render(
        args.run_dir,
        args.output_dir,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        full_route=args.full_route,
        fps=args.fps,
        camera_name=args.camera,
        center_time_seconds=args.center_time_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
