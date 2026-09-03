#!/usr/bin/env python3
"""Render auditable front/model-input/BEV GIFs from one closed-loop run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-seconds", type=float, default=2.0)
    parser.add_argument("--post-seconds", type=float, default=3.0)
    parser.add_argument("--full-route", action="store_true")
    parser.add_argument("--auto-critical-window", action="store_true")
    parser.add_argument("--center-time-seconds", type=float)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def load_rows(trace_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("control trace is empty")
    return rows


def font(size: int) -> ImageFont.ImageFont:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_row(rows_by_step: dict[int, dict[str, Any]], frame_index: int):
    target_step = frame_index * 10
    nearest_step = min(rows_by_step, key=lambda step: abs(step - target_step))
    return rows_by_step[nearest_step]


def critical_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        safety = row.get("closedloop_safety") or {}
        value = safety.get("min_obb_collision_ttc_seconds")
        if value is not None:
            candidates.append((float(value), int(row["step"]), row))
    return min(candidates, default=(None, None, None))[-1]


def choose_time_window(
    rows: list[dict[str, Any]],
    *,
    pre_seconds: float,
    post_seconds: float,
    full_route: bool,
    auto_critical_window: bool,
    center_time_seconds: float | None = None,
) -> tuple[float, float, str]:
    first_time = float(rows[0]["sim_time_seconds"])
    last_time = float(rows[-1]["sim_time_seconds"])
    if full_route:
        return first_time, last_time, "full_route"
    if center_time_seconds is not None:
        center = float(center_time_seconds)
        if not first_time <= center <= last_time:
            raise RuntimeError("explicit center time falls outside the trace")
        return (
            max(first_time, center - pre_seconds),
            min(last_time, center + post_seconds),
            "explicit_center_time",
        )
    active = [row for row in rows if row.get("corruption_active")]
    if active:
        return (
            max(first_time, float(active[0]["sim_time_seconds"]) - pre_seconds),
            min(last_time, float(active[-1]["sim_time_seconds"]) + post_seconds),
            "corruption_event_window",
        )
    if auto_critical_window:
        row = critical_row(rows)
        if row is None:
            raise RuntimeError("trace has no finite OBB-TTC critical row")
        center = float(row["sim_time_seconds"])
        return (
            max(first_time, center - pre_seconds),
            min(last_time, center + post_seconds),
            "minimum_finite_obb_ttc_window",
        )
    return first_time, last_time, "full_route_no_corruption"


def metric_text(row: dict[str, Any]) -> str:
    safety = row.get("closedloop_safety") or {}
    actor = safety.get("critical_actor") or {}
    ttc = safety.get("min_obb_collision_ttc_seconds")
    gap = safety.get("min_obb_separating_axis_gap_m")
    ttc_text = "--" if ttc is None else f"{float(ttc):.2f}s"
    gap_text = "--" if gap is None else f"{float(gap):.2f}m"
    actor_text = "--"
    if actor:
        actor_text = f"{actor.get('category', '?')}#{actor.get('actor_id', '?')}"
    state = "CORRUPT" if row.get("corruption_active") else "clean"
    return (
        f"t={float(row['sim_time_seconds']):.2f}s  "
        f"p={float(row['route_progress']):.3f}  "
        f"v={float(row['speed']):.2f}m/s  {state}  "
        f"TTC={ttc_text}  gap={gap_text}  actor={actor_text}"
    )


def render_channel(
    frame_dir: Path,
    output_path: Path,
    *,
    label: str,
    rows_by_step: dict[int, dict[str, Any]],
    start_time: float,
    end_time: float,
    fps: float,
    max_size: tuple[int, int],
) -> dict[str, Any]:
    selected = []
    for path in sorted(frame_dir.glob("*.png"), key=lambda item: int(item.stem)):
        row = nearest_row(rows_by_step, int(path.stem))
        time_value = float(row["sim_time_seconds"])
        if start_time <= time_value <= end_time:
            selected.append((path, row))
    if not selected:
        raise RuntimeError(f"no frames selected from {frame_dir}")
    text_font = font(18)
    frames = []
    for path, row in selected:
        image = Image.open(path).convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image, "RGBA")
        caption = f"{label} | {metric_text(row)}"
        box = draw.textbbox((0, 0), caption, font=text_font)
        draw.rectangle((0, 0, image.width, box[3] + 12), fill=(0, 0, 0, 185))
        draw.text((7, 5), caption, font=text_font, fill=(255, 255, 255, 255))
        frames.append(image.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
    duration_ms = max(20, int(round(1000.0 / fps)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return {
        "output": str(output_path.resolve()),
        "sha256": sha256(output_path),
        "frame_count": len(frames),
        "first_source_frame": int(selected[0][0].stem),
        "last_source_frame": int(selected[-1][0].stem),
    }


def main() -> int:
    args = parse_args()
    if args.pre_seconds < 0 or args.post_seconds < 0:
        raise ValueError("pre/post seconds must be non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    trace_path = find_one(args.run_dir, "records_*/**/control_trace.jsonl")
    scenario_dir = trace_path.parent
    rows = load_rows(trace_path)
    rows_by_step = {int(row["step"]): row for row in rows}
    start_time, end_time, basis = choose_time_window(
        rows,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        full_route=args.full_route,
        auto_critical_window=args.auto_critical_window,
        center_time_seconds=args.center_time_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    channels = {}
    specifications = (
        ("raw_front", "rgb_front", "raw front", (720, 405)),
        (
            "model_input_front",
            "rgb_front_model_input",
            "legacy reconstructed ORION preview",
            (720, 405),
        ),
        (
            "exact_model_tensor_front",
            "rgb_front_model_tensor",
            "exact ORION 640x640 front tensor",
            (640, 640),
        ),
        ("bev", "bev", "external BEV", (512, 512)),
    )
    for key, directory, label, max_size in specifications:
        frame_dir = scenario_dir / directory
        if not frame_dir.is_dir():
            continue
        channels[key] = render_channel(
            frame_dir,
            args.output_dir / f"{key}.gif",
            label=label,
            rows_by_step=rows_by_step,
            start_time=start_time,
            end_time=end_time,
            fps=args.fps,
            max_size=max_size,
        )
    if not channels:
        raise RuntimeError("run has no supported front or BEV frame directories")
    report = {
        "schema": "orion.closedloop_front_bev_gifs.v1",
        "run_dir": str(args.run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "selection_basis": basis,
        "time_range_seconds": [start_time, end_time],
        "fps": args.fps,
        "channels": channels,
    }
    manifest = args.output_dir / "visualization_manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
