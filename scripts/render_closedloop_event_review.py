#!/usr/bin/env python3
"""Render a clean-trace front/BEV contact sheet for event preregistration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-min", type=float, default=0.0)
    parser.add_argument("--progress-max", type=float, default=0.35)
    parser.add_argument("--max-panels", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--auto-critical-window",
        action="store_true",
        help="center the review on the trace-wide minimum finite OBB-TTC",
    )
    parser.add_argument("--pre-seconds", type=float, default=3.0)
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument(
        "--on-path-region",
        type=float,
        nargs=4,
        metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"),
    )
    parser.add_argument(
        "--off-path-region",
        type=float,
        nargs=4,
        metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"),
    )
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def load_font(size: int) -> ImageFont.ImageFont:
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


def nearest_row(rows_by_step: dict[int, dict[str, Any]], frame: int) -> dict[str, Any]:
    target_step = frame * 10
    step = min(rows_by_step, key=lambda value: abs(value - target_step))
    return rows_by_step[step]


def sample_evenly(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def _actor_sort_key(actor: dict[str, Any]) -> tuple[int, float]:
    ttc = actor.get("obb_collision_ttc_seconds")
    if ttc is not None and math.isfinite(float(ttc)):
        return 0, float(ttc)
    gap = actor.get("obb_separating_axis_gap_m")
    if gap is not None and math.isfinite(float(gap)):
        return 1, float(gap)
    return 2, math.inf


def critical_actor_for_row(row: dict[str, Any]) -> dict[str, Any] | None:
    actors = row.get("closedloop_safety", {}).get("actors", [])
    candidates = [actor for actor in actors if _actor_sort_key(actor)[0] < 2]
    return min(candidates, key=_actor_sort_key) if candidates else None


def critical_trace_event(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    for row in rows:
        for actor in row.get("closedloop_safety", {}).get("actors", []):
            ttc = actor.get("obb_collision_ttc_seconds")
            if ttc is not None and math.isfinite(float(ttc)):
                candidates.append((float(ttc), int(row["step"]), row, actor))
    if not candidates:
        raise RuntimeError("trace contains no finite OBB collision TTC")
    _, _, row, actor = min(candidates, key=lambda item: (item[0], item[1]))
    return row, actor


def safety_label(row: dict[str, Any]) -> str:
    actor = critical_actor_for_row(row)
    if actor is None:
        return "OBB-TTC=--  gap=--  actor=none"
    category = actor.get("category", "actor")
    actor_id = actor.get("actor_id", "?")
    ttc = actor.get("obb_collision_ttc_seconds")
    gap = actor.get("obb_separating_axis_gap_m")
    ttc_text = "--" if ttc is None else f"{float(ttc):.2f}s"
    gap_text = "--" if gap is None else f"{float(gap):.2f}m"
    return f"OBB-TTC={ttc_text}  gap={gap_text}  actor={category}#{actor_id}"


def validate_region(values: list[float] | None, name: str):
    if values is None:
        return None
    top, left, bottom, right = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (top, left, bottom, right)):
        raise ValueError(f"{name} coordinates must be finite")
    if not (0.0 <= top < bottom <= 1.0 and 0.0 <= left < right <= 1.0):
        raise ValueError(f"{name} must be normalized and non-empty")
    return top, left, bottom, right


def draw_region_overlay(
    image: Image.Image,
    region: tuple[float, float, float, float] | None,
    *,
    color: str,
    label: str,
) -> None:
    if region is None:
        return
    top, left, bottom, right = region
    box = (
        round(left * image.width),
        round(top * image.height),
        round(right * image.width),
        round(bottom * image.height),
    )
    draw = ImageDraw.Draw(image)
    width = max(2, round(image.width / 300))
    draw.rectangle(box, outline=color, width=width)
    font = load_font(max(12, round(image.width / 38)))
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = min(max(0, box[0] + 4), max(0, image.width - text_width - 6))
    text_y = min(max(0, box[1] + 4), max(0, image.height - text_height - 6))
    draw.rectangle(
        (text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2),
        fill="#111111",
    )
    draw.text((text_x, text_y), label, fill=color, font=font)


def make_panel(
    front_path: Path,
    bev_path: Path,
    row: dict[str, Any],
    *,
    panel_width: int = 720,
    on_path_region: tuple[float, float, float, float] | None = None,
    off_path_region: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    front = Image.open(front_path).convert("RGB")
    bev = Image.open(bev_path).convert("RGB")
    image_height = 300
    front.thumbnail((panel_width * 2 // 3, image_height), Image.Resampling.LANCZOS)
    bev.thumbnail((panel_width // 3, image_height), Image.Resampling.LANCZOS)
    draw_region_overlay(front, on_path_region, color="#ff5d5d", label="ON")
    draw_region_overlay(front, off_path_region, color="#42d9ff", label="OFF")
    header_height = 72
    panel = Image.new("RGB", (panel_width, image_height + header_height), "#111111")
    front_x = max(0, (panel_width * 2 // 3 - front.width) // 2)
    bev_x = panel_width * 2 // 3 + max(0, (panel_width // 3 - bev.width) // 2)
    panel.paste(front, (front_x, header_height))
    panel.paste(bev, (bev_x, header_height))
    draw = ImageDraw.Draw(panel)
    font = load_font(18)
    primary_label = (
        f"frame={front_path.stem}  t={float(row['sim_time_seconds']):.2f}s  "
        f"progress={float(row['route_progress']):.4f}  "
        f"speed={float(row['speed']):.2f}m/s"
    )
    draw.text((10, 7), primary_label, fill="white", font=font)
    draw.text((10, 35), safety_label(row), fill="#ffd166", font=font)
    draw.line(
        (panel_width * 2 // 3, header_height, panel_width * 2 // 3, panel.height),
        fill="#e4b84c",
        width=2,
    )
    return panel


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.progress_min < args.progress_max <= 1.0):
        raise ValueError("require 0 <= progress-min < progress-max <= 1")
    if args.columns <= 0 or args.max_panels <= 0:
        raise ValueError("columns and max-panels must be positive")
    if args.pre_seconds < 0.0 or args.post_seconds < 0.0:
        raise ValueError("pre-seconds and post-seconds must be non-negative")
    on_path_region = validate_region(args.on_path_region, "on-path region")
    off_path_region = validate_region(args.off_path_region, "off-path region")
    if (on_path_region is None) != (off_path_region is None):
        raise ValueError("on-path and off-path regions must be provided together")

    trace_path = find_one(args.run_dir, "records_*/**/control_trace.jsonl")
    scenario_dir = trace_path.parent
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_step = {int(row["step"]): row for row in rows}
    front_paths = sorted(
        (scenario_dir / "rgb_front").glob("*.png"), key=lambda path: int(path.stem)
    )
    critical_row = None
    critical_actor = None
    if args.auto_critical_window:
        critical_row, critical_actor = critical_trace_event(rows)
        critical_time = float(critical_row["sim_time_seconds"])
        time_min = critical_time - args.pre_seconds
        time_max = critical_time + args.post_seconds
    candidates = []
    for front_path in front_paths:
        row = nearest_row(rows_by_step, int(front_path.stem))
        progress = float(row["route_progress"])
        bev_path = scenario_dir / "bev" / front_path.name
        in_window = (
            time_min <= float(row["sim_time_seconds"]) <= time_max
            if args.auto_critical_window
            else args.progress_min <= progress <= args.progress_max
        )
        if in_window and bev_path.is_file():
            candidates.append((front_path, bev_path, row))
    selected = sample_evenly(candidates, args.max_panels)
    if not selected:
        raise RuntimeError("no saved front/BEV frame lies in the requested progress range")

    panels = [
        make_panel(
            front,
            bev,
            row,
            on_path_region=on_path_region,
            off_path_region=off_path_region,
        )
        for front, bev, row in selected
    ]
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    rows_count = math.ceil(len(panels) / args.columns)
    sheet = Image.new(
        "RGB", (cell_width * args.columns, cell_height * rows_count), "#202020"
    )
    for index, panel in enumerate(panels):
        x = (index % args.columns) * cell_width
        y = (index // args.columns) * cell_height
        sheet.paste(panel, (x, y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "clean_event_front_bev_contact_sheet.png"
    sheet.save(sheet_path)
    manifest = {
        "schema": "orion.closedloop_clean_event_review.v1",
        "run_dir": str(args.run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "selection_basis": (
            "minimum_finite_obb_ttc_window"
            if args.auto_critical_window
            else "route_progress_window"
        ),
        "progress_range": (
            None if args.auto_critical_window else [args.progress_min, args.progress_max]
        ),
        "time_range_seconds": (
            [time_min, time_max] if args.auto_critical_window else None
        ),
        "critical_event": (
            {
                "step": int(critical_row["step"]),
                "sim_time_seconds": float(critical_row["sim_time_seconds"]),
                "route_progress": float(critical_row["route_progress"]),
                "speed_mps": float(critical_row["speed"]),
                "actor_id": critical_actor.get("actor_id"),
                "actor_category": critical_actor.get("category"),
                "actor_type_id": critical_actor.get("type_id"),
                "obb_ttc_seconds": float(
                    critical_actor["obb_collision_ttc_seconds"]
                ),
                "obb_separating_axis_gap_m": critical_actor.get(
                    "obb_separating_axis_gap_m"
                ),
            }
            if critical_row is not None and critical_actor is not None
            else None
        ),
        "proposed_regions": (
            {
                "on_path": list(on_path_region),
                "off_path": list(off_path_region),
            }
            if on_path_region is not None and off_path_region is not None
            else None
        ),
        "candidate_frame_count": len(candidates),
        "selected": [
            {
                "front": str(front.resolve()),
                "bev": str(bev.resolve()),
                "step": int(row["step"]),
                "sim_time_seconds": float(row["sim_time_seconds"]),
                "route_progress": float(row["route_progress"]),
                "speed": float(row["speed"]),
            }
            for front, bev, row in selected
        ],
        "contact_sheet": str(sheet_path.resolve()),
    }
    manifest_path = args.output_dir / "clean_event_review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
