#!/usr/bin/env python3
"""Render auditable front/BEV GIFs and traces for one learned closed-loop run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    return parser.parse_args()


def _find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {pattern} below {root}")
    return paths[0]


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _nearest_trace_row(
    rows_by_step: dict[int, dict[str, Any]], frame_index: int
) -> dict[str, Any]:
    target = frame_index * 10
    return rows_by_step[min(rows_by_step, key=lambda step: abs(step - target))]


def _make_gif(
    frame_dir: Path,
    destination: Path,
    *,
    label: str,
    rows_by_step: dict[int, dict[str, Any]],
    max_size: tuple[int, int],
) -> None:
    paths = sorted(frame_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError(f"no PNG frames in {frame_dir}")
    font = _font(18)
    frames = []
    for path in paths:
        row = _nearest_trace_row(rows_by_step, int(path.stem))
        image = Image.open(path).convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image, "RGBA")
        corruption = "DROP" if row["corruption_active"] else "valid"
        text = (
            f"{label} | t={float(row['sim_time_seconds']):.2f}s | {corruption} | "
            f"UQ={float(row['raw_uq_score']):.3f} | "
            f"gov={float(row['risk']['intensity']):.2f} | "
            f"v={float(row['speed']):.2f}m/s"
        )
        box = draw.textbbox((0, 0), text, font=font)
        draw.rectangle((0, 0, image.width, box[3] + 12), fill=(0, 0, 0, 178))
        draw.text((7, 5), text, font=font, fill=(255, 255, 255, 255))
        frames.append(image.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        disposal=2,
    )


def _event_indices(rows: list[dict[str, Any]]) -> tuple[int, int]:
    indices = [index for index, row in enumerate(rows) if row["corruption_active"]]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError("trace does not contain one contiguous corruption event")
    return indices[0], indices[-1]


def _render_trace_plot(
    rows: list[dict[str, Any]], destination: Path, threshold: float
) -> None:
    first, last = _event_indices(rows)
    times = [float(row["sim_time_seconds"]) for row in rows]
    scores = [float(row["raw_uq_score"]) for row in rows]
    intensities = [float(row["risk"]["intensity"]) for row in rows]
    speeds = [float(row["speed"]) for row in rows]
    event_start = times[first]
    event_end = times[last] + statistics.median(
        right - left for left, right in zip(times, times[1:]) if right > left
    )
    canvas = Image.new("RGB", (1200, 840), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = _font(27)
    label_font = _font(19)
    tick_font = _font(15)
    draw.text(
        (70, 22),
        "Route146 learned operational-trigger controlled stop",
        font=title_font,
        fill="#111111",
    )
    series = (
        (scores, "learned observation score", threshold, (117, 73, 168), (0.0, 1.0)),
        (intensities, "governor intensity", None, (207, 91, 62), (0.0, 1.0)),
        (
            speeds,
            "ego speed (m/s)",
            0.25,
            (35, 116, 171),
            (min(0.0, min(speeds)), max(5.0, max(speeds))),
        ),
    )
    left, right = 225, 1165
    panel_height, panel_gap, top = 205, 48, 92
    t_min, t_max = min(times), max(times)

    def x_of(value: float) -> float:
        return left + (value - t_min) / max(t_max - t_min, 1e-9) * (right - left)

    for index, (values, label, reference, color, limits) in enumerate(series):
        y_top = top + index * (panel_height + panel_gap)
        y_bottom = y_top + panel_height
        low, high = limits

        def y_of(value: float) -> float:
            return y_bottom - (value - low) / max(high - low, 1e-9) * panel_height

        draw.rectangle(
            (x_of(event_start), y_top, x_of(event_end), y_bottom),
            fill=(20, 20, 20, 28),
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            grid_y = y_bottom - fraction * panel_height
            draw.line((left, grid_y, right, grid_y), fill=(0, 0, 0, 35), width=1)
        draw.rectangle((left, y_top, right, y_bottom), outline=(0, 0, 0, 100), width=1)
        if reference is not None:
            ref_y = y_of(reference)
            for x in range(left, right, 13):
                draw.line((x, ref_y, min(x + 7, right), ref_y), fill=(211, 139, 47, 210), width=2)
        points = [(x_of(time), y_of(value)) for time, value in zip(times, values)]
        draw.line(points, fill=(*color, 255), width=3, joint="curve")
        draw.text((15, y_top + 82), label, font=label_font, fill="#222222")
        draw.text((left - 55, y_top - 7), f"{high:.2f}", font=tick_font, fill="#444444")
        draw.text((left - 55, y_bottom - 12), f"{low:.2f}", font=tick_font, fill="#444444")
    axis_bottom = top + 2 * (panel_height + panel_gap) + panel_height
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = t_min + fraction * (t_max - t_min)
        x = x_of(value)
        draw.line((x, axis_bottom, x, axis_bottom + 7), fill="#333333", width=1)
        draw.text((x - 18, axis_bottom + 10), f"{value:.1f}", font=tick_font, fill="#333333")
    draw.text((650, 814), "simulation time (s)", font=label_font, fill="#222222")
    draw.text(
        (x_of(event_start) + 5, top + 6),
        "front-camera dropout",
        font=tick_font,
        fill="#333333",
    )
    canvas.save(destination)


def main() -> int:
    args = _parse_args()
    trace_path = _find_one(args.run_dir, "records_*/**/control_trace.jsonl")
    scenario_dir = trace_path.parent
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("control trace is empty")
    rows_by_step = {int(row["step"]): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw_front_gif": args.output_dir / "route146_pairwise_stop_raw_front.gif",
        "model_input_front_gif": (
            args.output_dir / "route146_pairwise_stop_model_input_front.gif"
        ),
        "bev_gif": args.output_dir / "route146_pairwise_stop_bev.gif",
        "trace_plot": args.output_dir / "route146_pairwise_stop_trace.png",
    }
    _make_gif(
        scenario_dir / "rgb_front",
        outputs["raw_front_gif"],
        label="raw front",
        rows_by_step=rows_by_step,
        max_size=(720, 405),
    )
    _make_gif(
        scenario_dir / "rgb_front_model_input",
        outputs["model_input_front_gif"],
        label="ORION front input",
        rows_by_step=rows_by_step,
        max_size=(720, 405),
    )
    _make_gif(
        scenario_dir / "bev",
        outputs["bev_gif"],
        label="external BEV",
        rows_by_step=rows_by_step,
        max_size=(512, 512),
    )
    _render_trace_plot(rows, outputs["trace_plot"], args.trigger_threshold)
    first, last = _event_indices(rows)
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_steps": len(rows),
        "event_start_seconds": rows[first]["sim_time_seconds"],
        "event_last_active_seconds": rows[last]["sim_time_seconds"],
        "event_median_score": statistics.median(
            float(row["raw_uq_score"]) for row in rows[first : last + 1]
        ),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    summary_path = args.output_dir / "visualization_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
