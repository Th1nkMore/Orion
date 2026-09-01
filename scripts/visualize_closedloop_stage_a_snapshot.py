#!/usr/bin/env python3
"""Create auditable local plots from the closed-loop Stage-A snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # GIF/contact-sheet export does not require matplotlib.
    plt = None


LOW_SPEED_MPS = 0.25
UQ_THRESHOLD = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_trace(run_dir: Path) -> tuple[Path | None, list[dict]]:
    paths = list(run_dir.glob("records_*/**/control_trace.jsonl"))
    if not paths:
        return None, []
    return paths[0], [
        json.loads(line) for line in paths[0].read_text().splitlines() if line
    ]


def longest_low_speed(records: list[dict]) -> dict:
    best = {"duration": 0.0, "start": None, "end": None}
    start = None
    padded = records + [None]
    for index, record in enumerate(padded):
        below = record is not None and abs(float(record["speed"])) < LOW_SPEED_MPS
        if below and start is None:
            start = index
        if not below and start is not None:
            end = index - 1
            duration = (
                float(records[end]["sim_time_seconds"])
                - float(records[start]["sim_time_seconds"])
            )
            if duration > best["duration"]:
                best = {
                    "duration": duration,
                    "start": float(records[start]["sim_time_seconds"]),
                    "end": float(records[end]["sim_time_seconds"]),
                }
            start = None
    return best


def summarize(run_dir: Path, override: dict) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    trace_path, records = read_trace(run_dir)
    speeds = [float(row["speed"]) for row in records]
    scores = [
        float(row["raw_uq_score"])
        for row in records
        if row.get("raw_uq_score") is not None
    ]
    low = longest_low_speed(records)
    summary = {
        "run": run_dir.name,
        "route_index": int(manifest["pilot_route_index"]),
        "condition": manifest["pilot_condition"],
        "status": override["status"],
        "official_outcome": bool(override["official_outcome"]),
        "note": override["note"],
        "trace_path": str(trace_path) if trace_path else None,
        "trace_steps": len(records),
        "sim_time_end_s": records[-1]["sim_time_seconds"] if records else None,
        "mean_speed_mps": sum(speeds) / len(speeds) if speeds else None,
        "minimum_speed_mps": min(speeds) if speeds else None,
        "mean_raw_uq": sum(scores) / len(scores) if scores else None,
        "maximum_raw_uq": max(scores) if scores else None,
        "uq_above_0_4_fraction": (
            sum(score > UQ_THRESHOLD for score in scores) / len(scores)
            if scores else None
        ),
        "longest_below_0_25_s": low["duration"] if records else None,
        "longest_low_speed_start_s": low["start"],
        "longest_low_speed_end_s": low["end"],
    }
    return summary, records


def plot_traces(runs: list[tuple[dict, list[dict]]], destination: Path) -> None:
    if plt is None:
        return
    traced = [(summary, rows) for summary, rows in runs if rows]
    fig, axes = plt.subplots(
        2, len(traced), figsize=(7.0 * len(traced), 7.2), squeeze=False,
        sharex="col",
    )
    for column, (summary, rows) in enumerate(traced):
        times = [float(row["sim_time_seconds"]) for row in rows]
        speeds = [float(row["speed"]) for row in rows]
        scores = [float(row["raw_uq_score"]) for row in rows]
        speed_ax, uq_ax = axes[0][column], axes[1][column]
        speed_ax.plot(times, speeds, color="#2474b5", linewidth=1.7)
        speed_ax.axhline(LOW_SPEED_MPS, color="#d65f5f", linestyle="--", linewidth=1)
        start = summary["longest_low_speed_start_s"]
        end = summary["longest_low_speed_end_s"]
        if start is not None and end is not None:
            speed_ax.axvspan(start, end, color="#d65f5f", alpha=0.16)
        speed_ax.set_title(
            f"route {summary['route_index']} · {summary['status'].replace('_', ' ')}"
        )
        speed_ax.set_ylabel("ego speed (m/s)")
        speed_ax.grid(alpha=0.22)
        uq_ax.plot(times, scores, color="#8d5bb7", linewidth=1.7)
        uq_ax.axhline(UQ_THRESHOLD, color="#e28e2c", linestyle="--", linewidth=1)
        uq_ax.set_xlabel("simulation time (s)")
        uq_ax.set_ylabel("raw Density score")
        uq_ax.set_ylim(-0.03, 1.03)
        uq_ax.grid(alpha=0.22)
    fig.suptitle("Stage-A traces — completed and diagnostic runs", fontsize=15)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def nearest_existing(directory: Path, preferred: list[int]) -> list[Path]:
    existing = sorted(directory.glob("*.png"))
    if not existing:
        return []
    by_index = {int(path.stem): path for path in existing}
    indices = sorted(by_index)
    selected = []
    for target in preferred:
        index = min(indices, key=lambda value: abs(value - target))
        path = by_index[index]
        if path not in selected:
            selected.append(path)
    return selected


def contact_sheet(
    runs: list[tuple[dict, list[dict]]], destination: Path
) -> None:
    rows = []
    for summary, records in runs:
        if not records:
            continue
        run_dir = Path(summary["trace_path"]).parents[2]
        front_dirs = list(run_dir.glob("records_*/**/rgb_front"))
        if not front_dirs:
            continue
        last_frame = max(0, (len(records) - 1) // 10)
        preferred = [0, round(last_frame * 0.25), round(last_frame * 0.5),
                     round(last_frame * 0.75), last_frame]
        rows.append((summary, nearest_existing(front_dirs[0], preferred)))
    if not rows:
        return
    thumb_w, thumb_h, label_h = 320, 180, 52
    columns = max(len(images) for _, images in rows)
    canvas = Image.new("RGB", (columns * thumb_w, len(rows) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (summary, images) in enumerate(rows):
        y0 = row_index * (thumb_h + label_h)
        draw.text(
            (8, y0 + 5),
            f"route {summary['route_index']} · {summary['status'].replace('_', ' ')}",
            fill="black",
        )
        for column, path in enumerate(images):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = column * thumb_w + (thumb_w - image.width) // 2
            y = y0 + label_h + (thumb_h - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((column * thumb_w + 8, y0 + 25), f"saved frame {path.stem}", fill="black")
    canvas.save(destination)


def gif_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_gif(
    frame_dir: Path,
    destination: Path,
    *,
    route_index: int,
    condition: str,
    status: str,
    view_label: str,
    max_size: tuple[int, int],
    black_frame_indices: set[int] | None = None,
) -> None:
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        return
    font = gif_font(21)
    frames = []
    for path in paths:
        frame = Image.open(path).convert("RGB")
        if black_frame_indices is not None and int(path.stem) in black_frame_indices:
            # camera_dropout is applied after image preprocessing.  In normalized
            # space it is exactly -mean/std, i.e. an all-black RGB image.
            frame = Image.new("RGB", frame.size, "black")
        frame.thumbnail(max_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(frame, "RGBA")
        label = (
            f"route {route_index} | {condition} | {view_label} | "
            f"{status.replace('_', ' ')} | "
            f"t={int(path.stem) * 0.5:.1f}s"
        )
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((0, 0, frame.width, box[3] + 14), fill=(0, 0, 0, 170))
        draw.text((8, 6), label, font=font, fill=(255, 255, 255, 255))
        frames.append(frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        disposal=2,
    )


def render_gifs(
    runs: list[tuple[dict, list[dict]]], out_dir: Path
) -> None:
    for summary, records in runs:
        if not records:
            continue
        run_dir = Path(summary["trace_path"]).parents[2]
        scenario_dirs = list(run_dir.glob("records_*/*"))
        if not scenario_dirs:
            continue
        scenario_dir = scenario_dirs[0]
        condition = summary["condition"]
        stem = f"route{summary['route_index']}_{condition}"
        make_gif(
            scenario_dir / "rgb_front",
            out_dir / f"{stem}_raw_front.gif",
            route_index=summary["route_index"],
            condition=condition,
            status=summary["status"],
            view_label="raw scene front",
            max_size=(720, 405),
        )
        if condition.startswith("front_corrupt_"):
            active_frame_indices = {
                int(row["step"]) // 10
                for row in records
                if row.get("corruption_active") and int(row["step"]) % 10 == 0
            }
            make_gif(
                scenario_dir / "rgb_front",
                out_dir / f"{stem}_agent_input_front.gif",
                route_index=summary["route_index"],
                condition=condition,
                status=summary["status"],
                view_label="agent input front",
                max_size=(720, 405),
                black_frame_indices=active_frame_indices,
            )
        elif condition == "clean_off":
            # Preserve the concise historical filename for the clean input view.
            make_gif(
                scenario_dir / "rgb_front",
                out_dir / f"{stem}_front.gif",
                route_index=summary["route_index"],
                condition=condition,
                status=summary["status"],
                view_label="agent input front",
                max_size=(720, 405),
            )
        make_gif(
            scenario_dir / "bev",
            out_dir / f"{stem}_bev.gif",
            route_index=summary["route_index"],
            condition=condition,
            status=summary["status"],
            view_label="external BEV",
            max_size=(512, 512),
        )


def paired_diagnostics(runs: list[tuple[dict, list[dict]]]) -> list[dict]:
    """Compare time-aligned clean and corrupted traces without claiming causality."""
    clean_by_route = {
        summary["route_index"]: (summary, rows)
        for summary, rows in runs
        if summary["condition"] == "clean_off" and rows
    }
    diagnostics = []
    for corrupt_summary, corrupt_rows in runs:
        if corrupt_summary["condition"] != "front_corrupt_off" or not corrupt_rows:
            continue
        clean_pair = clean_by_route.get(corrupt_summary["route_index"])
        if clean_pair is None:
            continue
        clean_summary, clean_rows = clean_pair
        count = min(len(clean_rows), len(corrupt_rows))
        clean_scores = [float(row["raw_uq_score"]) for row in clean_rows[:count]]
        corrupt_scores = [
            float(row["raw_uq_score"]) for row in corrupt_rows[:count]
        ]
        deltas = [
            corrupt - clean
            for clean, corrupt in zip(clean_scores, corrupt_scores)
        ]
        diagnostics.append({
            "route_index": corrupt_summary["route_index"],
            "clean_run": clean_summary["run"],
            "corrupt_run": corrupt_summary["run"],
            "corrupt_status": corrupt_summary["status"],
            "official_pair": bool(
                clean_summary["official_outcome"]
                and corrupt_summary["official_outcome"]
            ),
            "paired_steps": count,
            "paired_sim_time_end_s": corrupt_rows[count - 1]["sim_time_seconds"],
            "clean_mean_uq": statistics.fmean(clean_scores),
            "corrupt_mean_uq": statistics.fmean(corrupt_scores),
            "mean_paired_uq_delta": statistics.fmean(deltas),
            "median_paired_uq_delta": statistics.median(deltas),
            "corrupt_uq_greater_fraction": (
                sum(corrupt > clean for clean, corrupt in zip(clean_scores, corrupt_scores))
                / count
            ),
            "clean_uq_above_0_4_fraction": (
                sum(score > UQ_THRESHOLD for score in clean_scores) / count
            ),
            "corrupt_uq_above_0_4_fraction": (
                sum(score > UQ_THRESHOLD for score in corrupt_scores) / count
            ),
            "interpretation_limit": (
                "Diagnostic only unless official_pair=true; a full-route fixed "
                "dropout tests separation, not event-window timing or semantic grounding."
            ),
        })
    return diagnostics


def main() -> None:
    args = parse_args()
    overrides = json.loads((args.results_root / "status_overrides.json").read_text())
    runs = []
    for run_dir in sorted((args.results_root / "raw").glob("route*")):
        override = overrides["runs"].get(run_dir.name)
        if override is None:
            continue
        runs.append(summarize(run_dir, override))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_at": overrides["snapshot_at"],
        "warning": (
            "Only runs with official_outcome=true are completed safety outcomes; "
            "all other runs are diagnostic-only."
        ),
        "runs": [summary for summary, _ in runs],
        "paired_diagnostics": paired_diagnostics(runs),
    }
    (args.out_dir / "stage_a_snapshot_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    columns = sorted({key for summary, _ in runs for key in summary})
    with (args.out_dir / "stage_a_snapshot_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary for summary, _ in runs)
    plot_traces(runs, args.out_dir / "stage_a_clean_traces.png")
    contact_sheet(runs, args.out_dir / "stage_a_front_contact_sheet.png")
    render_gifs(runs, args.out_dir)


if __name__ == "__main__":
    main()
