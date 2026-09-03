"""Summarize UQ vision-adapter evaluation JSON files.

The script turns route-balanced aggregate results and optional per-sample
stratified results into CSV / Markdown tables and report-ready figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ("none", "shuffled", "correct")
COLORS = {
    "none": "#C62828",
    "shuffled": "#F9A825",
    "correct": "#1565C0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize UQ adapter results")
    parser.add_argument("--route-json", required=True)
    parser.add_argument("--stratified-json", default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def short_route_name(route: str) -> str:
    parts = route.split("_")
    route_part = next((part for part in parts if part.startswith("Route")), "")
    weather_part = next((part for part in parts if part.startswith("Weather")), "")
    scenario = parts[0]
    return "\n".join(item for item in (scenario, route_part, weather_part) if item)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def route_rows(route_payload: dict) -> list[dict]:
    results = route_payload["results"]
    routes = list(results["none"]["planning_by_route"].keys())
    rows = []
    for route in routes:
        row = {"route": route}
        for mode in MODES:
            metrics = results[mode]["planning_by_route"][route]
            row[f"{mode}_ade"] = float(metrics["ade"])
            row[f"{mode}_fde"] = float(metrics["fde"])
            row[f"{mode}_count"] = int(metrics["count"])
        row["correct_vs_none_pct"] = (
            (row["none_ade"] - row["correct_ade"]) / row["none_ade"] * 100.0
        )
        row["correct_vs_shuffled_pct"] = (
            (row["shuffled_ade"] - row["correct_ade"]) / row["shuffled_ade"] * 100.0
        )
        rows.append(row)
    rows.sort(key=lambda item: item["correct_vs_none_pct"], reverse=True)
    return rows


def write_route_tables(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "route_balanced_per_route.csv"
    fields = [
        "route",
        "none_ade",
        "shuffled_ade",
        "correct_ade",
        "none_fde",
        "shuffled_fde",
        "correct_fde",
        "correct_vs_none_pct",
        "correct_vs_shuffled_pct",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    md_path = out_dir / "route_balanced_per_route.md"
    lines = [
        "| Route | none ADE | shuffled ADE | correct ADE | correct vs none | correct vs shuffled |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {route} | {none:.3f} | {shuf:.3f} | {corr:.3f} | {n_imp:+.1f}% | {s_imp:+.1f}% |".format(
                route=row["route"],
                none=row["none_ade"],
                shuf=row["shuffled_ade"],
                corr=row["correct_ade"],
                n_imp=row["correct_vs_none_pct"],
                s_imp=row["correct_vs_shuffled_pct"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_route_ade(rows: list[dict], out_dir: Path) -> None:
    labels = [short_route_name(row["route"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=160)
    for offset, mode in zip((-width, 0.0, width), MODES):
        values = [row[f"{mode}_ade"] for row in rows]
        ax.bar(x + offset, values, width, label=mode, color=COLORS[mode], alpha=0.9)
    ax.set_ylabel("ADE / m")
    ax.set_title("Route-balanced planning ADE under camera dropout corruption")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "route_balanced_ade_by_route.png")
    plt.close(fig)


def plot_route_improvement(rows: list[dict], out_dir: Path) -> None:
    labels = [short_route_name(row["route"]) for row in rows]
    values = [row["correct_vs_none_pct"] for row in rows]
    colors = ["#2E7D32" if value >= 0 else "#C62828" for value in values]
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
    ax.bar(np.arange(len(rows)), values, color=colors, alpha=0.88)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylabel("ADE improvement of correct vs none / %")
    ax.set_title("Where UQ adapter helps or hurts")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "route_balanced_correct_improvement.png")
    plt.close(fig)


def stratified_rows(payload: dict) -> list[dict]:
    rows = []
    for group, group_payload in payload.get("stratified", {}).items():
        row = {"group": group, "count": int(group_payload["count"])}
        for mode in MODES:
            row[f"{mode}_ade"] = float(group_payload["modes"][mode]["ade"])
        row["correct_vs_none_pct"] = (
            (row["none_ade"] - row["correct_ade"]) / row["none_ade"] * 100.0
        )
        row["correct_vs_shuffled_pct"] = (
            (row["shuffled_ade"] - row["correct_ade"]) / row["shuffled_ade"] * 100.0
        )
        rows.append(row)
    rows.sort(key=lambda item: item["group"])
    return rows


def write_stratified_outputs(payload: dict, out_dir: Path) -> None:
    rows = stratified_rows(payload)
    if not rows:
        return
    csv_path = out_dir / "uq_stratified_summary.csv"
    fields = [
        "group",
        "count",
        "none_ade",
        "shuffled_ade",
        "correct_ade",
        "correct_vs_none_pct",
        "correct_vs_shuffled_pct",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    md_path = out_dir / "uq_stratified_summary.md"
    lines = [
        "| UQ group | Count | none ADE | shuffled ADE | correct ADE | correct vs none | correct vs shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {count} | {none:.3f} | {shuf:.3f} | {corr:.3f} | {n_imp:+.1f}% | {s_imp:+.1f}% |".format(
                group=row["group"],
                count=row["count"],
                none=row["none_ade"],
                shuf=row["shuffled_ade"],
                corr=row["correct_ade"],
                n_imp=row["correct_vs_none_pct"],
                s_imp=row["correct_vs_shuffled_pct"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [row["group"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    for offset, mode in zip((-width, 0.0, width), MODES):
        ax.bar(
            x + offset,
            [row[f"{mode}_ade"] for row in rows],
            width,
            label=mode,
            color=COLORS[mode],
            alpha=0.9,
        )
    ax.set_ylabel("ADE / m")
    ax.set_title("Planning ADE stratified by Density UQ score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "uq_stratified_ade.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    route_payload = load_json(args.route_json)
    rows = route_rows(route_payload)
    write_route_tables(rows, out_dir)
    plot_route_ade(rows, out_dir)
    plot_route_improvement(rows, out_dir)

    if args.stratified_json:
        write_stratified_outputs(load_json(args.stratified_json), out_dir)

    print(f"Wrote summary outputs to {out_dir}")


if __name__ == "__main__":
    main()
