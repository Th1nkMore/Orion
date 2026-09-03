#!/usr/bin/env python3
"""Render compact optimization and gate diagnostics for a v10 staged smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT_SCHEMA = "orion.stage2l_v10_staged_smoke.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v10 plot")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported v10 report schema")

    phase_order = (
        "A_map_pretrain",
        "B_risk_alignment",
        "C_language_grounding",
    )
    colors = {"A_map_pretrain": "#2878B5", "B_risk_alignment": "#F07C19", "C_language_grounding": "#5B9A4D"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    offset = 0
    plotted = False
    for phase_name in phase_order:
        phase = report.get("phases", {}).get(phase_name)
        if not phase:
            continue
        history = list(phase.get("history", []))
        steps = [offset + int(item["optimizer_step"]) for item in history]
        if phase_name in {"A_map_pretrain", "B_risk_alignment"}:
            axes[0].plot(
                steps,
                [float(item["map_loss"]) for item in history],
                color=colors[phase_name],
                linewidth=2,
                label=phase_name,
            )
            if phase_name == "B_risk_alignment":
                axes[0].plot(
                    steps,
                    [float(item["ranking_loss"]) for item in history],
                    color=colors[phase_name],
                    linestyle="--",
                    alpha=0.8,
                    label="B ranking loss",
                )
        else:
            axes[0].plot(
                steps,
                [float(item["mean_target_nll"]) for item in history],
                color=colors[phase_name],
                linewidth=2,
                label="C target NLL",
            )
        axes[1].plot(
            steps,
            [float(item["gradient_norm_before_clip"]) for item in history],
            color=colors[phase_name],
            linewidth=1.8,
            label=phase_name,
        )
        offset += len(history)
        plotted = plotted or bool(history)

    if not plotted:
        raise ValueError("v10 report contains no optimizer history")
    for axis in axes:
        axis.grid(True, linewidth=0.6, alpha=0.25)
        axis.set_xlabel("Cumulative optimizer step")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Objective component")
    axes[0].set_title("Stage2-L v10 staged objectives")
    axes[1].set_ylabel("Gradient norm before clip")
    axes[1].set_title("Optimization stability")
    figure.suptitle(
        "%s | completed: %s"
        % (report.get("status"), ", ".join(report.get("completed_phases", []))),
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
