"""Plot balanced Reliability QA intervention results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uq_estimator.risk_qa import reliability_level, reliability_percentile


LEVELS = ("very low", "low", "moderate", "high", "very high")
DISPLAY_LEVELS = ("Very low", "Low", "Moderate", "High", "Very high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--controls-json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def confusion(records: list[dict], mode: str, score_key: str) -> np.ndarray:
    level_index = {level: index for index, level in enumerate(LEVELS)}
    matrix = np.zeros((len(LEVELS), len(LEVELS)), dtype=np.int64)
    for record in records:
        prediction = record["outputs"][mode]["parsed_level"]
        if prediction not in level_index:
            continue
        target = reliability_level(
            reliability_percentile(record[score_key])
        )
        matrix[level_index[target], level_index[prediction]] += 1
    return matrix


def draw_confusion(ax, matrix: np.ndarray, title: str) -> None:
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sum > 0,
    )
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = normalized[row, column]
            color = "white" if value > 0.55 else "#1f2937"
            ax.text(
                column,
                row,
                f"{matrix[row, column]}\n{value:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    ax.set_xticks(range(len(LEVELS)), DISPLAY_LEVELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LEVELS)), DISPLAY_LEVELS)
    ax.set_xlabel("Generated reliability level")
    ax.set_ylabel("Target reliability level")
    ax.set_title(title, fontweight="bold")
    return image


def main() -> None:
    args = parse_args()
    evaluation = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    controls = json.loads(Path(args.controls_json).read_text(encoding="utf-8"))
    records = evaluation["records"]
    summary = evaluation["summary"]

    correct_matrix = confusion(records, "correct", "correct_uq_score")
    shuffled_matrix = confusion(records, "shuffled", "shuffled_uq_score")
    none_parse = np.mean([
        record["outputs"]["none"]["parsed_level"] is not None
        for record in controls["records"]
    ])
    zero_parse = np.mean([
        record["outputs"]["zero"]["parsed_level"] is not None
        for record in controls["records"]
    ])

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.95]},
    )
    draw_confusion(
        axes[0], correct_matrix, "Correct UQ intervention"
    )
    draw_confusion(
        axes[1], shuffled_matrix, "Shuffled UQ intervention"
    )
    labels = ("C\nparse", "C\nacc.", "S\nparse", "S\nacc.",
              "CF\nresp.", "None", "Zero")
    values = (
        summary["correct"]["parse_rate"],
        summary["correct"]["accuracy"],
        summary["shuffled"]["parse_rate"],
        summary["shuffled"]["accuracy"],
        summary["intervention"]["response_rate"],
        none_parse,
        zero_parse,
    )
    colors = ("#2563eb", "#2563eb", "#0f766e", "#0f766e",
              "#ca8a04", "#6b7280", "#6b7280")
    bars = axes[2].bar(range(len(values)), values, color=colors, width=0.72)
    axes[2].set_ylim(0.0, 1.08)
    axes[2].set_xticks(range(len(labels)), labels)
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].set_ylabel("Rate")
    axes[2].set_title("Reliability-language evidence", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Continuous UQ tokens produce calibrated, intervention-sensitive "
        "LLM reliability statements",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.06, right=0.98, bottom=0.23, top=0.82, wspace=0.34
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
