#!/usr/bin/env python3
"""Build a local round-2 dashboard from existing lightweight result files.

This script only consumes JSON/TXT assets that already exist in the repo.
It does not load ORION checkpoints or run model inference.
Historical replay-control results are used as diagnostics only; they are not
the paper-aligned closed-loop headline metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round-2 dashboard assets")
    parser.add_argument("--openloop", default="results/eval_openloop_v3.json")
    parser.add_argument(
        "--replay-baseline",
        "--closedloop-baseline",
        dest="replay_baseline",
        default="results/closedloop_baseline_50.json",
    )
    parser.add_argument(
        "--replay-film",
        "--closedloop-film",
        dest="replay_film",
        default="results/closedloop_replay_v3.json",
    )
    parser.add_argument("--bev-report", default="results/bev_noattn/report.txt")
    parser.add_argument("--feasibility", default="results/feasibility/feasibility_report.md")
    parser.add_argument("--out-dir", default="results/round2_dashboard")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bev_report(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"Condition : (?P<label>.+?)\n"
        r"\s+Frames\s+:\s+(?P<frames>\d+)\n"
        r"\s+Mean\s+:\s+(?P<mean>[0-9.]+)\n"
        r"\s+Std\s+:\s+(?P<std>[0-9.]+)",
        re.MULTILINE,
    )
    out: dict[str, dict[str, float]] = {}
    for match in pattern.finditer(text):
        out[match.group("label")] = {
            "frames": int(match.group("frames")),
            "mean": float(match.group("mean")),
            "std": float(match.group("std")),
        }
    delta_match = re.search(r"Δ mean uncertainty \(adverse − normal\): (?P<delta>[+-]?[0-9.]+)", text)
    out["delta"] = {"mean": float(delta_match.group("delta")) if delta_match else 0.0}
    return out


def parse_feasibility(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    items = {}
    for qid, title, conclusion in re.findall(
        r"## (Q\d+)：(.+?)\n\n\*\*结论：(.+?)\*\*",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ):
        items[qid] = {"title": title.strip(), "conclusion": conclusion.strip()}
    return items


def feasibility_display_lines(items: dict[str, dict[str, str]]) -> list[str]:
    english = {
        "Q1": "Flash attention blocks attn extraction; mitigation exists.",
        "Q2": "BEV queries are learned, non-grid points; use k-NN interpolation.",
        "Q3": "Poses-cls distribution is still missing and needs server extraction.",
        "Q4": "Plan-anchor coverage is diverse enough for per-mode cost.",
        "Q5": "LLM-free loading still needs server-side verification.",
    }
    return [f"{qid}: {english.get(qid, item['conclusion'])}" for qid, item in sorted(items.items())]


def write_csv(out_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(block: dict, name: str) -> float:
    if name in block:
        return block[name]
    mean_name = f"{name}_mean"
    if mean_name in block:
        return block[mean_name]
    raise KeyError(name)


def normalize_replay(data: dict) -> dict:
    if "scenario_results" not in data:
        return data

    scenario_results = list(data["scenario_results"].values())
    split_map = {
        "all": scenario_results,
        "normal": [row for row in scenario_results if not row.get("is_adverse", False)],
        "adverse": [row for row in scenario_results if row.get("is_adverse", False)],
    }

    for split, rows in split_map.items():
        block = data.get("aggregate", {}).get(split, {})
        if rows and "avg_speed" not in block and "avg_speed_mean" not in block:
            block["avg_speed"] = sum(row.get("avg_speed", 0.0) for row in rows) / len(rows)
        data["aggregate"][split] = block
    return data


def create_summary_payload(
    openloop: dict,
    replay_baseline: dict,
    replay_film: dict,
    bev: dict,
    feasibility: dict,
) -> dict:
    return {
        "openloop": {
            "n_samples": openloop["n_samples"],
            "auroc": openloop["auroc"],
            "normal": openloop["stats"]["normal"],
            "adverse": openloop["stats"]["adverse"],
        },
        "replay": {
            "baseline": replay_baseline["aggregate"],
            "film_v3": replay_film["aggregate"],
            "delta_all": {
                "traj_ade_3s": metric_value(replay_film["aggregate"]["all"], "traj_ade_3s") - metric_value(replay_baseline["aggregate"]["all"], "traj_ade_3s"),
                "collision_rate_3s": metric_value(replay_film["aggregate"]["all"], "collision_rate_3s") - metric_value(replay_baseline["aggregate"]["all"], "collision_rate_3s"),
                "avg_speed": metric_value(replay_film["aggregate"]["all"], "avg_speed") - metric_value(replay_baseline["aggregate"]["all"], "avg_speed"),
                "control_mae_brake": metric_value(replay_film["aggregate"]["all"], "control_mae_brake") - metric_value(replay_baseline["aggregate"]["all"], "control_mae_brake"),
            },
        },
        "bev_ipm": bev,
        "feasibility": feasibility,
    }


def save_markdown_summary(out_path: Path, payload: dict) -> None:
    openloop = payload["openloop"]
    replay = payload["replay"]
    bev = payload["bev_ipm"]
    feasibility = payload["feasibility"]

    lines = [
        "# Round-2 Dashboard Summary",
        "",
        "## Current Baseline",
        "",
        f"- Open-loop AUROC: `{openloop['auroc']:.3f}` over `{openloop['n_samples']}` samples.",
        f"- Open-loop normal: `UQ={openloop['normal']['uq_mean']:.3f}`, `L2@3s={openloop['normal']['l2_3s_mean']:.3f}`, `Col@3s={openloop['normal']['col_3s_mean']:.4f}`.",
        f"- Open-loop adverse: `UQ={openloop['adverse']['uq_mean']:.3f}`, `L2@3s={openloop['adverse']['l2_3s_mean']:.3f}`, `Col@3s={openloop['adverse']['col_3s_mean']:.4f}`.",
        f"- Replay diagnostic baseline ADE@3s: `{metric_value(replay['baseline']['all'], 'traj_ade_3s'):.3f}`.",
        f"- Replay diagnostic FiLM ADE@3s: `{metric_value(replay['film_v3']['all'], 'traj_ade_3s'):.3f}`.",
        f"- Replay collision delta (FiLM - baseline): `{replay['delta_all']['collision_rate_3s']:+.4f}`.",
        "",
        "## Conservative Shortcut Evidence",
        "",
        f"- Normal replay ADE worsens from `{metric_value(replay['baseline']['normal'], 'traj_ade_3s'):.3f}` to `{metric_value(replay['film_v3']['normal'], 'traj_ade_3s'):.3f}`.",
        f"- Adverse replay collision changes from `{metric_value(replay['baseline']['adverse'], 'collision_rate_3s'):.4f}` to `{metric_value(replay['film_v3']['adverse'], 'collision_rate_3s'):.4f}`.",
        f"- All-scenario replay brake MAE changes from `{metric_value(replay['baseline']['all'], 'control_mae_brake'):.3f}` to `{metric_value(replay['film_v3']['all'], 'control_mae_brake'):.3f}`.",
        f"- All-scenario replay average speed changes from `{metric_value(replay['baseline']['all'], 'avg_speed'):.3f}` to `{metric_value(replay['film_v3']['all'], 'avg_speed'):.3f}`.",
        "",
        "## BEV / Feasibility",
        "",
        f"- IPM BEV uncertainty delta: `{bev['delta']['mean']:+.4f}` adverse-minus-normal.",
        f"- Normal BEV mean: `{bev.get('Normal (CloudySunset)', {}).get('mean', 0.0):.4f}`.",
        f"- Adverse BEV mean: `{bev.get('Adverse (HardRainNight)', {}).get('mean', 0.0):.4f}`.",
        "",
        "## Feasibility Snapshot",
        "",
    ]
    for qid in sorted(feasibility):
        item = feasibility[qid]
        lines.append(f"- `{qid}` {item['title']}: {item['conclusion']}")

    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Treat this folder as the local source of truth for round-2 planning.",
            "- Use open-loop as the main round-2 gate; treat replay plots here as historical diagnostics only.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figures(out_dir: Path, fmt: str, payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {
        "baseline": "#1f77b4",
        "film": "#d62728",
        "normal": "#4C78A8",
        "adverse": "#E45756",
        "bev": "#54A24B",
    }

    openloop = payload["openloop"]
    baseline = payload["replay"]["baseline"]["aggregate"] if "aggregate" in payload["replay"]["baseline"] else payload["replay"]["baseline"]
    film = payload["replay"]["film_v3"]["aggregate"] if "aggregate" in payload["replay"]["film_v3"] else payload["replay"]["film_v3"]
    bev = payload["bev_ipm"]

    # Figure 1: current best overview
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    splits = ["normal", "adverse"]
    x = np.arange(len(splits))
    width = 0.35
    axes[0].bar(x - width / 2, [openloop["normal"]["l2_3s_mean"], openloop["adverse"]["l2_3s_mean"]], width, color=colors["baseline"], label="L2@3s")
    axes[0].bar(x + width / 2, [openloop["normal"]["col_3s_mean"], openloop["adverse"]["col_3s_mean"]], width, color=colors["film"], label="Col@3s")
    axes[0].set_xticks(x, ["Normal", "Adverse"])
    axes[0].set_title("Open-Loop Split Snapshot")
    axes[0].set_ylabel("Metric Value")
    axes[0].legend()

    metrics = ["traj_ade_3s", "collision_rate_3s", "avg_speed_mean"]
    baseline_vals = [
        metric_value(baseline["all"], "traj_ade_3s"),
        metric_value(baseline["all"], "collision_rate_3s"),
        metric_value(baseline["all"], "avg_speed"),
    ]
    film_vals = [
        metric_value(film["all"], "traj_ade_3s"),
        metric_value(film["all"], "collision_rate_3s"),
        metric_value(film["all"], "avg_speed"),
    ]
    xx = np.arange(len(metrics))
    axes[1].bar(xx - width / 2, baseline_vals, width, color=colors["baseline"], label="Baseline")
    axes[1].bar(xx + width / 2, film_vals, width, color=colors["film"], label="FiLM v3")
    axes[1].set_xticks(xx, ["ADE@3s", "Col@3s", "AvgSpeed"])
    axes[1].set_title("Replay Diagnostic Snapshot")
    axes[1].legend()
    fig.suptitle(f"Round-2 Current Best Summary (AUROC={openloop['auroc']:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_current_best_overview.{fmt}")
    plt.close(fig)

    # Figure 2: safety-efficiency tradeoff
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, split, label in zip(axes, ["normal", "adverse"], ["Normal", "Adverse"]):
        b_ade = metric_value(baseline[split], "traj_ade_3s")
        f_ade = metric_value(film[split], "traj_ade_3s")
        b_col = metric_value(baseline[split], "collision_rate_3s")
        f_col = metric_value(film[split], "collision_rate_3s")
        ax.scatter([b_ade], [b_col], s=110, color=colors["baseline"], label="Baseline")
        ax.scatter([f_ade], [f_col], s=110, color=colors["film"], label="FiLM v3")
        ax.annotate("", xy=(f_ade, f_col), xytext=(b_ade, b_col), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#555"})
        ax.set_xlabel("ADE@3s")
        ax.set_ylabel("Collision Rate@3s")
        ax.set_title(f"{label}: Replay Safety vs Efficiency")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_safety_efficiency_tradeoff.{fmt}")
    plt.close(fig)

    # Figure 3: conservative shortcut evidence
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    splits = ["all", "normal", "adverse"]
    baseline_brake = [metric_value(baseline[s], "control_mae_brake") for s in splits]
    film_brake = [metric_value(film[s], "control_mae_brake") for s in splits]
    baseline_speed = [metric_value(baseline[s], "avg_speed") for s in splits]
    film_speed = [metric_value(film[s], "avg_speed") for s in splits]
    baseline_ade = [metric_value(baseline[s], "traj_ade_3s") for s in splits]
    film_ade = [metric_value(film[s], "traj_ade_3s") for s in splits]
    xx = np.arange(len(splits))

    axes[0].bar(xx - width / 2, baseline_brake, width, color=colors["baseline"])
    axes[0].bar(xx + width / 2, film_brake, width, color=colors["film"])
    axes[0].set_xticks(xx, ["All", "Normal", "Adverse"])
    axes[0].set_title("Brake MAE")

    axes[1].bar(xx - width / 2, baseline_speed, width, color=colors["baseline"])
    axes[1].bar(xx + width / 2, film_speed, width, color=colors["film"])
    axes[1].set_xticks(xx, ["All", "Normal", "Adverse"])
    axes[1].set_title("Average Speed")

    axes[2].bar(xx - width / 2, baseline_ade, width, color=colors["baseline"], label="Baseline")
    axes[2].bar(xx + width / 2, film_ade, width, color=colors["film"], label="FiLM v3")
    axes[2].set_xticks(xx, ["All", "Normal", "Adverse"])
    axes[2].set_title("ADE@3s")
    axes[2].legend()
    fig.suptitle("Conservative Shortcut Evidence")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_conservative_shortcut_evidence.{fmt}")
    plt.close(fig)

    # Figure 4: BEV + feasibility panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bev_names = ["Normal (CloudySunset)", "Adverse (HardRainNight)"]
    bev_means = [bev.get(name, {}).get("mean", 0.0) for name in bev_names]
    bev_stds = [bev.get(name, {}).get("std", 0.0) for name in bev_names]
    axes[0].bar([0, 1], bev_means, yerr=bev_stds, color=[colors["normal"], colors["adverse"]], capsize=5)
    axes[0].set_xticks([0, 1], ["Normal", "Adverse"])
    axes[0].set_title(f"IPM-BEV Uncertainty (Δ={bev['delta']['mean']:+.3f})")
    axes[0].set_ylabel("Mean Uncertainty")

    axes[1].axis("off")
    feasibility_lines = feasibility_display_lines(payload["feasibility"])
    axes[1].text(
        0.0,
        1.0,
        "Feasibility Snapshot\n\n" + "\n".join(feasibility_lines),
        va="top",
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_bev_feasibility_snapshot.{fmt}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    openloop = load_json(args.openloop)
    baseline = normalize_replay(load_json(args.replay_baseline))
    film = normalize_replay(load_json(args.replay_film))
    bev = parse_bev_report(args.bev_report)
    feasibility = parse_feasibility(args.feasibility)

    payload = create_summary_payload(openloop, baseline, film, bev, feasibility)

    best_rows = [
        {
            "source": "openloop",
            "split": "normal",
            "uq_mean": openloop["stats"]["normal"]["uq_mean"],
            "l2_3s": openloop["stats"]["normal"]["l2_3s_mean"],
            "col_3s": openloop["stats"]["normal"]["col_3s_mean"],
        },
        {
            "source": "openloop",
            "split": "adverse",
            "uq_mean": openloop["stats"]["adverse"]["uq_mean"],
            "l2_3s": openloop["stats"]["adverse"]["l2_3s_mean"],
            "col_3s": openloop["stats"]["adverse"]["col_3s_mean"],
        },
        {
            "source": "replay_baseline",
            "split": "all",
            "uq_mean": "",
            "l2_3s": metric_value(baseline["aggregate"]["all"], "traj_ade_3s"),
            "col_3s": metric_value(baseline["aggregate"]["all"], "collision_rate_3s"),
        },
        {
            "source": "replay_film_v3",
            "split": "all",
            "uq_mean": film["aggregate"]["all"]["uq_score_mean"],
            "l2_3s": metric_value(film["aggregate"]["all"], "traj_ade_3s"),
            "col_3s": metric_value(film["aggregate"]["all"], "collision_rate_3s"),
        },
    ]
    write_csv(out_dir / "current_best_table.csv", best_rows, ["source", "split", "uq_mean", "l2_3s", "col_3s"])

    with open(out_dir / "dashboard_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    save_markdown_summary(out_dir / "summary.md", payload)
    build_figures(out_dir, args.format, payload)

    print(f"Wrote dashboard assets to {out_dir}")


if __name__ == "__main__":
    main()
