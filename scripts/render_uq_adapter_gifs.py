"""Render GIFs for UQ vision-adapter planning interventions.

This script runs a small evaluation subset and renders BEV trajectory GIFs for
none / shuffled / correct UQ modes. It is intended for reports and slides, not
for full metric evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, ProgressBar, load_checkpoint, set_random_seed

from scripts.train_film import custom_wrap_fp16_model, forward_film_training
from scripts.train_uq_token import (
    build_shuffled_uq_lookup,
    filter_dataset_by_split,
    get_shuffled_uq,
    route_from_info,
    sample_id_from_batch,
)
from uq_estimator.corruptions import corrupt_batch_images
from uq_estimator.density import get_uq_state_dict
from uq_estimator.training import freeze_for_uq_token_training, load_uq_token_weights


COLORS = {
    "gt": "#2E7D32",
    "none": "#D32F2F",
    "shuffled": "#F9A825",
    "correct": "#1565C0",
}
MODE_LABELS = {
    "none": "none",
    "shuffled": "shuffled UQ",
    "correct": "correct UQ",
}
EGO_LENGTH = 4.5
EGO_WIDTH = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render UQ vision-adapter trajectory GIFs"
    )
    parser.add_argument("--config", default="adzoo/orion/configs/orion_stage3_infer.py")
    parser.add_argument("--checkpoint", default="ckpts/Orion.pth")
    parser.add_argument("--density-checkpoint", default="checkpoints/density_uq/best.pt")
    parser.add_argument("--descriptor-cache", default="data/density_uq/descriptors.pt")
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--init-adaptation", required=True)
    parser.add_argument("--out-dir", default="results/uq_adapter_gifs")
    parser.add_argument(
        "--routes",
        nargs="+",
        default=[
            "VehicleTurningRoute_Town15_Route504_Weather10",
            "BlockedIntersection_Town03_Route135_Weather5",
            "YieldToEmergencyVehicle_Town04_Route166_Weather10",
        ],
    )
    parser.add_argument("--frames-per-route", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-corruption", action="store_true")
    parser.add_argument(
        "--corruption",
        choices=("blur", "dark", "camera_dropout"),
        default="camera_dropout",
    )
    parser.add_argument("--corruption-severity", type=int, default=1)
    parser.add_argument(
        "--modes",
        default="none,shuffled,correct",
        help="comma-separated modes from none, shuffled, correct",
    )
    return parser.parse_args()


def _select_route_infos(infos: list[dict], routes: list[str], limit: int, stride: int):
    grouped: dict[str, list[dict]] = {route: [] for route in routes}
    route_set = set(routes)
    for info in infos:
        route = route_from_info(info)
        if route in route_set:
            grouped[route].append(info)

    selected = []
    counts = {}
    for route in routes:
        route_infos = sorted(grouped.get(route, []), key=lambda item: item["frame_idx"])
        sampled = route_infos[:: max(stride, 1)][:limit]
        if not sampled:
            raise RuntimeError(f"No samples found for route {route}")
        selected.extend(sampled)
        counts[route] = len(sampled)
    return selected, counts


def _build_model(args: argparse.Namespace):
    density_payload = torch.load(
        args.density_checkpoint, map_location="cpu", weights_only=True
    )
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.frozen = True
    cfg.model.use_lora = True
    cfg.model.use_uq_token = True
    cfg.model.use_uq_vision_adapter = True
    cfg.model.use_uncertainty_l2 = False
    cfg.model.use_bev_uncertainty = False
    cfg.model.pts_bbox_head.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = args.density_checkpoint
    cfg.model.pts_bbox_head.transformer.use_uncertainty = False
    cfg.data.test.ann_file = args.ann_file

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        custom_wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.pts_bbox_head.uq_estimator.load_state_dict(
        get_uq_state_dict(density_payload), strict=False
    )
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    freeze_for_uq_token_training(model)
    loaded = load_uq_token_weights(model, args.init_adaptation)
    print(f"[GIF] loaded {loaded} adaptation tensors from {args.init_adaptation}")

    model.cuda()
    model.eval()
    model.pts_bbox_head.with_dn = False
    model.pts_bbox_head.uq_estimator.eval()
    if hasattr(model.lm_head, "gradient_checkpointing_disable"):
        model.lm_head.gradient_checkpointing_disable()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return cfg, model, density_payload["split_assignment"]


def _trajectory_ade(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    distance = np.linalg.norm(pred[valid] - target[valid], axis=-1)
    return float(distance.mean())


def _as_abs_traj(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().float().cpu()
    while value.ndim > 2:
        value = value[0]
    return value.cumsum(dim=0).numpy()


def _collect_records(args, model, loader, infos, shuffled_lookup):
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    records_by_route: dict[str, list[dict]] = {}
    progress = ProgressBar(len(infos))
    with torch.no_grad():
        for index, (data, info) in enumerate(zip(loader, infos)):
            route = route_from_info(info)
            frame_idx = int(info["frame_idx"])
            sample_id = sample_id_from_batch(data)
            if args.eval_corruption:
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                base_data = corrupt_batch_images(
                    copy.deepcopy(data), args.corruption, args.corruption_severity
                )
            else:
                base_data = copy.deepcopy(data)

            frame_record = {
                "route": route,
                "frame_idx": frame_idx,
                "sample_id": sample_id,
                "modes": {},
            }
            gt_abs = None
            mask_np = None
            for mode in modes:
                mode_data = copy.deepcopy(base_data)
                mode_lookup = shuffled_lookup if mode == "shuffled" else None
                output = forward_film_training(
                    model,
                    mode_data,
                    lambda_col=0.0,
                    lambda_uq_consistency=0.0,
                    lambda_plan=0.0,
                    lambda_vae=0.0,
                    lambda_vlm=0.0,
                    lambda_ground=0.0,
                    uq_mode=mode,
                    shuffled_uq=get_shuffled_uq(mode_data, mode_lookup),
                    grounding_only=False,
                    token_input="score_direction",
                    conditioning="vision_adapter",
                )
                if output is None:
                    continue
                pred_abs = _as_abs_traj(output["planning_prediction"])
                gt_abs = _as_abs_traj(output["planning_target"])
                mask = output["planning_mask"].detach().float().cpu()
                while mask.ndim > 1:
                    mask = mask[0]
                mask_np = mask.numpy()
                target_score = float(output["target_score"].detach().float().mean().cpu())
                predicted_score = float(
                    output["predicted_score"].detach().float().mean().cpu()
                )
                frame_record["modes"][mode] = {
                    "pred": pred_abs.tolist(),
                    "ade": _trajectory_ade(pred_abs, gt_abs, mask_np),
                    "target_score": target_score,
                    "predicted_score": predicted_score,
                }
            if gt_abs is not None:
                frame_record["gt"] = gt_abs.tolist()
                frame_record["mask"] = mask_np.tolist()
                records_by_route.setdefault(route, []).append(frame_record)
            progress.update()
    return records_by_route


def _render_frame(route: str, frame_record: dict, route_summary: dict):
    trajectories = [np.asarray(frame_record["gt"], dtype=np.float32)]
    for mode in ("none", "shuffled", "correct"):
        if mode in frame_record["modes"]:
            trajectories.append(np.asarray(frame_record["modes"][mode]["pred"]))
    all_points = np.concatenate(trajectories, axis=0)
    max_abs = max(float(np.abs(all_points).max()), 2.0)
    xlim = max_abs + 1.0
    ylim_top = max(float(all_points[:, 1].max()) + 2.0, 6.0)
    ylim_bottom = min(float(all_points[:, 1].min()) - 1.0, -2.0)

    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=120)
    fig.patch.set_facecolor("#F7F7F2")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(ylim_bottom, ylim_top)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D7D7D0", linewidth=0.8, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.set_xlabel("lateral / m")
    ax.set_ylabel("forward / m")

    ego = Rectangle(
        (-EGO_WIDTH / 2, -EGO_LENGTH / 2),
        EGO_WIDTH,
        EGO_LENGTH,
        edgecolor="#111111",
        facecolor="#BDBDBD",
        linewidth=1.8,
        zorder=4,
    )
    ax.add_patch(ego)
    ax.annotate(
        "",
        xy=(0, EGO_LENGTH / 2 + 0.8),
        xytext=(0, EGO_LENGTH / 2),
        arrowprops=dict(arrowstyle="->", color="#111111", lw=2),
        zorder=5,
    )

    def plot_traj(points, color, label, marker, linestyle="-", linewidth=2.5):
        points = np.asarray(points, dtype=np.float32)
        ax.plot(
            points[:, 0],
            points[:, 1],
            linestyle,
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=3,
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=48,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )

    plot_traj(frame_record["gt"], COLORS["gt"], "GT", "s", linewidth=3.0)
    if "none" in frame_record["modes"]:
        plot_traj(
            frame_record["modes"]["none"]["pred"],
            COLORS["none"],
            "none",
            "o",
            linestyle="--",
        )
    if "shuffled" in frame_record["modes"]:
        plot_traj(
            frame_record["modes"]["shuffled"]["pred"],
            COLORS["shuffled"],
            "shuffled UQ",
            "^",
            linestyle="-.",
        )
    if "correct" in frame_record["modes"]:
        plot_traj(
            frame_record["modes"]["correct"]["pred"],
            COLORS["correct"],
            "correct UQ",
            "D",
            linewidth=3.0,
        )

    title = route.replace("_", " ")
    ax.set_title(f"{title}\nframe {frame_record['frame_idx']:05d}", fontsize=11)
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0.92)

    score = None
    if "correct" in frame_record["modes"]:
        score = frame_record["modes"]["correct"]["target_score"]
    lines = []
    if score is not None:
        lines.append(f"Density UQ score: {score:.3f}")
    for mode in ("none", "shuffled", "correct"):
        if mode in frame_record["modes"]:
            lines.append(f"{MODE_LABELS[mode]} ADE: {frame_record['modes'][mode]['ade']:.3f} m")
    if route_summary:
        lines.append("")
        lines.append(f"route mean ADE")
        lines.append(f"none: {route_summary.get('none', float('nan')):.3f}")
        lines.append(f"shuffled: {route_summary.get('shuffled', float('nan')):.3f}")
        lines.append(f"correct: {route_summary.get('correct', float('nan')):.3f}")

    ax.text(
        0.98,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#333333", alpha=0.92),
    )

    fig.tight_layout()
    fig.canvas.draw()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = buffer.reshape(height, width, 4)[:, :, :3].copy()
    plt.close(fig)
    return image


def _route_summary(records: list[dict]) -> dict[str, float]:
    summary = {}
    for mode in ("none", "shuffled", "correct"):
        values = [
            record["modes"][mode]["ade"]
            for record in records
            if mode in record["modes"] and np.isfinite(record["modes"][mode]["ade"])
        ]
        if values:
            summary[mode] = float(np.mean(values))
    return summary


def _render_gifs(records_by_route: dict[str, list[dict]], out_dir: Path, fps: int):
    gif_dir = out_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_paths = []
    summaries = {}
    for route, records in records_by_route.items():
        summary = _route_summary(records)
        summaries[route] = summary
        frames = [_render_frame(route, record, summary) for record in records]
        path = gif_dir / f"{route}.gif"
        imageio.mimsave(path, frames, fps=fps, loop=0)
        gif_paths.append(str(path))
        print(f"[GIF] wrote {path} ({len(frames)} frames)")
    return gif_paths, summaries


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, model, assignment = _build_model(args)
    dataset = build_dataset(cfg.data.test)
    filter_dataset_by_split(dataset, assignment, "calibration")
    selected_infos, counts = _select_route_infos(
        dataset.data_infos, args.routes, args.frames_per_route, args.frame_stride
    )
    dataset.data_infos = selected_infos
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f"[GIF] selected route counts: {json.dumps(counts, sort_keys=True)}")

    shuffled_lookup = None
    if "shuffled" in [item.strip() for item in args.modes.split(",")]:
        shuffled_lookup = build_shuffled_uq_lookup(
            args.descriptor_cache,
            args.density_checkpoint,
            assignment,
            "calibration",
            args.seed,
        )

    records_by_route = _collect_records(args, model, loader, selected_infos, shuffled_lookup)
    cache_path = out_dir / "trajectory_records.pt"
    torch.save(records_by_route, cache_path)
    gif_paths, summaries = _render_gifs(records_by_route, out_dir, args.fps)
    payload = {
        "routes": args.routes,
        "selected_counts": counts,
        "corruption": args.corruption if args.eval_corruption else None,
        "corruption_severity": args.corruption_severity if args.eval_corruption else None,
        "summaries": summaries,
        "gifs": gif_paths,
        "cache": str(cache_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[GIF] wrote {summary_path}")


if __name__ == "__main__":
    main()
