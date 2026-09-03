"""Train the explicit density-UQ token projector and LLM LoRA adapters."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, ProgressBar, load_checkpoint, set_random_seed

from scripts.train_film import custom_wrap_fp16_model, forward_film_training
from uq_estimator.density import get_uq_state_dict
from uq_estimator.density import DensityUQEstimator
from uq_estimator.corruptions import corrupt_batch_images
from uq_estimator.training import (
    count_parameter_groups,
    freeze_for_uq_token_training,
    load_uq_token_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORION UQ tokens")
    parser.add_argument("--config", default="adzoo/orion/configs/orion_stage3_infer.py")
    parser.add_argument("--checkpoint", default="ckpts/Orion.pth")
    parser.add_argument("--density-checkpoint", default="checkpoints/density_uq/best.pt")
    parser.add_argument(
        "--descriptor-cache", default="data/density_uq/descriptors.pt"
    )
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--split", choices=("train", "calibration"), default="train")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-col", type=float, default=1.0)
    parser.add_argument("--col-margin", type=float, default=4.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-plan", type=float, default=1.0)
    parser.add_argument("--lambda-vae", type=float, default=0.1)
    parser.add_argument("--lambda-vlm", type=float, default=0.001)
    parser.add_argument("--lambda-ground", type=float, default=1.0)
    parser.add_argument("--lambda-pair", type=float, default=0.1)
    parser.add_argument("--lambda-pair-rank", type=float, default=0.1)
    parser.add_argument("--pair-rank-margin", type=float, default=0.01)
    parser.add_argument("--paired-corruption", action="store_true")
    parser.add_argument("--counterfactual-pair", action="store_true")
    parser.add_argument(
        "--corruption",
        choices=("blur", "dark", "camera_dropout"),
        default="blur",
    )
    parser.add_argument("--corruption-severity", type=int, default=2)
    parser.add_argument("--eval-corruption", action="store_true")
    parser.add_argument(
        "--uq-mode",
        choices=("correct", "zero", "shuffled", "none"),
        default="correct",
    )
    parser.add_argument("--grounding-only", action="store_true")
    parser.add_argument("--counterfactual-grounding", action="store_true")
    parser.add_argument(
        "--token-input",
        choices=("score_direction", "score_only"),
        default="score_direction",
    )
    parser.add_argument(
        "--conditioning",
        choices=("token", "vision_adapter"),
        default="token",
    )
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument(
        "--eval-route-balanced",
        action="store_true",
        help="evaluate an equal number of calibration samples from each route",
    )
    parser.add_argument(
        "--eval-route-samples",
        type=int,
        default=50,
        help="samples per route for --eval-route-balanced",
    )
    parser.add_argument(
        "--eval-route-limit",
        type=int,
        default=None,
        help="maximum number of calibration routes for route-balanced eval",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="optional path to save intervention evaluation JSON",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-route-balanced", action="store_true")
    parser.add_argument(
        "--train-route-samples",
        type=int,
        default=100,
        help="uniformly spaced samples per route for route-balanced training",
    )
    parser.add_argument(
        "--train-route-limit",
        type=int,
        default=None,
        help="optional maximum number of routes for route-balanced training",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-adaptation",
        default=None,
        help="load adaptation weights but start with a fresh optimizer",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-planning",
        action="store_true",
        help="include trajectory ADE/FDE during evaluation",
    )
    parser.add_argument(
        "--intervention-eval-after-train",
        action="store_true",
    )
    parser.add_argument(
        "--eval-modes",
        default="none,zero,shuffled,correct",
        help="comma-separated UQ modes for --eval-only",
    )
    parser.add_argument("--out", default="checkpoints/uq_token/best.pt")
    parser.add_argument(
        "--lambda-clean-preservation",
        type=float,
        default=0.0,
        help="keep clean conditioned planning features close to none-mode features",
    )
    return parser.parse_args()


def route_from_info(info: dict) -> str:
    folder = str(info.get("folder", "")).replace("\\", "/").rstrip("/")
    if not folder:
        raise KeyError("Dataset info has no route folder")
    return folder.rsplit("/", 1)[-1]


def filter_dataset_by_split(dataset, assignment: dict[str, str], split: str) -> int:
    selected = [
        info for info in dataset.data_infos
        if assignment.get(route_from_info(info)) == split
    ]
    if not selected:
        raise RuntimeError(f"No dataset samples matched route split {split!r}")
    dataset.data_infos = selected
    dataset.flag = np.zeros(len(selected), dtype=np.uint8)
    return len(selected)


def select_route_balanced_samples(
    dataset,
    samples_per_route: int,
    route_limit: int | None = None,
    sampling: str = "head",
) -> dict[str, int]:
    route_samples: dict[str, list[dict]] = {}
    for info in dataset.data_infos:
        route = route_from_info(info)
        if route not in route_samples:
            if route_limit is not None and len(route_samples) >= route_limit:
                continue
            route_samples[route] = []
        route_samples[route].append(info)

    if sampling not in {"head", "uniform"}:
        raise ValueError(f"Unknown route sampling mode: {sampling}")
    for route, infos in route_samples.items():
        if len(infos) <= samples_per_route:
            continue
        if sampling == "head":
            route_samples[route] = infos[:samples_per_route]
        else:
            indices = np.linspace(
                0, len(infos) - 1, samples_per_route, dtype=np.int64
            )
            route_samples[route] = [infos[int(index)] for index in indices]

    selected = [
        info
        for route in route_samples
        for info in route_samples[route]
    ]
    if not selected:
        raise RuntimeError("No samples selected for route-balanced evaluation")
    dataset.data_infos = selected
    dataset.flag = np.zeros(len(selected), dtype=np.uint8)
    return {route: len(items) for route, items in route_samples.items()}


def add_route_mean_planning(results: dict[str, dict]) -> dict[str, dict]:
    for mode, result in results.items():
        by_route = result.get("planning_by_route", {})
        valid = [
            metrics for metrics in by_route.values()
            if metrics.get("count", 0) > 0
        ]
        if not valid:
            continue
        result["route_mean_planning"] = {
            "routes": len(valid),
            "count": int(sum(metrics.get("count", 0) for metrics in valid)),
            "ade": float(np.mean([metrics["ade"] for metrics in valid])),
            "fde": float(np.mean([metrics["fde"] for metrics in valid])),
        }
    return results


def sample_id_from_batch(data: dict) -> str:
    value = data["img_metas"]
    while isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "data"):
        value = value.data
        while isinstance(value, (list, tuple)):
            value = value[0]
    filenames = value.get("filename", [])
    if not filenames:
        raise KeyError("img_metas contains no filename")
    parts = str(filenames[0]).replace("\\", "/").split("/")
    route = parts[-4]
    frame = Path(parts[-1]).stem
    return f"{route}__{frame}.pt"


def build_shuffled_uq_lookup(
    descriptor_cache: str,
    density_checkpoint: str,
    assignment: dict[str, str],
    split: str,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cache = torch.load(descriptor_cache, map_location="cpu", weights_only=True)
    model = DensityUQEstimator.from_checkpoint(density_checkpoint).eval()
    filenames = cache["filenames"]
    routes = cache["routes"]
    selected = [
        index for index, route in enumerate(routes)
        if assignment.get(route) == split
    ]
    descriptors = cache["descriptors"][selected].float()
    active_parts = []
    score_parts = []
    with torch.no_grad():
        for chunk in descriptors.split(512):
            _, score, _, active = model.encode_descriptor(chunk)
            active_parts.append(active)
            score_parts.append(score)
    active = torch.cat(active_parts)
    score = torch.cat(score_parts)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(selected), generator=generator)
    source = torch.roll(order, shifts=1)
    lookup = {}
    for target_position, source_position in zip(order.tolist(), source.tolist()):
        filename = filenames[selected[target_position]]
        lookup[filename] = (
            active[source_position].clone(),
            score[source_position].clone(),
        )
    return lookup


def get_shuffled_uq(
    data: dict,
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if lookup is None:
        return None
    sample_id = sample_id_from_batch(data)
    active, score = lookup[sample_id]
    return active.unsqueeze(0).cuda(), score.unsqueeze(0).cuda()


def set_pair_forward_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grounding_metrics(
    predictions: list[float],
    targets: list[float],
) -> dict[str, float]:
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if len(predicted) < 2:
        return {
            "count": int(len(predicted)),
            "mae": float(np.abs(predicted - target).mean()) if len(predicted) else float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    return {
        "count": int(len(predicted)),
        "mae": float(np.abs(predicted - target).mean()),
        "pearson": float(pearsonr(predicted, target).statistic),
        "spearman": float(spearmanr(predicted, target).statistic),
    }


def trajectory_metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> dict[str, float]:
    if not predictions:
        return {"count": 0, "ade": float("nan"), "fde": float("nan")}

    prediction = torch.cat(predictions).float().cumsum(dim=1)
    target = torch.cat(targets).float().cumsum(dim=1)
    mask = torch.cat(masks).float()
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    valid_steps = mask.sum()
    ade = (distance * mask).sum() / valid_steps.clamp_min(1.0)

    fde_values = []
    for sample_distance, sample_mask in zip(distance, mask):
        valid = torch.nonzero(sample_mask > 0, as_tuple=False).flatten()
        if len(valid):
            fde_values.append(sample_distance[valid[-1]])
    fde = (
        torch.stack(fde_values).mean()
        if fde_values else torch.tensor(float("nan"))
    )
    return {
        "count": int(prediction.shape[0]),
        "ade": float(ade),
        "fde": float(fde),
    }


def evaluate_grounding(
    model,
    data_loader,
    args,
    shuffled_lookup=None,
    uq_mode=None,
) -> dict[str, float]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            if (
                not args.eval_route_balanced
                and args.eval_max_samples is not None
                and index >= args.eval_max_samples
            ):
                break
            shuffled_uq = get_shuffled_uq(data, shuffled_lookup)
            output = forward_film_training(
                model,
                data,
                lambda_uq_consistency=0.0,
                lambda_plan=0.0,
                lambda_vae=0.0,
                lambda_vlm=0.0,
                lambda_ground=1.0,
                uq_mode=uq_mode or args.uq_mode,
                shuffled_uq=shuffled_uq,
                grounding_only=True,
                token_input=args.token_input,
                conditioning=args.conditioning,
            )
            if output is None:
                continue
            predictions.extend(output["predicted_score"].cpu().flatten().tolist())
            targets.extend(output["target_score"].cpu().flatten().tolist())
    model.train()
    return grounding_metrics(predictions, targets)


def evaluate_planning(
    model,
    data_loader,
    args,
    shuffled_lookup=None,
    uq_mode=None,
) -> dict[str, dict[str, float]]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    grounding_predictions: list[float] = []
    grounding_targets: list[float] = []
    route_records: dict[str, dict[str, list[torch.Tensor]]] = {}
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            if (
                not args.eval_route_balanced
                and args.eval_max_samples is not None
                and index >= args.eval_max_samples
            ):
                break
            route = sample_id_from_batch(data).split("__", 1)[0]
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            if args.eval_corruption:
                data = corrupt_batch_images(
                    data, args.corruption, args.corruption_severity
                )
            output = forward_film_training(
                model,
                data,
                lambda_col=0.0,
                lambda_uq_consistency=0.0,
                lambda_plan=0.0,
                lambda_vae=0.0,
                lambda_vlm=0.0,
                lambda_ground=0.0,
                uq_mode=uq_mode or args.uq_mode,
                shuffled_uq=get_shuffled_uq(data, shuffled_lookup),
                grounding_only=False,
                token_input=args.token_input,
                conditioning=args.conditioning,
            )
            if output is None:
                continue
            predictions.append(output["planning_prediction"].cpu())
            targets.append(output["planning_target"].cpu())
            masks.append(output["planning_mask"].cpu())
            route_entry = route_records.setdefault(
                route, {"predictions": [], "targets": [], "masks": []}
            )
            route_entry["predictions"].append(
                output["planning_prediction"].cpu()
            )
            route_entry["targets"].append(output["planning_target"].cpu())
            route_entry["masks"].append(output["planning_mask"].cpu())
            grounding_predictions.extend(
                output["predicted_score"].cpu().flatten().tolist()
            )
            grounding_targets.extend(
                output["target_score"].cpu().flatten().tolist()
            )
    model.train()
    result = {
        "planning": trajectory_metrics(predictions, targets, masks),
        "grounding": grounding_metrics(
            grounding_predictions,
            grounding_targets,
        ),
    }
    result["planning_by_route"] = {
        route: trajectory_metrics(
            values["predictions"],
            values["targets"],
            values["masks"],
        )
        for route, values in route_records.items()
    }
    return result


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_loss: float,
    args: argparse.Namespace,
    history: list[dict],
) -> None:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("uq_token_projector.")
        or name.startswith("uq_vision_adapter.")
        or name.startswith("uq_grounding_head.")
        or "lora_" in name
    }
    payload = {
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "args": vars(args),
        "history": history,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_adaptation_checkpoint(path: str, model, optimizer=None, scheduler=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    load_uq_token_weights(model, payload)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def main() -> None:
    args = parse_args()
    if args.init_adaptation and args.resume:
        raise ValueError("--init-adaptation and --resume are mutually exclusive")
    set_random_seed(args.seed, deterministic=True)

    density_payload = torch.load(
        args.density_checkpoint, map_location="cpu", weights_only=True
    )
    assignment = density_payload["split_assignment"]

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.frozen = True
    cfg.model.use_lora = True
    cfg.model.use_uq_token = True
    cfg.model.use_uq_vision_adapter = (
        args.conditioning == "vision_adapter"
    )
    cfg.model.use_uncertainty_l2 = False
    cfg.model.use_bev_uncertainty = False
    cfg.model.pts_bbox_head.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = args.density_checkpoint
    cfg.model.pts_bbox_head.transformer.use_uncertainty = False
    cfg.data.test.ann_file = args.ann_file

    dataset = build_dataset(cfg.data.test)
    split_size = filter_dataset_by_split(dataset, assignment, args.split)
    train_route_balanced_counts = None
    if args.train_route_balanced:
        if args.max_samples is not None:
            raise ValueError(
                "--max-samples cannot be combined with --train-route-balanced"
            )
        train_route_balanced_counts = select_route_balanced_samples(
            dataset,
            args.train_route_samples,
            args.train_route_limit,
            sampling="uniform",
        )
    elif args.max_samples is not None:
        dataset.data_infos = dataset.data_infos[:args.max_samples]
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=True,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    eval_dataset = build_dataset(cfg.data.test)
    eval_size = filter_dataset_by_split(eval_dataset, assignment, "calibration")
    route_balanced_counts = None
    if args.eval_route_balanced:
        route_balanced_counts = select_route_balanced_samples(
            eval_dataset,
            args.eval_route_samples,
            args.eval_route_limit,
        )
    elif args.eval_offset:
        eval_dataset.data_infos = eval_dataset.data_infos[args.eval_offset:]
        eval_dataset.flag = np.zeros(len(eval_dataset), dtype=np.uint8)
    if args.eval_max_samples is not None and not args.eval_route_balanced:
        eval_dataset.data_infos = eval_dataset.data_infos[:args.eval_max_samples]
        eval_dataset.flag = np.zeros(len(eval_dataset), dtype=np.uint8)
    eval_loader = build_dataloader(
        eval_dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(
        f"[UQToken] split={args.split}, matched={split_size}, "
        f"training_samples={len(dataset)}"
    )
    if train_route_balanced_counts is not None:
        print(
            f"[UQToken] route-balanced train: "
            f"{json.dumps(train_route_balanced_counts)}"
        )
    print(
        f"[UQToken] calibration matched={eval_size}, "
        f"evaluation_samples={len(eval_dataset)}"
    )
    if route_balanced_counts is not None:
        print(
            f"[UQToken] route-balanced eval: "
            f"{json.dumps(route_balanced_counts)}"
        )

    train_shuffled_lookup = None
    eval_shuffled_lookup = None
    if (
        args.uq_mode == "shuffled"
        or args.counterfactual_grounding
        or args.counterfactual_pair
    ):
        train_shuffled_lookup = build_shuffled_uq_lookup(
            args.descriptor_cache,
            args.density_checkpoint,
            assignment,
            args.split,
            args.seed,
        )
        eval_shuffled_lookup = build_shuffled_uq_lookup(
            args.descriptor_cache,
            args.density_checkpoint,
            assignment,
            "calibration",
            args.seed,
        )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        custom_wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.pts_bbox_head.uq_estimator.load_state_dict(
        get_uq_state_dict(density_payload), strict=False
    )
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]

    groups = freeze_for_uq_token_training(model)
    counts = count_parameter_groups(groups)
    print(f"[UQToken] trainable parameters: {json.dumps(counts)}")
    for group, parameters in groups.items():
        print(f"[UQToken] {group}: {len(parameters)} tensors")
    if args.init_adaptation:
        loaded = load_uq_token_weights(model, args.init_adaptation)
        print(
            f"[UQToken] initialized {loaded} adaptation tensors from "
            f"{args.init_adaptation}"
        )

    model.cuda()
    model.train()
    model.pts_bbox_head.with_dn = False
    model.pts_bbox_head.uq_estimator.eval()
    if hasattr(model.lm_head, "gradient_checkpointing_disable"):
        model.lm_head.gradient_checkpointing_disable()

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for _, parameter in (
                        groups.get("projector", [])
                        + groups.get("vision_adapter", [])
                    )
                ],
                "lr": args.lr_projector,
            },
            {
                "params": [parameter for _, parameter in groups["lora"]],
                "lr": args.lr_lora,
            },
            {
                "params": [parameter for _, parameter in groups["grounding"]],
                "lr": args.lr_projector,
            },
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * max(len(data_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min(args.lr_projector, args.lr_lora) * 0.01
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    history: list[dict] = []
    if args.resume:
        payload = load_adaptation_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("step", 0))
        best_loss = float(payload.get("best_loss", best_loss))
        history = list(payload.get("history", history))

    if args.eval_only:
        results = {}
        for mode in [item.strip() for item in args.eval_modes.split(",") if item.strip()]:
            if mode not in {"none", "zero", "shuffled", "correct"}:
                raise ValueError(f"Unsupported evaluation UQ mode: {mode}")
            mode_lookup = None
            if mode == "shuffled":
                mode_lookup = build_shuffled_uq_lookup(
                    args.descriptor_cache,
                    args.density_checkpoint,
                    assignment,
                    "calibration",
                    args.seed,
                )
            if args.eval_planning:
                results[mode] = evaluate_planning(
                    model,
                    eval_loader,
                    args,
                    shuffled_lookup=mode_lookup,
                    uq_mode=mode,
                )
            else:
                results[mode] = evaluate_grounding(
                    model,
                    eval_loader,
                    args,
                    shuffled_lookup=mode_lookup,
                    uq_mode=mode,
                )
        if args.eval_planning:
            results = add_route_mean_planning(results)
        label = "planning and grounding" if args.eval_planning else "grounding"
        payload = {
            "label": label,
            "eval_route_balanced": bool(args.eval_route_balanced),
            "route_balanced_counts": route_balanced_counts,
            "results": results,
        }
        print(f"[UQToken] intervention {label}: {json.dumps(results)}")
        if args.eval_output:
            output_path = Path(args.eval_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"[UQToken] wrote evaluation results to {output_path}")
        return

    optimizer.zero_grad(set_to_none=True)
    gradient_audited = False
    for epoch in range(start_epoch, args.epochs):
        losses = []
        progress = ProgressBar(len(data_loader))
        for step, data in enumerate(data_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            paired_log = {}
            if args.paired_corruption:
                pair_seed = args.seed + global_step
                clean_baseline_feature = None
                if args.lambda_clean_preservation > 0:
                    set_pair_forward_seed(pair_seed)
                    with torch.no_grad():
                        clean_baseline_output = forward_film_training(
                            model,
                            copy.deepcopy(data),
                            lambda_col=0.0,
                            lambda_uq_consistency=0.0,
                            lambda_plan=0.0,
                            lambda_vae=0.0,
                            lambda_vlm=0.0,
                            lambda_ground=0.0,
                            uq_mode="none",
                            grounding_only=args.grounding_only,
                            token_input=args.token_input,
                            conditioning=args.conditioning,
                        )
                    if clean_baseline_output is not None:
                        clean_baseline_feature = clean_baseline_output[
                            "planning_feature_raw"
                        ].detach()
                    del clean_baseline_output
                set_pair_forward_seed(pair_seed)
                clean_output = forward_film_training(
                    model,
                    copy.deepcopy(data),
                    lambda_col=0.0,
                    lambda_uq_consistency=0.0,
                    lambda_plan=0.0,
                    lambda_vae=0.0,
                    lambda_vlm=0.0,
                    lambda_ground=0.0,
                    uq_mode=args.uq_mode,
                    shuffled_uq=get_shuffled_uq(
                        data, train_shuffled_lookup
                    ),
                    grounding_only=args.grounding_only,
                    token_input=args.token_input,
                    conditioning=args.conditioning,
                )
                if clean_output is None:
                    progress.update()
                    continue
                clean_reference_feature = clean_output[
                    "planning_feature_raw"
                ].detach()
                clean_reference_loss = clean_output["loss"].detach()
                clean_reference_score = clean_output[
                    "target_score"
                ].detach()
                clean_preservation_loss = clean_output["loss"].new_zeros(())
                if clean_baseline_feature is not None:
                    clean_preservation_loss = (
                        clean_output["planning_feature_raw"]
                        - clean_baseline_feature
                    ).float().pow(2).mean()
                    (
                        args.lambda_clean_preservation
                        * clean_preservation_loss
                        / args.grad_accum
                    ).backward()
                del clean_output

                shuffled_reference_feature = None
                if args.counterfactual_pair and not args.grounding_only:
                    set_pair_forward_seed(pair_seed)
                    shuffled_output = forward_film_training(
                        model,
                        corrupt_batch_images(
                            data,
                            args.corruption,
                            args.corruption_severity,
                        ),
                        lambda_col=0.0,
                        lambda_uq_consistency=0.0,
                        lambda_plan=0.0,
                        lambda_vae=0.0,
                        lambda_vlm=0.0,
                        lambda_ground=0.0,
                        uq_mode="shuffled",
                        shuffled_uq=get_shuffled_uq(
                            data, train_shuffled_lookup
                        ),
                        grounding_only=False,
                        token_input=args.token_input,
                        conditioning=args.conditioning,
                    )
                    shuffled_reference_feature = shuffled_output[
                        "planning_feature_raw"
                    ].detach()
                    del shuffled_output

                corrupted_data = corrupt_batch_images(
                    data, args.corruption, args.corruption_severity
                )
                set_pair_forward_seed(pair_seed)
                output = forward_film_training(
                    model,
                    corrupted_data,
                    lambda_col=args.lambda_col,
                    col_margin=args.col_margin,
                    lambda_uq_consistency=args.lambda_consistency,
                    lambda_plan=args.lambda_plan,
                    lambda_vae=args.lambda_vae,
                    lambda_vlm=args.lambda_vlm,
                    lambda_ground=args.lambda_ground,
                    uq_mode=args.uq_mode,
                    shuffled_uq=get_shuffled_uq(
                        data, train_shuffled_lookup
                    ),
                    grounding_only=args.grounding_only,
                    counterfactual_grounding=args.counterfactual_grounding,
                    token_input=args.token_input,
                    conditioning=args.conditioning,
                )
                if output is None:
                    progress.update()
                    continue
                if args.grounding_only:
                    pair_loss = output["loss"].new_zeros(())
                    pair_rank_loss = output["loss"].new_zeros(())
                else:
                    correct_difference = (
                        output["planning_feature_raw"]
                        - clean_reference_feature
                    ).float().pow(2).mean()
                    pair_loss = correct_difference
                    pair_rank_loss = output["loss"].new_zeros(())
                    if args.counterfactual_pair:
                        shuffled_difference = (
                            shuffled_reference_feature
                            - clean_reference_feature
                        ).float().pow(2).mean()
                        shuffled_distance = shuffled_difference
                        pair_rank_loss = torch.relu(
                            pair_loss
                            - shuffled_distance.detach()
                            + args.pair_rank_margin
                        )
                loss = (
                    output["loss"]
                    + args.lambda_pair * pair_loss
                    + args.lambda_pair_rank * pair_rank_loss
                )
                paired_log = {
                    "clean_reference": float(clean_reference_loss),
                    "corrupted_total": float(output["loss"].detach()),
                    "pair_consistency": float(pair_loss.detach()),
                    "pair_rank": float(pair_rank_loss.detach()),
                    "clean_preservation": float(
                        clean_preservation_loss.detach()
                    ),
                    "weighted_clean_preservation": float(
                        args.lambda_clean_preservation
                        * clean_preservation_loss.detach()
                    ),
                    "clean_uq_score": float(
                        clean_reference_score.mean()
                    ),
                    "corrupted_uq_score": float(
                        output["target_score"].mean()
                    ),
                }
            else:
                output = forward_film_training(
                    model,
                    copy.deepcopy(data),
                    lambda_col=args.lambda_col,
                    col_margin=args.col_margin,
                    lambda_uq_consistency=args.lambda_consistency,
                    lambda_plan=args.lambda_plan,
                    lambda_vae=args.lambda_vae,
                    lambda_vlm=args.lambda_vlm,
                    lambda_ground=args.lambda_ground,
                    uq_mode=args.uq_mode,
                    shuffled_uq=get_shuffled_uq(
                        data, train_shuffled_lookup
                    ),
                    grounding_only=args.grounding_only,
                    counterfactual_grounding=args.counterfactual_grounding,
                    token_input=args.token_input,
                    conditioning=args.conditioning,
                )
                if output is None:
                    progress.update()
                    continue
                loss = output["loss"]
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
            step_record = {
                "epoch": epoch + 1,
                "step": global_step + 1,
                **{
                    key: float(value)
                    for key, value in output.get("log_vars", {}).items()
                },
                **paired_log,
                "token_norm": float(getattr(model, "uq_token_norm", 0.0)),
                "lr_projector": optimizer.param_groups[0]["lr"],
                "lr_lora": optimizer.param_groups[1]["lr"],
            }
            history.append(step_record)
            if (
                args.log_interval > 0
                and (global_step + 1) % args.log_interval == 0
            ) or global_step == 0:
                print(f"\n[UQToken] step {global_step + 1}: {json.dumps(step_record)}")

            if not gradient_audited:
                conditioning_parameters = (
                    groups.get("projector", [])
                    + groups.get("vision_adapter", [])
                )
                projector_has_grad = any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for _, parameter in conditioning_parameters
                )
                lora_has_grad = any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for _, parameter in groups["lora"]
                )
                grounding_has_grad = any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for _, parameter in groups["grounding"]
                )
                frozen_has_grad = any(
                    parameter.grad is not None
                    for name, parameter in model.named_parameters()
                    if not parameter.requires_grad
                )
                projector_required = args.uq_mode != "none"
                grounding_required = args.lambda_ground > 0
                if (
                    (projector_required and not projector_has_grad)
                    or not lora_has_grad
                    or (grounding_required and not grounding_has_grad)
                    or frozen_has_grad
                ):
                    raise RuntimeError(
                        "Gradient audit failed: "
                        f"projector={projector_has_grad}, "
                        f"lora={lora_has_grad}, "
                        f"grounding={grounding_has_grad}, "
                        f"frozen={frozen_has_grad}"
                    )
                print("[UQToken] gradient audit passed")
                gradient_audited = True

            if (global_step + 1) % args.grad_accum == 0:
                trainable = [
                    parameter
                    for parameters in groups.values()
                    for _, parameter in parameters
                ]
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            global_step += 1
            progress.update()

        if losses:
            mean_loss = float(np.mean(losses))
            validation = evaluate_grounding(
                model,
                eval_loader,
                args,
                shuffled_lookup=eval_shuffled_lookup,
            )
            print(
                f"\n[UQToken] epoch={epoch + 1}, loss={mean_loss:.6f}, "
                f"token_norm={float(getattr(model, 'uq_token_norm', 0.0)):.6f}, "
                f"grounding={json.dumps(validation)}"
            )
            history.append({
                "epoch": epoch + 1,
                "validation_grounding": validation,
            })
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_checkpoint(
                    args.out,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    best_loss,
                    args,
                    history,
                )
                print(f"[UQToken] saved {args.out}")
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(f"[UQToken] complete, best_loss={best_loss:.6f}")
    if args.intervention_eval_after_train:
        intervention = {}
        for mode in ("none", "zero", "shuffled", "correct"):
            mode_lookup = None
            if mode == "shuffled":
                mode_lookup = build_shuffled_uq_lookup(
                    args.descriptor_cache,
                    args.density_checkpoint,
                    assignment,
                    "calibration",
                    args.seed,
                )
            intervention[mode] = evaluate_grounding(
                model,
                eval_loader,
                args,
                shuffled_lookup=mode_lookup,
                uq_mode=mode,
            )
        print(
            f"[UQToken] post-train intervention grounding: "
            f"{json.dumps(intervention)}"
        )


if __name__ == "__main__":
    main()
