"""Train the explicit density-UQ token projector and LLM LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--eval-max-samples", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=None)
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
            if args.eval_max_samples is not None and index >= args.eval_max_samples:
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
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            if args.eval_max_samples is not None and index >= args.eval_max_samples:
                break
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
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
            )
            if output is None:
                continue
            predictions.append(output["planning_prediction"].cpu())
            targets.append(output["planning_target"].cpu())
            masks.append(output["planning_mask"].cpu())
            grounding_predictions.extend(
                output["predicted_score"].cpu().flatten().tolist()
            )
            grounding_targets.extend(
                output["target_score"].cpu().flatten().tolist()
            )
    model.train()
    return {
        "planning": trajectory_metrics(predictions, targets, masks),
        "grounding": grounding_metrics(
            grounding_predictions,
            grounding_targets,
        ),
    }


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
    cfg.model.use_uncertainty_l2 = False
    cfg.model.use_bev_uncertainty = False
    cfg.model.pts_bbox_head.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = args.density_checkpoint
    cfg.model.pts_bbox_head.transformer.use_uncertainty = False
    cfg.data.test.ann_file = args.ann_file

    dataset = build_dataset(cfg.data.test)
    split_size = filter_dataset_by_split(dataset, assignment, args.split)
    if args.max_samples is not None:
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
    if args.eval_max_samples is not None:
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
    print(
        f"[UQToken] calibration matched={eval_size}, "
        f"evaluation_samples={len(eval_dataset)}"
    )

    train_shuffled_lookup = None
    eval_shuffled_lookup = None
    if args.uq_mode == "shuffled" or args.counterfactual_grounding:
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
                "params": [parameter for _, parameter in groups["projector"]],
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
        label = "planning and grounding" if args.eval_planning else "grounding"
        print(f"[UQToken] intervention {label}: {json.dumps(results)}")
        return

    optimizer.zero_grad(set_to_none=True)
    gradient_audited = False
    for epoch in range(start_epoch, args.epochs):
        losses = []
        progress = ProgressBar(len(data_loader))
        for step, data in enumerate(data_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            output = forward_film_training(
                model,
                data,
                lambda_col=args.lambda_col,
                col_margin=args.col_margin,
                lambda_uq_consistency=args.lambda_consistency,
                lambda_plan=args.lambda_plan,
                lambda_vae=args.lambda_vae,
                lambda_vlm=args.lambda_vlm,
                lambda_ground=args.lambda_ground,
                uq_mode=args.uq_mode,
                shuffled_uq=get_shuffled_uq(data, train_shuffled_lookup),
                grounding_only=args.grounding_only,
                counterfactual_grounding=args.counterfactual_grounding,
                token_input=args.token_input,
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
                projector_has_grad = any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for _, parameter in groups["projector"]
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
                if (
                    (projector_required and not projector_has_grad)
                    or not lora_has_grad
                    or not grounding_has_grad
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
