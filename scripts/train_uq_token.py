"""Train the explicit density-UQ token projector and LLM LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, ProgressBar, load_checkpoint, set_random_seed

from scripts.train_film import custom_wrap_fp16_model, forward_film_training
from uq_estimator.density import get_uq_state_dict
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
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
        if name.startswith("uq_token_projector.") or "lora_" in name
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
    print(
        f"[UQToken] split={args.split}, matched={split_size}, "
        f"training_samples={len(dataset)}"
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
                "token_norm": float(model.uq_token_norm),
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
                frozen_has_grad = any(
                    parameter.grad is not None
                    for name, parameter in model.named_parameters()
                    if not parameter.requires_grad
                )
                if not projector_has_grad or not lora_has_grad or frozen_has_grad:
                    raise RuntimeError(
                        "Gradient audit failed: "
                        f"projector={projector_has_grad}, "
                        f"lora={lora_has_grad}, frozen={frozen_has_grad}"
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
            print(
                f"\n[UQToken] epoch={epoch + 1}, loss={mean_loss:.6f}, "
                f"token_norm={float(model.uq_token_norm):.6f}"
            )
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


if __name__ == "__main__":
    main()
