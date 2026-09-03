"""Run per-sample UQ vision-adapter evaluation and stratify by UQ score."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, ProgressBar, load_checkpoint, set_random_seed

from scripts.train_film import custom_wrap_fp16_model, forward_film_training
from scripts.train_uq_token import (
    add_route_mean_planning,
    build_shuffled_uq_lookup,
    filter_dataset_by_split,
    get_shuffled_uq,
    route_from_info,
    sample_id_from_batch,
    select_route_balanced_samples,
    trajectory_metrics,
)
from uq_estimator.corruptions import corrupt_batch_images
from uq_estimator.density import get_uq_state_dict
from uq_estimator.training import freeze_for_uq_token_training, load_uq_token_weights


MODES = ("none", "shuffled", "correct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified UQ adapter eval")
    parser.add_argument("--config", default="adzoo/orion/configs/orion_stage3_infer.py")
    parser.add_argument("--checkpoint", default="ckpts/Orion.pth")
    parser.add_argument("--density-checkpoint", default="checkpoints/density_uq/best.pt")
    parser.add_argument("--descriptor-cache", default="data/density_uq/descriptors.pt")
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--init-adaptation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-route-samples", type=int, default=50)
    parser.add_argument("--eval-route-limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-corruption", action="store_true")
    parser.add_argument(
        "--corruption",
        choices=("blur", "dark", "camera_dropout"),
        default="camera_dropout",
    )
    parser.add_argument("--corruption-severity", type=int, default=1)
    parser.add_argument("--bins", choices=("median", "tertile"), default="median")
    return parser.parse_args()


def build_adapter_model(args: argparse.Namespace):
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
    print(f"[Stratified] loaded {loaded} adaptation tensors")

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


def sample_ade(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = prediction.detach().float().cpu().cumsum(dim=1)
    tgt = target.detach().float().cpu().cumsum(dim=1)
    valid = mask.detach().float().cpu()
    distance = torch.linalg.vector_norm(pred - tgt, dim=-1)
    denom = valid.sum().clamp_min(1.0)
    return float((distance * valid).sum() / denom)


def sample_fde(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = prediction.detach().float().cpu().cumsum(dim=1)
    tgt = target.detach().float().cpu().cumsum(dim=1)
    valid = mask.detach().float().cpu()
    distance = torch.linalg.vector_norm(pred - tgt, dim=-1)
    values = []
    for sample_distance, sample_mask in zip(distance, valid):
        indices = torch.nonzero(sample_mask > 0, as_tuple=False).flatten()
        if len(indices):
            values.append(sample_distance[indices[-1]])
    if not values:
        return float("nan")
    return float(torch.stack(values).mean())


def aggregate_records(records: list[dict]) -> dict:
    output = {}
    for mode in MODES:
        predictions = []
        targets = []
        masks = []
        for record in records:
            mode_record = record["modes"].get(mode)
            if mode_record is None:
                continue
            predictions.append(torch.tensor(mode_record["prediction"]).unsqueeze(0))
            targets.append(torch.tensor(record["target"]).unsqueeze(0))
            masks.append(torch.tensor(record["mask"]).unsqueeze(0))
        output[mode] = trajectory_metrics(predictions, targets, masks)
    return output


def assign_groups(records: list[dict], bins: str) -> dict[str, list[dict]]:
    scores = np.asarray([record["target_score"] for record in records], dtype=np.float64)
    if bins == "median":
        threshold = float(np.median(scores))
        groups = {"low_uq": [], "high_uq": []}
        for record in records:
            key = "high_uq" if record["target_score"] >= threshold else "low_uq"
            groups[key].append(record)
        return groups

    low, high = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    groups = {"low_uq": [], "mid_uq": [], "high_uq": []}
    for record in records:
        if record["target_score"] < low:
            groups["low_uq"].append(record)
        elif record["target_score"] < high:
            groups["mid_uq"].append(record)
        else:
            groups["high_uq"].append(record)
    return groups


def evaluate(args, model, data_loader, shuffled_lookup) -> tuple[list[dict], dict]:
    records = []
    route_records: dict[str, list[dict]] = {}
    progress = ProgressBar(len(data_loader))
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            route = sample_id_from_batch(data).split("__", 1)[0]
            sample_id = sample_id_from_batch(data)
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            base_data = copy.deepcopy(data)
            if args.eval_corruption:
                base_data = corrupt_batch_images(
                    base_data, args.corruption, args.corruption_severity
                )
            record = {"sample_id": sample_id, "route": route, "modes": {}}
            target = None
            mask = None
            target_score = None
            for mode in MODES:
                # Keep every stochastic operation paired across intervention
                # modes.  Seeding only once before this loop makes each mode
                # consume a different random stream and confounds the UQ
                # intervention with sampling noise.
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                mode_data = copy.deepcopy(base_data)
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
                    shuffled_uq=get_shuffled_uq(
                        mode_data, shuffled_lookup if mode == "shuffled" else None
                    ),
                    grounding_only=False,
                    token_input="score_direction",
                    conditioning="vision_adapter",
                )
                if output is None:
                    continue
                prediction = output["planning_prediction"].detach().float().cpu()
                target = output["planning_target"].detach().float().cpu()
                mask = output["planning_mask"].detach().float().cpu()
                target_score = float(output["target_score"].detach().float().mean().cpu())
                record["modes"][mode] = {
                    "ade": sample_ade(prediction, target, mask),
                    "fde": sample_fde(prediction, target, mask),
                    "prediction": prediction.squeeze(0).tolist(),
                }
            if set(record["modes"].keys()) == set(MODES) and target is not None:
                record["target"] = target.squeeze(0).tolist()
                record["mask"] = mask.squeeze(0).tolist()
                record["target_score"] = target_score
                records.append(record)
                route_records.setdefault(route, []).append(record)
            progress.update()
    return records, route_records


def summarize(args, records: list[dict], route_records: dict[str, list[dict]]) -> dict:
    planning = aggregate_records(records)
    by_route = {
        route: aggregate_records(route_items)
        for route, route_items in route_records.items()
    }
    nested = {
        mode: {
            "planning": planning[mode],
            "planning_by_route": {
                route: metrics[mode] for route, metrics in by_route.items()
            },
        }
        for mode in MODES
    }
    add_route_mean_planning(nested)

    groups = assign_groups(records, args.bins)
    stratified = {
        group: {"count": len(group_records), "modes": aggregate_records(group_records)}
        for group, group_records in groups.items()
        if group_records
    }

    return {
        "config": {
            "bins": args.bins,
            "corruption": args.corruption if args.eval_corruption else None,
            "corruption_severity": args.corruption_severity if args.eval_corruption else None,
            "eval_route_samples": args.eval_route_samples,
            "eval_route_limit": args.eval_route_limit,
            "seed": args.seed,
        },
        "count": len(records),
        "planning": planning,
        "planning_by_route": by_route,
        "route_mean_planning": {
            mode: nested[mode].get("route_mean_planning", {}) for mode in MODES
        },
        "stratified": stratified,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)
    cfg, model, assignment = build_adapter_model(args)

    dataset = build_dataset(cfg.data.test)
    filter_dataset_by_split(dataset, assignment, "calibration")
    route_counts = select_route_balanced_samples(
        dataset,
        args.eval_route_samples,
        args.eval_route_limit,
    )
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f"[Stratified] route-balanced counts: {json.dumps(route_counts)}")
    shuffled_lookup = build_shuffled_uq_lookup(
        args.descriptor_cache,
        args.density_checkpoint,
        assignment,
        "calibration",
        args.seed,
    )
    records, route_records = evaluate(args, model, loader, shuffled_lookup)
    payload = summarize(args, records, route_records)
    payload["route_balanced_counts"] = route_counts

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Stratified] wrote {output}")
    print(
        "[Stratified] summary: "
        + json.dumps(
            {
                "count": payload["count"],
                "planning": payload["planning"],
                "stratified": payload["stratified"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
