"""Align ORION language output with explicit counterfactual Risk QA targets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, ProgressBar, load_checkpoint, set_random_seed
from mmcv.datasets.data_utils.constants import DEFAULT_IMAGE_TOKEN
from mmcv.datasets.data_utils.data_utils import preprocess

from scripts.eval_risk_qa import (
    apply_uq_mode,
    info_sample_id,
    prepare_visual_context,
)
from scripts.train_film import custom_wrap_fp16_model
from scripts.train_uq_token import (
    build_shuffled_uq_lookup,
    filter_dataset_by_split,
    sample_id_from_batch,
)
from uq_estimator.density import get_uq_state_dict
from uq_estimator.density import DensityUQEstimator
from uq_estimator.risk_qa import (
    RISK_QA_QUESTION,
    RELIABILITY_QA_QUESTION,
    build_risk_qa_answer,
    mask_to_final_supervised_span,
    render_natural_risk_qa_answer,
    render_critical_object_context,
    render_reliability_answer,
    render_risk_synthesis_answer,
    render_risk_qa_answer,
    select_balanced_sample_ids,
    select_critical_objects,
)
from uq_estimator.training import load_uq_token_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="adzoo/orion/configs/orion_stage3_infer.py"
    )
    parser.add_argument("--checkpoint", default="ckpts/Orion.pth")
    parser.add_argument("--init-adaptation", default=None)
    parser.add_argument(
        "--density-checkpoint", default="checkpoints/density_uq/best.pt"
    )
    parser.add_argument(
        "--descriptor-cache", default="data/density_uq/descriptors.pt"
    )
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--balanced-per-level", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-projector", type=float, default=5e-5)
    parser.add_argument("--lr-lora", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--answer-style",
        choices=("structured", "natural", "level_only", "synthesis"),
        default="level_only",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def risk_qa_tokens(
    tokenizer,
    answer: str,
    answer_style: str,
    critical_objects=(),
    reliability_answer: str | None = None,
):
    if answer_style == "synthesis":
        sources = [[
            {
                "from": "human",
                "value": (
                    f"{DEFAULT_IMAGE_TOKEN}\nYou are driving a car. "
                    "Identify the critical objects in the scene."
                ),
            },
            {
                "from": "gpt",
                "value": render_critical_object_context(critical_objects),
            },
            {"from": "human", "value": RELIABILITY_QA_QUESTION},
            {"from": "gpt", "value": reliability_answer},
            {
                "from": "human",
                "value": (
                    "Combine the critical-object facts and visual reliability "
                    "into one concise risk summary. Do not add new objects."
                ),
            },
            {"from": "gpt", "value": answer},
        ]]
        converted = preprocess(sources, tokenizer, has_image=True)
        labels = mask_to_final_supervised_span(converted["labels"][0])
        return converted["input_ids"][0], labels
    if answer_style == "level_only":
        task = RELIABILITY_QA_QUESTION
    else:
        task = (
            f"{RISK_QA_QUESTION}\n"
            "State the visual reliability level, list critical objects and "
            "their relative positions, then give a short risk assessment."
        )
    question = (
        f"{DEFAULT_IMAGE_TOKEN}\nYou are driving a car. {task}"
    )
    sources = [[
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer},
    ]]
    converted = preprocess(sources, tokenizer, has_image=True)
    return converted["input_ids"][0], converted["labels"][0]


def freeze_for_risk_qa(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    groups = {"projector": [], "lora": []}
    for name, parameter in model.named_parameters():
        if name.startswith("uq_token_projector."):
            groups["projector"].append((name, parameter))
        elif "lora_" in name:
            groups["lora"].append((name, parameter))
        else:
            continue
        parameter.requires_grad = True
    if not groups["projector"] or not groups["lora"]:
        raise RuntimeError("Risk QA requires projector and LoRA parameters")
    return groups


def language_loss(
    model,
    vision,
    tokenizer,
    answer,
    answer_style,
    critical_objects=(),
    reliability_answer=None,
):
    input_ids, labels = risk_qa_tokens(
        tokenizer,
        answer,
        answer_style,
        critical_objects=critical_objects,
        reliability_answer=reliability_answer,
    )
    input_ids = input_ids.unsqueeze(0).cuda()
    labels = labels.unsqueeze(0).cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id or 0)
    output = model.lm_head(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        images=vision,
        use_cache=False,
    )
    return output.loss


def save_checkpoint(path, model, optimizer, args, step, history):
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("uq_token_projector.") or "lora_" in name
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state,
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "history": history,
        },
        output,
    )


def build_density_score_lookup(
    descriptor_cache: str,
    density_checkpoint,
    allowed_sample_ids: set[str],
) -> dict[str, float]:
    cache = torch.load(
        descriptor_cache, map_location="cpu", weights_only=True
    )
    density = DensityUQEstimator.from_checkpoint(density_checkpoint).eval()
    selected = [
        index for index, filename in enumerate(cache["filenames"])
        if filename in allowed_sample_ids
    ]
    score_lookup = {}
    with torch.no_grad():
        for start in range(0, len(selected), 512):
            indices = selected[start:start + 512]
            _, scores, _, _ = density.encode_descriptor(
                cache["descriptors"][indices].float()
            )
            for index, score in zip(indices, scores.flatten().tolist()):
                score_lookup[cache["filenames"][index]] = float(score)
    return score_lookup


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)
    density_payload = torch.load(
        args.density_checkpoint, map_location="cpu", weights_only=True
    )
    assignment = density_payload["split_assignment"]
    with open(args.ann_file, "rb") as handle:
        infos = pickle.load(handle)
    info_lookup = {info_sample_id(info): info for info in infos}

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
    matched = filter_dataset_by_split(dataset, assignment, args.split)
    if args.balanced_per_level is not None:
        info_by_name = {
            info_sample_id(info): info for info in dataset.data_infos
        }
        score_lookup = build_density_score_lookup(
            args.descriptor_cache,
            density_payload,
            set(info_by_name),
        )
        selected_ids, level_counts = select_balanced_sample_ids(
            score_lookup,
            args.balanced_per_level,
            args.seed,
        )
        dataset.data_infos = [info_by_name[name] for name in selected_ids]
        print(f"[RiskQA] balanced levels: {json.dumps(level_counts)}")
    else:
        dataset.data_infos = dataset.data_infos[:args.max_samples]
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=True,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    shuffled_lookup = build_shuffled_uq_lookup(
        args.descriptor_cache,
        args.density_checkpoint,
        assignment,
        args.split,
        args.seed,
    )
    print(f"[RiskQA] matched={matched}, training_samples={len(dataset)}")

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        custom_wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.pts_bbox_head.uq_estimator.load_state_dict(
        get_uq_state_dict(density_payload), strict=False
    )
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    if args.init_adaptation:
        loaded = load_uq_token_weights(model, args.init_adaptation)
        print(f"[RiskQA] initialized {loaded} adaptation tensors")

    groups = freeze_for_risk_qa(model)
    print(
        "[RiskQA] trainable parameters: "
        + json.dumps({
            key: sum(parameter.numel() for _, parameter in values)
            for key, values in groups.items()
        })
    )
    model.cuda().eval()
    model.pts_bbox_head.with_dn = False
    model.pts_bbox_head.uq_estimator.eval()
    if hasattr(model.lm_head, "gradient_checkpointing_disable"):
        model.lm_head.gradient_checkpointing_disable()

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

    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    gradient_audited = False
    for epoch in range(args.epochs):
        progress = ProgressBar(len(loader))
        for batch in loader:
            if global_step >= args.max_steps:
                break
            name = sample_id_from_batch(batch)
            info = info_lookup[name]
            objects = select_critical_objects(
                info["gt_boxes"],
                info["gt_names"],
            )
            context = prepare_visual_context(model, batch)
            shuffled_active, shuffled_score = shuffled_lookup[name]
            shuffled_uq = (
                shuffled_active.unsqueeze(0).cuda(),
                shuffled_score.unsqueeze(0).cuda(),
            )
            correct_vision = apply_uq_mode(
                model,
                *context,
                mode="correct",
                shuffled_uq=shuffled_uq,
            )
            shuffled_vision = apply_uq_mode(
                model,
                *context,
                mode="shuffled",
                shuffled_uq=shuffled_uq,
            )
            render_answer = (
                render_risk_synthesis_answer
                if args.answer_style == "synthesis"
                else (
                    render_reliability_answer
                    if args.answer_style == "level_only"
                    else (
                        render_natural_risk_qa_answer
                        if args.answer_style == "natural"
                        else render_risk_qa_answer
                    )
                )
            )
            correct_risk = build_risk_qa_answer(
                float(context[2].item()), objects
            )
            shuffled_risk = build_risk_qa_answer(
                float(shuffled_score.item()), objects
            )
            correct_answer = render_answer(correct_risk)
            shuffled_answer = render_answer(shuffled_risk)
            correct_loss = language_loss(
                model,
                correct_vision,
                model.tokenizer,
                correct_answer,
                args.answer_style,
                critical_objects=objects,
                reliability_answer=render_reliability_answer(correct_risk),
            )
            shuffled_loss = language_loss(
                model,
                shuffled_vision,
                model.tokenizer,
                shuffled_answer,
                args.answer_style,
                critical_objects=objects,
                reliability_answer=render_reliability_answer(shuffled_risk),
            )
            loss = 0.5 * (correct_loss + shuffled_loss)
            (loss / args.grad_accum).backward()

            if not gradient_audited:
                audit = {
                    key: any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and parameter.grad.abs().sum() > 0
                        for _, parameter in values
                    )
                    for key, values in groups.items()
                }
                if not all(audit.values()):
                    raise RuntimeError(f"Gradient audit failed: {audit}")
                print(f"[RiskQA] gradient audit passed: {audit}")
                gradient_audited = True

            if (global_step + 1) % args.grad_accum == 0:
                trainable = [
                    parameter
                    for values in groups.values()
                    for _, parameter in values
                ]
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            record = {
                "epoch": epoch + 1,
                "step": global_step + 1,
                "loss": float(loss.detach()),
                "correct_loss": float(correct_loss.detach()),
                "shuffled_loss": float(shuffled_loss.detach()),
                "correct_score": float(context[2].item()),
                "shuffled_score": float(shuffled_score.item()),
            }
            history.append(record)
            global_step += 1
            if (
                global_step == 1
                or global_step % args.log_interval == 0
            ):
                print(f"\n[RiskQA] {json.dumps(record)}")
            progress.update()
        if global_step >= args.max_steps:
            break

    save_checkpoint(
        args.out,
        model,
        optimizer,
        args,
        global_step,
        history,
    )
    print(f"\n[RiskQA] saved {args.out}, steps={global_step}")


if __name__ == "__main__":
    main()
