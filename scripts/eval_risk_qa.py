"""Generate structured Risk QA answers under UQ-token interventions."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, load_checkpoint, set_random_seed
from mmcv.datasets.data_utils import conversation as conversation_lib
from mmcv.datasets.data_utils.constants import DEFAULT_IMAGE_TOKEN
from mmcv.datasets.data_utils.data_utils import tokenizer_image_token

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
    parse_natural_risk_qa_answer,
    parse_reliability_answer,
    parse_risk_qa_answer,
    render_natural_risk_qa_answer,
    render_reliability_answer,
    render_risk_qa_answer,
    reliability_level,
    reliability_percentile,
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
    parser.add_argument(
        "--adaptation-checkpoint",
        default=None,
    )
    parser.add_argument(
        "--density-checkpoint", default="checkpoints/density_uq/best.pt"
    )
    parser.add_argument(
        "--descriptor-cache", default="data/density_uq/descriptors.pt"
    )
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--balanced-per-level", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes", default="none,zero,shuffled,correct"
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--answer-style",
        choices=("structured", "natural", "level_only"),
        default="level_only",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_generation_prompt(tokenizer, answer_style="natural") -> torch.Tensor:
    conversation = conversation_lib.default_conversation.copy()
    if answer_style == "level_only":
        task = RELIABILITY_QA_QUESTION
    else:
        task = (
            f"{RISK_QA_QUESTION}\n"
            "State the visual reliability level, list critical objects and "
            "their relative positions, then give a short risk assessment."
        )
    question = f"{DEFAULT_IMAGE_TOKEN}\nYou are driving a car. {task}"
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return tokenizer_image_token(
        conversation.get_prompt(),
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0)


def prepare_visual_context(model, batch: dict):
    orion = model.module if hasattr(model, "module") else model
    data = copy.deepcopy(batch)
    img_metas = data["img_metas"]
    special_keys = {
        "img",
        "input_ids",
        "gt_bboxes_3d",
        "gt_attr_labels",
        "vlm_labels",
        "img_metas",
    }
    for key in list(data):
        if key in special_keys:
            continue
        data[key] = data[key][0][0].unsqueeze(0).cuda()
    data["img"] = data["img"][0].cuda()
    data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0]
    data.pop("gt_attr_labels", None)

    data["img_feats"] = orion.extract_feat(data["img"])
    if data["img"].dim() == 4:
        data["img"] = data["img"].unsqueeze(0)
    img_metas_list = img_metas[0]
    data.pop("img_metas", None)
    location = orion.prepare_location(img_metas_list, **data)
    pos_embed = orion.position_embeding(data, location, img_metas_list)
    _, det_query, uncertainty_embedding, uncertainty_score = (
        orion.pts_bbox_head(img_metas_list, pos_embed, **data)
    )
    uq_output = orion.pts_bbox_head.uq_output
    active_embedding = uq_output.active_embedding
    _, map_query = orion.map_head(img_metas_list, pos_embed, **data)
    baseline_vision = torch.cat((det_query.clone(), map_query.clone()), dim=1)
    return (
        baseline_vision,
        uncertainty_embedding,
        uncertainty_score,
        active_embedding,
    )


def apply_uq_mode(
    model,
    baseline_vision,
    uncertainty_embedding,
    uncertainty_score,
    active_embedding,
    mode,
    shuffled_uq=None,
):
    orion = model.module if hasattr(model, "module") else model
    if mode == "none":
        return baseline_vision
    if mode == "zero":
        return orion._append_uq_tokens(
            baseline_vision,
            torch.zeros_like(uncertainty_embedding),
            torch.zeros_like(uncertainty_score),
            active_embedding=torch.zeros_like(active_embedding),
        )
    if mode == "correct":
        return orion._append_uq_tokens(
            baseline_vision,
            uncertainty_embedding,
            uncertainty_score,
            active_embedding=torch.zeros_like(active_embedding),
        )
    if mode == "shuffled":
        if shuffled_uq is None:
            raise ValueError("shuffled mode requires shuffled UQ")
        shuffled_active, shuffled_score = shuffled_uq
        return orion._append_uq_tokens(
            baseline_vision,
            uncertainty_embedding,
            shuffled_score,
            active_embedding=torch.zeros_like(shuffled_active),
        )
    raise ValueError(f"Unsupported UQ mode: {mode}")


def info_sample_id(info: dict) -> str:
    return (
        f"{Path(info['folder']).name}__"
        f"{int(info['frame_idx']):05d}.pt"
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


def summarize_level_outputs(records: list[dict]) -> dict:
    level_order = ("very low", "low", "moderate", "high", "very high")
    level_index = {level: index for index, level in enumerate(level_order)}
    summary = {}
    for mode, score_key in (
        ("correct", "correct_uq_score"),
        ("shuffled", "shuffled_uq_score"),
    ):
        if not records or mode not in records[0]["outputs"]:
            continue
        predictions = []
        targets = []
        for record in records:
            prediction = record["outputs"][mode]["parsed_level"]
            target = reliability_level(
                reliability_percentile(record[score_key])
            )
            if prediction in level_index:
                predictions.append(level_index[prediction])
                targets.append(level_index[target])
        count = len(records)
        parsed = len(predictions)
        summary[mode] = {
            "count": count,
            "parse_rate": parsed / count if count else 0.0,
            "accuracy": (
                sum(pred == target for pred, target in zip(
                    predictions, targets
                )) / count
                if count else 0.0
            ),
            "ordinal_mae": (
                float(np.mean(np.abs(
                    np.asarray(predictions) - np.asarray(targets)
                )))
                if parsed else float("nan")
            ),
            "spearman": (
                float(spearmanr(predictions, targets).statistic)
                if parsed >= 2 else float("nan")
            ),
        }

    if all(mode in summary for mode in ("correct", "shuffled")):
        eligible = 0
        changed = 0
        for record in records:
            correct_target = reliability_level(
                reliability_percentile(record["correct_uq_score"])
            )
            shuffled_target = reliability_level(
                reliability_percentile(record["shuffled_uq_score"])
            )
            if correct_target == shuffled_target:
                continue
            eligible += 1
            changed += (
                record["outputs"]["correct"]["parsed_level"]
                != record["outputs"]["shuffled"]["parsed_level"]
            )
        summary["intervention"] = {
            "eligible": eligible,
            "changed": changed,
            "response_rate": changed / eligible if eligible else float("nan"),
        }
    return summary


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unsupported = set(modes).difference({"none", "zero", "shuffled", "correct"})
    if unsupported:
        raise ValueError(f"Unsupported modes: {sorted(unsupported)}")

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
    filter_dataset_by_split(dataset, assignment, args.split)
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
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    shuffled_lookup = build_shuffled_uq_lookup(
        args.descriptor_cache,
        args.density_checkpoint,
        assignment,
        args.split,
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
    if args.adaptation_checkpoint:
        loaded = load_uq_token_weights(model, args.adaptation_checkpoint)
        print(f"[RiskQA] loaded {loaded} adaptation tensors")
    model.cuda().eval()
    model.pts_bbox_head.with_dn = False
    model.pts_bbox_head.uq_estimator.eval()

    prompt_ids = build_generation_prompt(
        model.tokenizer, args.answer_style
    ).cuda()
    records = []
    with torch.no_grad():
        for batch in loader:
            name = sample_id_from_batch(batch)
            info = info_lookup[name]
            objects = select_critical_objects(
                info["gt_boxes"],
                info["gt_names"],
            )
            context = prepare_visual_context(model, batch)
            correct_score = float(context[2].item())
            shuffled_active, shuffled_score = shuffled_lookup[name]
            shuffled_uq = (
                shuffled_active.unsqueeze(0).cuda(),
                shuffled_score.unsqueeze(0).cuda(),
            )
            render_answer = (
                render_reliability_answer
                if args.answer_style == "level_only"
                else (
                    render_natural_risk_qa_answer
                    if args.answer_style == "natural"
                    else render_risk_qa_answer
                )
            )
            target_by_mode = {
                "correct": render_answer(
                    build_risk_qa_answer(correct_score, objects)
                ),
                "shuffled": render_answer(
                    build_risk_qa_answer(
                        float(shuffled_score.item()),
                        objects,
                    )
                ),
            }
            outputs = {}
            for mode in modes:
                vision = apply_uq_mode(
                    model,
                    *context,
                    mode=mode,
                    shuffled_uq=shuffled_uq,
                )
                output_ids = model.lm_head.generate(
                    inputs=prompt_ids,
                    images=vision,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
                text = model.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0]
                try:
                    if args.answer_style == "level_only":
                        parsed_level = parse_reliability_answer(text)
                        parsed_objects = ()
                        parsed_percentile = None
                    elif args.answer_style == "natural":
                        parsed_level, parsed_objects = (
                            parse_natural_risk_qa_answer(text)
                        )
                        parsed_percentile = None
                    else:
                        parsed = parse_risk_qa_answer(text)
                        parsed_level = parsed.reliability_level
                        parsed_objects = tuple(
                            item.category for item in parsed.critical_objects
                        )
                        parsed_percentile = parsed.reliability_percentile
                    parse_error = None
                except ValueError as error:
                    parse_error = str(error)
                    parsed_percentile = None
                    parsed_level = None
                    parsed_objects = ()
                outputs[mode] = {
                    "text": text,
                    "parse_error": parse_error,
                    "parsed_percentile": parsed_percentile,
                    "parsed_level": parsed_level,
                    "parsed_objects": parsed_objects,
                }
            records.append(
                {
                    "sample_id": name,
                    "correct_uq_score": correct_score,
                    "shuffled_uq_score": float(shuffled_score.item()),
                    "critical_objects": [
                        {
                            "category": item.category,
                            "position": item.position,
                            "distance_m": item.distance_m,
                        }
                        for item in objects
                    ],
                    "targets": target_by_mode,
                    "outputs": outputs,
                }
            )
            print(f"[RiskQA] generated {name}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "summary": summarize_level_outputs(records),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[RiskQA] saved {output}")


if __name__ == "__main__":
    main()
