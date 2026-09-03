"""Measure whether reliability dialogue history changes ORION planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.datasets.data_utils import conversation as conversation_lib
from mmcv.datasets.data_utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    EGO_WAYPOINT_TOKEN,
)
from mmcv.datasets.data_utils.data_utils import tokenizer_image_token
from mmcv.models import build_model
from mmcv.utils import Config, load_checkpoint, set_random_seed

from scripts.eval_risk_qa import (
    apply_uq_mode,
    build_density_score_lookup,
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
from uq_estimator.risk_qa import (
    RELIABILITY_QA_QUESTION,
    build_risk_qa_answer,
    render_reliability_answer,
    select_balanced_sample_ids,
)
from uq_estimator.training import load_uq_token_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="adzoo/orion/configs/orion_stage3_infer.py"
    )
    parser.add_argument("--checkpoint", default="ckpts/Orion.pth")
    parser.add_argument("--adaptation-checkpoint", required=True)
    parser.add_argument(
        "--density-checkpoint", default="checkpoints/density_uq/best.pt"
    )
    parser.add_argument(
        "--descriptor-cache", default="data/density_uq/descriptors.pt"
    )
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--balanced-per-level", type=int, default=2)
    parser.add_argument("--max-valid", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def unwrap_input_ids(batch: dict) -> torch.Tensor:
    value = batch["input_ids"]
    while isinstance(value, list):
        value = value[0]
    return value.unsqueeze(0).cuda()


def build_planning_prompt(tokenizer, reliability_answer: str) -> torch.Tensor:
    conversation = conversation_lib.default_conversation.copy()
    conversation.append_message(
        conversation.roles[0],
        f"{DEFAULT_IMAGE_TOKEN}\nYou are driving a car. "
        f"{RELIABILITY_QA_QUESTION}",
    )
    conversation.append_message(
        conversation.roles[1], reliability_answer
    )
    conversation.append_message(
        conversation.roles[0],
        "Please provide the planning trajectory for the ego car without "
        "reasons.",
    )
    conversation.append_message(
        conversation.roles[1],
        f"Here is the planning trajectory {EGO_WAYPOINT_TOKEN}",
    )
    return tokenizer_image_token(
        conversation.get_prompt(),
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0).cuda()


def decode_active_trajectory(
    model,
    ego_feature: torch.Tensor,
    batch: dict,
    seed: int,
) -> torch.Tensor:
    orion = model.module if hasattr(model, "module") else model
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    current_states = ego_feature.float().unsqueeze(1)
    sample, _ = orion.distribution_forward(current_states, None, None)
    states_hs, _ = orion.future_states_predict(
        1, sample, current_states, current_states
    )
    ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
    predictions = []
    for index in range(6):
        prediction = orion.ego_fut_decoder(
            ego_query_hs[index]
        ).reshape(1, orion.ego_fut_mode, 2)
        predictions.append(prediction)
    trajectories = torch.stack(predictions, dim=2)
    command = batch["ego_fut_cmd"][0][:, 0, 0].cuda()
    return (trajectories * command[..., None, None]).sum(dim=1)


def trajectory_ade(
    prediction: torch.Tensor,
    batch: dict,
) -> float:
    target = batch["ego_fut_trajs"][0][:, 0].cuda()
    mask = batch["ego_fut_masks"][0][:, 0, 0].cuda()
    valid = mask.bool()
    if not valid.any():
        return float("nan")
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return float(distance[valid].mean())


def pair_displacement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right, dim=-1).mean())


def has_valid_future(batch: dict) -> bool:
    mask = batch["ego_fut_masks"][0][0]
    return bool(mask.sum().item() > 0)


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
    filter_dataset_by_split(dataset, assignment, args.split)
    level_counts = {}
    if args.balanced_per_level > 0:
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
    loaded = load_uq_token_weights(model, args.adaptation_checkpoint)
    print(f"[RiskPlanning] loaded {loaded} adaptation tensors")
    if level_counts:
        print(f"[RiskPlanning] balanced levels: {json.dumps(level_counts)}")
    model.cuda().eval()
    model.pts_bbox_head.with_dn = False
    model.pts_bbox_head.uq_estimator.eval()

    records = []
    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            if len(records) >= args.max_valid:
                break
            if not has_valid_future(batch):
                continue
            name = sample_id_from_batch(batch)
            context = prepare_visual_context(model, batch)
            correct_score = float(context[2].item())
            shuffled_active, shuffled_score = shuffled_lookup[name]
            shuffled_uq = (
                shuffled_active.unsqueeze(0).cuda(),
                shuffled_score.unsqueeze(0).cuda(),
            )
            vision = apply_uq_mode(
                model,
                *context,
                mode="correct",
                shuffled_uq=shuffled_uq,
            )
            correct_answer = render_reliability_answer(
                build_risk_qa_answer(correct_score, ())
            )
            shuffled_answer = render_reliability_answer(
                build_risk_qa_answer(float(shuffled_score.item()), ())
            )
            prompts = {
                "baseline": unwrap_input_ids(batch),
                "correct_text": build_planning_prompt(
                    model.tokenizer, correct_answer
                ),
                "shuffled_text": build_planning_prompt(
                    model.tokenizer, shuffled_answer
                ),
            }
            features = {}
            trajectories = {}
            ade = {}
            for mode, prompt in prompts.items():
                feature = model.lm_head.inference_ego(
                    inputs=prompt,
                    images=vision,
                    use_cache=True,
                    return_ego_feature=True,
                )
                features[mode] = feature
                trajectories[mode] = decode_active_trajectory(
                    model, feature, batch, args.seed + sample_index
                )
                ade[mode] = trajectory_ade(trajectories[mode], batch)
            records.append({
                "sample_id": name,
                "correct_uq_score": correct_score,
                "shuffled_uq_score": float(shuffled_score.item()),
                "correct_answer": correct_answer,
                "shuffled_answer": shuffled_answer,
                "ade": ade,
                "hidden_l2": {
                    "correct_vs_baseline": float(torch.linalg.vector_norm(
                        features["correct_text"] - features["baseline"]
                    )),
                    "shuffled_vs_baseline": float(torch.linalg.vector_norm(
                        features["shuffled_text"] - features["baseline"]
                    )),
                    "correct_vs_shuffled": float(torch.linalg.vector_norm(
                        features["correct_text"] - features["shuffled_text"]
                    )),
                },
                "trajectory_l2": {
                    "correct_vs_baseline": pair_displacement(
                        trajectories["correct_text"],
                        trajectories["baseline"],
                    ),
                    "shuffled_vs_baseline": pair_displacement(
                        trajectories["shuffled_text"],
                        trajectories["baseline"],
                    ),
                    "correct_vs_shuffled": pair_displacement(
                        trajectories["correct_text"],
                        trajectories["shuffled_text"],
                    ),
                },
            })
            print(f"[RiskPlanning] evaluated {name}")

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for record in records:
            value = record
            for key in path:
                value = value[key]
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "count": len(records),
        "ade": {
            mode: mean(("ade", mode))
            for mode in ("baseline", "correct_text", "shuffled_text")
        },
        "hidden_l2": {
            key: mean(("hidden_l2", key))
            for key in (
                "correct_vs_baseline",
                "shuffled_vs_baseline",
                "correct_vs_shuffled",
            )
        },
        "trajectory_l2": {
            key: mean(("trajectory_l2", key))
            for key in (
                "correct_vs_baseline",
                "shuffled_vs_baseline",
                "correct_vs_shuffled",
            )
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print(f"[RiskPlanning] saved {output}")


if __name__ == "__main__":
    main()
