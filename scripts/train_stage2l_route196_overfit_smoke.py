#!/usr/bin/env python3
"""Real ORION/VLM Stage2-L engineering overfit smoke on Route196.

The smoke trains the spatial UQ tokenizer, the existing ORION LLM LoRA
parameters, and an explicit task-relevance-map head.  Frozen ORION det/map
visual tokens are concatenated with 600 UQ tokens inside the same multimodal
``<image>`` span.  R is decoded from the final-layer hidden states at those UQ
token positions, after VLM self-attention over visual, route-text and UQ input.

This is deliberately a one-event optimization sanity check.  It may establish
that the implementation can learn the intended semantics, but it is never
reported as a generalization or safety result.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mmcv.datasets.data_utils import conversation as conversation_lib
from mmcv.datasets.data_utils.constants import DEFAULT_IMAGE_TOKEN
from mmcv.datasets.data_utils.data_utils import preprocess, tokenizer_image_token
from mmcv.models import build_model
from mmcv.utils import Config, load_checkpoint, set_random_seed

from scripts.scenario_factory_lib import sha256_file
from team_code.orion_b2d_agent import resolve_local_model_paths
from uq_estimator.stage2l_pilot import (
    balanced_driving_epoch,
    driving_stance_counts,
    matched_answer_preference_loss,
    parse_planning_stance,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRelevanceMapHead,
    UQComponentTokenizer,
    fixed_task_risk,
    matched_task_risk_ranking_loss,
    task_relevance_loss,
)


SCHEMA = "orion.stage2l_route196_overfit_smoke.v1"
IMAGE_TOKEN_INDEX = -200
ORION_VISUAL_TOKENS = 529
SPATIAL_UQ_TOKENS = 600
VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _load_records(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 20 or {row["counterfactual"]["variant"] for row in rows} != set(VARIANTS):
        raise ValueError("Route196 smoke requires exactly five variants x four QA families")
    if len({row["event_id"] for row in rows}) != 1:
        raise ValueError("engineering overfit smoke must contain exactly one event")
    return rows


def _resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def _load_inputs(
    records: Iterable[Mapping[str, Any]], records_path: Path
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
    components: Dict[str, torch.Tensor] = {}
    relevance = None
    route_context = None
    for row in records:
        variant = row["counterfactual"]["variant"]
        if variant not in components:
            ref = row["model_input"]["stage1_observation_uq"]
            path = _resolve(ref["path"], records_path.parent)
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("Stage1 UQ sidecar hash mismatch")
            with np.load(path) as archive:
                value = archive[ref["component_key"]].astype(np.float32)
            if value.shape != (4, 6, 40, 40, 3):
                raise ValueError("unexpected Stage1 component shape")
            components[variant] = torch.from_numpy(value).unsqueeze(0)
        sidecar_ref = row["target"]["map_sidecar"]
        sidecar = _resolve(sidecar_ref["path"], records_path.parent)
        if sha256_file(sidecar) != sidecar_ref["sha256"]:
            raise ValueError("QA map sidecar hash mismatch")
        with np.load(sidecar) as archive:
            current_relevance = archive[sidecar_ref["relevance_key"]].astype(np.float32)
        if relevance is None:
            relevance = current_relevance
        elif not np.array_equal(relevance, current_relevance):
            raise ValueError("matched counterfactuals must share one relevance target")
        current_route = row["model_input"]["route_context"]["payload"]
        if route_context is None:
            route_context = current_route
        elif route_context != current_route:
            raise ValueError("matched counterfactuals must share route context")
    if set(components) != set(VARIANTS) or relevance is None:
        raise ValueError("incomplete Route196 counterfactual inputs")
    target = torch.from_numpy(relevance).unsqueeze(0)
    target = F.adaptive_avg_pool2d(target, (10, 10))
    return components, target, route_context


def _route_text(route_context: Mapping[str, Any]) -> str:
    command_names = {0: "LEFT", 1: "RIGHT", 2: "STRAIGHT", 3: "FOLLOW_LANE", 4: "CHANGE_LEFT", 5: "CHANGE_RIGHT"}
    command = route_context.get("command")
    # Saved closed-loop metadata uses the 1-based CARLA RoadOption encoding.
    command_text = command_names.get(int(command) - 1, str(command))
    points = route_context["orion_unmodified_plan_right_forward_m"]
    rendered = "; ".join("(%.2f, %.2f)" % (float(x), float(y)) for x, y in points)
    return "Route command: %s. Frozen ORION future path points (right_m, forward_m): %s." % (
        command_text, rendered
    )


def _training_tokens(tokenizer, row: Mapping[str, Any], route_text: str, answer: str = None):
    question = "%s\n%s\n%s" % (
        DEFAULT_IMAGE_TOKEN,
        route_text,
        row["conversation"][0]["value"],
    )
    target = row["conversation"][1]["value"] if answer is None else answer
    converted = preprocess(
        [[{"from": "human", "value": question}, {"from": "gpt", "value": target}]],
        tokenizer,
        has_image=True,
    )
    return converted["input_ids"][0], converted["labels"][0]


def _prompt_tokens(tokenizer, row: Mapping[str, Any], route_text: str) -> torch.Tensor:
    conversation = conversation_lib.default_conversation.copy()
    conversation.append_message(
        conversation.roles[0],
        "%s\n%s\n%s" % (DEFAULT_IMAGE_TOKEN, route_text, row["conversation"][0]["value"]),
    )
    conversation.append_message(conversation.roles[1], None)
    return tokenizer_image_token(
        conversation.get_prompt(), tokenizer, return_tensors="pt"
    ).unsqueeze(0)


def _load_orion_lm(config_path: Path, checkpoint_path: Path):
    cfg = Config.fromfile(str(config_path))
    resolve_local_model_paths(cfg)
    cfg.model.frozen = False
    cfg.model.use_lora = True
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    lm = model.lm_head
    tokenizer = model.tokenizer
    model.lm_head = None
    del model
    gc.collect()
    for name, parameter in lm.named_parameters():
        parameter.requires_grad = "lora_" in name
    if not any(parameter.requires_grad for parameter in lm.parameters()):
        raise RuntimeError("ORION LLM exposes no trainable LoRA parameters")
    lm.config.use_cache = False
    lm.cuda().train()
    return lm, tokenizer


def _forward_record(
    *,
    lm,
    tokenizer,
    uq_tokenizer: UQComponentTokenizer,
    relevance_head: TaskRelevanceMapHead,
    baseline_vision: torch.Tensor,
    components: torch.Tensor,
    relevance_target: torch.Tensor,
    on_components: torch.Tensor,
    off_components: torch.Tensor,
    row: Mapping[str, Any],
    route_text: str,
    answer: str = None,
) -> Dict[str, torch.Tensor]:
    input_ids, labels = _training_tokens(tokenizer, row, route_text, answer=answer)
    input_ids = input_ids.unsqueeze(0).cuda(non_blocking=True)
    labels = labels.unsqueeze(0).cuda(non_blocking=True)
    attention = input_ids.ne(tokenizer.pad_token_id or 0)
    uq = uq_tokenizer(components.cuda(non_blocking=True))
    vision = torch.cat((baseline_vision, uq.tokens), dim=1)
    image_start = int(torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].item())
    output = lm(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        images=vision,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = output.hidden_states[-1]
    uq_start = image_start + ORION_VISUAL_TOKENS
    uq_hidden = hidden[:, uq_start:uq_start + SPATIAL_UQ_TOKENS]
    if uq_hidden.shape[1:] != (SPATIAL_UQ_TOKENS, 4096):
        raise RuntimeError("VLM-fused UQ hidden-state span is malformed")
    logits = relevance_head(uq_hidden.reshape(1, 6, 10, 10, 4096))
    map_loss = task_relevance_loss(logits, relevance_target)
    current_risk = fixed_task_risk(uq.latest_scalar_uq, logits)
    on_uq = uq_tokenizer(on_components.cuda(non_blocking=True))
    off_uq = uq_tokenizer(off_components.cuda(non_blocking=True))
    on_risk = fixed_task_risk(on_uq.latest_scalar_uq, logits)
    off_risk = fixed_task_risk(off_uq.latest_scalar_uq, logits)
    ranking = matched_task_risk_ranking_loss(on_risk, off_risk, margin=0.2)
    return {
        "language_loss": output.loss,
        "map_loss": map_loss,
        "ranking_loss": ranking,
        "relevance_logits": logits,
        "current_risk": current_risk,
        "current_peak": current_risk.flatten(1).amax(dim=1).mean(),
        "on_peak": on_risk.flatten(1).amax(dim=1).mean(),
        "off_peak": off_risk.flatten(1).amax(dim=1).mean(),
    }


@torch.no_grad()
def _evaluate(
    *, lm, tokenizer, uq_tokenizer, relevance_head, baseline_vision,
    components, relevance_target, records, route_text,
) -> Dict[str, Any]:
    lm.eval()
    uq_tokenizer.eval()
    relevance_head.eval()
    language_losses = []
    map_losses = []
    on_peaks = []
    off_peaks = []
    for row in records:
        result = _forward_record(
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head,
            baseline_vision=baseline_vision,
            components=components[row["counterfactual"]["variant"]],
            relevance_target=relevance_target,
            on_components=components["on_path_uq"],
            off_components=components["off_path_uq"],
            row=row,
            route_text=route_text,
        )
        language_losses.append(float(result["language_loss"].item()))
        map_losses.append(float(result["map_loss"].item()))
        on_peaks.append(float(result["on_peak"].item()))
        off_peaks.append(float(result["off_peak"].item()))

    implication = {
        row["counterfactual"]["variant"]: row
        for row in records if row["question_family"] == "driving_implication"
    }
    expected_nll = {}
    alternative_nll = {}
    maintain_answer = implication["zero_uq"]["conversation"][1]["value"]
    conservative_answer = implication["on_path_uq"]["conversation"][1]["value"]
    for variant in ("zero_uq", "off_path_uq", "on_path_uq"):
        row = implication[variant]
        expected = _forward_record(
            lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head, baseline_vision=baseline_vision,
            components=components[variant], relevance_target=relevance_target,
            on_components=components["on_path_uq"], off_components=components["off_path_uq"],
            row=row, route_text=route_text,
        )
        alternative_answer = conservative_answer if variant != "on_path_uq" else maintain_answer
        alternative = _forward_record(
            lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head, baseline_vision=baseline_vision,
            components=components[variant], relevance_target=relevance_target,
            on_components=components["on_path_uq"], off_components=components["off_path_uq"],
            row=row, route_text=route_text, answer=alternative_answer,
        )
        expected_nll[variant] = float(expected["language_loss"].item())
        alternative_nll[variant] = float(alternative["language_loss"].item())

    generated = {}
    for variant in ("zero_uq", "off_path_uq", "on_path_uq"):
        row = implication[variant]
        prompt = _prompt_tokens(tokenizer, row, route_text).cuda()
        uq = uq_tokenizer(components[variant].cuda())
        vision = torch.cat((baseline_vision, uq.tokens), dim=1)
        output_ids = lm.generate(
            inputs=prompt,
            images=vision,
            do_sample=False,
            num_beams=1,
            max_new_tokens=72,
            use_cache=True,
        )
        generated[variant] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    lm.train()
    uq_tokenizer.train()
    relevance_head.train()
    return {
        "mean_language_nll": float(np.mean(language_losses)),
        "mean_relevance_bce": float(np.mean(map_losses)),
        "mean_on_path_peak_risk": float(np.mean(on_peaks)),
        "mean_off_path_peak_risk": float(np.mean(off_peaks)),
        "mean_on_minus_off": float(np.mean(np.asarray(on_peaks) - np.asarray(off_peaks))),
        "expected_answer_nll": expected_nll,
        "counterfactual_answer_nll": alternative_nll,
        "generated_driving_implication": generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-protocol",
        type=Path,
        help="Required preregistered protocol when balanced sampling is enabled.",
    )
    parser.add_argument(
        "--launch-amendment",
        type=Path,
        help="Required single-run authorization when balanced sampling is enabled.",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-head", type=float, default=3e-4)
    parser.add_argument("--lambda-map", type=float, default=2.0)
    parser.add_argument("--lambda-ranking", type=float, default=1.0)
    parser.add_argument("--lambda-answer-preference", type=float, default=0.0)
    parser.add_argument("--answer-preference-margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument(
        "--balance-driving-stances",
        action="store_true",
        help="Oversample only minority driving-implication stances per epoch.",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Stage2-L smoke output")
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.lambda_answer_preference < 0.0 or args.answer_preference_margin < 0.0:
        raise ValueError("answer preference weight and margin must be non-negative")
    training_protocol = None
    launch_amendment = None
    if args.balance_driving_stances:
        if args.training_protocol is None or args.launch_amendment is None:
            raise ValueError(
                "balanced driving sampling requires --training-protocol and "
                "--launch-amendment"
            )
        training_protocol = _load_json(args.training_protocol.resolve())
        if (
            training_protocol.get("schema")
            != "orion.stage2l_uq_language_grounding_protocol.v1"
            or training_protocol.get("sampling", {}).get(
                "balance_driving_stances"
            )
            is not True
        ):
            raise ValueError("training protocol does not authorize balanced sampling")
        maximum = int(
            training_protocol.get("route196_new_objective_smoke", {}).get(
                "maximum_steps", -1
            )
        )
        if maximum < 1 or args.max_steps > maximum:
            raise ValueError("max-steps exceeds the preregistered smoke maximum")
        preference = training_protocol.get("losses", {}).get(
            "matched_answer_preference"
        )
        if args.lambda_answer_preference > 0.0:
            if not isinstance(preference, dict) or (
                float(preference.get("weight", -1.0))
                != args.lambda_answer_preference
                or float(preference.get("margin", -1.0))
                != args.answer_preference_margin
            ):
                raise ValueError(
                    "answer preference CLI values disagree with the protocol"
                )
        elif preference is not None:
            raise ValueError(
                "protocol requires a non-zero matched answer preference loss"
            )
        launch_amendment = _load_json(args.launch_amendment.resolve())
        launch_locks = launch_amendment.get("launch_locks", {})
        launch_authorization_key = training_protocol.get(
            "launch_authorization_key",
            "route196_balanced_language_smoke_allowed",
        )
        if (
            launch_amendment.get("schema")
            != "orion.scenario_factory.amendment.v1"
            or launch_locks.get(launch_authorization_key) is not True
            or launch_locks.get("stage2l_pilot_training_allowed") is not False
        ):
            raise ValueError(
                "launch amendment must authorize only the Route196 balanced smoke"
            )
    elif args.training_protocol is not None or args.launch_amendment is not None:
        raise ValueError(
            "--training-protocol/--launch-amendment are only valid with "
            "--balance-driving-stances"
        )
    if not torch.cuda.is_available():
        raise SystemExit("real Stage2-L smoke requires CUDA")
    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)

    records_path = args.records.resolve()
    records = _load_records(records_path)
    components, relevance_target, route_context = _load_inputs(records, records_path)
    route_text = _route_text(route_context)
    cache = torch.load(args.visual_cache.resolve(), map_location="cpu")
    if cache.get("schema") != "orion.closedloop_visual_context_cache.v1":
        raise ValueError("unsupported ORION visual-context cache")
    if cache.get("metadata", {}).get("event_id") != records[0]["event_id"]:
        raise ValueError("visual cache and QA records identify different events")
    baseline_vision = cache["baseline_vision"].float().cuda()
    if tuple(baseline_vision.shape) != (1, ORION_VISUAL_TOKENS, 4096):
        raise ValueError("visual cache shape mismatch")
    relevance_target = relevance_target.cuda()

    lm, tokenizer = _load_orion_lm(args.config.resolve(), args.checkpoint.resolve())
    uq_tokenizer = UQComponentTokenizer(model_dim=4096, hidden_dim=256, grid_hw=(10, 10)).cuda()
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    lora_parameters = [parameter for parameter in lm.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate_lora},
            {"params": list(uq_tokenizer.parameters()) + list(relevance_head.parameters()), "lr": args.learning_rate_head},
        ],
        weight_decay=1e-4,
    )

    before = _evaluate(
        lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
        relevance_head=relevance_head, baseline_vision=baseline_vision,
        components=components, relevance_target=relevance_target,
        records=records, route_text=route_text,
    )
    history = []
    implication = {
        row["counterfactual"]["variant"]: row
        for row in records if row["question_family"] == "driving_implication"
    }
    maintain_answer = implication["zero_uq"]["conversation"][1]["value"]
    conservative_answer = implication["on_path_uq"]["conversation"][1]["value"]
    sampling_rng = random.Random(args.seed)
    epoch_rows = (
        balanced_driving_epoch(records, sampling_rng)
        if args.balance_driving_stances else list(records)
    )
    if not args.balance_driving_stances:
        sampling_rng.shuffle(epoch_rows)
    for step in range(1, args.max_steps + 1):
        if step > 1 and (step - 1) % len(epoch_rows) == 0:
            epoch_rows = (
                balanced_driving_epoch(records, sampling_rng)
                if args.balance_driving_stances else list(records)
            )
            if not args.balance_driving_stances:
                sampling_rng.shuffle(epoch_rows)
        row = epoch_rows[(step - 1) % len(epoch_rows)]
        optimizer.zero_grad(set_to_none=True)
        result = _forward_record(
            lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head, baseline_vision=baseline_vision,
            components=components[row["counterfactual"]["variant"]],
            relevance_target=relevance_target,
            on_components=components["on_path_uq"],
            off_components=components["off_path_uq"],
            row=row, route_text=route_text,
        )
        preference_loss = result["language_loss"].new_zeros(())
        if (
            args.lambda_answer_preference > 0.0
            and row["question_family"] == "driving_implication"
        ):
            target_stance = row["target"]["structured_summary"][
                "planning_implication"
            ]["stance"]
            alternative_answer = (
                conservative_answer
                if target_stance == "maintain" else maintain_answer
            )
            alternative = _forward_record(
                lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
                relevance_head=relevance_head, baseline_vision=baseline_vision,
                components=components[row["counterfactual"]["variant"]],
                relevance_target=relevance_target,
                on_components=components["on_path_uq"],
                off_components=components["off_path_uq"],
                row=row, route_text=route_text, answer=alternative_answer,
            )
            preference_loss = matched_answer_preference_loss(
                result["language_loss"], alternative["language_loss"],
                margin=args.answer_preference_margin,
            )
        loss = (
            result["language_loss"]
            + args.lambda_map * result["map_loss"]
            + args.lambda_ranking * result["ranking_loss"]
            + args.lambda_answer_preference * preference_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            lora_parameters + list(uq_tokenizer.parameters()) + list(relevance_head.parameters()),
            1.0,
        )
        optimizer.step()
        row_log = {
            "step": step,
            "variant": row["counterfactual"]["variant"],
            "question_family": row["question_family"],
            "loss": float(loss.item()),
            "language_nll": float(result["language_loss"].item()),
            "relevance_bce": float(result["map_loss"].item()),
            "ranking_loss": float(result["ranking_loss"].item()),
            "answer_preference_loss": float(preference_loss.item()),
            "on_minus_off": float((result["on_peak"] - result["off_peak"]).item()),
        }
        history.append(row_log)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LRoute196] " + json.dumps(row_log, sort_keys=True), flush=True)

    after = _evaluate(
        lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
        relevance_head=relevance_head, baseline_vision=baseline_vision,
        components=components, relevance_target=relevance_target,
        records=records, route_text=route_text,
    )
    expected = after["expected_answer_nll"]
    alternative = after["counterfactual_answer_nll"]
    generated_stances = {
        variant: parse_planning_stance(text)
        for variant, text in after["generated_driving_implication"].items()
    }
    on_path_target_stance = implication["on_path_uq"]["target"][
        "structured_summary"
    ]["planning_implication"]["stance"]
    checks = {
        "optimization_reduces_language_nll": after["mean_language_nll"] < before["mean_language_nll"],
        "optimization_reduces_relevance_bce": after["mean_relevance_bce"] < before["mean_relevance_bce"],
        "on_path_risk_exceeds_off_path_by_margin": after["mean_on_minus_off"] >= 0.2,
        "zero_uq_prefers_nonconservative_answer": expected["zero_uq"] < alternative["zero_uq"],
        "off_path_prefers_nonconservative_answer": expected["off_path_uq"] < alternative["off_path_uq"],
        "on_path_prefers_conservative_answer": expected["on_path_uq"] < alternative["on_path_uq"],
        "generated_zero_uq_is_maintain": generated_stances["zero_uq"] == "maintain",
        "generated_off_path_is_maintain": generated_stances["off_path_uq"] == "maintain",
        "generated_on_path_matches_target_stance": generated_stances["on_path_uq"] == on_path_target_stance,
    }
    status = "engineering_overfit_pass" if all(checks.values()) else "engineering_overfit_failed_gate"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_route196_overfit.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "status": status,
            "engineering_overfit_only": True,
            "uq_tokenizer": uq_tokenizer.state_dict(),
            "relevance_head": relevance_head.state_dict(),
            "lora": {name: value.detach().cpu() for name, value in lm.state_dict().items() if "lora_" in name},
            "steps": args.max_steps,
        },
        checkpoint_path,
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "engineering_overfit_only": True,
        "formal_training_ready": False,
        "event_count": 1,
        "qa_record_count": len(records),
        "steps": args.max_steps,
        "sampling": {
            "balance_driving_stances": args.balance_driving_stances,
            "source_driving_stance_counts": driving_stance_counts(records),
            "effective_epoch_record_count": len(epoch_rows),
            "non_driving_records_repeated": False,
        },
        "answer_preference_objective": {
            "weight": args.lambda_answer_preference,
            "margin": args.answer_preference_margin,
            "driving_implication_only": True,
            "matched_same_input_opposite_stance": args.lambda_answer_preference > 0.0,
        },
        "architecture": {
            "orion_native_visual_tokens": ORION_VISUAL_TOKENS,
            "orion_native_visual_token_breakdown": {
                "detection_head": 273,
                "map_head": 256,
            },
            "spatial_uq_tokens": SPATIAL_UQ_TOKENS,
            "uq_token_layout": [6, 10, 10],
            "relevance_decoded_from_vlm_final_hidden_states": True,
            "trainable": ["ORION LLM LoRA", "UQ component tokenizer", "task relevance map head"],
            "frozen": ["ORION visual encoder", "ORION detection/map heads", "Stage1 observation-UQ adapter"],
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
        },
        "before": before,
        "after": after,
        "generated_stances_after": generated_stances,
        "checks": checks,
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "inputs": {
            "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
            "visual_cache": {"path": str(args.visual_cache.resolve()), "sha256": sha256_file(args.visual_cache.resolve())},
            "orion_checkpoint_sha256": cache["metadata"]["orion_checkpoint"]["sha256"],
        },
        "history": history,
        "claim_boundary": "One-event Route196 optimization sanity check only; no held-out semantic, generalization, trajectory, closed-loop, or safety claim.",
    }
    if training_protocol is not None:
        report["inputs"]["training_protocol"] = {
            "path": str(args.training_protocol.resolve()),
            "sha256": sha256_file(args.training_protocol.resolve()),
            "status": training_protocol.get("status"),
        }
        report["inputs"]["launch_amendment"] = {
            "path": str(args.launch_amendment.resolve()),
            "sha256": sha256_file(args.launch_amendment.resolve()),
            "status": launch_amendment.get("status"),
        }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path), "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
