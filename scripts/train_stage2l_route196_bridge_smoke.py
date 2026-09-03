#!/usr/bin/env python3
"""Bounded Route196 smoke for the explicit two-pass R/K language bridge.

Pass 1 uses U-independent spatial queries, ORION visual tokens, and route text
to predict task relevance R. Pass 2 appends compact K=U*R bridge tokens to the
original ORION visual and Stage1-UQ tokens for QA generation. No trajectory,
direct control, Density UQ, or hard governor is trained or used.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping

import numpy as np
import torch

from mmcv.utils import set_random_seed

from scripts.train_stage2l_route196_overfit_smoke import (
    IMAGE_TOKEN_INDEX,
    ORION_VISUAL_TOKENS,
    SPATIAL_UQ_TOKENS,
    _load_inputs,
    _load_json,
    _load_orion_lm,
    _load_records,
    _prompt_tokens,
    _route_text,
    _training_tokens,
)
from uq_estimator.stage2l_pilot import (
    balanced_driving_epoch,
    driving_stance_counts,
    matched_answer_preference_loss,
    parse_planning_stance,
    sha256_file,
)
from uq_estimator.stage2l_bridge_runtime import (
    build_two_pass_language_conditioning,
    extract_relevance_query_grid,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    fixed_task_risk,
    matched_task_risk_ranking_loss,
    task_relevance_loss,
)


SCHEMA = "orion.stage2l_route196_bridge_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
BRIDGE_TOKENS = 7


def _relevance_pass(
    *, lm, tokenizer, relevance_queries, relevance_head, baseline_vision,
    relevance_target, map_row, route_text,
) -> Dict[str, torch.Tensor]:
    prompt = _prompt_tokens(tokenizer, map_row, route_text).cuda(non_blocking=True)
    attention = prompt.ne(tokenizer.pad_token_id or 0)
    queries = relevance_queries(
        batch_size=1,
        views=6,
        device=baseline_vision.device,
        dtype=baseline_vision.dtype,
    )
    vision = torch.cat((baseline_vision, queries), dim=1)
    output = lm(
        input_ids=prompt,
        attention_mask=attention,
        images=vision,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    query_grid = extract_relevance_query_grid(
        output.hidden_states[-1], prompt,
        image_token_index=IMAGE_TOKEN_INDEX,
        visual_token_count=ORION_VISUAL_TOKENS,
        views=6,
        grid_h=10,
        grid_w=10,
    )
    logits = relevance_head(query_grid)
    return {
        "relevance_logits": logits,
        "map_loss": task_relevance_loss(logits, relevance_target),
    }


def _condition_variant(
    *, uq_tokenizer, risk_bridge, baseline_vision, components,
    relevance_logits,
) -> Dict[str, torch.Tensor]:
    conditioned = build_two_pass_language_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        baseline_vision=baseline_vision,
        components=components,
        relevance_logits=relevance_logits,
    )
    vision = conditioned.vision_tokens
    expected_tokens = ORION_VISUAL_TOKENS + SPATIAL_UQ_TOKENS + BRIDGE_TOKENS
    if vision.shape[1:] != (expected_tokens, 4096):
        raise RuntimeError("two-pass bridge vision span is malformed")
    return {
        "vision": vision,
        "task_risk": conditioned.task_risk,
        "bridge_global_features": conditioned.bridge_global_features,
    }


def _language_loss(
    *, lm, tokenizer, vision, row: Mapping[str, Any], route_text: str,
    answer: str = None,
) -> torch.Tensor:
    input_ids, labels = _training_tokens(
        tokenizer, row, route_text, answer=answer
    )
    input_ids = input_ids.unsqueeze(0).cuda(non_blocking=True)
    labels = labels.unsqueeze(0).cuda(non_blocking=True)
    attention = input_ids.ne(tokenizer.pad_token_id or 0)
    output = lm(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        images=vision,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return output.loss


@torch.no_grad()
def _generate(
    *, lm, tokenizer, vision, row: Mapping[str, Any], route_text: str,
) -> str:
    prompt = _prompt_tokens(tokenizer, row, route_text).cuda(non_blocking=True)
    output_ids = lm.generate(
        inputs=prompt,
        images=vision,
        do_sample=False,
        num_beams=1,
        max_new_tokens=72,
        use_cache=True,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


@torch.no_grad()
def _evaluate(
    *, lm, tokenizer, uq_tokenizer, relevance_queries, relevance_head,
    risk_bridge, baseline_vision, components, relevance_target, records,
    route_text, map_row,
) -> Dict[str, Any]:
    modules = (lm, uq_tokenizer, relevance_queries, relevance_head, risk_bridge)
    for module in modules:
        module.eval()
    relevance = _relevance_pass(
        lm=lm, tokenizer=tokenizer, relevance_queries=relevance_queries,
        relevance_head=relevance_head, baseline_vision=baseline_vision,
        relevance_target=relevance_target, map_row=map_row,
        route_text=route_text,
    )
    conditioned = {
        variant: _condition_variant(
            uq_tokenizer=uq_tokenizer, risk_bridge=risk_bridge,
            baseline_vision=baseline_vision, components=value,
            relevance_logits=relevance["relevance_logits"],
        )
        for variant, value in components.items()
    }
    language_nll = [
        float(_language_loss(
            lm=lm, tokenizer=tokenizer,
            vision=conditioned[row["counterfactual"]["variant"]]["vision"],
            row=row, route_text=route_text,
        ).item())
        for row in records
    ]
    implication = {
        row["counterfactual"]["variant"]: row
        for row in records if row["question_family"] == "driving_implication"
    }
    maintain_answer = implication["zero_uq"]["conversation"][1]["value"]
    conservative_answer = implication["on_path_uq"]["conversation"][1]["value"]
    expected_nll = {}
    alternative_nll = {}
    generated = {}
    for variant in ("zero_uq", "off_path_uq", "on_path_uq"):
        row = implication[variant]
        vision = conditioned[variant]["vision"]
        expected_nll[variant] = float(_language_loss(
            lm=lm, tokenizer=tokenizer, vision=vision, row=row,
            route_text=route_text,
        ).item())
        alternative = (
            conservative_answer if variant != "on_path_uq" else maintain_answer
        )
        alternative_nll[variant] = float(_language_loss(
            lm=lm, tokenizer=tokenizer, vision=vision, row=row,
            route_text=route_text, answer=alternative,
        ).item())
        generated[variant] = _generate(
            lm=lm, tokenizer=tokenizer, vision=vision, row=row,
            route_text=route_text,
        )
    on_peak = conditioned["on_path_uq"]["task_risk"].flatten(1).amax(dim=1)
    off_peak = conditioned["off_path_uq"]["task_risk"].flatten(1).amax(dim=1)
    metrics = {
        "mean_language_nll": float(np.mean(language_nll)),
        "relevance_bce": float(relevance["map_loss"].item()),
        "on_minus_off_peak_task_risk": float((on_peak - off_peak).mean().item()),
        "expected_answer_nll": expected_nll,
        "counterfactual_answer_nll": alternative_nll,
        "generated_driving_implication": generated,
        "generated_stances": {
            key: parse_planning_stance(value) for key, value in generated.items()
        },
        "bridge_global_features": {
            key: value["bridge_global_features"].detach().cpu().tolist()
            for key, value in conditioned.items()
        },
    }
    for module in modules:
        module.train()
    return metrics


def _validate_authorization(
    protocol: Dict[str, Any],
    amendment: Dict[str, Any],
    *,
    protocol_path: Path,
    output_dir: Path,
    max_steps: int,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported bridge training protocol")
    bridge = protocol.get("task_risk_language_bridge", {})
    if (
        bridge.get("passes") != 2
        or bridge.get("relevance_queries_receive_uq") is not False
        or bridge.get("bridge_token_count") != BRIDGE_TOKENS
    ):
        raise ValueError("protocol does not describe the implemented bridge")
    maximum = int(protocol.get("route196_bridge_smoke", {}).get("maximum_steps", -1))
    if max_steps < 1 or max_steps > maximum:
        raise ValueError("max-steps exceeds bridge-smoke authorization")
    key = protocol.get("launch_authorization_key")
    locks = amendment.get("launch_locks", {})
    authorized = amendment.get("authorized_run", {})
    validated = amendment.get("validated_inputs", {})
    project_root = Path(__file__).resolve().parents[1]
    current_hashes = {
        "training_protocol_sha256": sha256_file(protocol_path.resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "uq_relevance_bridge_sha256": sha256_file(
            project_root / "uq_estimator" / "uq_relevance_tokenizer.py"
        ),
        "two_pass_runtime_sha256": sha256_file(
            project_root / "uq_estimator" / "stage2l_bridge_runtime.py"
        ),
    }
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or locks.get(key) is not True
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or int(authorized.get("maximum_submissions", 0)) != 1
        or authorized.get("fresh_initialization_from_original_orion_checkpoint") is not True
        or Path(str(authorized.get("output_root", ""))).resolve()
        != output_dir.resolve()
        or any(validated.get(name) != value for name, value in current_hashes.items())
    ):
        raise ValueError("bridge launch amendment is not active and fail-closed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-head", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite bridge smoke output")
    protocol = _load_json(args.training_protocol.resolve())
    amendment = _load_json(args.launch_amendment.resolve())
    _validate_authorization(
        protocol,
        amendment,
        protocol_path=args.training_protocol,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    if not torch.cuda.is_available():
        raise SystemExit("real Stage2-L bridge smoke requires CUDA")
    losses = protocol["losses"]
    lambda_map = float(losses["dense_soft_relevance_bce"])
    lambda_ranking = float(losses["matched_onpath_offpath_risk_ranking"])
    preference = losses["matched_answer_preference"]
    lambda_preference = float(preference["weight"])
    preference_margin = float(preference["margin"])
    if float(losses.get("trajectory", -1.0)) != 0.0:
        raise ValueError("bridge smoke may not train trajectory loss")

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    records_path = args.records.resolve()
    records = _load_records(records_path)
    if amendment["authorized_run"].get("event_id") != records[0]["event_id"]:
        raise ValueError("bridge launch amendment authorizes a different event")
    components, relevance_target, route_context = _load_inputs(records, records_path)
    route_text = _route_text(route_context)
    map_row = next(
        row for row in records
        if row["counterfactual"]["variant"] == "observed"
        and row["question_family"] == "task_relevance"
    )
    cache = torch.load(args.visual_cache.resolve(), map_location="cpu")
    if cache.get("schema") != "orion.closedloop_visual_context_cache.v1":
        raise ValueError("unsupported ORION visual-context cache")
    if cache.get("metadata", {}).get("event_id") != records[0]["event_id"]:
        raise ValueError("visual cache and QA records identify different events")
    baseline_vision = cache["baseline_vision"].float().cuda()
    if baseline_vision.shape != (1, ORION_VISUAL_TOKENS, 4096):
        raise ValueError("visual cache shape mismatch")
    relevance_target = relevance_target.cuda()

    lm, tokenizer = _load_orion_lm(args.config.resolve(), args.checkpoint.resolve())
    uq_tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    risk_bridge = TaskRiskLanguageBridge(model_dim=4096, hidden_dim=256).cuda()
    trainable_modules = (
        uq_tokenizer, relevance_queries, relevance_head, risk_bridge
    )
    lora_parameters = [p for p in lm.parameters() if p.requires_grad]
    auxiliary_parameters = [
        p for module in trainable_modules for p in module.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate_lora},
            {"params": auxiliary_parameters, "lr": args.learning_rate_head},
        ],
        weight_decay=1e-4,
    )
    before = _evaluate(
        lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries, relevance_head=relevance_head,
        risk_bridge=risk_bridge, baseline_vision=baseline_vision,
        components=components, relevance_target=relevance_target,
        records=records, route_text=route_text, map_row=map_row,
    )
    implication = {
        row["counterfactual"]["variant"]: row
        for row in records if row["question_family"] == "driving_implication"
    }
    maintain_answer = implication["zero_uq"]["conversation"][1]["value"]
    conservative_answer = implication["on_path_uq"]["conversation"][1]["value"]
    rng = random.Random(args.seed)
    epoch_rows = balanced_driving_epoch(records, rng)
    history = []
    for step in range(1, args.max_steps + 1):
        if step > 1 and (step - 1) % len(epoch_rows) == 0:
            epoch_rows = balanced_driving_epoch(records, rng)
        row = epoch_rows[(step - 1) % len(epoch_rows)]
        optimizer.zero_grad(set_to_none=True)
        relevance = _relevance_pass(
            lm=lm, tokenizer=tokenizer, relevance_queries=relevance_queries,
            relevance_head=relevance_head, baseline_vision=baseline_vision,
            relevance_target=relevance_target, map_row=map_row,
            route_text=route_text,
        )
        variant = row["counterfactual"]["variant"]
        conditioned = _condition_variant(
            uq_tokenizer=uq_tokenizer, risk_bridge=risk_bridge,
            baseline_vision=baseline_vision, components=components[variant],
            relevance_logits=relevance["relevance_logits"],
        )
        language_loss = _language_loss(
            lm=lm, tokenizer=tokenizer, vision=conditioned["vision"],
            row=row, route_text=route_text,
        )
        preference_loss = language_loss.new_zeros(())
        if row["question_family"] == "driving_implication":
            stance = row["target"]["structured_summary"][
                "planning_implication"
            ]["stance"]
            alternative_answer = (
                conservative_answer if stance == "maintain" else maintain_answer
            )
            alternative_loss = _language_loss(
                lm=lm, tokenizer=tokenizer, vision=conditioned["vision"],
                row=row, route_text=route_text, answer=alternative_answer,
            )
            preference_loss = matched_answer_preference_loss(
                language_loss, alternative_loss, margin=preference_margin
            )
        on_uq = uq_tokenizer(components["on_path_uq"].cuda(non_blocking=True))
        off_uq = uq_tokenizer(components["off_path_uq"].cuda(non_blocking=True))
        on = fixed_task_risk(
            on_uq.latest_scalar_uq, relevance["relevance_logits"]
        )
        off = fixed_task_risk(
            off_uq.latest_scalar_uq, relevance["relevance_logits"]
        )
        ranking_loss = matched_task_risk_ranking_loss(on, off, margin=0.2)
        loss = (
            language_loss
            + lambda_map * relevance["map_loss"]
            + lambda_ranking * ranking_loss
            + lambda_preference * preference_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            lora_parameters + auxiliary_parameters, 1.0
        )
        optimizer.step()
        if step == 1 or step % args.log_interval == 0:
            item = {
                "step": step,
                "variant": variant,
                "question_family": row["question_family"],
                "loss": float(loss.item()),
                "language_nll": float(language_loss.item()),
                "relevance_bce": float(relevance["map_loss"].item()),
                "ranking_loss": float(ranking_loss.item()),
                "answer_preference_loss": float(preference_loss.item()),
            }
            history.append(item)
            print("[Stage2LBridge] " + json.dumps(item, sort_keys=True), flush=True)

    after = _evaluate(
        lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries, relevance_head=relevance_head,
        risk_bridge=risk_bridge, baseline_vision=baseline_vision,
        components=components, relevance_target=relevance_target,
        records=records, route_text=route_text, map_row=map_row,
    )
    expected = after["expected_answer_nll"]
    alternative = after["counterfactual_answer_nll"]
    target_on_stance = implication["on_path_uq"]["target"][
        "structured_summary"
    ]["planning_implication"]["stance"]
    checks = {
        "optimization_reduces_language_nll": after["mean_language_nll"] < before["mean_language_nll"],
        "optimization_reduces_relevance_bce": after["relevance_bce"] < before["relevance_bce"],
        "on_path_risk_exceeds_off_path_by_margin": after["on_minus_off_peak_task_risk"] >= 0.2,
        "zero_uq_prefers_maintain": expected["zero_uq"] < alternative["zero_uq"],
        "off_path_prefers_maintain": expected["off_path_uq"] < alternative["off_path_uq"],
        "on_path_prefers_conservative": expected["on_path_uq"] < alternative["on_path_uq"],
        "generated_zero_uq_is_maintain": after["generated_stances"]["zero_uq"] == "maintain",
        "generated_off_path_is_maintain": after["generated_stances"]["off_path_uq"] == "maintain",
        "generated_on_path_matches_target": after["generated_stances"]["on_path_uq"] == target_on_stance,
    }
    passed = all(checks.values())
    status = "engineering_bridge_overfit_pass" if passed else "engineering_bridge_overfit_failed_gate"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_route196_bridge.pt"
    torch.save({
        "schema": SCHEMA,
        "status": status,
        "uq_tokenizer": uq_tokenizer.state_dict(),
        "relevance_queries": relevance_queries.state_dict(),
        "relevance_head": relevance_head.state_dict(),
        "risk_bridge": risk_bridge.state_dict(),
        "lora": {
            name: value.detach().cpu() for name, value in lm.state_dict().items()
            if "lora_" in name
        },
        "steps": args.max_steps,
    }, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "status": status,
        "engineering_overfit_only": True,
        "formal_training_ready": False,
        "stage2l_pilot_training_ready": passed,
        "steps": args.max_steps,
        "sampling": {
            "balance_driving_stances": True,
            "source_driving_stance_counts": driving_stance_counts(records),
            "effective_epoch_record_count": len(epoch_rows),
        },
        "architecture": {
            "passes": 2,
            "relevance_queries_receive_uq": False,
            "bridge_token_count": BRIDGE_TOKENS,
            "task_risk": "K = U * sigmoid(R)",
            "direct_control": False,
            "trajectory": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
        },
        "before": before,
        "after": after,
        "checks": checks,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path.resolve()),
        },
        "inputs": {
            "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
            "visual_cache": {"path": str(args.visual_cache.resolve()), "sha256": sha256_file(args.visual_cache.resolve())},
            "training_protocol": {"path": str(args.training_protocol.resolve()), "sha256": sha256_file(args.training_protocol.resolve())},
            "launch_amendment": {"path": str(args.launch_amendment.resolve()), "sha256": sha256_file(args.launch_amendment.resolve())},
            "base_orion_checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint.resolve())},
        },
        "claim_boundary": "One-event bridge learnability smoke only; no held-out, trajectory, closed-loop, or safety claim."
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del baseline_vision
    gc.collect()
    print(json.dumps({"status": status, "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
