#!/usr/bin/env python3
"""Bounded Route196 smoke for the structured Stage2-L semantic bottleneck.

Pass 1 predicts task relevance R from U-independent spatial queries, ORION
vision, and route language.  Pass 2 forms K=U*R, predicts a three-way planning
stance from K, and appends the predicted soft stance token for QA generation.
Ground-truth stance labels are loss-only and are never forwarded as inputs.
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

from scripts.train_stage2l_route196_bridge_smoke import (
    BRIDGE_TOKENS,
    _generate,
    _language_loss,
    _relevance_pass,
)
from scripts.train_stage2l_route196_overfit_smoke import (
    ORION_VISUAL_TOKENS,
    SPATIAL_UQ_TOKENS,
    _load_inputs,
    _load_json,
    _load_orion_lm,
    _load_records,
    _route_text,
)
from uq_estimator.stage2l_pilot import (
    balanced_driving_epoch,
    driving_stance_counts,
    matched_answer_preference_loss,
    parse_planning_stance,
    sha256_file,
)
from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    PlanningStanceSemanticBottleneck,
    encode_planning_stances,
    planning_stance_loss,
)
from uq_estimator.stage2l_semantic_runtime import (
    build_structured_semantic_conditioning,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    fixed_task_risk,
    matched_task_risk_ranking_loss,
)


SCHEMA = "orion.stage2l_route196_structured_semantic_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
SEMANTIC_TOKENS = 1


def _target_stance(row: Mapping[str, Any]) -> str:
    return str(
        row["target"]["structured_summary"]["planning_implication"]["stance"]
    )


def _condition_variant(
    *,
    uq_tokenizer,
    risk_bridge,
    stance_bottleneck,
    baseline_vision,
    components,
    relevance_logits,
) -> Dict[str, torch.Tensor]:
    conditioned = build_structured_semantic_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        stance_bottleneck=stance_bottleneck,
        baseline_vision=baseline_vision,
        components=components,
        relevance_logits=relevance_logits,
    )
    expected_tokens = (
        ORION_VISUAL_TOKENS + SPATIAL_UQ_TOKENS + BRIDGE_TOKENS + SEMANTIC_TOKENS
    )
    if conditioned.vision_tokens.shape[1:] != (expected_tokens, 4096):
        raise RuntimeError("structured semantic vision span is malformed")
    return {
        "vision": conditioned.vision_tokens,
        "task_risk": conditioned.task_risk,
        "bridge_global_features": conditioned.bridge_global_features,
        "stance_logits": conditioned.stance_logits,
        "stance_probabilities": conditioned.stance_probabilities,
        "predicted_stance_indices": conditioned.predicted_stance_indices,
    }


@torch.no_grad()
def _evaluate(
    *,
    lm,
    tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    stance_bottleneck,
    baseline_vision,
    components,
    relevance_target,
    records,
    route_text,
    map_row,
) -> Dict[str, Any]:
    modules = (
        lm,
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
        stance_bottleneck,
    )
    for module in modules:
        module.eval()
    relevance = _relevance_pass(
        lm=lm,
        tokenizer=tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        baseline_vision=baseline_vision,
        relevance_target=relevance_target,
        map_row=map_row,
        route_text=route_text,
    )
    conditioned = {
        variant: _condition_variant(
            uq_tokenizer=uq_tokenizer,
            risk_bridge=risk_bridge,
            stance_bottleneck=stance_bottleneck,
            baseline_vision=baseline_vision,
            components=value,
            relevance_logits=relevance["relevance_logits"],
        )
        for variant, value in components.items()
    }
    language_nll = [
        float(
            _language_loss(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned[row["counterfactual"]["variant"]]["vision"],
                row=row,
                route_text=route_text,
            ).item()
        )
        for row in records
    ]
    implication = {
        row["counterfactual"]["variant"]: row
        for row in records
        if row["question_family"] == "driving_implication"
    }
    maintain_answer = implication["zero_uq"]["conversation"][1]["value"]
    conservative_answer = implication["on_path_uq"]["conversation"][1]["value"]
    expected_nll: Dict[str, float] = {}
    alternative_nll: Dict[str, float] = {}
    generated: Dict[str, str] = {}
    generated_stances: Dict[str, Any] = {}
    structured_predictions: Dict[str, str] = {}
    structured_probabilities: Dict[str, Any] = {}
    structured_target_probabilities: Dict[str, float] = {}
    structured_losses = []
    for variant, row in implication.items():
        value = conditioned[variant]
        target_stance = _target_stance(row)
        target = encode_planning_stances(
            [target_stance], device=value["stance_logits"].device
        )
        structured_losses.append(
            planning_stance_loss(value["stance_logits"], target)
        )
        predicted_index = int(value["predicted_stance_indices"].item())
        structured_predictions[variant] = PLANNING_STANCES[predicted_index]
        probabilities = value["stance_probabilities"][0]
        structured_probabilities[variant] = {
            name: float(probabilities[index].item())
            for index, name in enumerate(PLANNING_STANCES)
        }
        structured_target_probabilities[variant] = float(
            probabilities[int(target.item())].item()
        )
        generated[variant] = _generate(
            lm=lm,
            tokenizer=tokenizer,
            vision=value["vision"],
            row=row,
            route_text=route_text,
        )
        generated_stances[variant] = parse_planning_stance(generated[variant])

    for variant in ("zero_uq", "off_path_uq", "on_path_uq"):
        row = implication[variant]
        vision = conditioned[variant]["vision"]
        expected_nll[variant] = float(
            _language_loss(
                lm=lm,
                tokenizer=tokenizer,
                vision=vision,
                row=row,
                route_text=route_text,
            ).item()
        )
        alternative = conservative_answer if variant != "on_path_uq" else maintain_answer
        alternative_nll[variant] = float(
            _language_loss(
                lm=lm,
                tokenizer=tokenizer,
                vision=vision,
                row=row,
                route_text=route_text,
                answer=alternative,
            ).item()
        )

    target_stances = {
        variant: _target_stance(row) for variant, row in implication.items()
    }
    structured_accuracy = float(
        np.mean(
            [
                structured_predictions[variant] == target
                for variant, target in target_stances.items()
            ]
        )
    )
    generated_accuracy = float(
        np.mean(
            [
                generated_stances[variant] == target
                for variant, target in target_stances.items()
            ]
        )
    )
    on_peak = conditioned["on_path_uq"]["task_risk"].flatten(1).amax(dim=1)
    off_peak = conditioned["off_path_uq"]["task_risk"].flatten(1).amax(dim=1)
    metrics = {
        "mean_language_nll": float(np.mean(language_nll)),
        "relevance_bce": float(relevance["map_loss"].item()),
        "mean_structured_stance_ce": float(
            torch.stack(structured_losses).mean().item()
        ),
        "structured_stance_accuracy": structured_accuracy,
        "generated_stance_accuracy": generated_accuracy,
        "minimum_structured_target_probability": min(
            structured_target_probabilities.values()
        ),
        "on_minus_off_peak_task_risk": float((on_peak - off_peak).mean().item()),
        "expected_answer_nll": expected_nll,
        "counterfactual_answer_nll": alternative_nll,
        "structured_stance_predictions": structured_predictions,
        "structured_stance_probabilities": structured_probabilities,
        "structured_target_probabilities": structured_target_probabilities,
        "target_stances": target_stances,
        "generated_driving_implication": generated,
        "generated_stances": generated_stances,
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
        raise ValueError("unsupported semantic training protocol")
    bridge = protocol.get("task_risk_language_bridge", {})
    semantic = protocol.get("structured_semantic_bottleneck", {})
    if (
        bridge.get("passes") != 2
        or bridge.get("relevance_queries_receive_uq") is not False
        or bridge.get("bridge_token_count") != BRIDGE_TOKENS
        or semantic.get("semantic_token_count") != SEMANTIC_TOKENS
        or semantic.get("ground_truth_stance_enters_forward") is not False
        or tuple(semantic.get("classes", ())) != PLANNING_STANCES
    ):
        raise ValueError("protocol does not describe the implemented semantics")
    maximum = int(
        protocol.get("route196_structured_semantic_smoke", {}).get(
            "maximum_steps", -1
        )
    )
    if max_steps < 1 or max_steps > maximum:
        raise ValueError("max-steps exceeds semantic-smoke authorization")
    key = protocol.get("launch_authorization_key")
    locks = amendment.get("launch_locks", {})
    authorized = amendment.get("authorized_run", {})
    validated = amendment.get("validated_inputs", {})
    project_root = Path(__file__).resolve().parents[1]
    current_hashes = {
        "training_protocol_sha256": sha256_file(protocol_path.resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "bridge_trainer_helpers_sha256": sha256_file(
            project_root / "scripts" / "train_stage2l_route196_bridge_smoke.py"
        ),
        "uq_relevance_bridge_sha256": sha256_file(
            project_root / "uq_estimator" / "uq_relevance_tokenizer.py"
        ),
        "two_pass_runtime_sha256": sha256_file(
            project_root / "uq_estimator" / "stage2l_bridge_runtime.py"
        ),
        "semantic_bottleneck_sha256": sha256_file(
            project_root / "uq_estimator" / "stage2l_semantic_bottleneck.py"
        ),
        "semantic_runtime_sha256": sha256_file(
            project_root / "uq_estimator" / "stage2l_semantic_runtime.py"
        ),
    }
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or locks.get(key) is not True
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or int(authorized.get("maximum_submissions", 0)) != 1
        or int(authorized.get("maximum_steps", -1)) != maximum
        or max_steps != maximum
        or authorized.get("fresh_initialization_from_original_orion_checkpoint")
        is not True
        or Path(str(authorized.get("output_root", ""))).resolve()
        != output_dir.resolve()
        or any(validated.get(name) != value for name, value in current_hashes.items())
    ):
        raise ValueError("semantic launch amendment is not active and fail-closed")


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
        raise FileExistsError("refusing to overwrite semantic smoke output")
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
        raise SystemExit("real Stage2-L semantic smoke requires CUDA")
    losses = protocol["losses"]
    lambda_map = float(losses["dense_soft_relevance_bce"])
    lambda_ranking = float(losses["matched_onpath_offpath_risk_ranking"])
    lambda_stance = float(losses["structured_stance_cross_entropy"])
    preference = losses["matched_answer_preference"]
    lambda_preference = float(preference["weight"])
    preference_margin = float(preference["margin"])
    if float(losses.get("trajectory", -1.0)) != 0.0:
        raise ValueError("semantic smoke may not train trajectory loss")

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    records_path = args.records.resolve()
    records = _load_records(records_path)
    if amendment["authorized_run"].get("event_id") != records[0]["event_id"]:
        raise ValueError("semantic launch amendment authorizes a different event")
    components, relevance_target, route_context = _load_inputs(records, records_path)
    route_text = _route_text(route_context)
    map_row = next(
        row
        for row in records
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
    stance_bottleneck = PlanningStanceSemanticBottleneck(
        model_dim=4096, hidden_dim=256
    ).cuda()
    trainable_modules = (
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
        stance_bottleneck,
    )
    lora_parameters = [parameter for parameter in lm.parameters() if parameter.requires_grad]
    auxiliary_parameters = [
        parameter for module in trainable_modules for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate_lora},
            {"params": auxiliary_parameters, "lr": args.learning_rate_head},
        ],
        weight_decay=1e-4,
    )
    before = _evaluate(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        stance_bottleneck=stance_bottleneck,
        baseline_vision=baseline_vision,
        components=components,
        relevance_target=relevance_target,
        records=records,
        route_text=route_text,
        map_row=map_row,
    )
    implication = {
        row["counterfactual"]["variant"]: row
        for row in records
        if row["question_family"] == "driving_implication"
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
            lm=lm,
            tokenizer=tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            baseline_vision=baseline_vision,
            relevance_target=relevance_target,
            map_row=map_row,
            route_text=route_text,
        )
        variant = row["counterfactual"]["variant"]
        conditioned = _condition_variant(
            uq_tokenizer=uq_tokenizer,
            risk_bridge=risk_bridge,
            stance_bottleneck=stance_bottleneck,
            baseline_vision=baseline_vision,
            components=components[variant],
            relevance_logits=relevance["relevance_logits"],
        )
        language_loss = _language_loss(
            lm=lm,
            tokenizer=tokenizer,
            vision=conditioned["vision"],
            row=row,
            route_text=route_text,
        )
        stance_loss = language_loss.new_zeros(())
        preference_loss = language_loss.new_zeros(())
        target_stance = None
        if row["question_family"] == "driving_implication":
            target_stance = _target_stance(row)
            target = encode_planning_stances(
                [target_stance], device=conditioned["stance_logits"].device
            )
            stance_loss = planning_stance_loss(
                conditioned["stance_logits"], target
            )
            alternative_answer = (
                conservative_answer if target_stance == "maintain" else maintain_answer
            )
            alternative_loss = _language_loss(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned["vision"],
                row=row,
                route_text=route_text,
                answer=alternative_answer,
            )
            preference_loss = matched_answer_preference_loss(
                language_loss, alternative_loss, margin=preference_margin
            )
        on_uq = uq_tokenizer(components["on_path_uq"].cuda(non_blocking=True))
        off_uq = uq_tokenizer(components["off_path_uq"].cuda(non_blocking=True))
        on = fixed_task_risk(on_uq.latest_scalar_uq, relevance["relevance_logits"])
        off = fixed_task_risk(off_uq.latest_scalar_uq, relevance["relevance_logits"])
        ranking_loss = matched_task_risk_ranking_loss(on, off, margin=0.2)
        loss = (
            language_loss
            + lambda_map * relevance["map_loss"]
            + lambda_ranking * ranking_loss
            + lambda_stance * stance_loss
            + lambda_preference * preference_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters + auxiliary_parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_interval == 0:
            item = {
                "step": step,
                "variant": variant,
                "question_family": row["question_family"],
                "target_stance": target_stance,
                "loss": float(loss.item()),
                "language_nll": float(language_loss.item()),
                "relevance_bce": float(relevance["map_loss"].item()),
                "ranking_loss": float(ranking_loss.item()),
                "structured_stance_ce": float(stance_loss.item()),
                "answer_preference_loss": float(preference_loss.item()),
            }
            history.append(item)
            print("[Stage2LSemantic] " + json.dumps(item, sort_keys=True), flush=True)

    after = _evaluate(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        stance_bottleneck=stance_bottleneck,
        baseline_vision=baseline_vision,
        components=components,
        relevance_target=relevance_target,
        records=records,
        route_text=route_text,
        map_row=map_row,
    )
    expected = after["expected_answer_nll"]
    alternative = after["counterfactual_answer_nll"]
    checks = {
        "optimization_reduces_language_nll": after["mean_language_nll"]
        < before["mean_language_nll"],
        "optimization_reduces_relevance_bce": after["relevance_bce"]
        < before["relevance_bce"],
        "optimization_reduces_structured_stance_ce": after[
            "mean_structured_stance_ce"
        ]
        < before["mean_structured_stance_ce"],
        "all_structured_stances_match_targets": after[
            "structured_stance_accuracy"
        ]
        == 1.0,
        "structured_target_probability_floor": after[
            "minimum_structured_target_probability"
        ]
        >= 0.55,
        "on_path_risk_exceeds_off_path_by_margin": after[
            "on_minus_off_peak_task_risk"
        ]
        >= 0.2,
        "zero_uq_prefers_maintain": expected["zero_uq"]
        < alternative["zero_uq"],
        "off_path_prefers_maintain": expected["off_path_uq"]
        < alternative["off_path_uq"],
        "on_path_prefers_conservative": expected["on_path_uq"]
        < alternative["on_path_uq"],
        "all_generated_stances_match_targets": after["generated_stance_accuracy"]
        == 1.0,
    }
    passed = all(checks.values())
    status = (
        "engineering_structured_semantic_overfit_pass"
        if passed
        else "engineering_structured_semantic_overfit_failed_gate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_route196_structured_semantic.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "status": status,
            "uq_tokenizer": uq_tokenizer.state_dict(),
            "relevance_queries": relevance_queries.state_dict(),
            "relevance_head": relevance_head.state_dict(),
            "risk_bridge": risk_bridge.state_dict(),
            "stance_bottleneck": stance_bottleneck.state_dict(),
            "planning_stances": PLANNING_STANCES,
            "lora": {
                name: value.detach().cpu()
                for name, value in lm.state_dict().items()
                if "lora_" in name
            },
            "steps": args.max_steps,
        },
        checkpoint_path,
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "engineering_overfit_only": True,
        "formal_training_ready": False,
        "stage2l_pilot_training_ready": False,
        "stage2l_pilot_migration_review_ready": passed,
        "stage2p_ready": False,
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
            "semantic_token_count": SEMANTIC_TOKENS,
            "semantic_token_uses_predicted_distribution": True,
            "ground_truth_stance_enters_forward": False,
            "planning_stances": PLANNING_STANCES,
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
            "records": {
                "path": str(records_path),
                "sha256": sha256_file(records_path),
            },
            "visual_cache": {
                "path": str(args.visual_cache.resolve()),
                "sha256": sha256_file(args.visual_cache.resolve()),
            },
            "training_protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
            },
            "launch_amendment": {
                "path": str(args.launch_amendment.resolve()),
                "sha256": sha256_file(args.launch_amendment.resolve()),
            },
            "base_orion_checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint.resolve()),
            },
        },
        "claim_boundary": "One-event structured semantic learnability smoke only; no held-out, trajectory, closed-loop, or safety claim.",
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
