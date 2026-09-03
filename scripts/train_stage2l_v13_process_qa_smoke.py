#!/usr/bin/env python3
"""Bounded Stage2-L v13 structured-process QA capacity smoke.

Both capacity arms preserve the same causal architecture:

* Stage1 U and the U tokenizer are frozen.
* A U-independent ORION/VLM pass predicts factorized route/actor R once and
  exposes its 600 view/grid hidden tokens.  A fixed, parameter-free 5x5
  average pool produces 150 language-pass R tokens without changing the
  full-resolution 10x10 R supervision.
* Stage1's 600 U tokens and those 150 pooled R hidden tokens enter the second
  ORION/VLM pass directly.  There is no learned K bridge and K is never a
  model input.
* The second pass is supervised on observation -> epistemic limit -> task
  binding -> decision process answers, while dense component R retains direct
  spatial supervision.

The ``lora`` arm trains the existing ORION LoRA.  The
``partial_unfreeze`` arm additionally trains the final decoder layers.  This
is an engineering capacity comparison, not formal Stage2-L or safety evidence.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import scripts.train_stage2l_v11_identifiable_smoke as v11
import scripts.train_stage2l_v121_factorized_r_smoke as v121
import scripts.train_stage2l_v122_vertical_slice_semantic_smoke as v122
from uq_estimator.stage2l_factorized_relevance_v121 import (
    FactorizedTaskRelevanceMapHead,
    factorized_relevance_terms_v121,
)
from uq_estimator.stage2l_identifiability import (
    REQUIRED_RISK_VARIANTS,
    audit_matched_task_risk,
)
from uq_estimator.stage2l_process_qa_v13 import (
    PROCESS_FAMILIES,
    TRAINING_ARMS,
    audit_matched_process_chains,
    build_process_step_row,
    build_structured_process_chain,
    build_structured_process_row,
    configure_trainable_scope,
    detached_conditioning_gradient_anchor,
)
from uq_estimator.uq_relevance_tokenizer import (
    ViewAlignedTaskRelevanceQueryTokenizer,
)


SCHEMA = "orion.stage2l_v13_process_qa_capacity_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v13_process_qa_capacity_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v13_process_qa_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v13_process_qa_launch.v1"
V121_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke.v1"


@dataclass(frozen=True)
class DirectMatchedConditioningV13:
    conditioning_by_variant: Mapping[str, torch.Tensor]
    latest_scalar_uq_by_variant: Mapping[str, torch.Tensor]
    no_u_ablation_vision: torch.Tensor
    task_risk_diagnostic: Any
    baseline_token_count: int
    direct_u_token_count: int
    direct_r_token_count: int
    k_input_token_count: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _process_module_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "uq_estimator"
        / "stage2l_process_qa_v13.py"
    )


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    cached = getattr(args, "_validated_inputs_cache", None)
    if cached is not None:
        return dict(cached)
    value = {
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "process_module_sha256": _sha256(_process_module_path()),
        "v122_lineage_helper_sha256": _sha256(
            Path(v122.__file__).resolve()
        ),
        "factorized_relevance_sha256": _sha256(
            Path(v122._factorized_path()).resolve()
        ),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "v11_records_sha256": _sha256(args.v11_records.resolve()),
        "dataset_audit_report_sha256": _sha256(
            args.dataset_audit_report.resolve()
        ),
        "view_feature_cache_sha256": _sha256(
            args.view_feature_cache.resolve()
        ),
        "u_tokenizer_checkpoint_sha256": _sha256(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "v121_checkpoint_sha256": _sha256(args.v121_checkpoint.resolve()),
        "v121_report_sha256": _sha256(args.v121_report.resolve()),
        "v121_terminal_validation_sha256": _sha256(
            args.v121_terminal_validation.resolve()
        ),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
    }
    setattr(args, "_validated_inputs_cache", dict(value))
    return value


def _process_rows(
    assets: v11.V11Assets, group_id: str, variant: str
) -> OrderedDict[str, Mapping[str, object]]:
    return OrderedDict(
        (family, assets.row(group_id, variant, family))
        for family in PROCESS_FAMILIES
    )


def audit_process_dataset(assets: v11.V11Assets) -> Dict[str, Any]:
    per_group: Dict[str, Any] = {}
    chain_count = 0
    for group_id in sorted(assets.group_rows):
        chains = {
            variant: build_structured_process_chain(
                _process_rows(assets, group_id, variant)
            )
            for variant in REQUIRED_RISK_VARIANTS
        }
        audit = audit_matched_process_chains(chains)
        per_group[group_id] = {
            "event_id": assets.group_event[group_id],
            "split": assets.group_split[group_id],
            "checks": dict(audit.checks),
            "passed": audit.passed,
        }
        chain_count += len(chains)
    failed = [key for key, value in per_group.items() if not value["passed"]]
    return {
        "group_count": len(per_group),
        "chain_count": chain_count,
        "step_target_count": chain_count * len(PROCESS_FAMILIES),
        "failed_group_ids": failed,
        "passed": not failed,
        "per_group": per_group,
    }


def audit_process_sequence_lengths(
    args: argparse.Namespace,
    assets: v11.V11Assets,
    *,
    image_token_count: int = 1279,
) -> Dict[str, Any]:
    """Tokenize every target and fail before GPU use if 2048 is exceeded."""

    from mmcv.utils import Config
    from transformers import AutoTokenizer
    import scripts.train_stage2l_mr1_smoke as base

    cfg = Config.fromfile(str(args.config.resolve()))
    tokenizer_path = Path(str(cfg.model.tokenizer))
    if not tokenizer_path.is_absolute():
        project_root = args.config.resolve().parents[3]
        tokenizer_path = (project_root / tokenizer_path).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError("ORION tokenizer directory is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    maxima = {"full_process": 0, "single_process_step": 0}
    longest = {}
    violations = []
    for group_id in sorted(assets.group_rows):
        for variant in REQUIRED_RISK_VARIANTS:
            rows = _process_rows(assets, group_id, variant)
            candidates = [("full_process", build_structured_process_row(rows))]
            candidates.extend(
                ("single_process_step", build_process_step_row(rows, family))
                for family in PROCESS_FAMILIES
            )
            for kind, row in candidates:
                input_ids, _ = base._training_tokens(
                    tokenizer,
                    row,
                    assets.route_text[group_id],
                    answer=str(row["conversation"][1]["value"]),
                )
                unexpanded = int(input_ids.numel())
                expanded = unexpanded - 1 + int(image_token_count)
                if expanded > maxima[kind]:
                    maxima[kind] = expanded
                    longest[kind] = {
                        "group_id": group_id,
                        "variant": variant,
                        "unexpanded_tokens": unexpanded,
                        "expanded_tokens": expanded,
                    }
                if expanded > 2048:
                    violations.append(
                        {
                            "group_id": group_id,
                            "variant": variant,
                            "kind": kind,
                            "expanded_tokens": expanded,
                        }
                    )
    return {
        "model_max_length": 2048,
        "image_token_count": int(image_token_count),
        "maximum_expanded_tokens": maxima,
        "longest": longest,
        "violation_count": len(violations),
        "violations": violations,
        "passed": not violations,
    }


def _protocol_checks(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    expected = _validated_inputs(args)
    expected.pop("trainer_sha256")
    training = protocol.get("training", {})
    architecture = protocol.get("architecture", {})
    arms = protocol.get("capacity_arms", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen_bounded_capacity_comparison"
        or protocol.get("input_sha256") != expected
        or architecture.get("task_relevance_owner") != "Stage2-L ORION/VLM"
        or architecture.get("u_enters_relevance_query") is not False
        or architecture.get("direct_u_tokens_enter_orion") is not True
        or architecture.get("direct_r_tokens_enter_orion") is not True
        or architecture.get("task_risk_language_bridge_present") is not False
        or architecture.get("k_used_as_model_input") is not False
        or architecture.get("stage1_trainable") is not False
        or architecture.get("u_tokenizer_trainable") is not False
        or architecture.get("trajectory_or_control_loss") is not False
        or set(arms) != set(TRAINING_ARMS)
        or int(training.get("optimizer_steps", 0)) != 200
        or int(training.get("process_steps_per_optimizer_step", 0)) != 1
        or int(training.get("partial_unfreeze_layers", 0)) != 4
        or training.get("automatic_extension") is not False
        or training.get("automatic_retry") is not False
    ):
        raise ValueError("v13 process-QA protocol is absent or stale")
    arm = arms[args.training_arm]
    if (
        arm.get("output_root") != str(args.output_dir.resolve())
        or bool(arm.get("train_orion_lora")) is not True
        or arm.get("task_risk_language_bridge_present") is not False
        or bool(arm.get("train_last_decoder_layers"))
        is not (args.training_arm == "partial_unfreeze")
    ):
        raise ValueError("v13 capacity arm differs from runtime arguments")


def _preflight(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    assets: v11.V11Assets,
) -> Dict[str, Any]:
    audit = audit_process_dataset(assets)
    if not audit["passed"]:
        raise ValueError("v13 process dataset semantic audit failed")
    sequence_lengths = audit_process_sequence_lengths(args, assets)
    if not sequence_lengths["passed"]:
        raise ValueError("v13 direct-token sequence exceeds ORION context")
    validation = _read_json(args.v121_terminal_validation.resolve())
    terminal = v122.validate_v121_terminal_validation(validation)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "v13_process_qa_preflight_pass_training_locked",
        "passed": True,
        "gpu_used": False,
        "training_started": False,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "output_root": str(args.output_dir.resolve()),
        "training_arm": args.training_arm,
        "event_count": len(assets.event_meta),
        "train_event_count": len(assets.event_groups["train"]),
        "dev_event_count": len(assets.event_groups["dev"]),
        "process_dataset_audit": audit,
        "sequence_length_audit": sequence_lengths,
        "v121_terminal_decision_retained": terminal,
        "architecture_invariants": {
            "r_computed_once_per_matched_group": True,
            "u_enters_r_query": False,
            "direct_stage1_u_tokens_enter_orion": True,
            "direct_r_hidden_tokens_enter_orion": True,
            "task_risk_language_bridge_present": False,
            "k_used_as_model_input": False,
            "language_image_token_count": 1279,
            "stage1_frozen": True,
            "u_tokenizer_frozen": True,
            "trajectory_or_control_loss": False,
        },
        "claim_boundary": (
            "CPU/data/lineage preflight for one v13 capacity arm; no model "
            "quality, planning, closed-loop or safety result."
        ),
    }


def _validate_launch(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    preflight = _read_json(args.trainer_preflight.resolve())
    launch = _read_json(args.launch_amendment.resolve())
    arms = launch.get("authorized_arms", {})
    locks = launch.get("locks", {})
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("training_arm") != args.training_arm
        or preflight.get("validated_inputs") != _validated_inputs(args)
        or preflight.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or preflight.get("output_root") != str(args.output_dir.resolve())
        or launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status")
        != "immutable_two_arm_process_qa_smoke_authorization"
        or launch.get("validated_inputs") != _validated_inputs(args)
        or launch.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or arms.get(args.training_arm, {}).get("preflight_sha256")
        != _sha256(args.trainer_preflight.resolve())
        or arms.get(args.training_arm, {}).get("output_root")
        != str(args.output_dir.resolve())
        or arms.get(args.training_arm, {}).get("maximum_submissions") != 1
        or arms.get(args.training_arm, {}).get("optimizer_steps") != 200
        or launch.get("automatic_retry") is not False
        or locks.get("bounded_v13_capacity_comparison_allowed") is not True
        or locks.get("formal_stage2l_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("closed_loop_allowed") is not False
        or locks.get("locked_test_allowed") is not False
    ):
        raise ValueError("v13 launch amendment or arm preflight differs")


def _factorized_forward(
    *, lm, tokenizer, queries, head, assets: v121.FactorizedAssets,
    group_id: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the U-independent R pass and expose its native hidden tokens."""

    v101 = v121._v101()
    baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
    features = assets.view_features[group_id].cuda(non_blocking=True)
    target = assets.component_relevance[group_id].cuda(non_blocking=True)
    row = assets.row(group_id, "observed", "task_relevance")
    prompt = v101._prompt_tokens(
        tokenizer, row, assets.route_text[group_id]
    ).cuda(non_blocking=True)
    attention = prompt.ne(tokenizer.pad_token_id or 0)
    query_inputs = queries(features)
    output = lm(
        input_ids=prompt,
        attention_mask=attention,
        images=torch.cat((baseline, query_inputs), dim=1),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    query_grid = v101.extract_relevance_query_grid(
        output.hidden_states[-1],
        prompt,
        image_token_index=v101.IMAGE_TOKEN_INDEX,
        visual_token_count=v101.ORION_VISUAL_TOKENS,
        views=6,
        grid_h=10,
        grid_w=10,
    )
    component_logits = head(query_grid).component_logits
    pooled_r = F.adaptive_avg_pool2d(
        query_grid.permute(0, 1, 4, 2, 3).reshape(6, 4096, 10, 10),
        (5, 5),
    )
    direct_r_tokens = pooled_r.reshape(1, 6, 4096, 5, 5).permute(
        0, 1, 3, 4, 2
    ).reshape(1, 150, 4096)
    if tuple(direct_r_tokens.shape) != (1, 150, 4096):
        raise RuntimeError("direct R hidden-token shape differs")
    return component_logits, target, direct_r_tokens, baseline


def _direct_matched_conditioning(
    *,
    component_logits: torch.Tensor,
    direct_r_tokens: torch.Tensor,
    baseline: torch.Tensor,
    lm_assets: v11.V11Assets,
    group_id: str,
    uq_tokenizer,
    protocol: Mapping[str, Any],
):
    """Concatenate native visual, direct U and direct R tokens only."""

    if tuple(baseline.shape) != (1, 529, 4096):
        raise RuntimeError("native ORION visual-token shape differs")
    if tuple(direct_r_tokens.shape) != (1, 150, 4096):
        raise RuntimeError("direct R token shape differs")
    frozen_r = direct_r_tokens.detach()
    tokenized = {}
    scalar_uq = {}
    for variant in REQUIRED_RISK_VARIANTS:
        value = uq_tokenizer(
            lm_assets.components[(group_id, variant)].cuda(non_blocking=True)
        )
        tokens = value.tokens.detach()
        if tuple(tokens.shape) != (1, 600, 4096):
            raise RuntimeError("Stage1 direct U token shape differs")
        tokenized[variant] = tokens
        scalar_uq[variant] = value.latest_scalar_uq.detach()
    conditioning = {
        variant: torch.cat((baseline, tokenized[variant], frozen_r), dim=1)
        for variant in REQUIRED_RISK_VARIANTS
    }
    if any(tuple(value.shape) != (1, 1279, 4096) for value in conditioning.values()):
        raise RuntimeError("direct-U/direct-R ORION conditioning shape differs")
    shared_union = v122.derived_union_logit(component_logits.detach())
    diagnostic = protocol["controlled_u_diagnostics"]
    risk_diagnostic = audit_matched_task_risk(
        shared_union,
        scalar_uq,
        required_on_over_off_margin=float(
            diagnostic["minimum_on_over_off_margin"]
        ),
        maximum_off_path_risk=float(
            diagnostic["maximum_off_path_risk_peak"]
        ),
        minimum_fraction=float(diagnostic["minimum_group_fraction"]),
    )
    return DirectMatchedConditioningV13(
        conditioning_by_variant=conditioning,
        latest_scalar_uq_by_variant=scalar_uq,
        no_u_ablation_vision=torch.cat(
            (baseline, tokenized["zero_uq"], frozen_r), dim=1
        ),
        task_risk_diagnostic=risk_diagnostic,
        baseline_token_count=529,
        direct_u_token_count=600,
        direct_r_token_count=150,
    )


def _process_answers(
    assets: v11.V11Assets, group_id: str
) -> Dict[str, str]:
    return {
        variant: build_structured_process_chain(
            _process_rows(assets, group_id, variant)
        ).answer
        for variant in REQUIRED_RISK_VARIANTS
    }


def _negative_variant(variant: str) -> str:
    return {
        "zero_uq": "on_path_uq",
        "on_path_uq": "off_path_uq",
        "off_path_uq": "on_path_uq",
        "view_shuffled_uq": "on_path_uq",
    }[variant]


@torch.no_grad()
def _evaluate_process_language(
    *,
    split: str,
    lm,
    tokenizer,
    uq_tokenizer,
    queries,
    head,
    lm_assets: v11.V11Assets,
    factorized_assets: v121.FactorizedAssets,
    protocol: Mapping[str, Any],
    answer_batch_size: int,
) -> Dict[str, Any]:
    import scripts.train_stage2l_mr1_smoke as base

    modules = (lm, uq_tokenizer, queries, head)
    for module in modules:
        module.eval()
    groups = [
        values[0] for _, values in sorted(lm_assets.event_groups[split].items())
    ]
    full_preferences = {variant: [] for variant in REQUIRED_RISK_VARIANTS}
    no_u_preferences = {variant: [] for variant in REQUIRED_RISK_VARIANTS}
    target_nlls = []
    step_nlls = {family: [] for family in PROCESS_FAMILIES}
    per_group = {}
    for group_id in groups:
        logits, _, direct_r_tokens, baseline = _factorized_forward(
            lm=lm,
            tokenizer=tokenizer,
            queries=queries,
            head=head,
            assets=factorized_assets,
            group_id=group_id,
        )
        conditioned = _direct_matched_conditioning(
            component_logits=logits,
            direct_r_tokens=direct_r_tokens,
            baseline=baseline,
            lm_assets=lm_assets,
            group_id=group_id,
            uq_tokenizer=uq_tokenizer,
            protocol=protocol,
        )
        answers = _process_answers(lm_assets, group_id)
        unique_answers = tuple(dict.fromkeys(answers.values()))
        reference_rows = _process_rows(
            lm_assets, group_id, REQUIRED_RISK_VARIANTS[0]
        )
        process_row = build_structured_process_row(reference_rows)
        no_u_values = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=conditioned.no_u_ablation_vision,
            row=process_row,
            route_text=lm_assets.route_text[group_id],
            answers=unique_answers,
            micro_batch_size=answer_batch_size,
        )
        no_u_by_answer = dict(zip(unique_answers, no_u_values))
        group_payload = {}
        for variant in REQUIRED_RISK_VARIANTS:
            rows = _process_rows(lm_assets, group_id, variant)
            row = build_structured_process_row(rows)
            values = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned.conditioning_by_variant[variant],
                row=row,
                route_text=lm_assets.route_text[group_id],
                answers=unique_answers,
                micro_batch_size=answer_batch_size,
            )
            by_answer = dict(zip(unique_answers, values))
            target = answers[variant]
            alternatives = [value for value in unique_answers if value != target]
            full_alt = torch.stack([by_answer[value] for value in alternatives]).amin()
            no_u_alt = torch.stack(
                [no_u_by_answer[value] for value in alternatives]
            ).amin()
            full_preferences[variant].append(bool(by_answer[target] < full_alt))
            no_u_preferences[variant].append(
                bool(no_u_by_answer[target] < no_u_alt)
            )
            target_nlls.append(float(by_answer[target].item()))
            if variant == "on_path_uq":
                for family in PROCESS_FAMILIES:
                    step = build_process_step_row(rows, family)
                    value = base._answer_nlls_mr1(
                        lm=lm,
                        tokenizer=tokenizer,
                        vision=conditioned.conditioning_by_variant[variant],
                        row=step,
                        route_text=lm_assets.route_text[group_id],
                        answers=(str(step["conversation"][1]["value"]),),
                        micro_batch_size=answer_batch_size,
                    )[0]
                    step_nlls[family].append(float(value.item()))
            group_payload[variant] = {
                "target_nll": float(by_answer[target].item()),
                "closest_counterfactual_nll": float(full_alt.item()),
                "target_preferred": bool(by_answer[target] < full_alt),
                "no_u_target_preferred": bool(
                    no_u_by_answer[target] < no_u_alt
                ),
            }
        per_group[group_id] = group_payload
    full_fraction = {
        key: float(np.mean(values)) for key, values in full_preferences.items()
    }
    no_u_fraction = {
        key: float(np.mean(values)) for key, values in no_u_preferences.items()
    }
    return {
        "split": split,
        "diagnostic_group_count": len(groups),
        "diagnostic_groups": groups,
        "mean_target_nll": float(np.mean(target_nlls)),
        "mean_step_nll": {
            key: float(np.mean(values)) for key, values in step_nlls.items()
        },
        "full_preference_fraction_by_variant": full_fraction,
        "full_overall_preference_fraction": float(
            np.mean(list(full_fraction.values()))
        ),
        "no_u_preference_fraction_by_variant": no_u_fraction,
        "no_u_overall_preference_fraction": float(
            np.mean(list(no_u_fraction.values()))
        ),
        "full_minus_no_u_preference_fraction": float(
            np.mean(list(full_fraction.values()))
            - np.mean(list(no_u_fraction.values()))
        ),
        "per_group": per_group,
    }


def _all_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _runtime_gradient_probe(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    lm,
    tokenizer,
    uq_tokenizer,
    queries,
    head,
    scope,
    lm_assets: v11.V11Assets,
    factorized_assets: v121.FactorizedAssets,
) -> Dict[str, Any]:
    """Exercise one real direct-token backward before the full baseline eval."""

    for module in (lm, queries, head):
        module.zero_grad(set_to_none=True)
    group_id = sorted(
        values[0]
        for _, values in sorted(lm_assets.event_groups["train"].items())
    )[0]
    variant = "on_path_uq"
    lm.train()
    queries.train()
    head.train()
    logits, target, direct_r_tokens, baseline = _factorized_forward(
        lm=lm,
        tokenizer=tokenizer,
        queries=queries,
        head=head,
        assets=factorized_assets,
        group_id=group_id,
    )
    training = protocol["training"]
    r_terms = factorized_relevance_terms_v121(
        logits,
        target,
        support_fraction_of_peak=float(
            training["r_objective"]["support_fraction"]
        ),
        inactive_view_background_anchor_weight=float(
            training["r_objective"]["inactive_view_background_anchor_weight"]
        ),
        empty_component_background_anchor_weight=float(
            training["r_objective"]["empty_component_background_anchor_weight"]
        ),
        route_component_weight=float(
            training["r_objective"]["route_component_weight"]
        ),
        actor_component_weight=float(
            training["r_objective"]["actor_component_weight"]
        ),
    )
    r_terms.loss.backward()
    r_gradient_groups = {
        name: sum(parameter.grad is not None for parameter in parameters)
        for name, parameters in scope.parameter_groups.items()
        if name in {"relevance_queries", "relevance_head"}
    }
    if any(r_gradient_groups.get(name, 0) == 0 for name in (
        "relevance_queries", "relevance_head"
    )):
        raise RuntimeError("v13 runtime probe found a disconnected R group")
    for module in (lm, queries, head):
        module.zero_grad(set_to_none=True)
    conditioned = _direct_matched_conditioning(
        component_logits=logits.detach(),
        direct_r_tokens=direct_r_tokens.detach(),
        baseline=baseline,
        lm_assets=lm_assets,
        group_id=group_id,
        uq_tokenizer=uq_tokenizer,
        protocol=protocol,
    )
    rows = _process_rows(lm_assets, group_id, variant)
    row = build_structured_process_row(rows)
    anchor = detached_conditioning_gradient_anchor(
        conditioned.conditioning_by_variant[variant]
    )
    import scripts.train_stage2l_mr1_smoke as base

    nll = base._answer_nlls_mr1(
        lm=lm,
        tokenizer=tokenizer,
        vision=anchor,
        row=row,
        route_text=lm_assets.route_text[group_id],
        answers=(_process_answers(lm_assets, group_id)[variant],),
        micro_batch_size=1,
    )[0]
    if not nll.requires_grad or nll.grad_fn is None:
        raise RuntimeError("v13 direct-token language loss has no gradient graph")
    nll.backward()
    language_gradient_groups = {
        name: sum(parameter.grad is not None for parameter in parameters)
        for name, parameters in scope.parameter_groups.items()
        if name in {"orion_lora", "partial_decoder"} and parameters
    }
    required_groups = {"orion_lora"}
    if args.training_arm == "partial_unfreeze":
        required_groups.add("partial_decoder")
    if any(
        language_gradient_groups.get(name, 0) == 0 for name in required_groups
    ):
        raise RuntimeError("v13 runtime probe found a disconnected language group")
    if anchor.grad is None or not bool(torch.isfinite(anchor.grad).all()):
        raise RuntimeError("v13 direct-token gradient anchor did not receive a finite gradient")
    if any(parameter.grad is not None for parameter in uq_tokenizer.parameters()):
        raise RuntimeError("v13 runtime probe gradient escaped into the U tokenizer")
    trainable = [
        parameter
        for module in (lm, queries, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not _all_finite(trainable):
        raise RuntimeError("v13 runtime probe produced non-finite gradients")
    result = {
        "status": "direct_token_backward_connected",
        "training_arm": args.training_arm,
        "group_id": group_id,
        "event_id": lm_assets.group_event[group_id],
        "variant": variant,
        "language_nll": float(nll.item()),
        "factorized_r_loss": float(r_terms.loss.item()),
        "r_gradient_parameter_counts": r_gradient_groups,
        "language_gradient_parameter_counts": language_gradient_groups,
        "conditioning_is_detached_leaf": bool(anchor.is_leaf),
        "u_tokenizer_gradient_parameter_count": 0,
        "optimizer_step_taken": False,
        "finite": True,
    }
    for module in (lm, queries, head):
        module.zero_grad(set_to_none=True)
    del anchor, nll, r_terms, conditioned, logits, target, direct_r_tokens
    torch.cuda.empty_cache()
    print("[Stage2LV13Probe] " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _train(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    lm,
    tokenizer,
    uq_tokenizer,
    queries,
    head,
    scope,
    lm_assets: v11.V11Assets,
    factorized_assets: v121.FactorizedAssets,
) -> list[Dict[str, Any]]:
    import scripts.train_stage2l_mr1_smoke as base

    uq_tokenizer.eval()
    for parameter in uq_tokenizer.parameters():
        parameter.requires_grad = False
    train = protocol["training"]
    learning_rates = train["learning_rates"]
    optimizer_groups = []
    for name, parameters in scope.parameter_groups.items():
        if not parameters:
            continue
        lr_key = {
            "orion_lora": "orion_lora",
            "partial_decoder": "partial_decoder",
            "relevance_queries": "relevance",
            "relevance_head": "relevance",
        }[name]
        optimizer_groups.append(
            {"params": list(parameters), "lr": float(learning_rates[lr_key])}
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(learning_rates["weight_decay"]),
    )
    trainable = [
        parameter
        for module in (lm, queries, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    sampler = v11.OneEventPerStepSampler(
        lm_assets.event_groups["train"], seed=args.seed
    )
    variants = tuple(REQUIRED_RISK_VARIANTS)
    families = tuple(PROCESS_FAMILIES)
    history = []
    for step in range(1, int(train["optimizer_steps"]) + 1):
        group_id = sampler.next()
        variant = variants[(step - 1) % len(variants)]
        family = families[(step - 1) % len(families)]
        lm.train()
        queries.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits, target, direct_r_tokens, baseline = _factorized_forward(
            lm=lm,
            tokenizer=tokenizer,
            queries=queries,
            head=head,
            assets=factorized_assets,
            group_id=group_id,
        )
        r_terms = factorized_relevance_terms_v121(
            logits,
            target,
            support_fraction_of_peak=float(train["r_objective"]["support_fraction"]),
            inactive_view_background_anchor_weight=float(
                train["r_objective"]["inactive_view_background_anchor_weight"]
            ),
            empty_component_background_anchor_weight=float(
                train["r_objective"]["empty_component_background_anchor_weight"]
            ),
            route_component_weight=float(
                train["r_objective"]["route_component_weight"]
            ),
            actor_component_weight=float(
                train["r_objective"]["actor_component_weight"]
            ),
        )
        weights = train["loss_weights"]
        weighted_r = float(weights["factorized_relevance"]) * r_terms.loss
        weighted_r.backward()
        r_loss_value = float(r_terms.loss.item())
        del r_terms, weighted_r

        # The language pass receives the detached native R hidden tokens and
        # frozen Stage1 U tokens directly.  Backward each language objective
        # separately so a 7B run never retains multiple decoder graphs.
        conditioned = _direct_matched_conditioning(
            component_logits=logits.detach(),
            direct_r_tokens=direct_r_tokens.detach(),
            baseline=baseline,
            lm_assets=lm_assets,
            group_id=group_id,
            uq_tokenizer=uq_tokenizer,
            protocol=protocol,
        )
        rows = _process_rows(lm_assets, group_id, variant)
        process_row = build_structured_process_row(rows)
        answers = _process_answers(lm_assets, group_id)
        negative = _negative_variant(variant)
        chain_anchor = detached_conditioning_gradient_anchor(
            conditioned.conditioning_by_variant[variant]
        )
        answer_values = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=chain_anchor,
            row=process_row,
            route_text=lm_assets.route_text[group_id],
            answers=(answers[variant], answers[negative]),
            micro_batch_size=args.answer_batch_size,
        )
        chain_nll, negative_nll = answer_values.unbind()
        preference = F.relu(
            float(train["preference_margin"])
            + chain_nll
            - negative_nll
        )
        weighted_chain = (
            float(weights["full_process_nll"]) * chain_nll
            + float(weights["counterfactual_preference"]) * preference
        )
        weighted_chain.backward()
        chain_nll_value = float(chain_nll.item())
        negative_nll_value = float(negative_nll.item())
        preference_value = float(preference.item())
        target_preferred = bool(chain_nll < negative_nll)
        del answer_values, chain_nll, negative_nll, chain_anchor
        del preference, weighted_chain
        step_row = build_process_step_row(rows, family)
        step_anchor = detached_conditioning_gradient_anchor(
            conditioned.conditioning_by_variant[variant]
        )
        step_nll = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=step_anchor,
            row=step_row,
            route_text=lm_assets.route_text[group_id],
            answers=(str(step_row["conversation"][1]["value"]),),
            micro_batch_size=args.answer_batch_size,
        )[0]
        weighted_step = float(weights["step_process_nll"]) * step_nll
        weighted_step.backward()
        step_nll_value = float(step_nll.item())
        del step_anchor
        loss_value = (
            float(weights["factorized_relevance"]) * r_loss_value
            + float(weights["full_process_nll"]) * chain_nll_value
            + float(weights["step_process_nll"]) * step_nll_value
            + float(weights["counterfactual_preference"]) * preference_value
        )
        if any(parameter.grad is not None for parameter in uq_tokenizer.parameters()):
            raise RuntimeError("v13 gradient escaped into the frozen U tokenizer")
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(train["gradient_clip_norm"])
        )
        finite = (
            bool(np.isfinite(loss_value))
            and bool(torch.isfinite(norm))
            and _all_finite(trainable)
        )
        if not finite:
            raise RuntimeError("v13 process-QA optimization became non-finite")
        optimizer.step()
        item = {
            "optimizer_step": step,
            "group_id": group_id,
            "event_id": lm_assets.group_event[group_id],
            "variant": variant,
            "process_family": family,
            "loss": loss_value,
            "factorized_r_loss": r_loss_value,
            "full_process_nll": chain_nll_value,
            "step_process_nll": step_nll_value,
            "negative_process_nll": negative_nll_value,
            "counterfactual_preference_loss": preference_value,
            "target_preferred": target_preferred,
            "gradient_norm_before_clip": float(norm.item()),
            "finite": finite,
            "training_arm": args.training_arm,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LV13] " + json.dumps(item, sort_keys=True), flush=True)
    del optimizer
    return history


def _checkpoint_state(
    *, lm, queries, head, scope, args, history,
) -> Dict[str, Any]:
    selected_layers = set(scope.partial_layer_indices)
    partial_state = {
        name: value.detach().cpu()
        for name, value in lm.state_dict().items()
        if (
            (match := re.search(r"(?:^|\.)layers\.(\d+)\.", name))
            is not None
            and int(match.group(1)) in selected_layers
            and "lora_" not in name
        )
    }
    return {
        "schema": SCHEMA,
        "status": "bounded_capacity_smoke_complete",
        "training_arm": args.training_arm,
        "optimizer_steps": len(history),
        "orion_lora": {
            name: value.detach().cpu()
            for name, value in lm.state_dict().items()
            if "lora_" in name
        },
        "partial_decoder": partial_state,
        "view_aligned_relevance_queries": {
            name: value.detach().cpu() for name, value in queries.state_dict().items()
        },
        "factorized_relevance_head": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "task_risk_language_bridge_present": False,
        "k_used_as_model_input": False,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "closed_loop_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--v121-checkpoint", type=Path, required=True)
    parser.add_argument("--v121-report", type=Path, required=True)
    parser.add_argument("--v121-terminal-validation", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-arm", choices=TRAINING_ARMS, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.dataset_manifest,
        args.v11_records,
        args.dataset_audit_report,
        args.view_feature_cache,
        args.u_tokenizer_checkpoint,
        args.v121_checkpoint,
        args.v121_report,
        args.v121_terminal_validation,
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v13 process-QA prerequisite is missing")
    if args.answer_batch_size < 1:
        raise ValueError("answer batch size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty v13 output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)

    v121._v101()._configure_base()
    lm_assets = v11.V11Assets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )
    factorized_assets = v121.FactorizedAssets(
        args.dataset_manifest, args.view_feature_cache
    )
    if set(lm_assets.group_rows) != set(factorized_assets.group_rows):
        raise ValueError("v13 language and factorized assets differ")
    if args.preflight_only:
        if args.trainer_preflight is not None or args.launch_amendment is not None:
            raise ValueError("v13 preflight cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("v13 preflight requires a fresh output path")
        value = _preflight(
            args=args, protocol=protocol, assets=lm_assets
        )
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": value["status"],
            "training_arm": args.training_arm,
            "chain_count": value["process_dataset_audit"]["chain_count"],
            "output": str(args.preflight_output.resolve()),
        }, sort_keys=True))
        return 0
    if args.preflight_output is not None or args.trainer_preflight is None or args.launch_amendment is None:
        raise ValueError("v13 training requires preflight and launch amendment")
    _validate_launch(args, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("v13 process-QA training requires CUDA")

    from mmcv.utils import set_random_seed
    import scripts.train_stage2l_mr1_smoke as stage2l_base
    import scripts.train_stage2l_v10_staged_smoke as v10

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, tokenizer = stage2l_base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    queries = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=4096,
        image_feature_dim=1024,
        hidden_dim=256,
        grid_hw=(10, 10),
        max_views=6,
    ).cuda()
    head = FactorizedTaskRelevanceMapHead(
        model_dim=4096, hidden_dim=256
    ).cuda()
    lineage = v122._load_v121_factorized_relevance(
        lm=lm,
        relevance_queries=queries,
        relevance_head=head,
        checkpoint_path=args.v121_checkpoint.resolve(),
    )
    uq_tokenizer = v10._load_frozen_u_tokenizer(
        args.u_tokenizer_checkpoint.resolve()
    ).cuda().eval()
    scope = configure_trainable_scope(
        lm=lm,
        relevance_queries=queries,
        relevance_head=head,
        arm=args.training_arm,
        partial_unfreeze_layers=int(
            protocol["training"]["partial_unfreeze_layers"]
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_gradient_probe = _runtime_gradient_probe(
        args=args,
        protocol=protocol,
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        queries=queries,
        head=head,
        scope=scope,
        lm_assets=lm_assets,
        factorized_assets=factorized_assets,
    )
    (args.output_dir / "runtime_gradient_probe.json").write_text(
        json.dumps(
            runtime_gradient_probe, indent=2, sort_keys=True, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )
    before = {
        split: _evaluate_process_language(
            split=split,
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            queries=queries,
            head=head,
            lm_assets=lm_assets,
            factorized_assets=factorized_assets,
            protocol=protocol,
            answer_batch_size=args.answer_batch_size,
        )
        for split in ("train", "dev")
    }
    history = _train(
        args=args,
        protocol=protocol,
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        queries=queries,
        head=head,
        scope=scope,
        lm_assets=lm_assets,
        factorized_assets=factorized_assets,
    )
    after = {
        split: _evaluate_process_language(
            split=split,
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            queries=queries,
            head=head,
            lm_assets=lm_assets,
            factorized_assets=factorized_assets,
            protocol=protocol,
            answer_batch_size=args.answer_batch_size,
        )
        for split in ("train", "dev")
    }
    checkpoint = _checkpoint_state(
        lm=lm,
        queries=queries,
        head=head,
        scope=scope,
        args=args,
        history=history,
    )
    torch.save(checkpoint, args.output_dir / "stage2l_v13_process_qa.pt")
    diagnostics = protocol["language_diagnostics"]
    quality = {
        "train_target_nll_improved": after["train"]["mean_target_nll"]
        < before["train"]["mean_target_nll"],
        "dev_target_nll_improved": after["dev"]["mean_target_nll"]
        < before["dev"]["mean_target_nll"],
        "dev_full_preference_above_no_u": after["dev"][
            "full_minus_no_u_preference_fraction"
        ]
        >= float(diagnostics["minimum_full_minus_no_u_fraction"]),
        "dev_on_path_preference": after["dev"][
            "full_preference_fraction_by_variant"
        ]["on_path_uq"]
        >= float(diagnostics["minimum_on_path_preference_fraction"]),
        "dev_zero_u_preference": after["dev"][
            "full_preference_fraction_by_variant"
        ]["zero_uq"]
        >= float(diagnostics["minimum_zero_u_preference_fraction"]),
    }
    report = {
        "schema": SCHEMA,
        "status": (
            "bounded_capacity_smoke_quality_diagnostics_passed"
            if all(quality.values())
            else "bounded_capacity_smoke_completed_with_soft_quality_failures"
        ),
        "training_arm": args.training_arm,
        "optimizer_steps": len(history),
        "lineage": lineage,
        "trainable_scope": {
            "arm": scope.arm,
            "parameter_counts": dict(scope.parameter_counts),
            "total_trainable_parameters": scope.total_trainable_parameters,
            "partial_layer_indices": list(scope.partial_layer_indices),
            "orion_trainable_parameter_name_count": len(
                scope.trainable_parameter_names
            ),
        },
        "process_dataset_audit": audit_process_dataset(lm_assets),
        "language_before": before,
        "runtime_gradient_probe": runtime_gradient_probe,
        "language_after": after,
        "quality_diagnostics": quality,
        "quality_diagnostics_passed": all(quality.values()),
        "history": history,
        "architecture_invariants": {
            "stage1_frozen": True,
            "u_tokenizer_frozen": True,
            "u_enters_relevance_query": False,
            "r_computed_once_per_matched_group": True,
            "direct_stage1_u_tokens_enter_orion": True,
            "direct_r_hidden_tokens_enter_orion": True,
            "task_risk_language_bridge_present": False,
            "k_used_as_model_input": False,
            "k_retained_as_posthoc_diagnostic_only": True,
            "factorized_r_directly_supervised": True,
            "orion_lora_trained_by_r_and_process_qa": True,
            "trajectory_or_control_loss": False,
        },
        "provenance": {
            "validated_inputs": _validated_inputs(args),
            "protocol_sha256": _sha256(args.training_protocol.resolve()),
            "preflight_sha256": _sha256(args.trainer_preflight.resolve()),
            "launch_amendment_sha256": _sha256(
                args.launch_amendment.resolve()
            ),
        },
        "locks": {
            "formal_stage2l_ready": False,
            "stage2p_ready": False,
            "closed_loop_eligible": False,
            "locked_test_read": False,
            "automatic_extension": False,
        },
        "claim_boundary": (
            "One bounded controlled-U capacity comparison arm. It tests "
            "whether ORION learns auditable process semantics; it is not "
            "learned-U, formal generalization, planning or safety evidence."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "training_arm": args.training_arm,
        "optimizer_steps": len(history),
        "quality_diagnostics_passed": all(quality.values()),
        "report": str((args.output_dir / "report.json").resolve()),
    }, sort_keys=True), flush=True)
    del lm, queries, head, uq_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
