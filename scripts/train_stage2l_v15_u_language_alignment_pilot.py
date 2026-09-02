#!/usr/bin/env python3
"""Align the existing Stage-1 U tokenizer with ORION language supervision.

This bounded Stage2-L1 pilot starts from the completed v14.1 LoRA.  It keeps
the observation-uncertainty estimator frozen, sends the existing U tokens
directly into ORION, and jointly trains only ORION LoRA plus the existing
language-facing U tokenizer.  Every field is supervised against all legal
answers.  Route, task relevance, risk, action, trajectory, and control are
absent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import scripts.train_stage2l_mr1_smoke as base
import scripts.train_stage2l_v10_staged_smoke as v10
import scripts.train_stage2l_v121_factorized_r_smoke as v121
import scripts.train_stage2l_v14_u_concept_lora_smoke as v14
from scripts.train_stage2l_route196_bridge_smoke import _generate
from uq_estimator.stage1_u_tokenizer_pretraining import (
    UQSummaryReconstructionHead,
    stage1_u_tokenizer_pretraining_terms,
)
from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
    build_explicit_u_qa_row,
    parse_strict_u_answer,
)
from uq_estimator.stage2l_u_concept_qa_v14 import (
    TAG_ORDER,
    U_VARIANTS,
)
from uq_estimator.stage2l_u_language_alignment_v15 import (
    SCHEMA as ALIGNMENT_SCHEMA,
    all_candidate_cross_entropy,
    build_balanced_field_schedule,
    exact_nll_gradient_coefficients,
    field_qa_and_candidates,
    target_margin,
)


SCHEMA = "orion.stage2l-v15-u-language-alignment-pilot/v1"
PROTOCOL_SCHEMA = "orion.stage2l-v15-u-language-alignment-protocol/v1"
PREFLIGHT_SCHEMA = "orion.stage2l-v15-u-language-alignment-preflight/v1"
V14_CHECKPOINT_SCHEMA = "orion.stage2l-v14-u-concept-lora-smoke/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _validated_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "v11_records_sha256": _sha256(args.v11_records.resolve()),
        "dataset_audit_report_sha256": _sha256(
            args.dataset_audit_report.resolve()
        ),
        "view_feature_cache_sha256": _sha256(args.view_feature_cache.resolve()),
        "u_tokenizer_checkpoint_sha256": _sha256(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "trained_v14_1_checkpoint_sha256": _sha256(
            args.trained_v14_checkpoint.resolve()
        ),
    }


def _protocol_checks(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    evaluation = protocol.get("evaluation", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "bounded_u_language_alignment_pilot"
        or protocol.get("input_sha256") != _validated_inputs(args)
        or protocol.get("alignment_schema") != ALIGNMENT_SCHEMA
        or architecture.get("direct_u_tokens_enter_orion") is not True
        or architecture.get("stage1_observation_estimator_trainable") is not False
        or architecture.get("u_tokenizer_existing_projection_trainable") is not True
        or architecture.get("new_bridge_or_r_input_present") is not False
        or architecture.get("route_task_risk_action_present") is not False
        or architecture.get("trajectory_or_control_loss") is not False
        or architecture.get("orion_trainable_scope") != "lora_only"
        or training.get("field_objective") != "all_candidate_cross_entropy"
        or training.get("task_agnostic_reconstruction_retention") is not True
        or int(training.get("optimizer_steps", 0)) != int(args.optimizer_steps)
        or training.get("automatic_retry") is not False
        or training.get("automatic_extension") is not False
        or evaluation.get("dev_groups") != 20
        or evaluation.get("u_states") != 120
        or evaluation.get("all_six_fields") is not True
        or Path(str(protocol.get("output_root", ""))).resolve()
        != args.output_dir.resolve()
    ):
        raise ValueError("v15 U-language alignment protocol is absent or stale")


def _load_v14_lora(lm, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu")
    if (
        payload.get("schema") != V14_CHECKPOINT_SCHEMA
        or payload.get("status") != "bounded_u_concept_lora_complete"
        or payload.get("optimizer_steps") != 200
        or payload.get("stage1_and_u_tokenizer_frozen") is not True
        or payload.get("r_k_route_risk_action_absent") is not True
    ):
        raise ValueError("v14.1 initialization checkpoint contract differs")
    state = payload.get("orion_lora", {})
    expected = {name for name in lm.state_dict() if "lora_" in name}
    if set(state) != expected:
        raise ValueError("v14.1 LoRA tensor keys differ")
    result = lm.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        "lora_" in name for name in result.missing_keys
    ):
        raise ValueError("v14.1 LoRA load was incomplete")
    return {
        "schema": payload["schema"],
        "optimizer_steps": payload["optimizer_steps"],
        "lora_tensor_count": len(state),
        "lora_parameter_count": sum(value.numel() for value in state.values()),
    }


def _training_schedule(
    assets: v14.UConceptAssets, optimizer_steps: int, seed: int
):
    return build_balanced_field_schedule(
        group_ids=assets.groups_for_split("train"),
        summaries=assets.summaries,
        optimizer_steps=optimizer_steps,
        seed=seed,
    )


def _schedule_audit(schedule, assets: v14.UConceptAssets) -> dict[str, Any]:
    fields = Counter(item.tag for item in schedule)
    values = {
        tag: Counter(item.target for item in schedule if item.tag == tag)
        for tag in TAG_ORDER
    }
    events = Counter(assets.group_event[item.group_id] for item in schedule)
    checks = {
        "all_steps_present": len(schedule) > 0,
        "fields_exactly_balanced": len(set(fields.values())) == 1,
        "all_canonical_values_presented": all(
            set(values[tag]) == set(FIELD_VOCABULARIES[tag]) for tag in TAG_ORDER
        ),
        "all_thirteen_train_events_presented": len(events) == 13,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "optimizer_steps": len(schedule),
        "presentations_by_field": dict(fields),
        "presentations_by_field_value": {
            tag: dict(counts) for tag, counts in values.items()
        },
        "presentations_by_event": dict(sorted(events.items())),
    }


def _sequence_audit(args: argparse.Namespace, assets: v14.UConceptAssets) -> dict:
    from mmcv.utils import Config
    from transformers import AutoTokenizer

    cfg = Config.fromfile(str(args.config.resolve()))
    tokenizer_path = Path(str(cfg.model.tokenizer))
    if not tokenizer_path.is_absolute():
        tokenizer_path = (args.config.resolve().parents[3] / tokenizer_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    maximum = 0
    longest = None
    for group_id in sorted(assets.group_audits):
        for variant in U_VARIANTS:
            summary = assets.summaries[(group_id, variant)]
            for tag in (None,) + TAG_ORDER:
                row = build_explicit_u_qa_row(summary, tag)
                ids, _ = base._training_tokens(
                    tokenizer,
                    row,
                    "",
                    answer=str(row["conversation"][1]["value"]),
                )
                expanded = int(ids.numel()) - 1 + 1129
                if expanded > maximum:
                    maximum = expanded
                    longest = {
                        "group_id": group_id,
                        "variant": variant,
                        "tag": tag,
                        "expanded_tokens": expanded,
                    }
    return {
        "model_max_length": 2048,
        "conditioning_tokens": 1129,
        "maximum_expanded_tokens": maximum,
        "longest": longest,
        "passed": maximum <= 2048,
    }


def _preflight(
    args: argparse.Namespace,
    assets: v14.UConceptAssets,
    schedule,
) -> dict[str, Any]:
    dataset = v14._dataset_audit(assets)
    schedule_audit = _schedule_audit(schedule, assets)
    sequences = _sequence_audit(args, assets)
    if not dataset["passed"] or not schedule_audit["passed"] or not sequences["passed"]:
        raise ValueError("v15 U-language alignment preflight failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "u_language_alignment_preflight_pass",
        "passed": True,
        "training_started": False,
        "gpu_used": False,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "output_root": str(args.output_dir.resolve()),
        "dataset_audit": dataset,
        "schedule_audit": schedule_audit,
        "sequence_audit": sequences,
        "architecture": {
            "direct_u_tokens_enter_orion": True,
            "stage1_observation_estimator_frozen": True,
            "existing_u_tokenizer_projection_trainable": True,
            "orion_lora_trainable": True,
            "task_agnostic_reconstruction_retention": True,
            "r_bridge_route_task_risk_action_absent": True,
            "all_candidate_field_supervision": True,
        },
    }


def _validate_preflight(args: argparse.Namespace) -> None:
    value = _read_json(args.trainer_preflight.resolve())
    if (
        value.get("schema") != PREFLIGHT_SCHEMA
        or value.get("passed") is not True
        or value.get("training_started") is not False
        or value.get("validated_inputs") != _validated_inputs(args)
        or value.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or value.get("trainer_sha256") != _sha256(Path(__file__).resolve())
        or value.get("output_root") != str(args.output_dir.resolve())
    ):
        raise ValueError("v15 U-language alignment preflight is absent or stale")


def _condition(
    *, baseline: torch.Tensor, components: torch.Tensor, uq_tokenizer
) -> torch.Tensor:
    tokens = uq_tokenizer(components.cuda(non_blocking=True)).tokens
    value = torch.cat((baseline.cuda(non_blocking=True).detach(), tokens), dim=1)
    if tuple(value.shape) != (1, 1129, 4096):
        raise RuntimeError("v15 direct visual/U conditioning shape differs")
    return value


@torch.no_grad()
def _score_candidates(
    *, lm, tokenizer, vision, row, answers: Sequence[str], answer_batch_size: int
) -> torch.Tensor:
    return base._answer_nlls_mr1(
        lm=lm,
        tokenizer=tokenizer,
        vision=vision,
        row=row,
        route_text="",
        answers=answers,
        micro_batch_size=answer_batch_size,
    ).detach()


def _backward_all_candidate_field(
    *,
    lm,
    tokenizer,
    uq_tokenizer,
    baseline,
    components,
    row,
    answers,
    target_index: int,
    answer_batch_size: int,
) -> dict[str, Any]:
    with torch.no_grad():
        scoring_vision = _condition(
            baseline=baseline, components=components, uq_tokenizer=uq_tokenizer
        )
        scored = _score_candidates(
            lm=lm,
            tokenizer=tokenizer,
            vision=scoring_vision,
            row=row,
            answers=answers,
            answer_batch_size=answer_batch_size,
        )
        coefficients = exact_nll_gradient_coefficients(scored, target_index)
        loss_value = float(all_candidate_cross_entropy(scored, target_index).item())
        nll_values = [float(value) for value in scored.cpu().tolist()]

    # Replaying one answer at a time avoids retaining every 7B-model graph.
    # The detached coefficient is the exact derivative d(CE)/d(candidate NLL).
    for answer, coefficient in zip(answers, coefficients):
        vision = _condition(
            baseline=baseline, components=components, uq_tokenizer=uq_tokenizer
        )
        nll = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=vision,
            row=row,
            route_text="",
            answers=(answer,),
            micro_batch_size=1,
        )[0]
        (coefficient.detach() * nll).backward()
    return {
        "all_candidate_cross_entropy": loss_value,
        "candidate_nlls": nll_values,
        "target_margin": target_margin(nll_values, target_index),
        "predicted_index": int(np.argmin(nll_values)),
    }


def _all_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _gradient_probe(
    *, lm, tokenizer, uq_tokenizer, assets: v14.UConceptAssets, lora_parameters
) -> dict[str, Any]:
    lm.train()
    lm.zero_grad(set_to_none=True)
    uq_tokenizer.zero_grad(set_to_none=True)
    group_id = assets.groups_for_split("train")[0]
    variant = "observed_u"
    summary = assets.summaries[(group_id, variant)]
    row, answers, target_index = field_qa_and_candidates(summary, "U_VIEW")
    vision = _condition(
        baseline=assets.visual_contexts[group_id],
        components=assets.variants[(group_id, variant)],
        uq_tokenizer=uq_tokenizer,
    )
    nll = base._answer_nlls_mr1(
        lm=lm,
        tokenizer=tokenizer,
        vision=vision,
        row=row,
        route_text="",
        answers=(answers[target_index],),
        micro_batch_size=1,
    )[0]
    nll.backward()
    lora_gradients = sum(value.grad is not None for value in lora_parameters)
    u_gradients = sum(
        value.grad is not None for value in uq_tokenizer.parameters()
    )
    if (
        lora_gradients == 0
        or u_gradients == 0
        or not _all_finite(lora_parameters)
        or not _all_finite(list(uq_tokenizer.parameters()))
    ):
        raise RuntimeError("v15 joint U-tokenizer/LoRA gradient probe failed")
    result = {
        "status": "direct_u_joint_language_alignment_backward_connected",
        "group_id": group_id,
        "variant": variant,
        "field": "U_VIEW",
        "target_nll": float(nll.item()),
        "lora_gradient_parameter_count": lora_gradients,
        "u_tokenizer_gradient_parameter_count": u_gradients,
        "finite": True,
        "optimizer_step_taken": False,
    }
    lm.zero_grad(set_to_none=True)
    uq_tokenizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def _reconstruction_metrics(uq_tokenizer, decoder, assets) -> dict[str, float]:
    uq_tokenizer.eval()
    decoder.eval()
    reconstruction = []
    zero_anchor = []
    with torch.no_grad():
        for group_id in assets.groups_for_split("dev"):
            components = assets.variants[(group_id, "observed_u")].cuda(
                non_blocking=True
            )
            terms = stage1_u_tokenizer_pretraining_terms(
                tokenizer=uq_tokenizer,
                reconstruction_head=decoder,
                components=components,
            )
            reconstruction.append(float(terms.reconstruction_loss.item()))
            zero_anchor.append(float(terms.zero_anchor_loss.item()))
    return {
        "mean_reconstruction_loss": float(np.mean(reconstruction)),
        "mean_zero_anchor_loss": float(np.mean(zero_anchor)),
    }


@torch.no_grad()
def _evaluate(
    *,
    lm,
    tokenizer,
    uq_tokenizer,
    assets: v14.UConceptAssets,
    answer_batch_size: int,
    generate: bool,
) -> dict[str, Any]:
    lm.eval()
    uq_tokenizer.eval()
    correct = defaultdict(list)
    margins = defaultdict(list)
    per_value = {tag: defaultdict(list) for tag in TAG_ORDER}
    expected_counts = {tag: Counter() for tag in TAG_ORDER}
    state_predictions = {}
    records = {}
    diagnostic_groups = {
        values[0] for _, values in sorted(assets.event_groups["dev"].items())
    }
    free_generation = {}
    for group_id in assets.groups_for_split("dev"):
        baseline = assets.visual_contexts[group_id]
        for variant in U_VARIANTS:
            summary = assets.summaries[(group_id, variant)]
            vision = _condition(
                baseline=baseline,
                components=assets.variants[(group_id, variant)],
                uq_tokenizer=uq_tokenizer,
            )
            predicted = {}
            state_correct = {}
            state_margins = {}
            for tag in TAG_ORDER:
                row, answers, target_index = field_qa_and_candidates(summary, tag)
                values = _score_candidates(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=vision,
                    row=row,
                    answers=answers,
                    answer_batch_size=answer_batch_size,
                )
                nlls = [float(value) for value in values.cpu().tolist()]
                predicted_value = FIELD_VOCABULARIES[tag][int(np.argmin(nlls))]
                target_value = summary.fields()[tag]
                is_correct = predicted_value == target_value
                margin = target_margin(nlls, target_index)
                predicted[tag] = predicted_value
                state_correct[tag] = is_correct
                state_margins[tag] = margin
                correct[tag].append(is_correct)
                margins[tag].append(margin)
                per_value[tag][target_value].append(is_correct)
                expected_counts[tag][target_value] += 1
            key = "%s::%s" % (group_id, variant)
            state_predictions[(group_id, variant)] = {
                "predicted": predicted,
                "expected": dict(summary.fields()),
            }
            records[key] = {
                "predicted": predicted,
                "expected": dict(summary.fields()),
                "correct_by_field": state_correct,
                "target_margin_by_field": state_margins,
                "field_accuracy": float(np.mean(list(state_correct.values()))),
            }
            if generate and group_id in diagnostic_groups:
                full_row = build_explicit_u_qa_row(summary)
                text = _generate(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=vision,
                    row=full_row,
                    route_text="",
                )
                try:
                    parsed = parse_strict_u_answer(text)
                    parse_error = None
                except ValueError as error:
                    parsed = {}
                    parse_error = str(error)
                free_generation[key] = {
                    "text": text,
                    "parsed": parsed,
                    "expected": dict(summary.fields()),
                    "strictly_parseable": parse_error is None,
                    "parse_error": parse_error,
                    "field_accuracy": float(
                        np.mean(
                            [
                                parsed.get(tag) == summary.fields()[tag]
                                for tag in TAG_ORDER
                            ]
                        )
                    ),
                }

    changed = []
    unchanged = []
    unchanged_correct = []
    for group_id in assets.groups_for_split("dev"):
        observed = state_predictions[(group_id, "observed_u")]
        for variant in U_VARIANTS:
            if variant == "observed_u":
                continue
            other = state_predictions[(group_id, variant)]
            for tag in TAG_ORDER:
                expected_changed = observed["expected"][tag] != other["expected"][tag]
                if expected_changed:
                    changed.append(
                        observed["predicted"][tag] == observed["expected"][tag]
                        and other["predicted"][tag] == other["expected"][tag]
                        and observed["predicted"][tag] != other["predicted"][tag]
                    )
                else:
                    invariant = observed["predicted"][tag] == other["predicted"][tag]
                    unchanged.append(invariant)
                    unchanged_correct.append(
                        invariant
                        and observed["predicted"][tag] == observed["expected"][tag]
                    )

    accuracy_by_tag = {
        tag: float(np.mean(correct[tag])) for tag in TAG_ORDER
    }
    balanced_by_tag = {
        tag: float(
            np.mean(
                [np.mean(values) for values in per_value[tag].values()]
            )
        )
        for tag in TAG_ORDER
    }
    majority_by_tag = {
        tag: max(expected_counts[tag].values()) / sum(expected_counts[tag].values())
        for tag in TAG_ORDER
    }
    nonzero_correct = [
        records["%s::%s" % (group_id, variant)]["correct_by_field"][tag]
        for group_id in assets.groups_for_split("dev")
        for variant in U_VARIANTS
        if variant != "zero_u"
        for tag in TAG_ORDER
        if tag != "U_PRESENT"
    ]
    generated_values = list(free_generation.values())
    return {
        "split": "dev",
        "dev_group_count": len(assets.groups_for_split("dev")),
        "u_state_count": len(records),
        "field_decision_count": len(records) * len(TAG_ORDER),
        "accuracy_by_tag": accuracy_by_tag,
        "balanced_accuracy_by_tag": balanced_by_tag,
        "majority_baseline_by_tag": majority_by_tag,
        "mean_target_margin_by_tag": {
            tag: float(np.mean(margins[tag])) for tag in TAG_ORDER
        },
        "positive_target_margin_fraction_by_tag": {
            tag: float(np.mean([value > 0.0 for value in margins[tag]]))
            for tag in TAG_ORDER
        },
        "nonzero_accuracy_excluding_presence": float(np.mean(nonzero_correct)),
        "counterfactual": {
            "changed_field_pair_count": len(changed),
            "changed_field_exact_response_fraction": float(np.mean(changed)),
            "unchanged_field_pair_count": len(unchanged),
            "unchanged_field_invariance_fraction": float(np.mean(unchanged)),
            "unchanged_field_correct_invariance_fraction": float(
                np.mean(unchanged_correct)
            ),
        },
        "free_generation": {
            "performed": generate,
            "sample_count": len(free_generation),
            "strict_parseable_fraction": (
                float(
                    np.mean(
                        [value["strictly_parseable"] for value in generated_values]
                    )
                )
                if generated_values
                else None
            ),
            "mean_field_accuracy": (
                float(np.mean([value["field_accuracy"] for value in generated_values]))
                if generated_values
                else None
            ),
            "samples": free_generation,
        },
        "records": records,
    }


def _train(
    *,
    args,
    protocol,
    lm,
    tokenizer,
    uq_tokenizer,
    decoder,
    assets,
    schedule,
    lora_parameters,
) -> list[dict[str, Any]]:
    training = protocol["training"]
    u_parameters = list(uq_tokenizer.parameters())
    decoder_parameters = list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": float(training["lora_learning_rate"]),
            },
            {
                "params": u_parameters,
                "lr": float(training["u_tokenizer_learning_rate"]),
            },
            {
                "params": decoder_parameters,
                "lr": float(training["reconstruction_head_learning_rate"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    all_trainable = lora_parameters + u_parameters + decoder_parameters
    history = []
    for step, example in enumerate(schedule, start=1):
        lm.train()
        uq_tokenizer.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        summary = assets.summaries[(example.group_id, example.variant)]
        row, answers, target_index = field_qa_and_candidates(summary, example.tag)
        components = assets.variants[(example.group_id, example.variant)]
        field = _backward_all_candidate_field(
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            baseline=assets.visual_contexts[example.group_id],
            components=components,
            row=row,
            answers=answers,
            target_index=target_index,
            answer_batch_size=args.answer_batch_size,
        )

        full_nll = None
        if step % int(training["full_answer_interval"]) == 0:
            full_row = build_explicit_u_qa_row(summary)
            vision = _condition(
                baseline=assets.visual_contexts[example.group_id],
                components=components,
                uq_tokenizer=uq_tokenizer,
            )
            full_nll_tensor = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=tokenizer,
                vision=vision,
                row=full_row,
                route_text="",
                answers=(str(full_row["conversation"][1]["value"]),),
                micro_batch_size=1,
            )[0]
            (
                float(training["full_answer_weight"]) * full_nll_tensor
            ).backward()
            full_nll = float(full_nll_tensor.item())

        terms = stage1_u_tokenizer_pretraining_terms(
            tokenizer=uq_tokenizer,
            reconstruction_head=decoder,
            components=components.cuda(non_blocking=True),
            zero_anchor_weight=float(training["zero_anchor_weight"]),
            smooth_l1_beta=float(training["smooth_l1_beta"]),
        )
        (float(training["reconstruction_weight"]) * terms.loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            all_trainable, float(training["gradient_clip_norm"])
        )
        finite = (
            np.isfinite(field["all_candidate_cross_entropy"])
            and np.isfinite(field["target_margin"])
            and bool(torch.isfinite(terms.loss))
            and bool(torch.isfinite(norm))
            and _all_finite(all_trainable)
        )
        if not finite:
            raise RuntimeError("v15 U-language alignment became non-finite")
        optimizer.step()
        item = {
            "optimizer_step": step,
            "group_id": example.group_id,
            "event_id": assets.group_event[example.group_id],
            "variant": example.variant,
            "field_tag": example.tag,
            "target_value": example.target,
            "candidate_count": len(answers),
            "all_candidate_cross_entropy": field[
                "all_candidate_cross_entropy"
            ],
            "target_margin": field["target_margin"],
            "prediction_correct": field["predicted_index"] == target_index,
            "full_answer_nll": full_nll,
            "reconstruction_loss": float(terms.reconstruction_loss.item()),
            "zero_anchor_loss": float(terms.zero_anchor_loss.item()),
            "gradient_norm_before_clip": float(norm.item()),
            "finite": finite,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LV15U] " + json.dumps(item, sort_keys=True), flush=True)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trained-v14-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=720)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.trained_v14_checkpoint,
        args.dataset_manifest,
        args.v11_records,
        args.dataset_audit_report,
        args.view_feature_cache,
        args.u_tokenizer_checkpoint,
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v15 U-language alignment prerequisite is missing")
    if args.optimizer_steps != 720 or args.answer_batch_size != 2:
        raise ValueError("v15 bounded pilot requires 720 steps and batch size 2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty v15 output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)
    v121._v101()._configure_base()
    assets = v14.UConceptAssets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )
    schedule = _training_schedule(assets, args.optimizer_steps, args.seed)

    if args.preflight_only:
        if args.trainer_preflight is not None:
            raise ValueError("v15 preflight cannot consume itself")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("v15 preflight requires a fresh output path")
        value = _preflight(args, assets, schedule)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "groups": value["dataset_audit"]["groups"],
                    "optimizer_steps": len(schedule),
                    "output": str(args.preflight_output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_output is not None or args.trainer_preflight is None:
        raise ValueError("v15 training requires a frozen trainer preflight")
    _validate_preflight(args)
    if not torch.cuda.is_available():
        raise RuntimeError("v15 U-language alignment pilot requires CUDA")

    from mmcv.utils import set_random_seed

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, tokenizer = base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    initialization = _load_v14_lora(lm, args.trained_v14_checkpoint)
    uq_tokenizer = v10._load_frozen_u_tokenizer(
        args.u_tokenizer_checkpoint.resolve()
    ).cuda()
    uq_tokenizer.requires_grad_(True)
    decoder = UQSummaryReconstructionHead(
        model_dim=4096, hidden_dim=256, component_dim=3
    ).cuda()
    lora_names = [
        name for name, parameter in lm.named_parameters() if parameter.requires_grad
    ]
    lora_parameters = [
        parameter for parameter in lm.parameters() if parameter.requires_grad
    ]
    if not lora_parameters or any("lora_" not in name for name in lora_names):
        raise RuntimeError("v15 ORION trainable scope is not LoRA-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = _gradient_probe(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        assets=assets,
        lora_parameters=lora_parameters,
    )
    (args.output_dir / "runtime_gradient_probe.json").write_text(
        json.dumps(probe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    before = {
        "dev": _evaluate(
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            assets=assets,
            answer_batch_size=args.answer_batch_size,
            generate=False,
        ),
        "reconstruction": _reconstruction_metrics(uq_tokenizer, decoder, assets),
    }
    history = _train(
        args=args,
        protocol=protocol,
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        decoder=decoder,
        assets=assets,
        schedule=schedule,
        lora_parameters=lora_parameters,
    )
    after = {
        "dev": _evaluate(
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            assets=assets,
            answer_batch_size=args.answer_batch_size,
            generate=True,
        ),
        "reconstruction": _reconstruction_metrics(uq_tokenizer, decoder, assets),
    }
    diagnostics = protocol["soft_diagnostics"]
    quality = {
        "nonzero_semantics_improved": after["dev"][
            "nonzero_accuracy_excluding_presence"
        ]
        > before["dev"]["nonzero_accuracy_excluding_presence"],
        "nonzero_semantics_above_v14_majority_baseline": after["dev"][
            "nonzero_accuracy_excluding_presence"
        ]
        >= float(diagnostics["minimum_nonzero_accuracy_excluding_presence"]),
        "view_balanced_accuracy": after["dev"]["balanced_accuracy_by_tag"][
            "U_VIEW"
        ]
        >= float(diagnostics["minimum_view_balanced_accuracy"]),
        "mean_changed_field_response": after["dev"]["counterfactual"][
            "changed_field_exact_response_fraction"
        ]
        >= float(diagnostics["minimum_changed_field_exact_response"]),
        "reconstruction_improved": after["reconstruction"][
            "mean_reconstruction_loss"
        ]
        < before["reconstruction"]["mean_reconstruction_loss"],
    }
    checkpoint = {
        "schema": SCHEMA,
        "status": "bounded_u_language_alignment_complete",
        "optimizer_steps": len(history),
        "initialization": initialization,
        "orion_lora": {
            name: value.detach().cpu()
            for name, value in lm.state_dict().items()
            if "lora_" in name
        },
        "u_tokenizer": {
            name: value.detach().cpu()
            for name, value in uq_tokenizer.state_dict().items()
        },
        "reconstruction_decoder_included": False,
        "stage1_observation_estimator_frozen": True,
        "task_free_u_language_alignment": True,
        "r_bridge_route_task_risk_action_absent": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "closed_loop_eligible": False,
    }
    checkpoint_path = args.output_dir / "stage2l_v15_u_language_alignment.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "status": (
            "bounded_u_language_alignment_soft_diagnostics_passed"
            if all(quality.values())
            else "bounded_u_language_alignment_completed_with_soft_failures"
        ),
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "optimizer_steps": len(history),
        "initialization": initialization,
        "dataset_audit": v14._dataset_audit(assets),
        "schedule_audit": _schedule_audit(schedule, assets),
        "runtime_gradient_probe": probe,
        "trainable_scope": {
            "orion": "lora_only",
            "lora_parameter_count": sum(value.numel() for value in lora_parameters),
            "u_tokenizer_parameter_count": sum(
                value.numel() for value in uq_tokenizer.parameters()
            ),
            "disposable_reconstruction_head_parameter_count": sum(
                value.numel() for value in decoder.parameters()
            ),
        },
        "before": before,
        "after": after,
        "soft_diagnostics": quality,
        "soft_diagnostics_passed": all(quality.values()),
        "history": history,
        "architecture_invariants": {
            "direct_u_tokens_enter_orion": True,
            "stage1_observation_estimator_trainable": False,
            "existing_u_tokenizer_projection_trainable": True,
            "new_bridge_or_r_input_present": False,
            "route_task_risk_action_present": False,
            "trajectory_or_control_loss": False,
            "density_uq_or_governor_used": False,
            "all_candidate_field_supervision": True,
            "task_agnostic_reconstruction_retention": True,
        },
        "provenance": {
            "validated_inputs": _validated_inputs(args),
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "protocol_sha256": _sha256(args.training_protocol.resolve()),
            "preflight_sha256": _sha256(args.trainer_preflight.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "locks": {
            "formal_stage2l_ready": False,
            "stage2p_ready": False,
            "closed_loop_eligible": False,
            "locked_test_read": False,
            "automatic_extension": False,
        },
        "claim_boundary": (
            "Task-free U-to-language alignment pilot only; no task relevance, "
            "planning, closed-loop, safety, or external-generalization claim."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_steps": len(history),
                "soft_diagnostics_passed": all(quality.values()),
                "report": str((args.output_dir / "report.json").resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del lm, uq_tokenizer, decoder
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
