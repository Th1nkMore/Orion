#!/usr/bin/env python3
"""Train ORION LoRA to read frozen Stage-1 U before task binding.

This bounded Stage2-L1 smoke uses the same frozen native visual context for a
matched group while changing only U.  Targets describe presence, camera,
coarse image region, level, temporal trend, and component identity.  Route,
R, K, driving risk, action, trajectory, control, Density UQ, and corruption
metadata are absent from both the prompt and the optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import scripts.train_stage2l_v11_identifiable_smoke as v11
import scripts.train_stage2l_v121_factorized_r_smoke as v121
from uq_estimator.stage2l_process_qa_v13 import (
    detached_conditioning_gradient_anchor,
)
from uq_estimator.stage2l_pilot import resolve_reference
from uq_estimator.stage2l_u_concept_qa_v14 import (
    TAG_ORDER,
    U_VARIANTS,
    audit_u_variant_group,
    build_distribution_preserving_u_variants,
    build_u_qa_row,
    render_u_answer,
    summarize_u_components,
)


SCHEMA = "orion.stage2l-v14-u-concept-lora-smoke/v1"
PROTOCOL_SCHEMA = "orion.stage2l-v14-u-concept-lora-protocol/v1"
PREFLIGHT_SCHEMA = "orion.stage2l-v14-u-concept-lora-preflight/v1"


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


class UConceptAssets:
    """Frozen 17-event vision cache plus real observed Stage-1 U histories."""

    def __init__(
        self,
        dataset_manifest: Path,
        view_feature_cache: Path,
        records_path: Path,
        audit_report_path: Path,
    ) -> None:
        self.base = v11.V11Assets(
            dataset_manifest,
            view_feature_cache,
            records_path,
            audit_report_path,
        )
        self.event_meta = self.base.event_meta
        self.event_groups = self.base.event_groups
        self.group_event = self.base.group_event
        self.group_split = self.base.group_split
        self.visual_contexts = self.base.visual_contexts
        self.records_path = self.base.records_path
        self.variants: dict[tuple[str, str], torch.Tensor] = {}
        self.summaries: dict[tuple[str, str], Any] = {}
        self.group_audits: dict[str, dict[str, Any]] = {}
        for group_id in sorted(self.base.group_rows):
            row = self.base.row(group_id, "observed", "task_relevance")
            reference = row["model_input"]["stage1_observation_uq"]
            if reference.get("source") != "frozen_stage1_observation_adapter":
                raise ValueError("U-concept source is not frozen observed Stage-1 U")
            path = resolve_reference(
                reference,
                self.records_path.parent,
                "v14 observed Stage1 U for %s" % group_id,
            )
            with np.load(path, allow_pickle=False) as archive:
                components = archive[reference["component_key"]].astype(np.float32)
            if components.shape != (4, 6, 40, 40, 3):
                raise ValueError("v14 observed Stage1 component shape differs")
            observed = torch.from_numpy(components).unsqueeze(0)
            variants = build_distribution_preserving_u_variants(observed, group_id)
            audit = audit_u_variant_group(variants)
            if not audit["passed"]:
                raise ValueError("v14 U variant audit failed for %s" % group_id)
            self.group_audits[group_id] = audit
            for name, value in variants.items():
                self.variants[(group_id, name)] = value
                self.summaries[(group_id, name)] = summarize_u_components(value)

    def groups_for_split(self, split: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                group_id
                for group_id, value in self.group_split.items()
                if value == split
            )
        )

    def answer(self, group_id: str, variant: str, tag: str | None = None) -> str:
        return render_u_answer(self.summaries[(group_id, variant)], tag)

    def row(self, group_id: str, variant: str, tag: str | None = None) -> dict:
        return build_u_qa_row(self.summaries[(group_id, variant)], tag)


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
    }


def _protocol_checks(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "bounded_u_concept_lora_smoke"
        or protocol.get("input_sha256") != _validated_inputs(args)
        or architecture.get("direct_u_tokens_enter_orion") is not True
        or architecture.get("r_input_present") is not False
        or architecture.get("route_or_task_prompt_present") is not False
        or architecture.get("stage1_trainable") is not False
        or architecture.get("u_tokenizer_trainable") is not False
        or architecture.get("orion_trainable_scope") != "lora_only"
        or architecture.get("trajectory_or_control_loss") is not False
        or int(training.get("optimizer_steps", 0)) != int(args.optimizer_steps)
        or training.get("automatic_retry") is not False
        or training.get("automatic_extension") is not False
        or Path(str(protocol.get("output_root", ""))).resolve()
        != args.output_dir.resolve()
    ):
        raise ValueError("v14 U-concept protocol is absent or stale")


def _dataset_audit(assets: UConceptAssets) -> dict[str, Any]:
    by_split = defaultdict(list)
    field_values: dict[str, dict[str, set[str]]] = {
        split: {tag: set() for tag in TAG_ORDER} for split in ("train", "dev")
    }
    for group_id, audit in assets.group_audits.items():
        split = assets.group_split[group_id]
        by_split[split].append(audit["distinct_answer_count"])
        for variant in U_VARIANTS:
            fields = assets.summaries[(group_id, variant)].fields()
            for tag, value in fields.items():
                field_values[split][tag].add(value)
    checks = {
        "event_count_17": len(assets.event_meta) == 17,
        "train_events_13": len(assets.event_groups["train"]) == 13,
        "dev_events_4": len(assets.event_groups["dev"]) == 4,
        "train_groups_60": len(assets.groups_for_split("train")) == 60,
        "dev_groups_20": len(assets.groups_for_split("dev")) == 20,
        "all_group_counterfactual_audits_pass": all(
            value["passed"] for value in assets.group_audits.values()
        ),
        "all_groups_have_four_distinct_answers": all(
            value["distinct_answer_count"] >= 4
            for value in assets.group_audits.values()
        ),
        "train_presence_has_yes_and_no": field_values["train"]["U_PRESENT"]
        == {"yes", "no"},
        "dev_presence_has_yes_and_no": field_values["dev"]["U_PRESENT"]
        == {"yes", "no"},
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "groups": len(assets.group_audits),
        "variant_presentations": len(assets.group_audits) * len(U_VARIANTS),
        "minimum_distinct_answers_per_group": min(
            value["distinct_answer_count"]
            for value in assets.group_audits.values()
        ),
        "field_vocabulary_by_split": {
            split: {tag: sorted(values) for tag, values in tags.items()}
            for split, tags in field_values.items()
        },
    }


def _sequence_audit(args: argparse.Namespace, assets: UConceptAssets) -> dict[str, Any]:
    from mmcv.utils import Config
    from transformers import AutoTokenizer
    import scripts.train_stage2l_mr1_smoke as base

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
            for tag in (None,) + TAG_ORDER:
                row = assets.row(group_id, variant, tag)
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
    protocol: Mapping[str, Any],
    assets: UConceptAssets,
) -> dict[str, Any]:
    dataset = _dataset_audit(assets)
    sequences = _sequence_audit(args, assets)
    if not dataset["passed"] or not sequences["passed"]:
        raise ValueError("v14 U-concept preflight failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "u_concept_lora_preflight_pass",
        "passed": True,
        "training_started": False,
        "gpu_used": False,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "output_root": str(args.output_dir.resolve()),
        "dataset_audit": dataset,
        "sequence_audit": sequences,
        "architecture": {
            "same_visual_context_across_u_variants": True,
            "observed_u_source_is_frozen_stage1": True,
            "nonzero_counterfactuals_preserve_observed_value_multiset": True,
            "direct_u_tokens_enter_orion": True,
            "r_k_route_risk_action_absent": True,
            "stage1_and_u_tokenizer_frozen": True,
            "orion_trainable_scope": "lora_only",
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
        raise ValueError("v14 U-concept preflight is absent or stale")


def _condition(
    *, baseline: torch.Tensor, components: torch.Tensor, uq_tokenizer
) -> torch.Tensor:
    with torch.no_grad():
        tokens = uq_tokenizer(components.cuda(non_blocking=True)).tokens.detach()
    value = torch.cat((baseline.cuda(non_blocking=True), tokens), dim=1)
    if tuple(value.shape) != (1, 1129, 4096):
        raise RuntimeError("v14 direct visual/U conditioning shape differs")
    return value


def _different_answer_variant(
    assets: UConceptAssets,
    group_id: str,
    variant: str,
    tag: str | None = None,
) -> str:
    target = assets.answer(group_id, variant, tag)
    start = U_VARIANTS.index(variant)
    for offset in range(1, len(U_VARIANTS)):
        candidate = U_VARIANTS[(start + offset) % len(U_VARIANTS)]
        if assets.answer(group_id, candidate, tag) != target:
            return candidate
    raise RuntimeError("U-concept group has no distinct negative answer")


def _parse_generated_fields(text: str) -> dict[str, str]:
    return {
        tag: match.group(1)
        for tag in TAG_ORDER
        if (
            match := re.search(
                r"<%s>\s*([A-Za-z0-9_]+)" % re.escape(tag), str(text)
            )
        )
        is not None
    }


@torch.no_grad()
def _evaluate(
    *,
    split: str,
    lm,
    tokenizer,
    uq_tokenizer,
    assets: UConceptAssets,
    answer_batch_size: int,
    generate: bool,
) -> dict[str, Any]:
    import scripts.train_stage2l_mr1_smoke as base
    from scripts.train_stage2l_route196_bridge_smoke import _generate

    lm.eval()
    uq_tokenizer.eval()
    diagnostic_groups = [
        values[0] for _, values in sorted(assets.event_groups[split].items())
    ]
    preferences = defaultdict(list)
    removed_preferences = defaultdict(list)
    target_nlls = []
    generated = {}
    per_group = {}
    for group_id in diagnostic_groups:
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        zero_answer = assets.answer(group_id, "zero_u")
        group_payload = {}
        for variant in U_VARIANTS:
            row = assets.row(group_id, variant)
            negative_variant = _different_answer_variant(
                assets, group_id, variant
            )
            target = assets.answer(group_id, variant)
            negative = assets.answer(group_id, negative_variant)
            vision = _condition(
                baseline=baseline,
                components=assets.variants[(group_id, variant)],
                uq_tokenizer=uq_tokenizer,
            )
            values = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=tokenizer,
                vision=vision,
                row=row,
                route_text="",
                answers=(target, negative),
                micro_batch_size=answer_batch_size,
            )
            target_value, negative_value = values.unbind()
            preferred = bool(target_value < negative_value)
            preferences[variant].append(preferred)
            target_nlls.append(float(target_value.item()))
            removed_target_preferred = None
            if variant != "zero_u" and target != zero_answer:
                removed = base._answer_nlls_mr1(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=baseline,
                    row=row,
                    route_text="",
                    answers=(target, zero_answer),
                    micro_batch_size=answer_batch_size,
                )
                removed_target_preferred = bool(removed[0] < removed[1])
                removed_preferences[variant].append(removed_target_preferred)
            group_payload[variant] = {
                "target_nll": float(target_value.item()),
                "negative_nll": float(negative_value.item()),
                "negative_variant": negative_variant,
                "target_preferred": preferred,
                "removed_u_target_preferred": removed_target_preferred,
            }
            if generate:
                text = _generate(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=vision,
                    row=row,
                    route_text="",
                )
                parsed = _parse_generated_fields(text)
                expected = assets.summaries[(group_id, variant)].fields()
                generated["%s::%s" % (group_id, variant)] = {
                    "text": text,
                    "parsed": parsed,
                    "expected": dict(expected),
                    "field_accuracy": float(
                        np.mean(
                            [parsed.get(tag) == expected[tag] for tag in TAG_ORDER]
                        )
                    ),
                    "all_fields_exact": parsed == dict(expected),
                }
        per_group[group_id] = group_payload
    by_variant = {
        variant: float(np.mean(preferences[variant])) for variant in U_VARIANTS
    }
    removed_by_variant = {
        variant: float(np.mean(values))
        for variant, values in removed_preferences.items()
    }
    full_nonzero = float(
        np.mean([by_variant[name] for name in U_VARIANTS if name != "zero_u"])
    )
    removed_nonzero = float(np.mean(list(removed_by_variant.values())))
    generation_values = list(generated.values())
    return {
        "split": split,
        "diagnostic_group_count": len(diagnostic_groups),
        "diagnostic_groups": diagnostic_groups,
        "mean_target_nll": float(np.mean(target_nlls)),
        "target_preference_fraction_by_variant": by_variant,
        "overall_target_preference_fraction": float(np.mean(list(by_variant.values()))),
        "nonzero_target_preference_fraction": full_nonzero,
        "removed_u_nonzero_target_preference_fraction": removed_nonzero,
        "full_minus_removed_u_preference_fraction": full_nonzero - removed_nonzero,
        "generation": {
            "performed": generate,
            "count": len(generated),
            "mean_field_accuracy": (
                float(np.mean([value["field_accuracy"] for value in generation_values]))
                if generation_values
                else None
            ),
            "exact_all_fields_fraction": (
                float(np.mean([value["all_fields_exact"] for value in generation_values]))
                if generation_values
                else None
            ),
            "samples": generated,
        },
        "per_group": per_group,
    }


def _all_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _gradient_probe(
    *, lm, tokenizer, uq_tokenizer, assets: UConceptAssets, trainable
) -> dict[str, Any]:
    import scripts.train_stage2l_mr1_smoke as base

    lm.train()
    lm.zero_grad(set_to_none=True)
    group_id = sorted(assets.groups_for_split("train"))[0]
    variant = "observed_u"
    row = assets.row(group_id, variant)
    vision = detached_conditioning_gradient_anchor(
        _condition(
            baseline=assets.visual_contexts[group_id],
            components=assets.variants[(group_id, variant)],
            uq_tokenizer=uq_tokenizer,
        )
    )
    nll = base._answer_nlls_mr1(
        lm=lm,
        tokenizer=tokenizer,
        vision=vision,
        row=row,
        route_text="",
        answers=(assets.answer(group_id, variant),),
        micro_batch_size=1,
    )[0]
    nll.backward()
    gradient_count = sum(parameter.grad is not None for parameter in trainable)
    if (
        not nll.requires_grad
        or gradient_count == 0
        or vision.grad is None
        or not bool(torch.isfinite(vision.grad).all())
        or not _all_finite(trainable)
        or any(parameter.grad is not None for parameter in uq_tokenizer.parameters())
    ):
        raise RuntimeError("v14 direct-U LoRA gradient probe failed")
    result = {
        "status": "direct_u_lora_backward_connected",
        "group_id": group_id,
        "variant": variant,
        "nll": float(nll.item()),
        "lora_gradient_parameter_count": gradient_count,
        "conditioning_is_detached_leaf": vision.is_leaf,
        "u_tokenizer_gradient_parameter_count": 0,
        "finite": True,
        "optimizer_step_taken": False,
    }
    lm.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def _train(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    lm,
    tokenizer,
    uq_tokenizer,
    assets: UConceptAssets,
    trainable,
) -> list[dict[str, Any]]:
    import scripts.train_stage2l_mr1_smoke as base

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sampler = v11.OneEventPerStepSampler(
        assets.event_groups["train"], seed=args.seed
    )
    history = []
    for step in range(1, args.optimizer_steps + 1):
        group_id = sampler.next()
        variant = U_VARIANTS[(step - 1) % len(U_VARIANTS)]
        tag = TAG_ORDER[(step - 1) % len(TAG_ORDER)]
        lm.train()
        optimizer.zero_grad(set_to_none=True)
        vision_value = _condition(
            baseline=assets.visual_contexts[group_id],
            components=assets.variants[(group_id, variant)],
            uq_tokenizer=uq_tokenizer,
        )

        full_row = assets.row(group_id, variant)
        negative_variant = _different_answer_variant(assets, group_id, variant)
        full_anchor = detached_conditioning_gradient_anchor(vision_value)
        full_values = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=full_anchor,
            row=full_row,
            route_text="",
            answers=(
                assets.answer(group_id, variant),
                assets.answer(group_id, negative_variant),
            ),
            micro_batch_size=args.answer_batch_size,
        )
        full_nll, negative_nll = full_values.unbind()
        full_preference = F.relu(
            float(training["preference_margin"]) + full_nll - negative_nll
        )
        full_loss = full_nll + float(training["preference_weight"]) * full_preference
        full_loss.backward()

        field_row = assets.row(group_id, variant, tag)
        field_negative_variant = _different_answer_variant(
            assets, group_id, variant, tag
        )
        field_anchor = detached_conditioning_gradient_anchor(vision_value)
        field_values = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=field_anchor,
            row=field_row,
            route_text="",
            answers=(
                assets.answer(group_id, variant, tag),
                assets.answer(group_id, field_negative_variant, tag),
            ),
            micro_batch_size=args.answer_batch_size,
        )
        field_nll, field_negative_nll = field_values.unbind()
        field_preference = F.relu(
            float(training["preference_margin"])
            + field_nll
            - field_negative_nll
        )
        field_loss = float(training["field_loss_weight"]) * (
            field_nll + float(training["preference_weight"]) * field_preference
        )
        field_loss.backward()

        if any(parameter.grad is not None for parameter in uq_tokenizer.parameters()):
            raise RuntimeError("v14 gradient escaped into frozen U tokenizer")
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        finite = (
            bool(torch.isfinite(full_loss))
            and bool(torch.isfinite(field_loss))
            and bool(torch.isfinite(norm))
            and _all_finite(trainable)
        )
        if not finite:
            raise RuntimeError("v14 U-concept LoRA optimization became non-finite")
        optimizer.step()
        item = {
            "optimizer_step": step,
            "group_id": group_id,
            "event_id": assets.group_event[group_id],
            "variant": variant,
            "field_tag": tag,
            "full_nll": float(full_nll.item()),
            "full_negative_nll": float(negative_nll.item()),
            "full_preference_loss": float(full_preference.item()),
            "field_nll": float(field_nll.item()),
            "field_negative_nll": float(field_negative_nll.item()),
            "field_preference_loss": float(field_preference.item()),
            "gradient_norm_before_clip": float(norm.item()),
            "finite": finite,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LV14U] " + json.dumps(item, sort_keys=True), flush=True)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=200)
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
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v14 U-concept prerequisite is missing")
    if args.optimizer_steps != 200 or args.answer_batch_size != 2:
        raise ValueError("v14 bounded smoke requires 200 steps and batch size 2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty v14 output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)
    v121._v101()._configure_base()
    assets = UConceptAssets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )

    if args.preflight_only:
        if args.trainer_preflight is not None:
            raise ValueError("v14 preflight cannot consume itself")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("v14 preflight requires a fresh output path")
        value = _preflight(args, protocol, assets)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": value["status"],
            "groups": value["dataset_audit"]["groups"],
            "variant_presentations": value["dataset_audit"]["variant_presentations"],
            "output": str(args.preflight_output.resolve()),
        }, sort_keys=True))
        return 0
    if args.preflight_output is not None or args.trainer_preflight is None:
        raise ValueError("v14 training requires a frozen trainer preflight")
    _validate_preflight(args)
    if not torch.cuda.is_available():
        raise RuntimeError("v14 U-concept LoRA smoke requires CUDA")

    from mmcv.utils import set_random_seed
    import scripts.train_stage2l_mr1_smoke as base
    import scripts.train_stage2l_v10_staged_smoke as v10

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, tokenizer = base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    uq_tokenizer = v10._load_frozen_u_tokenizer(
        args.u_tokenizer_checkpoint.resolve()
    ).cuda().eval()
    for parameter in uq_tokenizer.parameters():
        parameter.requires_grad = False
    trainable = [parameter for parameter in lm.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in lm.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("v14 ORION trainable scope is not LoRA-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = _gradient_probe(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        assets=assets,
        trainable=trainable,
    )
    (args.output_dir / "runtime_gradient_probe.json").write_text(
        json.dumps(probe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    before = {
        split: _evaluate(
            split=split,
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            assets=assets,
            answer_batch_size=args.answer_batch_size,
            generate=False,
        )
        for split in ("train", "dev")
    }
    history = _train(
        args=args,
        protocol=protocol,
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        assets=assets,
        trainable=trainable,
    )
    after = {
        split: _evaluate(
            split=split,
            lm=lm,
            tokenizer=tokenizer,
            uq_tokenizer=uq_tokenizer,
            assets=assets,
            answer_batch_size=args.answer_batch_size,
            generate=split == "dev",
        )
        for split in ("train", "dev")
    }
    gates = protocol["soft_diagnostics"]
    quality = {
        "train_nll_improved": after["train"]["mean_target_nll"]
        < before["train"]["mean_target_nll"],
        "dev_nll_improved": after["dev"]["mean_target_nll"]
        < before["dev"]["mean_target_nll"],
        "dev_target_preference": after["dev"]["overall_target_preference_fraction"]
        >= float(gates["minimum_dev_target_preference"]),
        "dev_zero_u_preference": after["dev"]["target_preference_fraction_by_variant"]["zero_u"]
        >= float(gates["minimum_dev_zero_preference"]),
        "dev_full_above_removed_u": after["dev"]["full_minus_removed_u_preference_fraction"]
        >= float(gates["minimum_full_minus_removed_u"]),
        "dev_generated_field_accuracy": (
            after["dev"]["generation"]["mean_field_accuracy"]
            >= float(gates["minimum_generated_field_accuracy"])
        ),
    }
    checkpoint = {
        "schema": SCHEMA,
        "status": "bounded_u_concept_lora_complete",
        "optimizer_steps": len(history),
        "orion_lora": {
            name: value.detach().cpu()
            for name, value in lm.state_dict().items()
            if "lora_" in name
        },
        "stage1_and_u_tokenizer_frozen": True,
        "r_k_route_risk_action_absent": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "closed_loop_eligible": False,
    }
    checkpoint_path = args.output_dir / "stage2l_v14_u_concept_lora.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "status": (
            "bounded_u_concept_lora_soft_diagnostics_passed"
            if all(quality.values())
            else "bounded_u_concept_lora_completed_with_soft_failures"
        ),
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "optimizer_steps": len(history),
        "dataset_audit": _dataset_audit(assets),
        "runtime_gradient_probe": probe,
        "trainable_scope": {
            "name": "orion_lora_only",
            "parameter_name_count": len(trainable_names),
            "parameter_count": sum(parameter.numel() for parameter in trainable),
        },
        "before": before,
        "after": after,
        "soft_diagnostics": quality,
        "soft_diagnostics_passed": all(quality.values()),
        "history": history,
        "architecture_invariants": {
            "direct_u_tokens_enter_orion": True,
            "same_visual_context_across_matched_u": True,
            "observed_u_from_frozen_stage1": True,
            "transformed_u_preserves_observed_value_multiset": True,
            "r_input_present": False,
            "k_input_present": False,
            "route_task_risk_action_prompt_present": False,
            "stage1_trainable": False,
            "u_tokenizer_trainable": False,
            "orion_trainable_scope": "lora_only",
            "trajectory_or_control_loss": False,
            "density_uq_or_governor_used": False,
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
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": len(history),
        "soft_diagnostics_passed": all(quality.values()),
        "report": str((args.output_dir / "report.json").resolve()),
    }, sort_keys=True), flush=True)
    del lm, uq_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
