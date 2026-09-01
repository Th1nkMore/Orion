#!/usr/bin/env python3
"""One bounded eight-event Stage2-L multi-route pre-experiment.

Stage1 U is frozen and task agnostic.  ORION/VLM owns dense relevance R,
fixed K=U*sigmoid(R), categorical task fields, and response stance.  Each
optimizer step samples one complete matched group from every train event so
route conflicts are visible before the update.  Dev events are evaluation
only.  This script contains no trajectory/control loss, Density UQ, governor,
automatic retry, or formal-training unlock.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.utils import set_random_seed

from scripts.scenario_factory_lib import sha256_file
from scripts.train_stage2l_v7_route151_smoke import (
    BRIDGE_TOKENS,
    CACHE_SCHEMA,
    IGNORE_INDEX,
    ORION_VISUAL_TOKENS,
    SEMANTIC_TOKENS,
    SPATIAL_UQ_TOKENS,
    _expanded_labels,
    _generate,
    _load_json,
    _load_orion_lm,
    _relevance_logits,
    _route_text,
    _training_tokens,
)
from scripts.train_stage2l_v9_route151_smoke import (
    TASK_RELEVANCE_FIELDS,
    _condition_variant_v9,
    _field_loss,
    _field_metrics,
)
from scripts.upgrade_stage2l_v9_qa_records import audit_records
from uq_estimator.stage2l_calibrated_objective import (
    geometry_normalized_task_risk_ranking_terms,
    relevance_support_metrics,
)
from uq_estimator.stage2l_matched_objective import (
    HARD_STANCE_VARIANTS,
    MATCHED_VARIANTS,
    partition_complete_matched_groups,
)
from uq_estimator.stage2l_pilot import resolve_reference
from uq_estimator.stage2l_qa_contract_v5 import (
    QUESTION_FAMILIES,
    deterministic_render_metrics,
    expected_semantic_fields,
)
from uq_estimator.stage2l_structured_field_head import (
    TASK_FIELD_VOCABULARIES,
    VLMTaskSemanticFieldHead,
    decode_task_field_predictions,
)
from uq_estimator.stage2l_support_aligned_objective_v9 import (
    support_aligned_relevance_terms,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


SCHEMA = "orion.stage2l_mr1_multiroute_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_mr1_training_protocol.v1"
DATASET_SCHEMA = "orion.stage2l_multiroute_smoke_dataset.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_mr1_trainer_preflight.v1"
EXPECTED_EVENT_COUNT = 8
EXPECTED_TRAIN_EVENT_COUNT = 6
EXPECTED_DEV_EVENT_COUNT = 2
EXPECTED_GROUP_COUNT = 37
EXPECTED_RECORD_COUNT = 740
ALLOWED_BOUNDED_OPTIMIZER_STEPS = (40, 80)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _read_records(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _answer_nlls_mr1(
    *,
    lm,
    tokenizer,
    vision: torch.Tensor,
    row: Mapping[str, Any],
    route_text: str,
    answers: Sequence[str],
    micro_batch_size: int,
) -> torch.Tensor:
    """Per-answer NLL, including the singleton supervised-answer case.

    The historical v7 helper intentionally requires a distinct negative for
    preference training.  MR1 has no preference loss, so auxiliary causal LM
    supervision validly supplies one target answer at a time.
    """

    if not answers:
        raise ValueError("at least one supervised answer is required")
    if micro_batch_size < 1:
        raise ValueError("answer micro-batch size must be positive")
    all_nlls = []
    for start in range(0, len(answers), micro_batch_size):
        chunk = answers[start : start + micro_batch_size]
        encoded = [
            _training_tokens(tokenizer, row, route_text, answer=answer)
            for answer in chunk
        ]
        lengths = [int(ids.shape[0]) for ids, _ in encoded]
        max_length = max(lengths)
        pad_id = int(tokenizer.pad_token_id or 0)
        input_ids = torch.full((len(chunk), max_length), pad_id, dtype=torch.long)
        labels = torch.full(
            (len(chunk), max_length), IGNORE_INDEX, dtype=torch.long
        )
        attention = torch.zeros((len(chunk), max_length), dtype=torch.bool)
        unpadded = []
        for index, (ids, target) in enumerate(encoded):
            length = lengths[index]
            input_ids[index, :length] = ids
            labels[index, :length] = target
            attention[index, :length] = True
            unpadded.append((ids, target))
        input_ids = input_ids.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        attention = attention.cuda(non_blocking=True)
        image_batch = vision.expand(len(chunk), -1, -1).contiguous()
        output = lm(
            input_ids=input_ids,
            attention_mask=attention,
            images=image_batch,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = output.logits
        expanded = torch.full(
            (len(chunk), logits.shape[1]),
            IGNORE_INDEX,
            dtype=torch.long,
            device=logits.device,
        )
        for index, (ids, target) in enumerate(unpadded):
            current = _expanded_labels(
                target.to(device=logits.device),
                ids.to(device=logits.device),
                visual_token_count=vision.shape[1],
            )
            if current.shape[0] > logits.shape[1]:
                raise RuntimeError("expanded MR1 supervision exceeds LM logits")
            expanded[index, : current.shape[0]] = current
        shift_logits = logits[:, :-1].float()
        shift_labels = expanded[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape(len(chunk), -1)
        valid = shift_labels.ne(IGNORE_INDEX)
        counts = valid.sum(dim=-1)
        if bool((counts == 0).any()):
            raise RuntimeError("supervised answer has no target tokens")
        all_nlls.append((token_loss * valid).sum(dim=-1) / counts)
    return torch.cat(all_nlls, dim=0)


class EventBalancedSampler:
    """Deterministically choose one group from every train event per step."""

    def __init__(
        self,
        event_groups: Mapping[str, Sequence[str]],
        *,
        seed: int,
    ) -> None:
        if len(event_groups) != EXPECTED_TRAIN_EVENT_COUNT:
            raise ValueError("MR1 sampler requires exactly six train events")
        if any(not values for values in event_groups.values()):
            raise ValueError("every train event must contain a group")
        self._rng = random.Random(seed)
        self._orders = {
            event: list(sorted(values)) for event, values in sorted(event_groups.items())
        }
        self._positions = {event: 0 for event in self._orders}
        for values in self._orders.values():
            self._rng.shuffle(values)

    def next(self) -> Tuple[str, ...]:
        selected = []
        for event in sorted(self._orders):
            order = self._orders[event]
            position = self._positions[event]
            if position >= len(order):
                self._rng.shuffle(order)
                position = 0
            selected.append(order[position])
            self._positions[event] = position + 1
        if len(selected) != len(set(selected)):
            raise RuntimeError("event-balanced sampler repeated a group in one step")
        return tuple(selected)


class MultiRouteAssets:
    """Hash-verified 8-event records, U/R sidecars, and ORION caches."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = _read_json(self.manifest_path)
        if self.manifest.get("schema") != DATASET_SCHEMA:
            raise ValueError("unsupported MR1 dataset manifest")
        if (
            self.manifest.get("event_count") != EXPECTED_EVENT_COUNT
            or self.manifest.get("train_event_count") != EXPECTED_TRAIN_EVENT_COUNT
            or self.manifest.get("dev_event_count") != EXPECTED_DEV_EVENT_COUNT
            or self.manifest.get("record_count") != EXPECTED_RECORD_COUNT
            or self.manifest.get("formal_stage2l_training_allowed") is not False
            or self.manifest.get("stage2p_allowed") is not False
            or self.manifest.get("review_boundary", {}).get(
                "eligible_for_bounded_preexperiment"
            )
            is not True
            or self.manifest.get("review_boundary", {}).get(
                "eligible_for_formal_training"
            )
            is not False
        ):
            raise ValueError("MR1 dataset scope or locks differ")

        self.records_path = Path(self.manifest["records"]["path"]).resolve()
        if (
            not self.records_path.is_file()
            or sha256_file(self.records_path)
            != self.manifest["records"]["sha256"]
        ):
            raise ValueError("MR1 records are absent or stale")
        self.records = _read_records(self.records_path)
        self.qa_audit = audit_records(self.records)
        self.groups = partition_complete_matched_groups(self.records)
        if (
            len(self.records) != EXPECTED_RECORD_COUNT
            or len(self.groups) != EXPECTED_GROUP_COUNT
            or self.qa_audit.get("passed") is not True
        ):
            raise ValueError("MR1 records fail V5 audit/counts")

        self.audit_artifacts = {}
        for name, ref in (
            ("aggregate", self.manifest["audit"]),
            ("train", self.manifest["split_audits"]["train"]),
            ("dev", self.manifest["split_audits"]["dev"]),
            ("references", self.manifest["reference_audit"]),
        ):
            path = Path(ref["path"]).resolve()
            if not path.is_file() or sha256_file(path) != ref["sha256"]:
                raise ValueError("MR1 %s audit is absent or stale" % name)
            value = _read_json(path)
            if value.get("passed") is not True:
                raise ValueError("MR1 %s audit did not pass" % name)
            self.audit_artifacts[name] = value

        self.rows: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
        self.group_rows: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
        self.group_event: Dict[str, str] = {}
        self.group_split: Dict[str, str] = {}
        for group in self.groups:
            group_id = str(group[0]["counterfactual"]["group_id"])
            events = {str(row["event_id"]) for row in group}
            splits = {str(row["split"]) for row in group}
            if len(events) != 1 or len(splits) != 1:
                raise ValueError("matched group crosses event or split")
            event = next(iter(events))
            split = next(iter(splits))
            if split not in ("train", "dev"):
                raise ValueError("MR1 may use only train/dev records")
            self.group_rows[group_id] = group
            self.group_event[group_id] = event
            self.group_split[group_id] = split
            for row in group:
                key = (
                    group_id,
                    str(row["counterfactual"]["variant"]),
                    str(row["question_family"]),
                )
                if key in self.rows:
                    raise ValueError("duplicate MR1 group/variant/family")
                self.rows[key] = row

        self.event_meta = {
            str(value["event_id"]): value for value in self.manifest["events"]
        }
        if set(self.event_meta) != set(self.group_event.values()):
            raise ValueError("MR1 event manifest and records differ")
        if any(
            self.event_meta[event]["split"] != self.group_split[group]
            for group, event in self.group_event.items()
        ):
            raise ValueError("MR1 event split differs from records")
        self.event_groups: Dict[str, Dict[str, Tuple[str, ...]]] = {
            split: {} for split in ("train", "dev")
        }
        for event, meta in sorted(self.event_meta.items()):
            split = str(meta["split"])
            groups = tuple(sorted(
                group
                for group, current_event in self.group_event.items()
                if current_event == event
            ))
            if len(groups) != int(meta["keyframe_count"]):
                raise ValueError("MR1 cache keyframe count differs")
            self.event_groups[split][event] = groups
        if (
            len(self.event_groups["train"]) != EXPECTED_TRAIN_EVENT_COUNT
            or len(self.event_groups["dev"]) != EXPECTED_DEV_EVENT_COUNT
        ):
            raise ValueError("MR1 6/2 split differs")

        self.visual_contexts: Dict[str, torch.Tensor] = {}
        for event, meta in sorted(self.event_meta.items()):
            cache_ref = meta["visual_cache"]
            cache_path = Path(cache_ref["path"]).resolve()
            if (
                not cache_path.is_file()
                or sha256_file(cache_path) != cache_ref["sha256"]
            ):
                raise ValueError("MR1 visual cache is absent/stale: %s" % event)
            cache = torch.load(cache_path, map_location="cpu")
            if cache.get("schema") != CACHE_SCHEMA:
                raise ValueError("unsupported MR1 visual cache schema")
            contexts = cache.get("contexts", {})
            if set(contexts) != set(self.event_groups[meta["split"]][event]):
                raise ValueError("MR1 visual cache group set differs: %s" % event)
            for group_id, value in contexts.items():
                if tuple(value.shape) != (1, ORION_VISUAL_TOKENS, 4096):
                    raise ValueError("MR1 ORION visual context shape mismatch")
                self.visual_contexts[str(group_id)] = value.detach().float().cpu()

        if set(self.visual_contexts) != set(self.group_rows):
            raise ValueError("MR1 records and combined visual caches differ")
        self.components: Dict[Tuple[str, str], torch.Tensor] = {}
        self.relevance: Dict[str, torch.Tensor] = {}
        self.route_text: Dict[str, str] = {}
        for group_id in sorted(self.group_rows):
            relevance_arrays = []
            route_payload = None
            for variant in MATCHED_VARIANTS:
                row = self.row(group_id, variant, "task_relevance")
                uq_ref = row["model_input"]["stage1_observation_uq"]
                uq_path = resolve_reference(
                    uq_ref,
                    self.records_path.parent,
                    "Stage1 U for %s/%s" % (group_id, variant),
                )
                with np.load(uq_path, allow_pickle=False) as archive:
                    components = archive[uq_ref["component_key"]].astype(np.float32)
                if components.shape != (4, 6, 40, 40, 3):
                    raise ValueError("unexpected MR1 Stage1 component shape")
                self.components[(group_id, variant)] = torch.from_numpy(
                    components
                ).unsqueeze(0)

                sidecar_ref = row["target"]["map_sidecar"]
                sidecar_path = resolve_reference(
                    sidecar_ref,
                    self.records_path.parent,
                    "R target for %s/%s" % (group_id, variant),
                )
                with np.load(sidecar_path, allow_pickle=False) as archive:
                    relevance = archive[
                        sidecar_ref["relevance_key"]
                    ].astype(np.float32)
                if relevance.shape != (6, 40, 40):
                    raise ValueError("unexpected MR1 task-relevance target shape")
                relevance_arrays.append(relevance)
                current_route = row["model_input"]["route_context"]["payload"]
                if route_payload is None:
                    route_payload = current_route
                elif current_route != route_payload:
                    raise ValueError("matched group changes route context")
            if any(
                not np.array_equal(relevance_arrays[0], value)
                for value in relevance_arrays[1:]
            ):
                raise ValueError("matched group changes the R target")
            target = torch.from_numpy(relevance_arrays[0]).unsqueeze(0)
            self.relevance[group_id] = F.adaptive_avg_pool2d(target, (10, 10))
            self.route_text[group_id] = _route_text(route_payload)

    def row(self, group_id: str, variant: str, family: str) -> Mapping[str, Any]:
        return self.rows[(str(group_id), str(variant), str(family))]

    def groups_for_split(self, split: str) -> Tuple[str, ...]:
        return tuple(sorted(
            group for group, value in self.group_split.items() if value == split
        ))

    def language_anchors(self, group_id: str) -> Tuple[Mapping[str, Any], ...]:
        rows = tuple(
            row
            for row in self.group_rows[group_id]
            if row["loss_policy"]["language_auxiliary_target"] is True
        )
        if len(rows) != 18:
            raise RuntimeError("complete MR1 group must expose 18 language anchors")
        return rows

    def field_targets(self, group_id: str, variant: str) -> Dict[str, str]:
        fields = dict(
            self.row(group_id, variant, "task_relevance")["target"][
                "vlm_task_field_targets"
            ]
        )
        if variant in HARD_STANCE_VARIANTS:
            fields.update(
                self.row(group_id, variant, "driving_implication")["target"][
                    "vlm_task_field_targets"
                ]
            )
        return fields


def _ranking_payload(
    *,
    group_ids: Sequence[str],
    on_uq: Mapping[str, torch.Tensor],
    off_uq: Mapping[str, torch.Tensor],
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    required_oracle_fraction: float,
) -> Dict[str, Any]:
    terms = geometry_normalized_task_risk_ranking_terms(
        torch.cat([on_uq[value] for value in group_ids], dim=0),
        torch.cat([off_uq[value] for value in group_ids], dim=0),
        torch.cat([logits[value] for value in group_ids], dim=0),
        torch.cat([targets[value] for value in group_ids], dim=0),
        required_oracle_fraction=required_oracle_fraction,
    )
    learned = terms.learned_gap.detach().cpu().tolist()
    oracle = terms.oracle_gap.detach().cpu().tolist()
    attained = terms.attained_fraction.detach().cpu().tolist()
    per_group = {
        group_id: {
            "learned_gap": float(learned[index]),
            "oracle_gap": float(oracle[index]),
            "attained_fraction": float(attained[index]),
            "positive_order": bool(learned[index] > 0.0),
        }
        for index, group_id in enumerate(group_ids)
    }
    return {
        "loss": float(terms.loss.item()),
        "minimum_attained_fraction": float(min(attained)),
        "mean_attained_fraction": float(np.mean(attained)),
        "positive_order_fraction": float(np.mean([value > 0.0 for value in learned])),
        "per_group": per_group,
    }


def _supported_macro_recall(metrics: Mapping[str, Any]) -> float:
    values = [
        float(value)
        for per_field in metrics["supported_class_recall"].values()
        for value in per_field.values()
    ]
    if not values:
        raise ValueError("no supported task-field classes were evaluated")
    return float(np.mean(values))


@torch.no_grad()
def _evaluate_split(
    *,
    split: str,
    lm,
    tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    field_head,
    assets: MultiRouteAssets,
    answer_batch_size: int,
    required_oracle_fraction: float,
    support_fraction: float,
    calibration_bce_weight: float,
    background_support_weight: float,
    background_probability_margin: float,
    generate_text: bool,
) -> Dict[str, Any]:
    modules = (lm, uq_tokenizer, relevance_queries, relevance_head, risk_bridge, field_head)
    for module in modules:
        module.eval()
    group_ids = assets.groups_for_split(split)
    logits_by_group = {}
    target_by_group = {}
    on_by_group = {}
    off_by_group = {}
    field_entries = []
    relevance_losses = []
    background_hinges = []
    predicted_for_render = {}
    target_summaries = {}
    diagnostic_conditioned = {}
    diagnostic_groups = {
        event: groups[0] for event, groups in assets.event_groups[split].items()
    }
    for group_id in group_ids:
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        target = assets.relevance[group_id].cuda(non_blocking=True)
        logits = _relevance_logits(
            lm=lm,
            tokenizer=tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            baseline_vision=baseline,
            relevance_target=target,
            map_row=assets.row(group_id, "observed", "task_relevance"),
            route_text=assets.route_text[group_id],
        )
        map_terms = support_aligned_relevance_terms(
            logits,
            target,
            support_fraction_of_peak=support_fraction,
            calibration_bce_weight=calibration_bce_weight,
            background_support_weight=background_support_weight,
            background_probability_margin=background_probability_margin,
        )
        relevance_losses.append(float(map_terms.loss.item()))
        background_hinges.append(float(map_terms.background_support_hinge.item()))
        conditioned = {
            variant: _condition_variant_v9(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                field_head=field_head,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=logits,
            )
            for variant in MATCHED_VARIANTS
        }
        logits_by_group[group_id] = logits
        target_by_group[group_id] = target
        on_by_group[group_id] = uq_tokenizer(
            assets.components[(group_id, "on_path_uq")].cuda(non_blocking=True)
        ).latest_scalar_uq
        off_by_group[group_id] = uq_tokenizer(
            assets.components[(group_id, "off_path_uq")].cuda(non_blocking=True)
        ).latest_scalar_uq
        for variant in MATCHED_VARIANTS:
            field_entries.append((
                group_id,
                variant,
                conditioned[variant]["field_probabilities"],
                assets.field_targets(group_id, variant),
            ))
        for variant in HARD_STANCE_VARIANTS:
            key = "%s::%s" % (group_id, variant)
            decoded = decode_task_field_predictions(
                conditioned[variant]["predicted_field_indices"]
            )[0]
            summary = assets.row(group_id, variant, "task_relevance")["target"][
                "structured_summary"
            ]
            target_summaries[key] = summary
            predicted_for_render[key] = {
                "observation_semantics": expected_semantic_fields(
                    "observation_semantics", summary
                ),
                "epistemic_limitation": expected_semantic_fields(
                    "epistemic_limitation", summary
                ),
                "task_relevance": {
                    field: decoded[field] for field in TASK_RELEVANCE_FIELDS
                },
                "driving_implication": {
                    "stance": decoded["stance"],
                    "direct_control": "no",
                    "response_basis": "observation_uncertainty",
                },
            }
        event = assets.group_event[group_id]
        if diagnostic_groups[event] == group_id:
            diagnostic_conditioned[group_id] = conditioned

    support = relevance_support_metrics(
        torch.cat([logits_by_group[value] for value in group_ids], dim=0),
        torch.cat([target_by_group[value] for value in group_ids], dim=0),
        support_fraction_of_peak=support_fraction,
    )
    ranking = _ranking_payload(
        group_ids=group_ids,
        on_uq=on_by_group,
        off_uq=off_by_group,
        logits=logits_by_group,
        targets=target_by_group,
        required_oracle_fraction=required_oracle_fraction,
    )
    per_event = {}
    for event, event_groups in assets.event_groups[split].items():
        per_event[event] = {
            "relevance_support": relevance_support_metrics(
                torch.cat([logits_by_group[value] for value in event_groups], dim=0),
                torch.cat([target_by_group[value] for value in event_groups], dim=0),
                support_fraction_of_peak=support_fraction,
            ),
            "ranking": _ranking_payload(
                group_ids=event_groups,
                on_uq=on_by_group,
                off_uq=off_by_group,
                logits=logits_by_group,
                targets=target_by_group,
                required_oracle_fraction=required_oracle_fraction,
            ),
        }
    task_fields = _field_metrics(field_entries)
    task_fields["supported_class_macro_recall"] = _supported_macro_recall(task_fields)
    render = deterministic_render_metrics(predicted_for_render, target_summaries)

    language_nlls = []
    generated = {}
    for event, group_id in diagnostic_groups.items():
        conditioned = diagnostic_conditioned[group_id]
        # One deterministic anchor per family and hard variant (12/event) is
        # enough to diagnose language direction without making it a gate.
        for variant in HARD_STANCE_VARIANTS:
            for family in QUESTION_FAMILIES:
                row = assets.row(group_id, variant, family)
                nll = _answer_nlls_mr1(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=conditioned[variant]["vision"],
                    row=row,
                    route_text=assets.route_text[group_id],
                    answers=(str(row["conversation"][1]["value"]),),
                    micro_batch_size=answer_batch_size,
                )[0]
                language_nlls.append(float(nll.item()))
        if generate_text:
            variant = "on_path_uq"
            row = assets.row(group_id, variant, "driving_implication")
            generated[event] = _generate(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned[variant]["vision"],
                row=row,
                route_text=assets.route_text[group_id],
            )
    result = {
        "split": split,
        "event_count": len(assets.event_groups[split]),
        "group_count": len(group_ids),
        "mean_support_aligned_relevance_loss": float(np.mean(relevance_losses)),
        "mean_background_support_hinge": float(np.mean(background_hinges)),
        "relevance_support": support,
        "ranking": ranking,
        "task_fields": task_fields,
        "deterministic_render": render,
        "per_event": per_event,
        "diagnostic_language_anchor_count": len(language_nlls),
        "diagnostic_mean_auxiliary_language_nll": float(np.mean(language_nlls)),
        "free_generation_diagnostic": generated,
    }
    for module in modules:
        module.train()
    return result


def _validate_protocol(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    project_root: Path,
    assets: MultiRouteAssets,
    max_optimizer_steps: int,
    language_anchors_per_step: int,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported MR1 training protocol")
    locks = protocol.get("launch_locks", {})
    if (
        locks.get("bounded_preexperiment_allowed_after_amendment") is not False
        or locks.get("formal_stage2l_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("immutable_amendment_required") is not True
    ):
        raise ValueError("MR1 base protocol does not preserve launch locks")
    run = protocol["bounded_preexperiment"]
    if (
        int(run["optimizer_steps"]) != max_optimizer_steps
        or int(run["train_events_per_step"]) != EXPECTED_TRAIN_EVENT_COUNT
        or int(run["language_anchors_per_step"]) != language_anchors_per_step
        or run.get("automatic_retry_or_extension") is not False
        or run.get("dev_labels_enter_optimizer") is not False
    ):
        raise ValueError("MR1 runtime arguments differ from protocol")
    expected_manifest = protocol["dataset"]
    if (
        sha256_file(assets.manifest_path) != expected_manifest["manifest_sha256"]
        or sha256_file(assets.records_path) != expected_manifest["records_sha256"]
    ):
        raise ValueError("MR1 dataset differs from protocol")
    for relative, expected in protocol["implementation_sources"].items():
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("MR1 implementation source mismatch: %s" % relative)
    if protocol.get("protocol_path") != str(protocol_path.resolve()):
        raise ValueError("MR1 protocol path is not the frozen platform path")


def _validated_input_hashes(
    *,
    assets: MultiRouteAssets,
    protocol_path: Path,
    trainer_preflight_path: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    cache_hashes = {
        event: sha256_file(Path(meta["visual_cache"]["path"]).resolve())
        for event, meta in sorted(assets.event_meta.items())
    }
    return {
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol_sha256": sha256_file(protocol_path),
        "trainer_preflight_sha256": sha256_file(trainer_preflight_path),
        "dataset_manifest_sha256": sha256_file(assets.manifest_path),
        "records_sha256": sha256_file(assets.records_path),
        "aggregate_audit_sha256": assets.manifest["audit"]["sha256"],
        "train_audit_sha256": assets.manifest["split_audits"]["train"]["sha256"],
        "dev_audit_sha256": assets.manifest["split_audits"]["dev"]["sha256"],
        "reference_audit_sha256": assets.manifest["reference_audit"]["sha256"],
        "visual_cache_sha256_by_event": cache_hashes,
        "orion_config_sha256": sha256_file(config_path),
        "base_orion_checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _validate_amendment(
    *,
    amendment: Mapping[str, Any],
    expected_hashes: Mapping[str, Any],
    output_dir: Path,
    max_optimizer_steps: int,
    answer_batch_size: int,
    language_anchors_per_step: int,
) -> None:
    run = amendment.get("authorized_run", {})
    locks = amendment.get("launch_locks", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or locks.get("stage2l_mr1_bounded_preexperiment_allowed") is not True
        or locks.get("formal_stage2l_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or int(run.get("maximum_submissions", 0)) != 1
        or int(run.get("maximum_optimizer_steps", -1)) != max_optimizer_steps
        or int(run.get("answer_micro_batch_size", -1)) != answer_batch_size
        or int(run.get("language_anchors_per_step", -1))
        != language_anchors_per_step
        or run.get("fresh_initialization_from_original_orion_checkpoint")
        is not True
        or run.get("automatic_retry_or_extension") is not False
        or run.get("formal_training") is not False
        or Path(str(run.get("output_root", ""))).resolve() != output_dir.resolve()
        or amendment.get("validated_inputs") != expected_hashes
    ):
        raise ValueError("MR1 amendment is absent, stale, or broader than one smoke")


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=40)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--language-anchors-per-step", type=int, default=6)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-head", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_optimizer_steps not in ALLOWED_BOUNDED_OPTIMIZER_STEPS:
        raise ValueError("MR1 bounded diagnostics allow exactly 40 or 80 optimizer steps")
    if args.answer_batch_size != 2 or args.language_anchors_per_step != 6:
        raise ValueError("MR1 answer batch/anchors are frozen at 2/6")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("ORION config/checkpoint prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite MR1 output")

    protocol_path = args.training_protocol.resolve()
    protocol = _read_json(protocol_path)
    project_root = Path(__file__).resolve().parents[1]
    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    assets = MultiRouteAssets(args.dataset_manifest)
    _validate_protocol(
        protocol=protocol,
        protocol_path=protocol_path,
        project_root=project_root,
        assets=assets,
        max_optimizer_steps=args.max_optimizer_steps,
        language_anchors_per_step=args.language_anchors_per_step,
    )

    if args.preflight_only:
        if args.launch_amendment is not None or args.trainer_preflight is not None:
            raise ValueError("locked MR1 preflight cannot receive launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("MR1 preflight needs a fresh output path")
        sampler = EventBalancedSampler(
            assets.event_groups["train"], seed=args.seed
        )
        first_two = [sampler.next(), sampler.next()]
        result = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "mr1_trainer_preflight_pass_training_locked",
            "passed": True,
            "training_started": False,
            "gpu_used": False,
            "dataset_manifest_sha256": sha256_file(assets.manifest_path),
            "records_sha256": sha256_file(assets.records_path),
            "train_events": sorted(assets.event_groups["train"]),
            "dev_events": sorted(assets.event_groups["dev"]),
            "train_group_count": len(assets.groups_for_split("train")),
            "dev_group_count": len(assets.groups_for_split("dev")),
            "first_two_balanced_units": first_two,
            "optimizer_steps": args.max_optimizer_steps,
            "primary_groups_per_step": EXPECTED_TRAIN_EVENT_COUNT,
            "language_anchors_per_step": args.language_anchors_per_step,
            "dev_labels_enter_optimizer": False,
            "formal_training_allowed": False,
            "stage2p_allowed": False,
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "training_protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
        }
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.preflight_output is not None:
        raise ValueError("real MR1 run cannot receive --preflight-output")
    if args.trainer_preflight is None or args.launch_amendment is None:
        raise ValueError("real MR1 run requires preflight and immutable amendment")
    trainer_preflight = _read_json(args.trainer_preflight.resolve())
    if (
        trainer_preflight.get("schema") != PREFLIGHT_SCHEMA
        or trainer_preflight.get("passed") is not True
        or trainer_preflight.get("training_started") is not False
        or trainer_preflight.get("trainer", {}).get("sha256")
        != sha256_file(Path(__file__).resolve())
        or trainer_preflight.get("training_protocol", {}).get("sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError("MR1 trainer preflight is absent or stale")
    expected_hashes = _validated_input_hashes(
        assets=assets,
        protocol_path=protocol_path,
        trainer_preflight_path=args.trainer_preflight.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
    )
    _validate_amendment(
        amendment=_read_json(args.launch_amendment.resolve()),
        expected_hashes=expected_hashes,
        output_dir=args.output_dir.resolve(),
        max_optimizer_steps=args.max_optimizer_steps,
        answer_batch_size=args.answer_batch_size,
        language_anchors_per_step=args.language_anchors_per_step,
    )
    if not torch.cuda.is_available():
        raise SystemExit("real MR1 smoke requires CUDA")

    losses = protocol["losses"]
    lambda_language = float(losses["auxiliary_language"]["weight"])
    lambda_map = float(losses["dense_relevance"]["weight"])
    lambda_ranking = float(losses["on_off_ranking"]["weight"])
    lambda_fields = float(losses["task_fields"]["weight"])
    required_oracle_fraction = float(
        losses["on_off_ranking"]["required_oracle_fraction"]
    )
    if any(
        float(losses[name]) != 0.0
        for name in ("language_preference", "trajectory", "direct_control")
    ):
        raise ValueError("MR1 may not train preference/trajectory/control losses")
    support_fraction = float(losses["dense_relevance"]["support_fraction"])
    calibration_bce_weight = float(
        losses["dense_relevance"]["calibration_bce_weight"]
    )
    background_support_weight = float(
        losses["dense_relevance"]["background_support_weight"]
    )
    background_probability_margin = float(
        losses["dense_relevance"]["background_probability_margin"]
    )
    class_counts = assets.audit_artifacts["train"]["task_field_class_counts"]

    lm, tokenizer = _load_orion_lm(args.config.resolve(), args.checkpoint.resolve())
    uq_tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    risk_bridge = TaskRiskLanguageBridge(model_dim=4096, hidden_dim=256).cuda()
    field_head = VLMTaskSemanticFieldHead(model_dim=4096, hidden_dim=256).cuda()
    auxiliary_modules = (
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
        field_head,
    )
    lora_parameters = [
        parameter for parameter in lm.parameters() if parameter.requires_grad
    ]
    auxiliary_parameters = [
        parameter for module in auxiliary_modules for parameter in module.parameters()
    ]
    trainable_parameters = lora_parameters + auxiliary_parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate_lora},
            {"params": auxiliary_parameters, "lr": args.learning_rate_head},
        ],
        weight_decay=1e-4,
    )
    evaluation_args = dict(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        field_head=field_head,
        assets=assets,
        answer_batch_size=args.answer_batch_size,
        required_oracle_fraction=required_oracle_fraction,
        support_fraction=support_fraction,
        calibration_bce_weight=calibration_bce_weight,
        background_support_weight=background_support_weight,
        background_probability_margin=background_probability_margin,
    )
    before = {
        split: _evaluate_split(**evaluation_args, split=split, generate_text=False)
        for split in ("train", "dev")
    }

    sampler = EventBalancedSampler(assets.event_groups["train"], seed=args.seed)
    train_group_ids = list(assets.groups_for_split("train"))
    language_rng = random.Random(args.seed + 1)
    language_rng.shuffle(train_group_ids)
    language_group_position = 0
    language_anchor_position = {group: 0 for group in train_group_ids}
    history = []
    runtime_fail_fast_passed = False
    for step in range(1, args.max_optimizer_steps + 1):
        primary_groups = sampler.next()
        if language_group_position >= len(train_group_ids):
            language_rng.shuffle(train_group_ids)
            language_group_position = 0
        language_group = train_group_ids[language_group_position]
        language_group_position += 1
        anchors = list(assets.language_anchors(language_group))
        start = language_anchor_position[language_group]
        selected_anchors = [
            anchors[(start + offset) % len(anchors)]
            for offset in range(args.language_anchors_per_step)
        ]
        language_anchor_position[language_group] = (
            start + args.language_anchors_per_step
        ) % len(anchors)

        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "language_nll": 0.0,
            "support_aligned_relevance": 0.0,
            "background_support_hinge": 0.0,
            "ranking_loss": 0.0,
            "minimum_attained_fraction": float("inf"),
            "task_field_loss": 0.0,
            "per_group_primary": [],
        }
        primary_denominator = float(len(primary_groups))
        for group_id in primary_groups:
            baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
            target = assets.relevance[group_id].cuda(non_blocking=True)
            relevance_logits = _relevance_logits(
                lm=lm,
                tokenizer=tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                baseline_vision=baseline,
                relevance_target=target,
                map_row=assets.row(group_id, "observed", "task_relevance"),
                route_text=assets.route_text[group_id],
            )
            map_terms = support_aligned_relevance_terms(
                relevance_logits,
                target,
                support_fraction_of_peak=support_fraction,
                calibration_bce_weight=calibration_bce_weight,
                background_support_weight=background_support_weight,
                background_probability_margin=background_probability_margin,
            )
            on_uq = uq_tokenizer(
                assets.components[(group_id, "on_path_uq")].cuda(non_blocking=True)
            ).latest_scalar_uq
            off_uq = uq_tokenizer(
                assets.components[(group_id, "off_path_uq")].cuda(non_blocking=True)
            ).latest_scalar_uq
            ranking_terms = geometry_normalized_task_risk_ranking_terms(
                on_uq,
                off_uq,
                relevance_logits,
                target,
                required_oracle_fraction=required_oracle_fraction,
            )
            conditioned = {
                variant: _condition_variant_v9(
                    uq_tokenizer=uq_tokenizer,
                    risk_bridge=risk_bridge,
                    field_head=field_head,
                    baseline_vision=baseline,
                    components=assets.components[(group_id, variant)].cuda(
                        non_blocking=True
                    ),
                    relevance_logits=relevance_logits,
                )
                for variant in MATCHED_VARIANTS
            }
            field_loss, _ = _field_loss(
                conditioned,
                {
                    variant: assets.field_targets(group_id, variant)
                    for variant in MATCHED_VARIANTS
                },
                class_counts,
            )
            primary_loss = (
                lambda_map * map_terms.loss
                + lambda_ranking * ranking_terms.loss
                + lambda_fields * field_loss
            ) / primary_denominator
            primary_loss.backward()
            attained = float(ranking_terms.attained_fraction.min().item())
            totals["loss"] += float(primary_loss.item())
            totals["support_aligned_relevance"] += (
                float(map_terms.loss.item()) / primary_denominator
            )
            totals["background_support_hinge"] += (
                float(map_terms.background_support_hinge.item())
                / primary_denominator
            )
            totals["ranking_loss"] += (
                float(ranking_terms.loss.item()) / primary_denominator
            )
            totals["minimum_attained_fraction"] = min(
                totals["minimum_attained_fraction"], attained
            )
            totals["task_field_loss"] += (
                float(field_loss.item()) / primary_denominator
            )
            totals["per_group_primary"].append({
                "event_id": assets.group_event[group_id],
                "group_id": group_id,
                "support_aligned_relevance": float(map_terms.loss.item()),
                "ranking_loss": float(ranking_terms.loss.item()),
                "minimum_attained_fraction": attained,
                "task_field_loss": float(field_loss.item()),
            })

        group_id = language_group
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        target = assets.relevance[group_id].cuda(non_blocking=True)
        for row in selected_anchors:
            variant = str(row["counterfactual"]["variant"])
            with torch.no_grad():
                logits = _relevance_logits(
                    lm=lm,
                    tokenizer=tokenizer,
                    relevance_queries=relevance_queries,
                    relevance_head=relevance_head,
                    baseline_vision=baseline,
                    relevance_target=target,
                    map_row=assets.row(group_id, "observed", "task_relevance"),
                    route_text=assets.route_text[group_id],
                )
            conditioned = _condition_variant_v9(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                field_head=field_head,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=logits,
            )
            language_loss = _answer_nlls_mr1(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned["vision"],
                row=row,
                route_text=assets.route_text[group_id],
                answers=(str(row["conversation"][1]["value"]),),
                micro_batch_size=args.answer_batch_size,
            )[0]
            objective = (
                lambda_language * language_loss / len(selected_anchors)
            )
            objective.backward()
            totals["loss"] += float(objective.item())
            totals["language_nll"] += (
                float(language_loss.item()) / len(selected_anchors)
            )

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        finite_loss = bool(np.isfinite(totals["loss"]))
        finite_gradient_norm = bool(torch.isfinite(gradient_norm))
        finite_gradients = _all_finite(
            parameter.grad
            for parameter in trainable_parameters
            if parameter.grad is not None
        )
        if step <= 2 and not (
            finite_loss and finite_gradient_norm and finite_gradients
        ):
            raise RuntimeError("MR1 first-two-step finite-value fail-fast")
        optimizer.step()
        if step == 2:
            runtime_fail_fast_passed = True
        item = {
            "optimizer_step": step,
            "primary_group_ids": list(primary_groups),
            "primary_event_ids": [
                assets.group_event[value] for value in primary_groups
            ],
            "primary_group_count": len(primary_groups),
            "language_group_id": group_id,
            "language_anchor_count": len(selected_anchors),
            "gradient_norm_before_clip": float(gradient_norm.item()),
            "finite_loss": finite_loss,
            "finite_gradient_norm": finite_gradient_norm,
            "finite_gradients": finite_gradients,
            **totals,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LMR1] " + json.dumps(item, sort_keys=True), flush=True)

    after = {
        split: _evaluate_split(
            **evaluation_args, split=split, generate_text=(split == "dev")
        )
        for split in ("train", "dev")
    }
    gates = protocol["release_gates"]
    train = after["train"]
    dev = after["dev"]
    checks = {
        "runtime_first_two_steps_finite": runtime_fail_fast_passed,
        "every_step_is_six_event_balanced": all(
            row["primary_group_count"] == EXPECTED_TRAIN_EVENT_COUNT
            and len(set(row["primary_event_ids"])) == EXPECTED_TRAIN_EVENT_COUNT
            and set(row["primary_event_ids"]) == set(assets.event_groups["train"])
            for row in history
        ),
        "dev_labels_never_enter_optimizer": all(
            all(assets.group_split[value] == "train" for value in row["primary_group_ids"])
            and assets.group_split[row["language_group_id"]] == "train"
            for row in history
        ),
        "train_relevance_foreground_recall": (
            train["relevance_support"]["foreground_recall"]
            >= gates["train_min_foreground_recall"]
        ),
        "train_relevance_background_fpr": (
            train["relevance_support"]["background_false_positive_rate"]
            <= gates["train_max_background_fpr"]
        ),
        "train_all_groups_positive_order": (
            train["ranking"]["positive_order_fraction"] == 1.0
        ),
        "train_all_groups_attain_margin": (
            train["ranking"]["minimum_attained_fraction"]
            >= gates["train_min_oracle_fraction"]
        ),
        "train_task_field_accuracy": (
            train["task_fields"]["overall_accuracy"]
            >= gates["train_min_task_field_accuracy"]
        ),
        "dev_relevance_foreground_recall": (
            dev["relevance_support"]["foreground_recall"]
            >= gates["dev_min_foreground_recall"]
        ),
        "dev_relevance_background_fpr": (
            dev["relevance_support"]["background_false_positive_rate"]
            <= gates["dev_max_background_fpr"]
        ),
        "dev_all_groups_positive_order": (
            dev["ranking"]["positive_order_fraction"] == 1.0
        ),
        "dev_all_groups_attain_margin": (
            dev["ranking"]["minimum_attained_fraction"]
            >= gates["dev_min_oracle_fraction"]
        ),
        "dev_task_field_accuracy": (
            dev["task_fields"]["overall_accuracy"]
            >= gates["dev_min_task_field_accuracy"]
        ),
        "dev_supported_class_macro_recall": (
            dev["task_fields"]["supported_class_macro_recall"]
            >= gates["dev_min_supported_class_macro_recall"]
        ),
        "dev_zero_uq_absence_semantics": (
            dev["task_fields"]["zero_uq_complete_field_accuracy"]
            >= gates["dev_min_zero_uq_complete_field_accuracy"]
        ),
        "dev_stance_accuracy": (
            dev["task_fields"]["per_field_accuracy"]["stance"]
            >= gates["dev_min_stance_accuracy"]
        ),
        "deterministic_render_parse": (
            dev["deterministic_render"]["semantic_parse_rate"] == 1.0
        ),
        "deterministic_render_fields": (
            dev["deterministic_render"]["semantic_field_accuracy"]
            >= gates["dev_min_render_field_accuracy"]
        ),
        "trajectory_control_density_and_governor_disabled": True,
    }
    diagnostics = {
        "train_auxiliary_language_nll_decreases": (
            after["train"]["diagnostic_mean_auxiliary_language_nll"]
            < before["train"]["diagnostic_mean_auxiliary_language_nll"]
        ),
        "dev_auxiliary_language_nll_decreases": (
            after["dev"]["diagnostic_mean_auxiliary_language_nll"]
            < before["dev"]["diagnostic_mean_auxiliary_language_nll"]
        ),
        "free_generation_is_release_evidence": False,
        "unsupported_spatial_classes_are_release_gates": False,
        "formal_human_per_frame_review_complete": False,
    }
    passed = all(checks.values())
    status = (
        "engineering_mr1_multiroute_smoke_pass"
        if passed
        else "engineering_mr1_multiroute_smoke_failed_gate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_mr1_multiroute_smoke.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "status": status,
            "engineering_preexperiment_only": True,
            "formal_training_ready": False,
            "stage2p_ready": False,
            "uq_tokenizer": uq_tokenizer.state_dict(),
            "relevance_queries": relevance_queries.state_dict(),
            "relevance_head": relevance_head.state_dict(),
            "risk_bridge": risk_bridge.state_dict(),
            "task_field_head": field_head.state_dict(),
            "lora": {
                name: value.detach().cpu()
                for name, value in lm.state_dict().items()
                if "lora_" in name
            },
            "optimizer_steps": len(history),
        },
        checkpoint_path,
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "engineering_preexperiment_only": True,
        "formal_training_ready": False,
        "stage2p_ready": False,
        "optimizer_steps": len(history),
        "primary_group_presentations": len(history) * EXPECTED_TRAIN_EVENT_COUNT,
        "language_anchor_presentations": (
            len(history) * args.language_anchors_per_step
        ),
        "train_events": sorted(assets.event_groups["train"]),
        "dev_events": sorted(assets.event_groups["dev"]),
        "before": before,
        "after": after,
        "checks": checks,
        "diagnostics": diagnostics,
        "history": history,
        "architecture": {
            "stage1_adapter_frozen_and_task_agnostic": True,
            "task_relevance_owned_by_vlm": True,
            "fixed_k_equals_u_times_sigmoid_r": True,
            "field_head_reads_u_and_k": True,
            "task_field_gradient_to_relevance_logits": False,
            "qa_language_gradient_to_relevance_logits": False,
            "dev_labels_enter_optimizer": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
            "trajectory_or_control_loss": False,
            "free_language_is_release_evidence": False,
        },
        "known_coverage_gaps": protocol["known_coverage_gaps"],
        "provenance": {
            "validated_inputs": expected_hashes,
            "dataset_manifest": str(assets.manifest_path),
            "training_protocol": str(protocol_path),
            "trainer_preflight": str(args.trainer_preflight.resolve()),
            "launch_amendment": str(args.launch_amendment.resolve()),
            "output_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
            },
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "checks": checks,
    }, indent=2, sort_keys=True))
    del optimizer, lm
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
