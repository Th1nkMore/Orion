#!/usr/bin/env python3
"""One bounded Stage2-L v11 U-identifiability engineering smoke.

The run freezes the v10.1 same-view contextual relevance path and the Stage1
U tokenizer.  It computes R once per matched group, reuses that exact tensor
for zero/on-path/off-path/shuffled U, and trains only the K-language bridge.
There is no trajectory, control, Density UQ, governor, formal split, or locked
test access in this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from scripts.audit_stage2l_v11_identifiability_dataset import audit_dataset
from scripts.scenario_factory_lib import sha256_file
from uq_estimator.stage2l_calibrated_objective import relevance_support_metrics
from uq_estimator.stage2l_factorized_runtime_v11 import (
    ContextualRelevancePassV11,
    MatchedVLMConditioningV11,
    build_matched_vlm_conditioning_v11,
)
from uq_estimator.stage2l_identifiability import (
    REQUIRED_RISK_VARIANTS,
    audit_answer_preferences,
)
from uq_estimator.stage2l_matched_objective import (
    partition_complete_matched_groups,
)
from uq_estimator.stage2l_pilot import resolve_reference
from uq_estimator.uq_relevance_tokenizer import (
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    ViewAlignedTaskRelevanceQueryTokenizer,
)


SCHEMA = "orion.stage2l_v11_identifiable_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v11_identifiable_smoke_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v11_identifiable_smoke_preflight.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
EXPECTED_RECORDS = 1600
EXPECTED_GROUPS = 80
EXPECTED_EVENTS = 17
EXPECTED_TRAIN_EVENTS = 13
EXPECTED_DEV_EVENTS = 4
V101_SCHEMA = "orion.stage2l_v101_view_aligned_phase_a.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class V11Assets:
    """Versioned v11 records plus immutable v10.1 visual feature caches."""

    def __init__(
        self,
        dataset_manifest: Path,
        view_feature_cache: Path,
        records_path: Path,
        audit_report_path: Path,
    ) -> None:
        import scripts.train_stage2l_mr1_smoke as base
        import scripts.train_stage2l_v101_view_aligned_phase_a as v101

        self.legacy = v101.PhaseAAssets(
            dataset_manifest.resolve(), view_feature_cache.resolve()
        )
        self.manifest_path = dataset_manifest.resolve()
        self.manifest = self.legacy.manifest
        self.records_path = records_path.resolve()
        report = _read_json(audit_report_path.resolve())
        if (
            report.get("v11_ready") is not True
            or report.get("metadata_passed") is not True
            or report.get("tensor_passed") is not True
            or report.get("record_count") != EXPECTED_RECORDS
            or report.get("group_count") != EXPECTED_GROUPS
            or report.get("records_sha256") != sha256_file(self.records_path)
        ):
            raise ValueError("v11 terminal dataset audit is absent or stale")
        self.dataset_audit = audit_dataset(
            self.records_path, verify_tensors=True
        )
        if self.dataset_audit.get("v11_ready") is not True:
            raise ValueError("v11 records fail a fresh full tensor audit")

        self.records = _read_records(self.records_path)
        groups = partition_complete_matched_groups(self.records)
        if len(self.records) != EXPECTED_RECORDS or len(groups) != EXPECTED_GROUPS:
            raise ValueError("v11 record/group counts differ")
        self.rows: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
        self.group_rows: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
        self.group_event: Dict[str, str] = {}
        self.group_split: Dict[str, str] = {}
        for group in groups:
            group_id = str(group[0]["counterfactual"]["group_id"])
            events = {str(row["event_id"]) for row in group}
            splits = {str(row["split"]) for row in group}
            if len(events) != 1 or len(splits) != 1:
                raise ValueError("v11 group crosses event or split")
            self.group_rows[group_id] = group
            self.group_event[group_id] = next(iter(events))
            self.group_split[group_id] = next(iter(splits))
            for row in group:
                key = (
                    group_id,
                    str(row["counterfactual"]["variant"]),
                    str(row["question_family"]),
                )
                if key in self.rows:
                    raise ValueError("duplicate v11 group/variant/family")
                self.rows[key] = row

        if (
            set(self.group_rows) != set(self.legacy.group_rows)
            or self.group_event != self.legacy.group_event
            or self.group_split != self.legacy.group_split
        ):
            raise ValueError("v11 records differ from frozen cache identities")
        self.event_meta = self.legacy.event_meta
        self.event_groups = self.legacy.event_groups
        self.visual_contexts = self.legacy.visual_contexts
        self.view_features = self.legacy.view_features
        self.relevance = self.legacy.relevance

        self.components: Dict[Tuple[str, str], torch.Tensor] = {}
        self.route_text: Dict[str, str] = {}
        for group_id in sorted(self.group_rows):
            route_payload = None
            for variant in REQUIRED_RISK_VARIANTS:
                row = self.row(group_id, variant, "task_relevance")
                reference = row["model_input"]["stage1_observation_uq"]
                path = resolve_reference(
                    reference,
                    self.records_path.parent,
                    "v11 Stage1 U for %s/%s" % (group_id, variant),
                )
                with np.load(path, allow_pickle=False) as archive:
                    components = archive[reference["component_key"]].astype(
                        np.float32
                    )
                if components.shape != (4, 6, 40, 40, 3):
                    raise ValueError("v11 Stage1 component shape differs")
                self.components[(group_id, variant)] = torch.from_numpy(
                    components
                ).unsqueeze(0)
                current_route = row["model_input"]["route_context"]
                if current_route.get("schema") != "orion.route_context.v2":
                    raise ValueError("v11 route context is not version v2")
                current_payload = current_route["payload"]
                if route_payload is None:
                    route_payload = current_payload
                elif current_payload != route_payload:
                    raise ValueError("v11 matched group changes route/ego context")
            self.route_text[group_id] = base._route_text(route_payload)

    def row(self, group_id: str, variant: str, family: str) -> Mapping[str, Any]:
        return self.rows[(str(group_id), str(variant), str(family))]

    def groups_for_split(self, split: str) -> Tuple[str, ...]:
        return tuple(
            sorted(
                group_id
                for group_id, value in self.group_split.items()
                if value == split
            )
        )

    def language_anchors(self, group_id: str) -> Tuple[Mapping[str, Any], ...]:
        rows = [
            self.row(group_id, variant, "task_relevance")
            for variant in REQUIRED_RISK_VARIANTS
        ]
        rows.extend(
            self.row(group_id, variant, "driving_implication")
            for variant in ("zero_uq", "off_path_uq", "on_path_uq")
        )
        if any(
            row.get("loss_policy", {}).get("language_auxiliary_target") is not True
            for row in rows
        ):
            raise ValueError("v11 selected a non-authorized language target")
        return tuple(rows)


class OneEventPerStepSampler:
    """Round-robin events while shuffling keyframes within each event."""

    def __init__(
        self, event_groups: Mapping[str, Sequence[str]], *, seed: int
    ) -> None:
        if len(event_groups) != EXPECTED_TRAIN_EVENTS:
            raise ValueError("v11 sampler requires thirteen train events")
        self._events = tuple(sorted(event_groups))
        self._rng = random.Random(seed)
        self._orders = {
            event: list(sorted(groups)) for event, groups in event_groups.items()
        }
        self._positions = {event: 0 for event in self._events}
        self._event_position = 0
        for values in self._orders.values():
            self._rng.shuffle(values)

    def next(self) -> str:
        event = self._events[self._event_position % len(self._events)]
        self._event_position += 1
        values = self._orders[event]
        position = self._positions[event]
        if position >= len(values):
            self._rng.shuffle(values)
            position = 0
        self._positions[event] = position + 1
        return values[position]


def _set_trainable(module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _load_v101_contextual_relevance(
    *, lm, relevance_queries, relevance_head, checkpoint_path: Path
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu")
    if (
        payload.get("schema") != V101_SCHEMA
        or payload.get("status") != "phase_a_failed_gate"
        or payload.get("optimizer_steps") != 120
        or payload.get("phase_a_only") is not True
        or payload.get("stage1_uq_loaded") is not False
        or payload.get("formal_stage2l_ready") is not False
        or payload.get("stage2p_ready") is not False
    ):
        raise ValueError("v11 contextual-R checkpoint contract differs")
    lm_state = lm.state_dict()
    lora = payload.get("lora", {})
    if {key for key in lm_state if "lora_" in key} != set(lora):
        raise ValueError("v11 contextual-R LoRA keys differ")
    result = lm.load_state_dict(lora, strict=False)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("v11 contextual-R LoRA load was incomplete")
    relevance_queries.load_state_dict(
        payload["view_aligned_relevance_queries"], strict=True
    )
    relevance_head.load_state_dict(payload["relevance_head"], strict=True)
    for module in (lm, relevance_queries, relevance_head):
        _set_trainable(module, False)
        module.eval()
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "optimizer_steps": payload["optimizer_steps"],
        "lora_tensor_count": len(lora),
        "query_tensor_count": len(payload["view_aligned_relevance_queries"]),
        "head_tensor_count": len(payload["relevance_head"]),
        "all_parameters_frozen": all(
            not parameter.requires_grad
            for module in (lm, relevance_queries, relevance_head)
            for parameter in module.parameters()
        ),
    }


def _checkpoint_contracts(args: argparse.Namespace) -> Dict[str, Any]:
    contextual = torch.load(
        args.v101_checkpoint.resolve(), map_location="cpu"
    )
    tokenizer = torch.load(
        args.u_tokenizer_checkpoint.resolve(), map_location="cpu"
    )
    if (
        contextual.get("schema") != V101_SCHEMA
        or contextual.get("status") != "phase_a_failed_gate"
        or contextual.get("optimizer_steps") != 120
    ):
        raise ValueError("v11 preflight contextual-R checkpoint differs")
    if (
        tokenizer.get("schema")
        != "orion.stage1_u_tokenizer_pretraining_run.v1"
        or tokenizer.get("status")
        != "bounded_task_agnostic_tokenizer_pretraining_pass"
        or tokenizer.get("task_agnostic") is not True
    ):
        raise ValueError("v11 preflight U-tokenizer checkpoint differs")
    return {
        "contextual_relevance": {
            "schema": contextual["schema"],
            "status": contextual["status"],
            "optimizer_steps": contextual["optimizer_steps"],
        },
        "u_tokenizer": {
            "schema": tokenizer["schema"],
            "status": tokenizer["status"],
            "task_agnostic": tokenizer["task_agnostic"],
        },
    }


def _runtime_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "uq_estimator"
        / "stage2l_factorized_runtime_v11.py"
    )


def _identifiability_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "uq_estimator"
        / "stage2l_identifiability.py"
    )


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "runtime_sha256": _sha256(_runtime_path()),
        "identifiability_audit_sha256": _sha256(_identifiability_path()),
        "parent_contract_sha256": _sha256(args.parent_contract.resolve()),
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
        "v101_checkpoint_sha256": _sha256(args.v101_checkpoint.resolve()),
        "v101_report_sha256": _sha256(args.v101_report.resolve()),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
    }


def _protocol_input_hashes(args: argparse.Namespace) -> Dict[str, str]:
    values = _validated_inputs(args)
    values.pop("trainer_sha256")
    return values


def _protocol_checks(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    architecture = protocol.get("architecture", {})
    required_false = (
        "u_enters_relevance_query",
        "stage1_trainable",
        "u_tokenizer_trainable",
        "contextual_relevance_trainable",
        "orion_lora_trainable",
        "learned_structured_field_head_used",
        "trajectory_or_control_loss",
        "density_uq_used",
        "governor_used",
    )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "frozen_bounded_identifiability_protocol_launch_locked"
        or protocol.get("input_sha256") != _protocol_input_hashes(args)
        or protocol.get("output_root") != str(args.output_dir.resolve())
        or architecture.get("only_trainable_module")
        != "TaskRiskLanguageBridge"
        or any(architecture.get(key) is not False for key in required_false)
        or protocol.get("launch_locks", {}).get("real_training_allowed")
        is not False
    ):
        raise ValueError("v11 bounded protocol is absent or stale")
    steps = int(protocol.get("training", {}).get("optimizer_steps", 0))
    anchors = int(protocol.get("training", {}).get("anchors_per_step", 0))
    if steps <= 0 or steps > 40 or anchors <= 0 or anchors > 2:
        raise ValueError("v11 bounded training ceiling differs")


def _group_conditioning(
    *,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    assets: V11Assets,
    group_id: str,
    protocol: Mapping[str, Any],
) -> Tuple[MatchedVLMConditioningV11, torch.Tensor]:
    import scripts.train_stage2l_v101_view_aligned_phase_a as v101

    target_holder: Dict[str, torch.Tensor] = {}

    def relevance_forward() -> ContextualRelevancePassV11:
        with torch.no_grad():
            logits, target = v101._map_logits(
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                group_id=group_id,
            )
            baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        target_holder["value"] = target
        return ContextualRelevancePassV11(
            baseline_vision=baseline, relevance_logits=logits
        )

    thresholds = protocol["controlled_u_gates"]
    result = build_matched_vlm_conditioning_v11(
        relevance_forward=relevance_forward,
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        components_by_variant={
            variant: assets.components[(group_id, variant)].cuda(
                non_blocking=True
            )
            for variant in REQUIRED_RISK_VARIANTS
        },
        risk_audit_kwargs={
            "required_on_over_off_margin": float(
                thresholds["minimum_on_over_off_margin"]
            ),
            "maximum_off_path_risk": float(
                thresholds["maximum_off_path_risk_peak"]
            ),
            "minimum_fraction": float(
                thresholds["minimum_group_fraction"]
            ),
        },
    )
    return result, target_holder["value"]


def _support_mask(targets: np.ndarray, fraction: float) -> np.ndarray:
    peaks = targets.reshape(targets.shape[0], -1).max(axis=1)
    shape = (targets.shape[0],) + (1,) * (targets.ndim - 1)
    return targets >= peaks.reshape(shape) * float(fraction)


def _safe_average_precision(scores: np.ndarray, truth: np.ndarray):
    import scripts.train_stage2l_v101_view_aligned_phase_a as v101

    return (
        v101._average_precision(scores, truth)
        if int(truth.astype(bool).sum()) > 0
        else None
    )


@torch.no_grad()
def _evaluate_factorization(
    *,
    split: str,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    assets: V11Assets,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    import scripts.train_stage2l_v101_view_aligned_phase_a as v101

    for module in (
        lm,
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
    ):
        module.eval()
    logits_all = []
    targets_all = []
    per_group = {}
    group_values = {
        "shared_r": [],
        "zero_exact": [],
        "matched_magnitude": [],
        "spatially_distinct": [],
        "on_over_off": [],
        "off_low": [],
    }
    event_groups: Dict[str, list[str]] = {}
    for group_id in assets.groups_for_split(split):
        result, target = _group_conditioning(
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            group_id=group_id,
            protocol=protocol,
        )
        audit = result.task_risk_audit
        logits_all.append(result.shared_relevance_logits)
        targets_all.append(target)
        event_groups.setdefault(assets.group_event[group_id], []).append(group_id)
        per_sample = audit.per_sample_gates
        group_values["shared_r"].append(result.relevance_invariance.all_invariant)
        group_values["zero_exact"].append(
            bool(per_sample["zero_u_exact"].all())
            and bool(per_sample["zero_task_risk_exact"].all())
        )
        group_values["matched_magnitude"].append(
            bool(per_sample["on_off_u_mass_matched"].all())
            and bool(per_sample["on_off_u_peak_matched"].all())
            and bool(per_sample["on_off_u_support_count_matched"].all())
        )
        group_values["spatially_distinct"].append(
            bool(per_sample["on_off_support_spatially_distinct"].all())
        )
        group_values["on_over_off"].append(
            bool(per_sample["on_over_off_margin"].all())
        )
        group_values["off_low"].append(
            bool(per_sample["off_path_risk_below_ceiling"].all())
        )
        per_group[group_id] = {
            "event_id": assets.group_event[group_id],
            "relevance_forward_call_count": result.relevance_forward_call_count,
            "maximum_relevance_drift": max(
                result.relevance_invariance.maximum_absolute_drift.values()
            ),
            "risk_peak": {
                variant: float(value.item())
                for variant, value in audit.risk_peak_by_variant.items()
            },
            "uncertainty_mass": {
                variant: float(value.item())
                for variant, value in audit.uncertainty_mass_by_variant.items()
            },
            "on_minus_off_risk_peak": float(audit.on_over_off_margin.item()),
            "on_minus_shuffled_risk_peak": float(
                audit.on_over_shuffled_margin.item()
            ),
            "gates": {
                key: bool(value.item()) for key, value in per_sample.items()
            },
            "structured_fields": {
                variant: dict(
                    conditioning.deterministic_semantics.structured_fields[0]
                )
                for variant, conditioning in result.conditioning_by_variant.items()
            },
        }

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    support_fraction = float(protocol["r_diagnostics"]["support_fraction_of_peak"])
    support = relevance_support_metrics(
        logits, targets, support_fraction_of_peak=support_fraction
    )
    probability = logits.sigmoid().float().cpu().numpy()
    target_np = targets.float().cpu().numpy()
    foreground = _support_mask(target_np, support_fraction)
    per_view = {}
    for index, camera in enumerate(CAMERA_ORDER):
        truth = foreground[:, index]
        per_view[camera] = {
            "average_precision": _safe_average_precision(
                probability[:, index], truth
            ),
            "foreground_count": int(truth.sum()),
            "foreground_prevalence": float(truth.mean()),
        }
    per_event = {}
    group_ids = list(assets.groups_for_split(split))
    index_by_group = {group_id: index for index, group_id in enumerate(group_ids)}
    for event_id, members in sorted(event_groups.items()):
        indices = [index_by_group[group_id] for group_id in members]
        truth = foreground[indices]
        per_event[event_id] = {
            "group_count": len(indices),
            "average_precision": _safe_average_precision(
                probability[indices], truth
            ),
            "foreground_prevalence": float(truth.mean()),
        }

    required_fraction = float(
        protocol["controlled_u_gates"]["minimum_group_fraction"]
    )
    fractions = {
        key: float(np.mean(values)) for key, values in group_values.items()
    }
    release_checks = {
        "shared_r_bitwise_exact": fractions["shared_r"] == 1.0,
        "zero_u_and_k_exact": fractions["zero_exact"] == 1.0,
        "on_off_magnitude_matched": fractions["matched_magnitude"] == 1.0,
        "on_off_support_spatially_distinct": fractions["spatially_distinct"]
        == 1.0,
        "on_over_off_fraction": fractions["on_over_off"] >= required_fraction,
        "off_path_low_risk_fraction": fractions["off_low"] >= required_fraction,
    }
    return {
        "split": split,
        "group_count": len(group_ids),
        "r_diagnostics": {
            "average_precision": v101._average_precision(
                probability, foreground
            ),
            "relevance_support": support,
            "per_event": per_event,
            "per_view": per_view,
        },
        "controlled_u_fractions": fractions,
        "release_checks": release_checks,
        "all_release_checks_passed": all(release_checks.values()),
        "per_group": per_group,
    }


def _task_relevance_rows(
    assets: V11Assets, group_id: str
) -> Dict[str, Mapping[str, Any]]:
    rows = {
        variant: assets.row(group_id, variant, "task_relevance")
        for variant in REQUIRED_RISK_VARIANTS
    }
    questions = {str(row["conversation"][0]["value"]) for row in rows.values()}
    if len(questions) != 1:
        raise ValueError("matched v11 task-relevance questions differ")
    answers = {variant: str(row["conversation"][1]["value"]) for variant, row in rows.items()}
    if any(
        not any(other != answer for other in answers.values())
        for answer in answers.values()
    ):
        raise ValueError("v11 task-relevance target has no counterfactual answer")
    return rows


@torch.no_grad()
def _evaluate_language(
    *,
    split: str,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    assets: V11Assets,
    protocol: Mapping[str, Any],
    answer_batch_size: int,
) -> Dict[str, Any]:
    import scripts.train_stage2l_mr1_smoke as base

    for module in (
        lm,
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
    ):
        module.eval()
    diagnostic_groups = [
        groups[0] for _, groups in sorted(assets.event_groups[split].items())
    ]
    full_target = {variant: [] for variant in REQUIRED_RISK_VARIANTS}
    full_counterfactual = {variant: [] for variant in REQUIRED_RISK_VARIANTS}
    no_u_target = {variant: [] for variant in REQUIRED_RISK_VARIANTS}
    no_u_counterfactual = {variant: [] for variant in REQUIRED_RISK_VARIANTS}
    per_group = {}
    for group_id in diagnostic_groups:
        result, _ = _group_conditioning(
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            group_id=group_id,
            protocol=protocol,
        )
        rows = _task_relevance_rows(assets, group_id)
        answers_by_variant = {
            variant: str(row["conversation"][1]["value"])
            for variant, row in rows.items()
        }
        unique_answers = tuple(dict.fromkeys(answers_by_variant.values()))
        reference_row = rows[REQUIRED_RISK_VARIANTS[0]]
        no_u_values = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=text_tokenizer,
            vision=result.no_u_ablation_vision,
            row=reference_row,
            route_text=assets.route_text[group_id],
            answers=unique_answers,
            micro_batch_size=answer_batch_size,
        )
        no_u_by_answer = {
            answer: no_u_values[index]
            for index, answer in enumerate(unique_answers)
        }
        group_result = {}
        for variant in REQUIRED_RISK_VARIANTS:
            row = rows[variant]
            full_values = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=text_tokenizer,
                vision=result.conditioning_by_variant[variant].vision_tokens,
                row=row,
                route_text=assets.route_text[group_id],
                answers=unique_answers,
                micro_batch_size=answer_batch_size,
            )
            full_by_answer = {
                answer: full_values[index]
                for index, answer in enumerate(unique_answers)
            }
            target_answer = answers_by_variant[variant]
            alternatives = tuple(
                answer for answer in unique_answers if answer != target_answer
            )
            full_target_value = full_by_answer[target_answer]
            full_counterfactual_value = torch.stack(
                [full_by_answer[answer] for answer in alternatives]
            ).amin()
            no_u_target_value = no_u_by_answer[target_answer]
            no_u_counterfactual_value = torch.stack(
                [no_u_by_answer[answer] for answer in alternatives]
            ).amin()
            full_target[variant].append(full_target_value.cpu())
            full_counterfactual[variant].append(
                full_counterfactual_value.cpu()
            )
            no_u_target[variant].append(no_u_target_value.cpu())
            no_u_counterfactual[variant].append(
                no_u_counterfactual_value.cpu()
            )
            group_result[variant] = {
                "target_answer": target_answer,
                "target_nll": float(full_target_value.item()),
                "closest_counterfactual_nll": float(
                    full_counterfactual_value.item()
                ),
                "target_preferred": bool(
                    full_target_value < full_counterfactual_value
                ),
                "no_u_target_nll": float(no_u_target_value.item()),
                "no_u_closest_counterfactual_nll": float(
                    no_u_counterfactual_value.item()
                ),
                "no_u_target_preferred": bool(
                    no_u_target_value < no_u_counterfactual_value
                ),
            }
        per_group[group_id] = group_result

    full_target_tensors = {
        key: torch.stack(values) for key, values in full_target.items()
    }
    full_counterfactual_tensors = {
        key: torch.stack(values) for key, values in full_counterfactual.items()
    }
    no_u_target_tensors = {
        key: torch.stack(values) for key, values in no_u_target.items()
    }
    no_u_counterfactual_tensors = {
        key: torch.stack(values) for key, values in no_u_counterfactual.items()
    }
    minimum = float(
        protocol["language_gates"]["minimum_per_variant_preference_fraction"]
    )
    full_audit = audit_answer_preferences(
        full_target_tensors,
        full_counterfactual_tensors,
        minimum_fraction=minimum,
    )
    no_u_audit = audit_answer_preferences(
        no_u_target_tensors,
        no_u_counterfactual_tensors,
        minimum_fraction=minimum,
    )
    full_overall = float(
        np.mean(list(full_audit.preference_fraction_by_variant.values()))
    )
    no_u_overall = float(
        np.mean(list(no_u_audit.preference_fraction_by_variant.values()))
    )
    mean_target_nll = float(
        torch.cat(list(full_target_tensors.values())).mean().item()
    )
    return {
        "split": split,
        "diagnostic_group_count": len(diagnostic_groups),
        "diagnostic_groups": diagnostic_groups,
        "mean_target_nll": mean_target_nll,
        "full_conditioning": {
            "preference_fraction_by_variant": dict(
                full_audit.preference_fraction_by_variant
            ),
            "gates": dict(full_audit.gates),
            "all_passed": full_audit.all_passed,
            "overall_preference_fraction": full_overall,
        },
        "no_u_ablation": {
            "kind": "baseline_orion_visual_tokens_only",
            "preference_fraction_by_variant": dict(
                no_u_audit.preference_fraction_by_variant
            ),
            "overall_preference_fraction": no_u_overall,
        },
        "full_minus_no_u_preference_fraction": full_overall - no_u_overall,
        "per_group": per_group,
    }


def _language_release_checks(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    factorization: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, bool]:
    gates = protocol["language_gates"]
    checks = {
        "controlled_u_train_passed": factorization["train"][
            "all_release_checks_passed"
        ],
        "controlled_u_dev_passed": factorization["dev"][
            "all_release_checks_passed"
        ],
        "train_target_nll_improved": after["train"]["mean_target_nll"]
        < before["train"]["mean_target_nll"],
        "dev_target_nll_improved": after["dev"]["mean_target_nll"]
        < before["dev"]["mean_target_nll"],
        "dev_every_variant_prefers_target": after["dev"][
            "full_conditioning"
        ]["all_passed"],
        "dev_no_u_preference_below_ceiling": after["dev"]["no_u_ablation"][
            "overall_preference_fraction"
        ]
        <= float(gates["maximum_no_u_preference_fraction"]),
        "dev_full_improves_over_no_u": after["dev"][
            "full_minus_no_u_preference_fraction"
        ]
        >= float(gates["minimum_full_minus_no_u_fraction"]),
    }
    return checks


def _train_language_bridge(
    *,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    assets: V11Assets,
    protocol: Mapping[str, Any],
    seed: int,
    answer_batch_size: int,
    log_interval: int,
) -> list[dict[str, Any]]:
    import scripts.train_stage2l_mr1_smoke as base

    frozen = (lm, uq_tokenizer, relevance_queries, relevance_head)
    for module in frozen:
        _set_trainable(module, False)
        module.eval()
        for parameter in module.parameters():
            parameter.grad = None
    _set_trainable(risk_bridge, True)
    risk_bridge.train()
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        risk_bridge.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sampler = OneEventPerStepSampler(
        assets.event_groups["train"], seed=seed
    )
    positions = {group_id: 0 for group_id in assets.groups_for_split("train")}
    steps = int(training["optimizer_steps"])
    anchors_per_step = int(training["anchors_per_step"])
    history = []
    for step in range(1, steps + 1):
        group_id = sampler.next()
        anchors = assets.language_anchors(group_id)
        start = positions[group_id]
        selected = tuple(
            anchors[(start + offset) % len(anchors)]
            for offset in range(anchors_per_step)
        )
        positions[group_id] += anchors_per_step
        optimizer.zero_grad(set_to_none=True)
        result, _ = _group_conditioning(
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            group_id=group_id,
            protocol=protocol,
        )
        losses = []
        anchors_used = []
        for row in selected:
            variant = str(row["counterfactual"]["variant"])
            nll = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=text_tokenizer,
                vision=result.conditioning_by_variant[variant].vision_tokens,
                row=row,
                route_text=assets.route_text[group_id],
                answers=(str(row["conversation"][1]["value"]),),
                micro_batch_size=answer_batch_size,
            )[0]
            losses.append(nll)
            anchors_used.append(
                {
                    "variant": variant,
                    "question_family": str(row["question_family"]),
                    "target_nll": float(nll.item()),
                }
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        if any(
            parameter.grad is not None
            for module in frozen
            for parameter in module.parameters()
        ):
            raise RuntimeError("v11 language gradient escaped the K bridge")
        norm = torch.nn.utils.clip_grad_norm_(risk_bridge.parameters(), 1.0)
        finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(norm)) and _all_finite(
            parameter.grad
            for parameter in risk_bridge.parameters()
            if parameter.grad is not None
        )
        if step <= 2 and not finite:
            raise RuntimeError("v11 first-two-step finite fail-fast")
        optimizer.step()
        item = {
            "optimizer_step": step,
            "group_id": group_id,
            "event_id": assets.group_event[group_id],
            "mean_target_nll": float(loss.item()),
            "gradient_norm_before_clip": float(norm.item()),
            "finite": finite,
            "relevance_forward_call_count": result.relevance_forward_call_count,
            "anchors": anchors_used,
            "trainable_scope": "TaskRiskLanguageBridge_only",
        }
        history.append(item)
        if step == 1 or step % log_interval == 0:
            print("[Stage2LV11] " + json.dumps(item, sort_keys=True), flush=True)
    del optimizer
    return history


def _preflight(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    assets: V11Assets,
) -> Dict[str, Any]:
    sampler = OneEventPerStepSampler(
        assets.event_groups["train"], seed=args.seed
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "v11_identifiability_preflight_pass_training_locked",
        "passed": True,
        "gpu_used": False,
        "training_started": False,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "output_root": str(args.output_dir.resolve()),
        "event_count": len(assets.event_meta),
        "group_count": len(assets.group_rows),
        "train_events": sorted(assets.event_groups["train"]),
        "dev_events": sorted(assets.event_groups["dev"]),
        "first_two_training_groups": [sampler.next(), sampler.next()],
        "fresh_dataset_audit": assets.dataset_audit,
        "checkpoint_contracts": _checkpoint_contracts(args),
        "architecture": dict(protocol["architecture"]),
        "launch_locks": dict(protocol["launch_locks"]),
        "claim_boundary": (
            "CPU/data/model-lineage preflight only. No ORION forward, "
            "training, learned semantics, planning or safety result."
        ),
    }


def _validate_launch(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    preflight = _read_json(args.trainer_preflight.resolve())
    amendment = _read_json(args.launch_amendment.resolve())
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("validated_inputs") != _validated_inputs(args)
        or preflight.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or preflight.get("output_root") != str(args.output_dir.resolve())
    ):
        raise ValueError("v11 preflight is absent or stale")
    authorized = amendment.get("authorized_run", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("status")
        != "immutable_v11_identifiability_smoke_authorization"
        or amendment.get("validated_inputs") != _validated_inputs(args)
        or amendment.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or amendment.get("preflight_sha256")
        != _sha256(args.trainer_preflight.resolve())
        or authorized.get("output_root") != str(args.output_dir.resolve())
        or authorized.get("maximum_submissions") != 1
        or authorized.get("automatic_retry") is not False
        or authorized.get("optimizer_steps")
        != int(protocol["training"]["optimizer_steps"])
        or amendment.get("launch_locks", {}).get(
            "stage2l_v11_bounded_smoke_allowed"
        )
        is not True
        or amendment.get("launch_locks", {}).get("formal_stage2l_allowed")
        is not False
    ):
        raise ValueError("v11 launch amendment is absent or stale")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-report", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--log-interval", type=int, default=5)
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
        args.v101_checkpoint,
        args.v101_report,
        args.parent_contract,
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v11 identifiability prerequisite is missing")
    if args.answer_batch_size < 1:
        raise ValueError("answer batch size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty v11 output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)
    import scripts.train_stage2l_mr1_smoke as base
    import scripts.train_stage2l_v101_view_aligned_phase_a as v101

    v101._configure_base()
    assets = V11Assets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )
    if args.preflight_only:
        if args.trainer_preflight is not None or args.launch_amendment is not None:
            raise ValueError("v11 preflight cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("v11 preflight requires a fresh output path")
        value = _preflight(args=args, protocol=protocol, assets=assets)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": value["status"],
            "event_count": value["event_count"],
            "group_count": value["group_count"],
            "output": str(args.preflight_output.resolve()),
        }, sort_keys=True))
        return 0
    if (
        args.preflight_output is not None
        or args.trainer_preflight is None
        or args.launch_amendment is None
    ):
        raise ValueError("real v11 run requires preflight and launch amendment")
    _validate_launch(args, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("real v11 identifiability smoke requires CUDA")

    from mmcv.utils import set_random_seed
    import scripts.train_stage2l_v10_staged_smoke as v10

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, text_tokenizer = base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    relevance_queries = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=4096,
        image_feature_dim=1024,
        hidden_dim=256,
        grid_hw=(10, 10),
        max_views=6,
    ).cuda()
    relevance_head = TaskRelevanceMapHead(
        model_dim=4096, hidden_dim=256
    ).cuda()
    contextual_lineage = _load_v101_contextual_relevance(
        lm=lm,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        checkpoint_path=args.v101_checkpoint,
    )
    uq_tokenizer = v10._load_frozen_u_tokenizer(
        args.u_tokenizer_checkpoint.resolve()
    ).cuda().eval()
    risk_bridge = TaskRiskLanguageBridge(
        model_dim=4096, hidden_dim=256, max_views=6
    ).cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factorization_before = {
        split: _evaluate_factorization(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            protocol=protocol,
        )
        for split in ("train", "dev")
    }
    controlled_passed = all(
        value["all_release_checks_passed"]
        for value in factorization_before.values()
    )
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "engineering_preexperiment_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "contextual_relevance_lineage": contextual_lineage,
        "factorization_before": factorization_before,
        "provenance": {
            "validated_inputs": _validated_inputs(args),
            "protocol_sha256": _sha256(args.training_protocol.resolve()),
            "preflight_sha256": _sha256(args.trainer_preflight.resolve()),
            "launch_amendment_sha256": _sha256(
                args.launch_amendment.resolve()
            ),
        },
        "locks": {
            "contextual_relevance_trained": False,
            "orion_lora_trained": False,
            "stage1_or_u_tokenizer_trained": False,
            "trajectory_or_control_loss_used": False,
            "density_uq_or_governor_used": False,
            "locked_test_read": False,
        },
    }
    if not controlled_passed:
        report.update({
            "status": "stopped_before_language_controlled_u_gate_failed",
            "optimizer_steps": 0,
            "language_before": None,
            "language_after": None,
            "history": [],
            "final_checks": {
                "controlled_u_train_passed": factorization_before["train"][
                    "all_release_checks_passed"
                ],
                "controlled_u_dev_passed": factorization_before["dev"][
                    "all_release_checks_passed"
                ],
            },
        })
    else:
        language_before = {
            split: _evaluate_language(
                split=split,
                lm=lm,
                text_tokenizer=text_tokenizer,
                uq_tokenizer=uq_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                risk_bridge=risk_bridge,
                assets=assets,
                protocol=protocol,
                answer_batch_size=args.answer_batch_size,
            )
            for split in ("train", "dev")
        }
        history = _train_language_bridge(
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            protocol=protocol,
            seed=args.seed,
            answer_batch_size=args.answer_batch_size,
            log_interval=args.log_interval,
        )
        language_after = {
            split: _evaluate_language(
                split=split,
                lm=lm,
                text_tokenizer=text_tokenizer,
                uq_tokenizer=uq_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                risk_bridge=risk_bridge,
                assets=assets,
                protocol=protocol,
                answer_batch_size=args.answer_batch_size,
            )
            for split in ("train", "dev")
        }
        checks = _language_release_checks(
            before=language_before,
            after=language_after,
            factorization=factorization_before,
            protocol=protocol,
        )
        passed = all(checks.values())
        report.update({
            "status": (
                "bounded_identifiability_gate_passed"
                if passed
                else "bounded_identifiability_gate_failed"
            ),
            "optimizer_steps": len(history),
            "language_before": language_before,
            "language_after": language_after,
            "history": history,
            "final_checks": checks,
        })
        torch.save(
            {
                "schema": SCHEMA,
                "status": report["status"],
                "optimizer_steps": len(history),
                "task_risk_language_bridge": {
                    key: value.detach().cpu()
                    for key, value in risk_bridge.state_dict().items()
                },
                "contextual_relevance_checkpoint_sha256": _sha256(
                    args.v101_checkpoint.resolve()
                ),
                "u_tokenizer_checkpoint_sha256": _sha256(
                    args.u_tokenizer_checkpoint.resolve()
                ),
                "formal_stage2l_ready": False,
                "stage2p_ready": False,
            },
            args.output_dir / "v11_bridge.pt",
        )
    report["claim_boundary"] = (
        "A passing result identifies controlled U use in the frozen-R "
        "Stage2-L engineering interface only. It does not validate learned "
        "Stage1 U, formal generalization, planning, closed loop or safety."
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "report": str((args.output_dir / "report.json").resolve()),
    }, sort_keys=True), flush=True)
    del lm, relevance_queries, relevance_head, uq_tokenizer, risk_bridge
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
