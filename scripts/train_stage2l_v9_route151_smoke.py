#!/usr/bin/env python3
"""Locked bounded Route151 smoke for VLM-owned Stage2-L task fields.

This file does not authorize a GPU run.  Real training is fail-closed behind
a separate immutable amendment.  Stage1 U remains frozen and task agnostic;
the VLM owns R, fixed K=U*sigmoid(R), categorical task fields, and stance.
Trajectory, direct control, Density UQ, and a hard governor are absent.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.utils import set_random_seed

from scripts.scenario_factory_lib import sha256_file
from scripts.train_stage2l_v7_route151_smoke import (
    BRIDGE_TOKENS,
    CACHE_SCHEMA,
    ORION_VISUAL_TOKENS,
    SEMANTIC_TOKENS,
    SPATIAL_UQ_TOKENS,
    _candidate_answer_nlls_v7,
    _generate,
    _load_json,
    _load_orion_lm,
    _relevance_logits,
    _route_text,
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
from uq_estimator.stage2l_semantic_runtime_v4 import (
    build_vlm_task_field_conditioning,
)
from uq_estimator.stage2l_structured_field_head import (
    TASK_FIELD_VOCABULARIES,
    VLMTaskSemanticFieldHead,
    dataset_frequency_balanced_partial_field_loss,
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


SCHEMA = "orion.stage2l_v9_vlm_task_field_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_vlm_task_field_training_protocol.v1"
QA_CONFIG_SCHEMA = "orion.uq_relevance_qa_factory_config.v5"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v9_route151_preflight.v1"
EXPECTED_EVENT_ID = "route151_step218"
TASK_RELEVANCE_FIELDS = (
    "relevance_level",
    "risk_level",
    "risk_view",
    "risk_region",
)


def _load_records(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = partition_complete_matched_groups(rows)
    audit = audit_records(rows)
    if len(groups) != 5 or not audit["passed"] or len(rows) != 100:
        raise ValueError("v9 smoke requires five audited matched groups")
    if {str(row.get("event_id", "")) for row in rows} != {EXPECTED_EVENT_ID}:
        raise ValueError("v9 smoke is restricted to Route151")
    if {str(row.get("split", "")) for row in rows} != {"train"}:
        raise ValueError("v9 smoke may consume only the frozen train split")
    return rows, audit


class Route151V9Assets:
    """Hash-resolved v5 records, frozen U maps, R targets, and visual cache."""

    def __init__(self, records_path: Path, visual_cache_path: Path) -> None:
        self.records_path = records_path.resolve()
        self.visual_cache_path = visual_cache_path.resolve()
        self.records, self.qa_audit = _load_records(self.records_path)
        self.groups = partition_complete_matched_groups(self.records)
        self.rows: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
        self.group_rows: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
        for group in self.groups:
            group_id = str(group[0]["counterfactual"]["group_id"])
            self.group_rows[group_id] = group
            for row in group:
                key = (
                    group_id,
                    str(row["counterfactual"]["variant"]),
                    str(row["question_family"]),
                )
                if key in self.rows:
                    raise ValueError("duplicate v9 group/variant/family")
                self.rows[key] = row

        cache = torch.load(self.visual_cache_path, map_location="cpu")
        if cache.get("schema") != CACHE_SCHEMA:
            raise ValueError("unsupported multiframe visual cache")
        contexts = cache.get("contexts", {})
        if set(contexts) != set(self.group_rows):
            raise ValueError("v9 records and visual cache groups differ")
        self.visual_contexts = {}
        for group_id, value in contexts.items():
            if tuple(value.shape) != (1, ORION_VISUAL_TOKENS, 4096):
                raise ValueError("ORION visual context shape mismatch")
            self.visual_contexts[str(group_id)] = value.detach().float().cpu()

        self.components: Dict[Tuple[str, str], torch.Tensor] = {}
        self.relevance: Dict[str, torch.Tensor] = {}
        self.route_text: Dict[str, str] = {}
        for group_id in self.group_rows:
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
                    components = archive[uq_ref["component_key"]].astype(
                        np.float32
                    )
                if components.shape != (4, 6, 40, 40, 3):
                    raise ValueError("unexpected Stage1 component shape")
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
                    raise ValueError("unexpected task-relevance target shape")
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
            self.relevance[group_id] = F.adaptive_avg_pool2d(
                target, (10, 10)
            )
            self.route_text[group_id] = _route_text(route_payload)

    def row(self, group_id: str, variant: str, family: str) -> Mapping[str, Any]:
        return self.rows[(str(group_id), str(variant), str(family))]

    def language_anchors(
        self, group_id: str
    ) -> Tuple[Mapping[str, Any], ...]:
        rows = tuple(
            row
            for row in self.group_rows[group_id]
            if row["loss_policy"]["language_auxiliary_target"] is True
        )
        if len(rows) != 18:
            raise RuntimeError("complete v9 group must expose 18 language anchors")
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


def _condition_variant_v9(
    *,
    uq_tokenizer,
    risk_bridge,
    field_head,
    baseline_vision,
    components,
    relevance_logits,
) -> Dict[str, Any]:
    conditioned = build_vlm_task_field_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        field_head=field_head,
        baseline_vision=baseline_vision,
        components=components,
        relevance_logits=relevance_logits,
    )
    expected = (
        ORION_VISUAL_TOKENS
        + SPATIAL_UQ_TOKENS
        + BRIDGE_TOKENS
        + SEMANTIC_TOKENS
    )
    if conditioned.vision_tokens.shape[1:] != (expected, 4096):
        raise RuntimeError("v9 conditioned vision-token span is malformed")
    return {
        "vision": conditioned.vision_tokens,
        "task_risk": conditioned.task_risk,
        "raw_observation_global_features": (
            conditioned.raw_observation_global_features
        ),
        "raw_task_risk_global_features": (
            conditioned.raw_task_risk_global_features
        ),
        "field_logits": conditioned.field_logits,
        "field_probabilities": conditioned.field_probabilities,
        "predicted_field_indices": conditioned.predicted_field_indices,
    }


def _field_loss(
    conditioned: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, str]],
    class_counts: Mapping[str, Mapping[str, int]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    logits: Dict[str, List[torch.Tensor]] = {
        field: [] for field in TASK_FIELD_VOCABULARIES
    }
    indices: Dict[str, List[int]] = {
        field: [] for field in TASK_FIELD_VOCABULARIES
    }
    for variant, fields in targets.items():
        for field, value in fields.items():
            vocabulary = TASK_FIELD_VOCABULARIES[field]
            logits[field].append(conditioned[variant]["field_logits"][field])
            indices[field].append(vocabulary.index(str(value)))
    active_logits = {
        field: torch.cat(values, dim=0)
        for field, values in logits.items()
        if values
    }
    active_targets = {
        field: torch.tensor(
            indices[field],
            dtype=torch.long,
            device=active_logits[field].device,
        )
        for field in active_logits
    }
    active_counts = {field: class_counts[field] for field in active_logits}
    return dataset_frequency_balanced_partial_field_loss(
        active_logits, active_targets, active_counts
    )


def _field_metrics(
    entries: Sequence[
        Tuple[str, str, Mapping[str, torch.Tensor], Mapping[str, str]]
    ]
) -> Dict[str, Any]:
    per_field_total = {field: 0 for field in TASK_FIELD_VOCABULARIES}
    per_field_correct = {field: 0 for field in TASK_FIELD_VOCABULARIES}
    class_total = {
        field: {value: 0 for value in vocabulary}
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
    }
    class_correct = {
        field: {value: 0 for value in vocabulary}
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
    }
    target_probabilities = []
    zero_exact = []
    for _, variant, probabilities, targets in entries:
        row_exact = True
        for field, target in targets.items():
            vocabulary = TASK_FIELD_VOCABULARIES[field]
            target_index = vocabulary.index(str(target))
            predicted_index = int(probabilities[field].argmax(dim=-1).item())
            correct = predicted_index == target_index
            per_field_total[field] += 1
            per_field_correct[field] += int(correct)
            class_total[field][target] += 1
            class_correct[field][target] += int(correct)
            target_probabilities.append(
                float(probabilities[field][0, target_index].item())
            )
            row_exact = row_exact and correct
        if variant == "zero_uq":
            zero_exact.append(row_exact)
    supported_class_recall = {
        field: {
            value: class_correct[field][value] / total
            for value, total in class_total[field].items()
            if total > 0
        }
        for field in TASK_FIELD_VOCABULARIES
    }
    total = sum(per_field_total.values())
    correct = sum(per_field_correct.values())
    return {
        "overall_accuracy": correct / total,
        "per_field_accuracy": {
            field: per_field_correct[field] / count
            for field, count in per_field_total.items()
            if count > 0
        },
        "supported_class_recall": supported_class_recall,
        "minimum_target_probability": min(target_probabilities),
        "zero_uq_complete_field_accuracy": float(np.mean(zero_exact)),
        "evaluated_field_count": total,
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
    field_head,
    assets: Route151V9Assets,
    answer_batch_size: int,
    required_oracle_fraction: float,
    support_fraction: float,
    calibration_bce_weight: float,
    background_support_weight: float,
    background_probability_margin: float,
    generate_text: bool,
) -> Dict[str, Any]:
    modules = (
        lm,
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
        field_head,
    )
    for module in modules:
        module.eval()
    relevance_logits_all = []
    relevance_targets_all = []
    on_uq_all = []
    off_uq_all = []
    relevance_losses = []
    background_support_hinges = []
    field_entries = []
    first_group_id = sorted(assets.group_rows)[0]
    first_conditioned = None
    for group_id in sorted(assets.group_rows):
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
        background_support_hinges.append(
            float(map_terms.background_support_hinge.item())
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
                relevance_logits=logits,
            )
            for variant in MATCHED_VARIANTS
        }
        relevance_logits_all.append(logits)
        relevance_targets_all.append(target)
        on_uq_all.append(
            uq_tokenizer(
                assets.components[(group_id, "on_path_uq")].cuda(
                    non_blocking=True
                )
            ).latest_scalar_uq
        )
        off_uq_all.append(
            uq_tokenizer(
                assets.components[(group_id, "off_path_uq")].cuda(
                    non_blocking=True
                )
            ).latest_scalar_uq
        )
        for variant in MATCHED_VARIANTS:
            field_entries.append((
                group_id,
                variant,
                conditioned[variant]["field_probabilities"],
                assets.field_targets(group_id, variant),
            ))
        if group_id == first_group_id:
            first_conditioned = conditioned

    all_logits = torch.cat(relevance_logits_all, dim=0)
    all_targets = torch.cat(relevance_targets_all, dim=0)
    support_metrics = relevance_support_metrics(
        all_logits,
        all_targets,
        support_fraction_of_peak=support_fraction,
    )
    ranking = geometry_normalized_task_risk_ranking_terms(
        torch.cat(on_uq_all, dim=0),
        torch.cat(off_uq_all, dim=0),
        all_logits,
        all_targets,
        required_oracle_fraction=required_oracle_fraction,
    )
    task_fields = _field_metrics(field_entries)

    target_nlls = []
    for row in assets.language_anchors(first_group_id):
        variant = str(row["counterfactual"]["variant"])
        nlls = _candidate_answer_nlls_v7(
            lm=lm,
            tokenizer=tokenizer,
            vision=first_conditioned[variant]["vision"],
            row=row,
            route_text=assets.route_text[first_group_id],
            answers=(str(row["conversation"][1]["value"]),),
            micro_batch_size=answer_batch_size,
        )
        target_nlls.append(float(nlls[0].item()))

    predicted_for_render = {}
    target_summaries = {}
    for variant in HARD_STANCE_VARIANTS:
        decoded = decode_task_field_predictions(
            first_conditioned[variant]["predicted_field_indices"]
        )[0]
        summary = assets.row(
            first_group_id, variant, "task_relevance"
        )["target"]["structured_summary"]
        target_summaries[variant] = summary
        predicted_for_render[variant] = {
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
    render_metrics = deterministic_render_metrics(
        predicted_for_render, target_summaries
    )

    generated = {}
    if generate_text:
        for variant in HARD_STANCE_VARIANTS:
            generated[variant] = {}
            for family in QUESTION_FAMILIES:
                row = assets.row(first_group_id, variant, family)
                generated[variant][family] = _generate(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=first_conditioned[variant]["vision"],
                    row=row,
                    route_text=assets.route_text[first_group_id],
                )
    result = {
        "mean_support_aligned_relevance_loss": float(
            np.mean(relevance_losses)
        ),
        "mean_background_support_hinge": float(
            np.mean(background_support_hinges)
        ),
        "relevance_support": support_metrics,
        "ranking": {
            "learned_gap": ranking.learned_gap.detach().cpu().tolist(),
            "oracle_gap": ranking.oracle_gap.detach().cpu().tolist(),
            "attained_fraction": (
                ranking.attained_fraction.detach().cpu().tolist()
            ),
            "minimum_attained_fraction": float(
                ranking.attained_fraction.min().item()
            ),
            "positive_order_fraction": float(
                ranking.learned_gap.gt(0.0).float().mean().item()
            ),
        },
        "task_fields": task_fields,
        "deterministic_render": render_metrics,
        "first_group_mean_auxiliary_language_nll": float(
            np.mean(target_nlls)
        ),
        "free_generation_diagnostic": generated,
    }
    for module in modules:
        module.train()
    return result


def _base_preflight(
    *,
    protocol: Mapping[str, Any],
    qa_config: Mapping[str, Any],
    v9_preflight: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
    project_root: Path,
    protocol_path: Path,
    qa_config_path: Path,
    dataset_audit_path: Path,
    reference_audit_path: Path,
    records_path: Path,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported v9 training protocol")
    if qa_config.get("schema") != QA_CONFIG_SCHEMA:
        raise ValueError("unsupported v9 QA config")
    locks = protocol.get("launch_locks", {})
    if (
        locks.get("real_orion_smoke_allowed") is not False
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("new_immutable_amendment_required") is not True
    ):
        raise ValueError("base v9 protocol does not preserve launch locks")
    expected_sources = protocol["implementation_sources"]
    for relative, expected in expected_sources.items():
        if sha256_file(project_root / relative) != expected:
            raise ValueError(
                "v9 implementation source hash mismatch: %s" % relative
            )
    dataset = protocol["route151_v9_dataset"]
    if (
        sha256_file(qa_config_path)
        != expected_sources[
            "configs/scenario_factory/qa_factory_v5_vlm_task_fields.json"
        ]
        or sha256_file(records_path) != dataset["records_sha256"]
        or sha256_file(dataset_audit_path) != dataset["audit_sha256"]
        or sha256_file(reference_audit_path)
        != dataset["reference_audit_sha256"]
        or dataset_audit.get("passed") is not True
        or reference_audit.get("passed") is not True
    ):
        raise ValueError("v9 dataset/QA prerequisite is stale")
    if (
        v9_preflight.get("schema") != PREFLIGHT_SCHEMA
        or v9_preflight.get("passed") is not True
        or v9_preflight.get("training_started") is not False
        or v9_preflight.get("real_orion_smoke_authorized") is not False
        or v9_preflight.get("protocol_sha256")
        != sha256_file(protocol_path)
        or any(not value for value in v9_preflight.get("checks", {}).values())
    ):
        raise ValueError("v9 architecture/data preflight is absent or stale")


def _validate_amendment(
    *,
    amendment: Mapping[str, Any],
    protocol_path: Path,
    qa_config_path: Path,
    v9_preflight_path: Path,
    dataset_audit_path: Path,
    reference_audit_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    records_path: Path,
    visual_cache_path: Path,
    output_dir: Path,
    max_optimizer_steps: int,
    answer_batch_size: int,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    expected_hashes = {
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol_sha256": sha256_file(protocol_path),
        "qa_factory_config_sha256": sha256_file(qa_config_path),
        "semantic_runtime_v4_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_semantic_runtime_v4.py"
        ),
        "structured_field_head_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_structured_field_head.py"
        ),
        "qa_contract_v5_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_qa_contract_v5.py"
        ),
        "v9_preflight_sha256": sha256_file(v9_preflight_path),
        "dataset_audit_sha256": sha256_file(dataset_audit_path),
        "reference_audit_sha256": sha256_file(reference_audit_path),
        "records_sha256": sha256_file(records_path),
        "visual_cache_sha256": sha256_file(visual_cache_path),
        "orion_config_sha256": sha256_file(config_path),
        "base_orion_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    authorized = amendment.get("authorized_run", {})
    locks = amendment.get("launch_locks", {})
    validated = amendment.get("validated_inputs", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or locks.get("stage2l_v9_route151_smoke_allowed") is not True
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or authorized.get("event_id") != EXPECTED_EVENT_ID
        or int(authorized.get("maximum_submissions", 0)) != 1
        or int(authorized.get("maximum_optimizer_steps", -1))
        != max_optimizer_steps
        or int(authorized.get("answer_micro_batch_size", -1))
        != answer_batch_size
        or authorized.get(
            "fresh_initialization_from_original_orion_checkpoint"
        )
        is not True
        or authorized.get("automatic_retry_or_extension") is not False
        or Path(str(authorized.get("output_root", ""))).resolve()
        != output_dir.resolve()
        or any(
            validated.get(name) != value
            for name, value in expected_hashes.items()
        )
    ):
        raise ValueError("Route151 v9 amendment is absent, stale, or broad")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--qa-config", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--v9-preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=20)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-head", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_optimizer_steps <= 0 or args.max_optimizer_steps > 60:
        raise ValueError("v9 bounded smoke must use 1..60 optimizer steps")
    if args.answer_batch_size != 2:
        raise ValueError("v9 answer micro-batch is frozen at 2")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("ORION config/checkpoint prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite v9 smoke output")
    protocol = _load_json(args.training_protocol.resolve())
    qa_config = _load_json(args.qa_config.resolve())
    v9_preflight = _load_json(args.v9_preflight.resolve())
    dataset_audit = _load_json(args.dataset_audit.resolve())
    reference_audit = _load_json(args.reference_audit.resolve())
    project_root = Path(__file__).resolve().parents[1]
    _base_preflight(
        protocol=protocol,
        qa_config=qa_config,
        v9_preflight=v9_preflight,
        dataset_audit=dataset_audit,
        reference_audit=reference_audit,
        project_root=project_root,
        protocol_path=args.training_protocol.resolve(),
        qa_config_path=args.qa_config.resolve(),
        dataset_audit_path=args.dataset_audit.resolve(),
        reference_audit_path=args.reference_audit.resolve(),
        records_path=args.records.resolve(),
    )
    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    assets = Route151V9Assets(args.records, args.visual_cache)
    if args.preflight_only:
        if args.launch_amendment is not None:
            raise ValueError("locked preflight must not receive an amendment")
        if args.preflight_output is None:
            raise ValueError("locked preflight requires an output path")
        if args.preflight_output.exists():
            raise FileExistsError("refusing to overwrite trainer preflight")
        result = {
            "status": "v9_route151_trainer_preflight_pass_training_locked",
            "event_id": EXPECTED_EVENT_ID,
            "qa_audit": assets.qa_audit,
            "group_ids": sorted(assets.group_rows),
            "proposed_optimizer_steps": args.max_optimizer_steps,
            "primary_groups_per_optimizer_step": len(assets.group_rows),
            "primary_records_per_optimizer_step": sum(
                len(value) for value in assets.group_rows.values()
            ),
            "language_groups_per_optimizer_step": 1,
            "group_interference_control": (
                "all five groups contribute primary gradients before every "
                "optimizer step"
            ),
            "proposal_is_authorization": False,
            "gradient_ownership": protocol["gradient_ownership"],
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "training_protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
            },
            "architecture_data_preflight": {
                "path": str(args.v9_preflight.resolve()),
                "sha256": sha256_file(args.v9_preflight.resolve()),
            },
            "training_started": False,
            "training_authorized": False,
            "gpu_used": False,
        }
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.preflight_output is not None:
        raise ValueError("real training must not receive --preflight-output")
    if args.launch_amendment is None:
        raise ValueError("real v9 smoke requires a separate launch amendment")
    amendment = _load_json(args.launch_amendment.resolve())
    _validate_amendment(
        amendment=amendment,
        protocol_path=args.training_protocol.resolve(),
        qa_config_path=args.qa_config.resolve(),
        v9_preflight_path=args.v9_preflight.resolve(),
        dataset_audit_path=args.dataset_audit.resolve(),
        reference_audit_path=args.reference_audit.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        records_path=args.records.resolve(),
        visual_cache_path=args.visual_cache.resolve(),
        output_dir=args.output_dir.resolve(),
        max_optimizer_steps=args.max_optimizer_steps,
        answer_batch_size=args.answer_batch_size,
    )
    if not torch.cuda.is_available():
        raise SystemExit("real v9 smoke requires CUDA")

    loss_config = protocol["losses_for_future_bounded_smoke"]
    lambda_language = float(
        loss_config["auxiliary_structured_qa_causal_language_modeling"][
            "weight"
        ]
    )
    lambda_map = float(
        loss_config["foreground_and_background_balanced_dense_relevance"][
            "weight"
        ]
    )
    lambda_ranking = float(
        loss_config["geometry_normalized_on_off_ranking"]["weight"]
    )
    required_oracle_fraction = float(
        loss_config["geometry_normalized_on_off_ranking"][
            "required_oracle_fraction"
        ]
    )
    lambda_fields = float(
        loss_config["dataset_frequency_balanced_partial_task_fields"][
            "weight"
        ]
    )
    if any(
        float(loss_config[name]) != 0.0
        for name in (
            "language_answer_preference",
            "trajectory",
            "direct_control",
            "observed_consequence_calibration",
        )
    ):
        raise ValueError("v9 smoke may not train preference/control losses")
    support_fraction = 0.1
    calibration_bce_weight = 0.1
    relevance_config = loss_config[
        "foreground_and_background_balanced_dense_relevance"
    ]
    background_support_weight = float(
        relevance_config["background_support_weight"]
    )
    background_probability_margin = float(
        relevance_config["background_probability_margin"]
    )
    class_counts = assets.qa_audit["task_field_class_counts"]

    lm, tokenizer = _load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    uq_tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_head = TaskRelevanceMapHead(
        model_dim=4096, hidden_dim=256
    ).cuda()
    risk_bridge = TaskRiskLanguageBridge(
        model_dim=4096, hidden_dim=256
    ).cuda()
    field_head = VLMTaskSemanticFieldHead(
        model_dim=4096, hidden_dim=256
    ).cuda()
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
        parameter
        for module in auxiliary_modules
        for parameter in module.parameters()
    ]
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
    before = _evaluate(**evaluation_args, generate_text=False)
    history = []
    group_ids = sorted(assets.group_rows)
    rng = random.Random(args.seed)
    language_group_order = list(group_ids)
    for step in range(1, args.max_optimizer_steps + 1):
        if (step - 1) % len(language_group_order) == 0:
            rng.shuffle(language_group_order)
        language_group_id = language_group_order[
            (step - 1) % len(language_group_order)
        ]
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "language_nll": 0.0,
            "support_aligned_relevance": 0.0,
            "background_support_hinge": 0.0,
            "ranking_loss": 0.0,
            "minimum_attained_fraction": float("inf"),
            "task_field_loss": 0.0,
            "per_field_loss": {
                field: 0.0 for field in TASK_FIELD_VOCABULARIES
            },
            "per_group_primary": [],
        }
        primary_denominator = float(len(group_ids))
        # Every optimizer update sees every matched group for the primary
        # R/ranking/field objectives.  This prevents the v8 failure mode where
        # a group passed its margin and was forgotten by later group updates.
        for primary_group_id in group_ids:
            baseline = assets.visual_contexts[primary_group_id].cuda(
                non_blocking=True
            )
            relevance_target = assets.relevance[primary_group_id].cuda(
                non_blocking=True
            )
            relevance_logits = _relevance_logits(
                lm=lm,
                tokenizer=tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                baseline_vision=baseline,
                relevance_target=relevance_target,
                map_row=assets.row(
                    primary_group_id, "observed", "task_relevance"
                ),
                route_text=assets.route_text[primary_group_id],
            )
            map_terms = support_aligned_relevance_terms(
                relevance_logits,
                relevance_target,
                support_fraction_of_peak=support_fraction,
                calibration_bce_weight=calibration_bce_weight,
                background_support_weight=background_support_weight,
                background_probability_margin=background_probability_margin,
            )
            on_uq = uq_tokenizer(
                assets.components[(primary_group_id, "on_path_uq")].cuda(
                    non_blocking=True
                )
            ).latest_scalar_uq
            off_uq = uq_tokenizer(
                assets.components[(primary_group_id, "off_path_uq")].cuda(
                    non_blocking=True
                )
            ).latest_scalar_uq
            ranking_terms = geometry_normalized_task_risk_ranking_terms(
                on_uq,
                off_uq,
                relevance_logits,
                relevance_target,
                required_oracle_fraction=required_oracle_fraction,
            )
            conditioned = {
                variant: _condition_variant_v9(
                    uq_tokenizer=uq_tokenizer,
                    risk_bridge=risk_bridge,
                    field_head=field_head,
                    baseline_vision=baseline,
                    components=assets.components[
                        (primary_group_id, variant)
                    ].cuda(non_blocking=True),
                    relevance_logits=relevance_logits,
                )
                for variant in MATCHED_VARIANTS
            }
            targets = {
                variant: assets.field_targets(primary_group_id, variant)
                for variant in MATCHED_VARIANTS
            }
            field_loss, per_field_loss = _field_loss(
                conditioned, targets, class_counts
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
            for field, value in per_field_loss.items():
                totals["per_field_loss"][field] += (
                    float(value.item()) / primary_denominator
                )
            totals["per_group_primary"].append({
                "group_id": primary_group_id,
                "support_aligned_relevance": float(map_terms.loss.item()),
                "background_support_hinge": float(
                    map_terms.background_support_hinge.item()
                ),
                "ranking_loss": float(ranking_terms.loss.item()),
                "minimum_attained_fraction": attained,
                "task_field_loss": float(field_loss.item()),
            })

        group_id = language_group_id
        group = assets.group_rows[group_id]
        anchors = assets.language_anchors(group_id)
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        relevance_target = assets.relevance[group_id].cuda(non_blocking=True)
        for row in anchors:
            variant = str(row["counterfactual"]["variant"])
            with torch.no_grad():
                current_logits = _relevance_logits(
                    lm=lm,
                    tokenizer=tokenizer,
                    relevance_queries=relevance_queries,
                    relevance_head=relevance_head,
                    baseline_vision=baseline,
                    relevance_target=relevance_target,
                    map_row=assets.row(
                        group_id, "observed", "task_relevance"
                    ),
                    route_text=assets.route_text[group_id],
                )
            current = _condition_variant_v9(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                field_head=field_head,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=current_logits,
            )
            nlls = _candidate_answer_nlls_v7(
                lm=lm,
                tokenizer=tokenizer,
                vision=current["vision"],
                row=row,
                route_text=assets.route_text[group_id],
                answers=(str(row["conversation"][1]["value"]),),
                micro_batch_size=args.answer_batch_size,
            )
            language_loss = nlls[0]
            language_objective = (
                lambda_language * language_loss / len(anchors)
            )
            language_objective.backward()
            totals["loss"] += float(language_objective.item())
            totals["language_nll"] += (
                float(language_loss.item()) / len(anchors)
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            lora_parameters + auxiliary_parameters, 1.0
        )
        optimizer.step()
        item = {
            "optimizer_step": step,
            "language_group_id": group_id,
            "primary_group_count": len(group_ids),
            "records_in_primary_optimizer_unit": sum(
                len(assets.group_rows[value]) for value in group_ids
            ),
            "records_in_language_optimizer_unit": len(group),
            "language_auxiliary_anchors": len(anchors),
            "optimizer_steps_inside_primary_sweep": 0,
            "gradient_norm_before_clip": float(gradient_norm.item()),
            **totals,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print(
                "[Stage2LV9Smoke] " + json.dumps(item, sort_keys=True),
                flush=True,
            )

    after = _evaluate(**evaluation_args, generate_text=True)
    gates = qa_config["release_gates_for_a_future_bounded_smoke"]
    supported_recalls = [
        value
        for per_field in after["task_fields"][
            "supported_class_recall"
        ].values()
        for value in per_field.values()
    ]
    checks = {
        "every_optimizer_step_covers_all_primary_groups": all(
            row["primary_group_count"] == 5
            and row["records_in_primary_optimizer_unit"] == 100
            and row["records_in_language_optimizer_unit"] == 20
            and row["optimizer_steps_inside_primary_sweep"] == 0
            for row in history
        ),
        "relevance_foreground_recall": (
            after["relevance_support"]["foreground_recall"]
            >= gates["dense_relevance_minimum_foreground_recall"]
        ),
        "relevance_background_fpr": (
            after["relevance_support"]["background_false_positive_rate"]
            <= gates[
                "dense_relevance_maximum_background_false_positive_rate"
            ]
        ),
        "all_groups_positive_on_off_order": (
            after["ranking"]["positive_order_fraction"] == 1.0
        ),
        "all_groups_attain_oracle_fraction": (
            after["ranking"]["minimum_attained_fraction"]
            >= gates["minimum_oracle_gap_fraction_for_every_group"]
        ),
        "all_observed_task_field_classes_recalled": all(
            value >= gates["observed_class_task_field_accuracy"]
            for value in supported_recalls
        ),
        "zero_uq_absence_semantics": (
            after["task_fields"]["zero_uq_complete_field_accuracy"]
            >= gates["zero_uq_absence_semantics_accuracy"]
        ),
        "hard_stance_accuracy": (
            after["task_fields"]["per_field_accuracy"]["stance"]
            >= gates["hard_stance_accuracy"]
        ),
        "deterministic_render_parse": (
            after["deterministic_render"]["semantic_parse_rate"]
            >= gates["deterministic_render_parse_rate"]
        ),
        "deterministic_render_fields": (
            after["deterministic_render"]["semantic_field_accuracy"]
            >= gates["deterministic_render_field_accuracy"]
        ),
        "gradient_ownership_frozen": (
            protocol["gradient_ownership"][
                "qa_language_loss_to_relevance_logits"
            ]
            is False
            and protocol["gradient_ownership"][
                "qa_language_loss_to_task_risk_bridge"
            ]
            is False
            and protocol["gradient_ownership"][
                "qa_language_loss_to_task_field_classifiers"
            ]
            is False
        ),
        "trajectory_and_control_remain_disabled": True,
    }
    diagnostics = {
        # The QA contract explicitly keeps free language outside release
        # evidence.  Report its optimization direction, but do not let an
        # auxiliary first-group NLL fluctuation override structured R/field
        # gates or determine the smoke status.
        "auxiliary_language_nll_decreases": (
            after["first_group_mean_auxiliary_language_nll"]
            < before["first_group_mean_auxiliary_language_nll"]
        ),
        "free_generation_is_release_evidence": False,
    }
    passed = all(checks.values())
    status = (
        "engineering_v9_vlm_task_field_smoke_pass"
        if passed
        else "engineering_v9_vlm_task_field_smoke_failed_gate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_v9_route151_smoke.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "status": status,
            "engineering_smoke_only": True,
            "formal_training_ready": False,
            "stage2l_pilot_training_ready": False,
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
        "engineering_smoke_only": True,
        "formal_training_ready": False,
        "stage2l_pilot_training_ready": False,
        "stage2p_ready": False,
        "event_id": EXPECTED_EVENT_ID,
        "optimizer_steps": len(history),
        "primary_record_equivalent_presentations": len(history) * 100,
        "language_record_equivalent_presentations": len(history) * 20,
        "qa_audit": assets.qa_audit,
        "before": before,
        "after": after,
        "checks": checks,
        "diagnostics": diagnostics,
        "history": history,
        "architecture": {
            "stage1_adapter_frozen_and_task_agnostic": True,
            "task_relevance_owned_by_vlm": True,
            "field_head_reads_u_and_k": True,
            "dense_relevance_owned_by_map_and_ranking_losses": True,
            "task_field_gradient_to_relevance_logits": False,
            "every_primary_update_covers_all_five_groups": True,
            "background_support_hinge_enabled": True,
            "qa_language_gradient_to_relevance_logits": False,
            "qa_language_gradient_to_task_risk_bridge": False,
            "qa_language_gradient_to_field_classifiers": False,
            "ground_truth_fields_enter_forward": False,
            "structured_task_fields_are_release_evidence": True,
            "free_language_is_release_evidence": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
            "trajectory_or_control_loss": False,
        },
        "provenance": {
            "records": {
                "path": str(args.records.resolve()),
                "sha256": sha256_file(args.records.resolve()),
            },
            "visual_cache": {
                "path": str(args.visual_cache.resolve()),
                "sha256": sha256_file(args.visual_cache.resolve()),
            },
            "qa_config": {
                "path": str(args.qa_config.resolve()),
                "sha256": sha256_file(args.qa_config.resolve()),
            },
            "training_protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
            },
            "v9_preflight": {
                "path": str(args.v9_preflight.resolve()),
                "sha256": sha256_file(args.v9_preflight.resolve()),
            },
            "dataset_audit": {
                "path": str(args.dataset_audit.resolve()),
                "sha256": sha256_file(args.dataset_audit.resolve()),
            },
            "reference_audit": {
                "path": str(args.reference_audit.resolve()),
                "sha256": sha256_file(args.reference_audit.resolve()),
            },
            "launch_amendment": {
                "path": str(args.launch_amendment.resolve()),
                "sha256": sha256_file(args.launch_amendment.resolve()),
            },
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "base_orion_checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint.resolve()),
            },
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path.resolve()),
            },
        },
        "claim_boundary": (
            "Route151 v9 engineering learnability smoke only; no held-out, "
            "trajectory, closed-loop, generalization or safety claim."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del lm
    gc.collect()
    print(
        json.dumps(
            {
                "status": status,
                "checks": checks,
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
