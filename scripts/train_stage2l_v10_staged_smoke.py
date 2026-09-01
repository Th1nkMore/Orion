#!/usr/bin/env python3
"""Accelerated, gate-staged Stage2-L v10 engineering smoke.

One job runs at most three bounded phases and stops at the first failed gate:

* A: VLM-owned dense task-relevance map R only.
* B: the same map objective plus matched K=U*R risk alignment.
* C: bridge-only auxiliary QA grounding with frozen U and frozen R path.

The run has no learned structured-field classifier, trajectory/control loss,
Density UQ, governor, native-glare input, locked-test data, or formal-training
unlock.  Every real run requires a hash-bound CPU preflight and immutable
launch amendment.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import random
import sys
import types
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from mmcv.utils import set_random_seed

from scripts.scenario_factory_lib import sha256_file
from uq_estimator.stage2l_calibrated_objective import (
    geometry_normalized_task_risk_ranking_terms,
    relevance_support_metrics,
)
from uq_estimator.stage2l_relevance_objective_v10 import (
    stage2l_relevance_objective_v10,
)
from uq_estimator.stage2l_semantic_runtime_v10 import (
    build_vlm_task_conditioning_v10,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


SCHEMA = "orion.stage2l_v10_staged_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v10_accelerated_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v10_staged_preflight.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
DATASET_SCHEMA = "orion.stage2l_expanded_coverage_dataset.v1"
EXPECTED_EVENT_COUNT = 17
EXPECTED_TRAIN_EVENT_COUNT = 13
EXPECTED_DEV_EVENT_COUNT = 4
EXPECTED_GROUP_COUNT = 80
EXPECTED_RECORD_COUNT = 1600
MATCHED_VARIANTS = (
    "observed",
    "zero_uq",
    "off_path_uq",
    "on_path_uq",
    "view_shuffled_uq",
)
base = None


def _load_base(*, require_real_agent: bool):
    """Load legacy dataset/model helpers without pulling CARLA into CPU preflight.

    The legacy helper chain imports the complete driving agent solely for its
    local-path resolver.  A data-only preflight never builds ORION, so a
    no-op resolver prevents an irrelevant CARLA shared-library dependency on
    the login node.  Real training always imports the real agent in a fresh
    compute-node process.
    """
    global base
    if base is not None:
        return base
    injected_stub = False
    module_name = "team_code.orion_b2d_agent"
    if not require_real_agent and module_name not in sys.modules:
        stub = types.ModuleType(module_name)

        def resolve_local_model_paths(config, project_root=None):
            del project_root
            return config

        stub.resolve_local_model_paths = resolve_local_model_paths
        sys.modules[module_name] = stub
        injected_stub = True
    try:
        base = importlib.import_module("scripts.train_stage2l_mr1_smoke")
    finally:
        if injected_stub:
            sys.modules.pop(module_name, None)
    return base


def _configure_assets() -> None:
    base.DATASET_SCHEMA = DATASET_SCHEMA
    base.EXPECTED_EVENT_COUNT = EXPECTED_EVENT_COUNT
    base.EXPECTED_TRAIN_EVENT_COUNT = EXPECTED_TRAIN_EVENT_COUNT
    base.EXPECTED_DEV_EVENT_COUNT = EXPECTED_DEV_EVENT_COUNT
    base.EXPECTED_GROUP_COUNT = EXPECTED_GROUP_COUNT
    base.EXPECTED_RECORD_COUNT = EXPECTED_RECORD_COUNT


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _load_frozen_u_tokenizer(checkpoint_path: Path) -> UQComponentTokenizer:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if (
        payload.get("schema") != "orion.stage1_u_tokenizer_pretraining_run.v1"
        or payload.get("status")
        != "bounded_task_agnostic_tokenizer_pretraining_pass"
        or payload.get("task_agnostic") is not True
        or payload.get("reconstruction_decoder_included") is not False
        or payload.get("stage2l_ready") is not False
    ):
        raise ValueError("U-tokenizer checkpoint is not the frozen task-agnostic artifact")
    tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10), max_views=6
    )
    tokenizer.load_state_dict(payload["uq_tokenizer"], strict=True)
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    return tokenizer


def _map_logits(
    *,
    lm,
    text_tokenizer,
    relevance_queries,
    relevance_head,
    assets,
    group_id: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
    target = assets.relevance[group_id].cuda(non_blocking=True)
    logits = base._relevance_logits(
        lm=lm,
        tokenizer=text_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        baseline_vision=baseline,
        relevance_target=target,
        map_row=assets.row(group_id, "observed", "task_relevance"),
        route_text=assets.route_text[group_id],
    )
    return logits, target, baseline


@torch.no_grad()
def _evaluate_map_split(
    *,
    split: str,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    assets,
    support_fraction: float,
    required_oracle_fraction: float,
) -> Dict[str, Any]:
    for module in (lm, uq_tokenizer, relevance_queries, relevance_head):
        module.eval()
    logits_all = []
    targets_all = []
    learned_gaps = []
    oracle_gaps = []
    attained = []
    map_losses = []
    per_group = {}
    for group_id in assets.groups_for_split(split):
        logits, target, _ = _map_logits(
            lm=lm,
            text_tokenizer=text_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            group_id=group_id,
        )
        map_terms = stage2l_relevance_objective_v10(
            logits,
            target,
            phase="map_pretrain",
            support_fraction_of_peak=support_fraction,
        )
        on_u = uq_tokenizer(
            assets.components[(group_id, "on_path_uq")].cuda(non_blocking=True)
        ).latest_scalar_uq
        off_u = uq_tokenizer(
            assets.components[(group_id, "off_path_uq")].cuda(non_blocking=True)
        ).latest_scalar_uq
        ranking = geometry_normalized_task_risk_ranking_terms(
            on_u,
            off_u,
            logits,
            target,
            required_oracle_fraction=required_oracle_fraction,
        )
        logits_all.append(logits)
        targets_all.append(target)
        map_losses.append(float(map_terms.map_loss.item()))
        learned = float(ranking.learned_gap.item())
        oracle = float(ranking.oracle_gap.item())
        fraction = float(ranking.attained_fraction.item())
        learned_gaps.append(learned)
        oracle_gaps.append(oracle)
        attained.append(fraction)
        per_group[group_id] = {
            "event_id": assets.group_event[group_id],
            "map_loss": float(map_terms.map_loss.item()),
            "learned_gap": learned,
            "oracle_gap": oracle,
            "attained_fraction": fraction,
            "positive_order": learned > 0.0,
        }
    support = relevance_support_metrics(
        torch.cat(logits_all, dim=0),
        torch.cat(targets_all, dim=0),
        support_fraction_of_peak=support_fraction,
    )
    return {
        "split": split,
        "group_count": len(logits_all),
        "mean_map_loss": float(np.mean(map_losses)),
        "relevance_support": support,
        "ranking": {
            "minimum_attained_fraction": float(min(attained)),
            "mean_attained_fraction": float(np.mean(attained)),
            "positive_order_fraction": float(
                np.mean([value > 0.0 for value in learned_gaps])
            ),
            "mean_learned_gap": float(np.mean(learned_gaps)),
            "mean_oracle_gap": float(np.mean(oracle_gaps)),
        },
        "per_group": per_group,
    }


def _phase_a_gate(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> Dict[str, bool]:
    train = metrics["train"]["relevance_support"]
    dev = metrics["dev"]["relevance_support"]
    return {
        "train_foreground_recall": train["foreground_recall"]
        >= float(gates["train_min_foreground_recall"]),
        "train_background_fpr": train["background_false_positive_rate"]
        <= float(gates["train_max_background_fpr"]),
        "dev_foreground_recall": dev["foreground_recall"]
        >= float(gates["dev_min_foreground_recall"]),
        "dev_background_fpr": dev["background_false_positive_rate"]
        <= float(gates["dev_max_background_fpr"]),
        "dev_probability_gap_positive": dev[
            "foreground_background_probability_gap"
        ]
        > 0.0,
    }


def _phase_b_gate(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> Dict[str, bool]:
    phase_a = _phase_a_gate(metrics, gates)
    dev = metrics["dev"]["ranking"]
    return {
        "phase_a_map_gates_retained": all(phase_a.values()),
        "dev_positive_order_fraction": dev["positive_order_fraction"]
        >= float(gates["dev_min_positive_order_fraction"]),
        "dev_mean_attained_fraction": dev["mean_attained_fraction"]
        >= float(gates["dev_min_mean_attained_fraction"]),
    }


def _driving_pair(assets, group_id: str):
    zero = assets.row(group_id, "zero_uq", "driving_implication")
    on = assets.row(group_id, "on_path_uq", "driving_implication")
    return zero, on


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
    assets,
    answer_batch_size: int,
) -> Dict[str, Any]:
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
    target_nlls = []
    preferences = []
    zero_preferences = []
    on_preferences = []
    per_group = {}
    for group_id in diagnostic_groups:
        logits, _, baseline = _map_logits(
            lm=lm,
            text_tokenizer=text_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            group_id=group_id,
        )
        zero_row, on_row = _driving_pair(assets, group_id)
        zero_answer = str(zero_row["conversation"][1]["value"])
        on_answer = str(on_row["conversation"][1]["value"])
        group_result = {}
        for variant, row, target_answer, alternative_answer in (
            ("zero_uq", zero_row, zero_answer, on_answer),
            ("on_path_uq", on_row, on_answer, zero_answer),
        ):
            conditioning = build_vlm_task_conditioning_v10(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=logits,
            )
            nlls = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=text_tokenizer,
                vision=conditioning.vision_tokens,
                row=row,
                route_text=assets.route_text[group_id],
                answers=(target_answer, alternative_answer),
                micro_batch_size=answer_batch_size,
            )
            target_nll = float(nlls[0].item())
            alternative_nll = float(nlls[1].item())
            preferred = target_nll < alternative_nll
            target_nlls.append(target_nll)
            preferences.append(preferred)
            (zero_preferences if variant == "zero_uq" else on_preferences).append(
                preferred
            )
            group_result[variant] = {
                "target_nll": target_nll,
                "alternative_nll": alternative_nll,
                "target_preferred": preferred,
                "deterministic_fields": dict(
                    conditioning.deterministic_semantics.structured_fields[0]
                ),
            }
        per_group[group_id] = group_result
    return {
        "split": split,
        "diagnostic_group_count": len(diagnostic_groups),
        "mean_target_nll": float(np.mean(target_nlls)),
        "target_preference_fraction": float(np.mean(preferences)),
        "zero_uq_target_preference_fraction": float(np.mean(zero_preferences)),
        "on_path_target_preference_fraction": float(np.mean(on_preferences)),
        "per_group": per_group,
    }


def _phase_c_gate(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    map_before: Mapping[str, Any],
    map_after: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> Dict[str, bool]:
    tolerance = float(gates["map_metric_retention_tolerance"])
    before_support = map_before["dev"]["relevance_support"]
    after_support = map_after["dev"]["relevance_support"]
    retained = all(
        abs(float(before_support[key]) - float(after_support[key])) <= tolerance
        for key in before_support
    )
    return {
        "train_target_nll_improved": after["train"]["mean_target_nll"]
        < before["train"]["mean_target_nll"],
        "dev_target_nll_improved": after["dev"]["mean_target_nll"]
        < before["dev"]["mean_target_nll"],
        "dev_target_preference_fraction": after["dev"][
            "target_preference_fraction"
        ]
        >= float(gates["dev_min_target_preference_fraction"]),
        "dev_zero_uq_target_preference": after["dev"][
            "zero_uq_target_preference_fraction"
        ]
        >= float(gates["dev_min_zero_uq_target_preference_fraction"]),
        "dev_on_path_target_preference": after["dev"][
            "on_path_target_preference_fraction"
        ]
        >= float(gates["dev_min_on_path_target_preference_fraction"]),
        "frozen_map_metrics_retained": retained,
    }


def _set_trainable(module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def _clear_gradients(modules: Iterable[Any]) -> None:
    """Discard gradients left by an earlier phase before changing ownership."""
    for module in modules:
        for parameter in module.parameters():
            parameter.grad = None


def _train_relevance_phase(
    *,
    phase: str,
    steps: int,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    assets,
    protocol,
    seed: int,
    log_interval: int,
) -> List[Dict[str, Any]]:
    if phase not in ("map_pretrain", "risk_alignment"):
        raise ValueError("unsupported relevance phase")
    _set_trainable(uq_tokenizer, False)
    lora_parameters = [value for value in lm.parameters() if value.requires_grad]
    _set_trainable(relevance_queries, True)
    _set_trainable(relevance_head, True)
    trainable = lora_parameters + list(relevance_queries.parameters()) + list(
        relevance_head.parameters()
    )
    if not trainable:
        raise RuntimeError("relevance phase exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": float(protocol["learning_rates"]["lora"])},
            {
                "params": list(relevance_queries.parameters())
                + list(relevance_head.parameters()),
                "lr": float(protocol["learning_rates"]["relevance"]),
            },
        ],
        weight_decay=float(protocol["learning_rates"]["weight_decay"]),
    )
    sampler = base.EventBalancedSampler(assets.event_groups["train"], seed=seed)
    history = []
    lm.train()
    relevance_queries.train()
    relevance_head.train()
    uq_tokenizer.eval()
    for step in range(1, steps + 1):
        groups = sampler.next()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "map_loss": 0.0,
            "ranking_loss": 0.0,
            "minimum_attained_fraction": None,
        }
        fractions = []
        for group_id in groups:
            logits, target, _ = _map_logits(
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                group_id=group_id,
            )
            kwargs = {}
            if phase == "risk_alignment":
                kwargs = {
                    "on_path_uq": uq_tokenizer(
                        assets.components[(group_id, "on_path_uq")].cuda(
                            non_blocking=True
                        )
                    ).latest_scalar_uq,
                    "off_path_uq": uq_tokenizer(
                        assets.components[(group_id, "off_path_uq")].cuda(
                            non_blocking=True
                        )
                    ).latest_scalar_uq,
                }
            terms = stage2l_relevance_objective_v10(
                logits,
                target,
                phase=phase,
                support_fraction_of_peak=float(
                    protocol["objective"]["support_fraction_of_peak"]
                ),
                required_oracle_fraction=float(
                    protocol["objective"]["required_oracle_fraction"]
                ),
                ranking_weight=float(protocol["objective"]["ranking_weight"]),
                **kwargs
            )
            objective = terms.loss / float(len(groups))
            objective.backward()
            totals["loss"] += float(objective.item())
            totals["map_loss"] += float(terms.map_loss.item()) / float(len(groups))
            if terms.ranking is not None:
                totals["ranking_loss"] += float(terms.ranking.loss.item()) / float(
                    len(groups)
                )
                fractions.extend(
                    float(value)
                    for value in terms.ranking.attained_fraction.detach().cpu()
                )
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        finite = (
            np.isfinite(totals["loss"])
            and bool(torch.isfinite(norm))
            and _all_finite(
                value.grad for value in trainable if value.grad is not None
            )
        )
        if step <= 2 and not finite:
            raise RuntimeError("v10 relevance first-two-step finite fail-fast")
        if any(value.grad is not None for value in uq_tokenizer.parameters()):
            raise RuntimeError("Stage2-L relevance phase modified frozen U tokenizer")
        optimizer.step()
        item = {
            "phase": phase,
            "optimizer_step": step,
            "primary_group_ids": list(groups),
            "primary_event_ids": [assets.group_event[value] for value in groups],
            "primary_group_count": len(groups),
            "gradient_norm_before_clip": float(norm.item()),
            "finite": bool(finite),
            **totals,
        }
        if fractions:
            item["minimum_attained_fraction"] = float(min(fractions))
        history.append(item)
        if step == 1 or step % log_interval == 0:
            print("[Stage2LV10] " + json.dumps(item, sort_keys=True), flush=True)
    del optimizer
    return history


def _select_language_anchors(
    assets,
    group_id: str,
    *,
    start: int,
    count: int,
) -> Tuple[Mapping[str, Any], ...]:
    allowed_families = {"task_relevance", "driving_implication"}
    anchors = [
        row
        for row in assets.language_anchors(group_id)
        if str(row["question_family"]) in allowed_families
    ]
    if not anchors:
        raise RuntimeError("v10 Phase C group has no task-risk language anchors")
    return tuple(anchors[(start + offset) % len(anchors)] for offset in range(count))


def _train_language_phase(
    *,
    steps: int,
    lm,
    text_tokenizer,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    assets,
    protocol,
    seed: int,
    answer_batch_size: int,
    log_interval: int,
) -> List[Dict[str, Any]]:
    frozen_modules = (lm, uq_tokenizer, relevance_queries, relevance_head)
    for module in frozen_modules:
        _set_trainable(module, False)
    # AdamW leaves the preceding Phase-B gradients attached after optimizer.step().
    # Clear them before the Phase-C escape check so it detects new leakage only.
    _clear_gradients(frozen_modules)
    _set_trainable(risk_bridge, True)
    optimizer = torch.optim.AdamW(
        risk_bridge.parameters(),
        lr=float(protocol["learning_rates"]["language_bridge"]),
        weight_decay=float(protocol["learning_rates"]["weight_decay"]),
    )
    train_groups = list(assets.groups_for_split("train"))
    rng = random.Random(seed)
    rng.shuffle(train_groups)
    positions = {group_id: 0 for group_id in train_groups}
    history = []
    lm.train()
    uq_tokenizer.eval()
    relevance_queries.eval()
    relevance_head.eval()
    risk_bridge.train()
    anchor_count = int(protocol["phases"]["C_language_grounding"]["anchors_per_step"])
    for step in range(1, steps + 1):
        group_id = train_groups[(step - 1) % len(train_groups)]
        if step > 1 and (step - 1) % len(train_groups) == 0:
            rng.shuffle(train_groups)
        anchors = _select_language_anchors(
            assets,
            group_id,
            start=positions[group_id],
            count=anchor_count,
        )
        positions[group_id] += anchor_count
        with torch.no_grad():
            logits, _, baseline = _map_logits(
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                group_id=group_id,
            )
        optimizer.zero_grad(set_to_none=True)
        nlls = []
        for row in anchors:
            variant = str(row["counterfactual"]["variant"])
            conditioning = build_vlm_task_conditioning_v10(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=logits,
            )
            nll = base._answer_nlls_mr1(
                lm=lm,
                tokenizer=text_tokenizer,
                vision=conditioning.vision_tokens,
                row=row,
                route_text=assets.route_text[group_id],
                answers=(str(row["conversation"][1]["value"]),),
                micro_batch_size=answer_batch_size,
            )[0]
            (nll / float(len(anchors))).backward()
            nlls.append(float(nll.item()))
        forbidden = list(lm.parameters()) + list(uq_tokenizer.parameters()) + list(
            relevance_queries.parameters()
        ) + list(relevance_head.parameters())
        if any(value.grad is not None for value in forbidden):
            raise RuntimeError("Phase C gradient escaped the language bridge")
        norm = torch.nn.utils.clip_grad_norm_(risk_bridge.parameters(), 1.0)
        finite = bool(torch.isfinite(norm)) and _all_finite(
            value.grad
            for value in risk_bridge.parameters()
            if value.grad is not None
        )
        if step <= 2 and not finite:
            raise RuntimeError("v10 language first-two-step finite fail-fast")
        optimizer.step()
        item = {
            "phase": "language_grounding",
            "optimizer_step": step,
            "group_id": group_id,
            "event_id": assets.group_event[group_id],
            "anchor_count": len(anchors),
            "mean_target_nll": float(np.mean(nlls)),
            "gradient_norm_before_clip": float(norm.item()),
            "finite": bool(finite),
            "trainable_scope": "task_risk_language_bridge_only",
        }
        history.append(item)
        if step == 1 or step % log_interval == 0:
            print("[Stage2LV10] " + json.dumps(item, sort_keys=True), flush=True)
    del optimizer
    return history


def _checkpoint_payload(
    *,
    status: str,
    completed_phases: Sequence[str],
    lm,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    u_tokenizer_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "engineering_preexperiment_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "completed_phases": list(completed_phases),
        "frozen_u_tokenizer_sha256": u_tokenizer_sha256,
        "uq_tokenizer_frozen": True,
        "uq_tokenizer": {
            key: value.detach().cpu() for key, value in uq_tokenizer.state_dict().items()
        },
        "relevance_queries": {
            key: value.detach().cpu()
            for key, value in relevance_queries.state_dict().items()
        },
        "relevance_head": {
            key: value.detach().cpu() for key, value in relevance_head.state_dict().items()
        },
        "risk_bridge": {
            key: value.detach().cpu() for key, value in risk_bridge.state_dict().items()
        },
        "lora": {
            key: value.detach().cpu()
            for key, value in lm.state_dict().items()
            if "lora_" in key
        },
        "learned_structured_field_head_used": False,
        "trajectory_or_control_loss_used": False,
    }


def _protocol_checks(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported v10 accelerated protocol")
    architecture = protocol.get("architecture", {})
    required_false = (
        "stage1_adapter_trainable",
        "uq_tokenizer_trainable",
        "learned_structured_field_head_used",
        "trajectory_training_enabled",
        "direct_control_training_enabled",
        "density_uq_used",
        "governor_used",
    )
    if any(architecture.get(key) is not False for key in required_false):
        raise ValueError("v10 accelerated responsibility contract differs")
    if architecture.get("task_relevance_owner") != "ORION/VLM":
        raise ValueError("v10 protocol moves task relevance outside ORION/VLM")
    if protocol.get("launch_locks", {}).get("real_training_allowed") is not False:
        raise ValueError("design protocol must remain training locked")


def _preflight(
    *,
    args,
    protocol,
    assets,
    uq_tokenizer,
) -> Dict[str, Any]:
    sampler = base.EventBalancedSampler(assets.event_groups["train"], seed=args.seed)
    first_two = [sampler.next(), sampler.next()]
    state = uq_tokenizer.state_dict()
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "stage2l_v10_staged_preflight_pass_training_locked",
        "passed": True,
        "training_started": False,
        "gpu_used": False,
        "dataset_manifest": {
            "path": str(args.dataset_manifest.resolve()),
            "sha256": sha256_file(args.dataset_manifest.resolve()),
        },
        "records_sha256": sha256_file(assets.records_path),
        "frozen_u_tokenizer": {
            "path": str(args.u_tokenizer_checkpoint.resolve()),
            "sha256": sha256_file(args.u_tokenizer_checkpoint.resolve()),
            "parameter_tensor_count": len(state),
            "all_parameters_frozen": all(
                not value.requires_grad for value in uq_tokenizer.parameters()
            ),
        },
        "train_events": sorted(assets.event_groups["train"]),
        "dev_events": sorted(assets.event_groups["dev"]),
        "first_two_event_balanced_units": first_two,
        "phase_steps": {
            key: int(value["optimizer_steps"])
            for key, value in protocol["phases"].items()
        },
        "trainer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "protocol": {
            "path": str(args.training_protocol.resolve()),
            "sha256": sha256_file(args.training_protocol.resolve()),
        },
        "formal_stage2l_allowed": False,
        "stage2p_allowed": False,
    }


def _validate_launch(args, preflight: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    amendment = _read_json(args.launch_amendment.resolve())
    expected = {
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.training_protocol.resolve()),
        "preflight_sha256": sha256_file(args.trainer_preflight.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
        "u_tokenizer_checkpoint_sha256": sha256_file(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "orion_config_sha256": sha256_file(args.config.resolve()),
        "orion_checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
    }
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("trainer", {}).get("sha256")
        != expected["trainer_sha256"]
        or preflight.get("protocol", {}).get("sha256")
        != expected["protocol_sha256"]
    ):
        raise ValueError("v10 staged preflight is stale")
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("status") != "immutable_single_run_authorization"
        or amendment.get("validated_inputs") != expected
        or amendment.get("authorized_run", {}).get("output_root")
        != str(args.output_dir.resolve())
        or amendment.get("authorized_run", {}).get("maximum_submissions") != 1
        or amendment.get("authorized_run", {}).get("automatic_retry") is not False
        or amendment.get("launch_locks", {}).get("stage2l_v10_bounded_smoke_allowed")
        is not True
        or amendment.get("launch_locks", {}).get("formal_stage2l_allowed")
        is not False
    ):
        raise ValueError("v10 staged launch amendment differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not all(
        path.is_file()
        for path in (
            args.config,
            args.checkpoint,
            args.dataset_manifest,
            args.u_tokenizer_checkpoint,
            args.training_protocol,
        )
    ):
        raise FileNotFoundError("v10 staged prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite v10 staged output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(protocol)
    if sha256_file(args.dataset_manifest.resolve()) != protocol["dataset"]["sha256"]:
        raise ValueError("v10 dataset hash differs from protocol")
    if (
        sha256_file(args.u_tokenizer_checkpoint.resolve())
        != protocol["frozen_u_tokenizer"]["sha256"]
    ):
        raise ValueError("v10 frozen U-tokenizer hash differs from protocol")
    _load_base(require_real_agent=not args.preflight_only)
    _configure_assets()
    assets = base.MultiRouteAssets(args.dataset_manifest.resolve())
    uq_tokenizer = _load_frozen_u_tokenizer(args.u_tokenizer_checkpoint.resolve())

    if args.preflight_only:
        if args.trainer_preflight is not None or args.launch_amendment is not None:
            raise ValueError("preflight cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("preflight needs a fresh output path")
        result = _preflight(
            args=args, protocol=protocol, assets=assets, uq_tokenizer=uq_tokenizer
        )
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.preflight_output is not None:
        raise ValueError("real run cannot receive preflight output")
    if args.trainer_preflight is None or args.launch_amendment is None:
        raise ValueError("real v10 staged run needs preflight and amendment")
    preflight = _read_json(args.trainer_preflight.resolve())
    _validate_launch(args, preflight, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("real v10 staged smoke requires CUDA")

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, text_tokenizer = base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    uq_tokenizer.cuda().eval()
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10), max_views=6
    ).cuda()
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    risk_bridge = TaskRiskLanguageBridge(
        model_dim=4096, hidden_dim=256, max_views=6
    ).cuda()
    support_fraction = float(protocol["objective"]["support_fraction_of_peak"])
    oracle_fraction = float(protocol["objective"]["required_oracle_fraction"])
    phase_gates = protocol["release_gates"]
    before = {
        split: _evaluate_map_split(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            support_fraction=support_fraction,
            required_oracle_fraction=oracle_fraction,
        )
        for split in ("train", "dev")
    }
    phases: Dict[str, Any] = {}
    completed = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_a_history = _train_relevance_phase(
        phase="map_pretrain",
        steps=int(protocol["phases"]["A_map_pretrain"]["optimizer_steps"]),
        lm=lm,
        text_tokenizer=text_tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        assets=assets,
        protocol=protocol,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    after_a = {
        split: _evaluate_map_split(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            support_fraction=support_fraction,
            required_oracle_fraction=oracle_fraction,
        )
        for split in ("train", "dev")
    }
    a_checks = _phase_a_gate(after_a, phase_gates)
    a_passed = all(a_checks.values())
    phases["A_map_pretrain"] = {
        "status": "pass" if a_passed else "failed_gate",
        "checks": a_checks,
        "history": phase_a_history,
        "metrics": after_a,
    }
    if a_passed:
        completed.append("A_map_pretrain")
    torch.save(
        _checkpoint_payload(
            status="phase_a_pass" if a_passed else "phase_a_failed_gate",
            completed_phases=completed,
            lm=lm,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            u_tokenizer_sha256=sha256_file(args.u_tokenizer_checkpoint.resolve()),
        ),
        args.output_dir / "phase_a.pt",
    )

    final_status = "stopped_after_phase_a_failed_gate"
    final_metrics = after_a
    if a_passed:
        phase_b_history = _train_relevance_phase(
            phase="risk_alignment",
            steps=int(protocol["phases"]["B_risk_alignment"]["optimizer_steps"]),
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            protocol=protocol,
            seed=args.seed + 1000,
            log_interval=args.log_interval,
        )
        after_b = {
            split: _evaluate_map_split(
                split=split,
                lm=lm,
                text_tokenizer=text_tokenizer,
                uq_tokenizer=uq_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                support_fraction=support_fraction,
                required_oracle_fraction=oracle_fraction,
            )
            for split in ("train", "dev")
        }
        b_checks = _phase_b_gate(after_b, phase_gates)
        b_passed = all(b_checks.values())
        phases["B_risk_alignment"] = {
            "status": "pass" if b_passed else "failed_gate",
            "checks": b_checks,
            "history": phase_b_history,
            "metrics": after_b,
        }
        if b_passed:
            completed.append("B_risk_alignment")
        torch.save(
            _checkpoint_payload(
                status="phase_b_pass" if b_passed else "phase_b_failed_gate",
                completed_phases=completed,
                lm=lm,
                uq_tokenizer=uq_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                risk_bridge=risk_bridge,
                u_tokenizer_sha256=sha256_file(
                    args.u_tokenizer_checkpoint.resolve()
                ),
            ),
            args.output_dir / "phase_b.pt",
        )
        final_status = "stopped_after_phase_b_failed_gate"
        final_metrics = after_b
        if b_passed:
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
                    answer_batch_size=args.answer_batch_size,
                )
                for split in ("train", "dev")
            }
            phase_c_history = _train_language_phase(
                steps=int(
                    protocol["phases"]["C_language_grounding"]["optimizer_steps"]
                ),
                lm=lm,
                text_tokenizer=text_tokenizer,
                uq_tokenizer=uq_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                risk_bridge=risk_bridge,
                assets=assets,
                protocol=protocol,
                seed=args.seed + 2000,
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
                    answer_batch_size=args.answer_batch_size,
                )
                for split in ("train", "dev")
            }
            after_c_map = {
                split: _evaluate_map_split(
                    split=split,
                    lm=lm,
                    text_tokenizer=text_tokenizer,
                    uq_tokenizer=uq_tokenizer,
                    relevance_queries=relevance_queries,
                    relevance_head=relevance_head,
                    assets=assets,
                    support_fraction=support_fraction,
                    required_oracle_fraction=oracle_fraction,
                )
                for split in ("train", "dev")
            }
            c_checks = _phase_c_gate(
                before=language_before,
                after=language_after,
                map_before=after_b,
                map_after=after_c_map,
                gates=phase_gates,
            )
            c_passed = all(c_checks.values())
            phases["C_language_grounding"] = {
                "status": "pass" if c_passed else "failed_gate",
                "trainable_scope": "task_risk_language_bridge_only",
                "checks": c_checks,
                "history": phase_c_history,
                "before": language_before,
                "after": language_after,
                "map_metrics_after": after_c_map,
            }
            if c_passed:
                completed.append("C_language_grounding")
            torch.save(
                _checkpoint_payload(
                    status="phase_c_pass" if c_passed else "phase_c_failed_gate",
                    completed_phases=completed,
                    lm=lm,
                    uq_tokenizer=uq_tokenizer,
                    relevance_queries=relevance_queries,
                    relevance_head=relevance_head,
                    risk_bridge=risk_bridge,
                    u_tokenizer_sha256=sha256_file(
                        args.u_tokenizer_checkpoint.resolve()
                    ),
                ),
                args.output_dir / "phase_c.pt",
            )
            final_status = (
                "all_bounded_v10_phases_pass"
                if c_passed
                else "stopped_after_phase_c_failed_gate"
            )
            final_metrics = after_c_map

    report = {
        "schema": SCHEMA,
        "status": final_status,
        "engineering_preexperiment_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "completed_phases": completed,
        "before": before,
        "final_map_metrics": final_metrics,
        "phases": phases,
        "provenance": {
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
            },
            "preflight": {
                "path": str(args.trainer_preflight.resolve()),
                "sha256": sha256_file(args.trainer_preflight.resolve()),
            },
            "launch_amendment": {
                "path": str(args.launch_amendment.resolve()),
                "sha256": sha256_file(args.launch_amendment.resolve()),
            },
            "dataset_manifest": {
                "path": str(args.dataset_manifest.resolve()),
                "sha256": sha256_file(args.dataset_manifest.resolve()),
            },
            "frozen_u_tokenizer": {
                "path": str(args.u_tokenizer_checkpoint.resolve()),
                "sha256": sha256_file(args.u_tokenizer_checkpoint.resolve()),
            },
        },
        "locks": {
            "learned_structured_field_head_used": False,
            "trajectory_or_control_loss_used": False,
            "density_uq_or_governor_used": False,
            "native_glare_used_for_training": False,
            "locked_test_read": False,
        },
        "claim_boundary": "Bounded 17-event engineering learnability smoke only; no formal Stage2-L, planning, closed-loop, generalization or safety claim.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": final_status, "completed_phases": completed},
            sort_keys=True,
        ),
        flush=True,
    )
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
