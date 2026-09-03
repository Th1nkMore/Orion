#!/usr/bin/env python3
"""Bounded real-ORION smoke for the Stage2-L v6 matched-group contract.

This is an implementation/learnability smoke only.  One optimizer update is
performed after accumulating every supervised loss from one complete
5-variant x 4-question-family group.  Observed and view-shuffled driving
answers remain diagnostic-only.  No trajectory, control, Density UQ, or
governor path is loaded or trained.
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
from scripts.train_stage2l_route196_bridge_smoke import (
    BRIDGE_TOKENS,
    _generate,
    _relevance_pass,
)
from scripts.train_stage2l_route196_overfit_smoke import (
    IMAGE_TOKEN_INDEX,
    ORION_VISUAL_TOKENS,
    SPATIAL_UQ_TOKENS,
    _load_orion_lm,
    _route_text,
    _training_tokens,
)
from uq_estimator.stage2l_matched_objective import (
    HARD_STANCE_VARIANTS,
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
    audit_matched_training_records,
    cross_family_answer_preference_loss,
    hard_language_supervision_allowed,
    partition_complete_matched_groups,
    same_variant_cross_family_answers,
)
from uq_estimator.stage2l_pilot import resolve_reference
from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    encode_planning_stances,
)
from uq_estimator.stage2l_semantic_bottleneck_v2 import (
    MagnitudePreservingPlanningStanceBottleneck,
)
from uq_estimator.stage2l_semantic_runtime_v2 import (
    build_magnitude_structured_conditioning,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
    fixed_task_risk,
    matched_task_risk_ranking_loss,
)


SCHEMA = "orion.stage2l_v6_matched_group_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v2"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
EXPECTED_EVENT_ID = "route151_step218"
SEMANTIC_TOKENS = 1
IGNORE_INDEX = -100


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _load_records(path: Path) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = partition_complete_matched_groups(rows)
    if len(groups) < 3 or len(groups) > 5:
        raise ValueError("v6 smoke requires three to five complete keyframe groups")
    if {str(row.get("event_id", "")) for row in rows} != {EXPECTED_EVENT_ID}:
        raise ValueError("v6 smoke is authorized only for Route151")
    if {str(row.get("split", "")) for row in rows} != {"train"}:
        raise ValueError("v6 smoke may consume only the formal train split")
    audit = audit_matched_training_records(rows)
    if (
        audit["record_count"] != 20 * len(groups)
        or audit["optimizer_step_count_per_epoch"] != len(groups)
        or audit["optimizer_steps_inside_group"] != 0
        or audit["hard_language_record_count"] != 18 * len(groups)
        or audit["hard_stance_record_count"] != 3 * len(groups)
    ):
        raise ValueError("v6 matched record audit is inconsistent")
    for row in rows:
        variant = str(row["counterfactual"]["variant"])
        family = str(row["question_family"])
        policy = row.get("loss_policy", {})
        expected_language = hard_language_supervision_allowed(variant, family)
        expected_stance = (
            family == "driving_implication" and variant in HARD_STANCE_VARIANTS
        )
        if (
            policy.get("optimizer_group_complete_before_step") is not True
            or policy.get("hard_language_target") is not expected_language
            or policy.get("cross_family_preference_anchor") is not expected_language
            or policy.get("hard_stance_target") is not expected_stance
        ):
            raise ValueError("per-record v6 loss policy mismatch")
    return rows


class Route151Assets:
    def __init__(self, records_path: Path, visual_cache_path: Path) -> None:
        self.records_path = records_path.resolve()
        self.visual_cache_path = visual_cache_path.resolve()
        self.records = _load_records(self.records_path)
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
                    raise ValueError("duplicate v6 group/variant/family")
                self.rows[key] = row

        cache = torch.load(self.visual_cache_path, map_location="cpu")
        if cache.get("schema") != CACHE_SCHEMA:
            raise ValueError("unsupported multiframe visual cache")
        contexts = cache.get("contexts", {})
        if set(contexts) != set(self.group_rows):
            raise ValueError("QA groups and visual-cache groups differ")
        self.visual_contexts = {}
        for group_id, value in contexts.items():
            if tuple(value.shape) != (1, ORION_VISUAL_TOKENS, 4096):
                raise ValueError("ORION visual context shape mismatch")
            self.visual_contexts[str(group_id)] = value.detach().float().cpu()

        self.components: Dict[Tuple[str, str], torch.Tensor] = {}
        self.relevance: Dict[str, torch.Tensor] = {}
        self.route_text: Dict[str, str] = {}
        for group_id, group in self.group_rows.items():
            relevance_arrays = []
            route_payload = None
            for variant in MATCHED_VARIANTS:
                row = self.row(group_id, variant, "task_relevance")
                uq_ref = row["model_input"]["stage1_observation_uq"]
                uq_path = resolve_reference(
                    uq_ref,
                    self.records_path.parent,
                    "Stage1 UQ for %s/%s" % (group_id, variant),
                )
                with np.load(uq_path) as archive:
                    components = archive[uq_ref["component_key"]].astype(np.float32)
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
                with np.load(sidecar_path) as archive:
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
            self.relevance[group_id] = F.adaptive_avg_pool2d(target, (10, 10))
            self.route_text[group_id] = _route_text(route_payload)

    def row(
        self, group_id: str, variant: str, family: str
    ) -> Mapping[str, Any]:
        return self.rows[(str(group_id), str(variant), str(family))]

    def supervised_anchors(
        self, group_id: str
    ) -> Tuple[Mapping[str, Any], ...]:
        rows = tuple(
            row
            for row in self.group_rows[group_id]
            if hard_language_supervision_allowed(
                str(row["counterfactual"]["variant"]),
                str(row["question_family"]),
            )
        )
        if len(rows) != 18:
            raise RuntimeError("complete v6 group must expose 18 language anchors")
        return rows


def _target_stance(row: Mapping[str, Any]) -> str:
    value = str(
        row["target"]["structured_summary"]["planning_implication"]["stance"]
    )
    if value not in PLANNING_STANCES:
        raise ValueError("unsupported planning stance")
    return value


def _condition_variant(
    *, uq_tokenizer, risk_bridge, stance_bottleneck, baseline_vision,
    components, relevance_logits,
) -> Dict[str, torch.Tensor]:
    conditioned = build_magnitude_structured_conditioning(
        uq_tokenizer=uq_tokenizer,
        risk_bridge=risk_bridge,
        stance_bottleneck=stance_bottleneck,
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
        raise RuntimeError("v6 conditioned vision-token span is malformed")
    return {
        "vision": conditioned.vision_tokens,
        "task_risk": conditioned.task_risk,
        "raw_global_features": conditioned.raw_global_features,
        "magnitude_features": conditioned.magnitude_features,
        "stance_logits": conditioned.stance_logits,
        "stance_probabilities": conditioned.stance_probabilities,
        "predicted_stance_indices": conditioned.predicted_stance_indices,
    }


def _expanded_labels(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    visual_token_count: int,
) -> torch.Tensor:
    locations = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
    if locations.numel() != 1:
        raise ValueError("each QA sequence must contain one image token")
    image_index = int(locations.item())
    ignored = torch.full(
        (int(visual_token_count),),
        IGNORE_INDEX,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat(
        (labels[:image_index], ignored, labels[image_index + 1 :]), dim=0
    )


def _candidate_answer_nlls(
    *, lm, tokenizer, vision: torch.Tensor, row: Mapping[str, Any],
    route_text: str, answers: Sequence[str], micro_batch_size: int,
) -> torch.Tensor:
    if len(answers) != len(QUESTION_FAMILIES):
        raise ValueError("candidate set must contain target plus three negatives")
    if micro_batch_size < 1 or micro_batch_size > len(answers):
        raise ValueError("answer micro-batch size is invalid")
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
        input_ids = torch.full(
            (len(chunk), max_length), pad_id, dtype=torch.long
        )
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
                raise RuntimeError("expanded supervision exceeds LM logits")
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
            raise RuntimeError("candidate answer has no supervised tokens")
        all_nlls.append((token_loss * valid).sum(dim=-1) / counts)
    return torch.cat(all_nlls, dim=0)


def _same_variant_answers(
    group: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]
) -> List[str]:
    negatives = same_variant_cross_family_answers(group, anchor)
    anchor_family = str(anchor["question_family"])
    return [
        str(anchor["conversation"][1]["value"]),
        *[
            negatives[family]
            for family in QUESTION_FAMILIES
            if family != anchor_family
        ],
    ]


@torch.no_grad()
def _evaluate(
    *, lm, tokenizer, uq_tokenizer, relevance_queries, relevance_head,
    risk_bridge, stance_bottleneck, assets: Route151Assets,
    answer_batch_size: int, generate_text: bool,
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
    relevance_bce = []
    on_minus_off = []
    target_probabilities = []
    hard_predictions = []
    raw_features = {}
    first_group_id = sorted(assets.group_rows)[0]
    first_conditioned = None
    for group_id in sorted(assets.group_rows):
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        target = assets.relevance[group_id].cuda(non_blocking=True)
        map_row = assets.row(group_id, "observed", "task_relevance")
        relevance = _relevance_pass(
            lm=lm,
            tokenizer=tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            baseline_vision=baseline,
            relevance_target=target,
            map_row=map_row,
            route_text=assets.route_text[group_id],
        )
        relevance_bce.append(float(relevance["map_loss"].item()))
        conditioned = {
            variant: _condition_variant(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                stance_bottleneck=stance_bottleneck,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)],
                relevance_logits=relevance["relevance_logits"],
            )
            for variant in MATCHED_VARIANTS
        }
        on_score = conditioned["on_path_uq"]["task_risk"].flatten(1).amax(1)
        off_score = conditioned["off_path_uq"]["task_risk"].flatten(1).amax(1)
        on_minus_off.append(float((on_score - off_score).mean().item()))
        raw_features[group_id] = {
            variant: conditioned[variant]["raw_global_features"][0]
            .detach()
            .cpu()
            .tolist()
            for variant in MATCHED_VARIANTS
        }
        for variant in HARD_STANCE_VARIANTS:
            row = assets.row(group_id, variant, "driving_implication")
            target_stance = _target_stance(row)
            target_index = PLANNING_STANCES.index(target_stance)
            probabilities = conditioned[variant]["stance_probabilities"][0]
            prediction = PLANNING_STANCES[
                int(conditioned[variant]["predicted_stance_indices"].item())
            ]
            target_probabilities.append(float(probabilities[target_index].item()))
            hard_predictions.append(prediction == target_stance)
        if group_id == first_group_id:
            first_conditioned = conditioned

    target_nlls = []
    preference_passes = []
    group = assets.group_rows[first_group_id]
    for row in assets.supervised_anchors(first_group_id):
        variant = str(row["counterfactual"]["variant"])
        nlls = _candidate_answer_nlls(
            lm=lm,
            tokenizer=tokenizer,
            vision=first_conditioned[variant]["vision"],
            row=row,
            route_text=assets.route_text[first_group_id],
            answers=_same_variant_answers(group, row),
            micro_batch_size=answer_batch_size,
        )
        target_nlls.append(float(nlls[0].item()))
        preference_passes.extend(
            bool(value >= 0.2)
            for value in (nlls[1:] - nlls[0]).detach().cpu().tolist()
        )

    generated = {}
    if generate_text:
        for variant in HARD_STANCE_VARIANTS:
            row = assets.row(first_group_id, variant, "driving_implication")
            generated[variant] = _generate(
                lm=lm,
                tokenizer=tokenizer,
                vision=first_conditioned[variant]["vision"],
                row=row,
                route_text=assets.route_text[first_group_id],
            )
    result = {
        "mean_relevance_bce": float(np.mean(relevance_bce)),
        "mean_on_minus_off_peak_task_risk": float(np.mean(on_minus_off)),
        "hard_stance_accuracy": float(np.mean(hard_predictions)),
        "minimum_hard_stance_target_probability": float(
            min(target_probabilities)
        ),
        "first_group_mean_hard_language_nll": float(np.mean(target_nlls)),
        "first_group_cross_family_margin_pass_fraction": float(
            np.mean(preference_passes)
        ),
        "raw_k_features": raw_features,
        "generated_driving_implications": generated,
    }
    for module in modules:
        module.train()
    return result


def _validate_authorization(
    *, protocol: Mapping[str, Any], amendment: Mapping[str, Any],
    protocol_path: Path, amendment_path: Path, config_path: Path,
    checkpoint_path: Path, records_path: Path, visual_cache_path: Path,
    output_dir: Path, max_optimizer_steps: int, answer_batch_size: int,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported v6 protocol schema")
    unit = protocol.get("matched_training_unit", {})
    locks = protocol.get("launch_locks", {})
    if (
        int(unit.get("optimizer_steps_inside_group", -1)) != 0
        or int(unit.get("optimizer_step_after_accumulating_complete_group", -1))
        != 1
        or locks.get("route196_retry_allowed") is not False
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
    ):
        raise ValueError("base v6 protocol does not preserve launch boundaries")
    authorized = amendment.get("authorized_run", {})
    amendment_locks = amendment.get("launch_locks", {})
    expected_hashes = {
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol_sha256": sha256_file(protocol_path.resolve()),
        "qa_factory_config_sha256": sha256_file(
            project_root
            / "configs/scenario_factory/qa_factory_v2_matched_supervision.json"
        ),
        "semantic_bottleneck_v2_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_semantic_bottleneck_v2.py"
        ),
        "semantic_runtime_v2_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_semantic_runtime_v2.py"
        ),
        "matched_objective_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_matched_objective.py"
        ),
        "records_sha256": sha256_file(records_path.resolve()),
        "visual_cache_sha256": sha256_file(visual_cache_path.resolve()),
        "orion_config_sha256": sha256_file(config_path.resolve()),
        "base_orion_checkpoint_sha256": sha256_file(checkpoint_path.resolve()),
    }
    validated = amendment.get("validated_inputs", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment_locks.get("stage2l_v6_route151_smoke_allowed") is not True
        or amendment_locks.get("stage2l_pilot_training_allowed") is not False
        or amendment_locks.get("stage2p_allowed") is not False
        or authorized.get("event_id") != EXPECTED_EVENT_ID
        or int(authorized.get("maximum_submissions", 0)) != 1
        or int(authorized.get("maximum_optimizer_steps", -1))
        != max_optimizer_steps
        or int(authorized.get("answer_micro_batch_size", -1))
        != answer_batch_size
        or authorized.get("fresh_initialization_from_original_orion_checkpoint")
        is not True
        or Path(str(authorized.get("output_root", ""))).resolve()
        != output_dir.resolve()
        or any(validated.get(name) != value for name, value in expected_hashes.items())
    ):
        raise ValueError(
            "Route151 v6 smoke amendment is absent, stale, or over-broad"
        )
    if sha256_file(amendment_path.resolve()) != amendment.get(
        "self_sha256_excluded", sha256_file(amendment_path.resolve())
    ):
        raise ValueError("unexpected amendment self-hash contract")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=20)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-head", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_optimizer_steps < 1 or args.answer_batch_size < 1:
        raise ValueError("smoke bounds must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite v6 smoke output")
    protocol = _load_json(args.training_protocol.resolve())
    amendment = _load_json(args.launch_amendment.resolve())
    _validate_authorization(
        protocol=protocol,
        amendment=amendment,
        protocol_path=args.training_protocol,
        amendment_path=args.launch_amendment,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        records_path=args.records,
        visual_cache_path=args.visual_cache,
        output_dir=args.output_dir,
        max_optimizer_steps=args.max_optimizer_steps,
        answer_batch_size=args.answer_batch_size,
    )
    losses = protocol["losses_prepared_not_activated"]
    lambda_language = float(losses["qa_causal_language_modeling"]["weight"])
    lambda_map = float(losses["dense_soft_relevance_bce"])
    lambda_ranking = float(losses["matched_onpath_offpath_risk_ranking"])
    lambda_stance = float(losses["hard_stance_cross_entropy"]["weight"])
    preference = losses["same_variant_cross_family_answer_preference"]
    lambda_preference = float(preference["weight"])
    preference_margin = float(preference["margin"])
    if any(
        float(losses.get(name, -1.0)) != 0.0
        for name in ("trajectory", "direct_control", "observed_consequence_calibration")
    ):
        raise ValueError("v6 smoke may not train consequence/control losses")

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    assets = Route151Assets(args.records, args.visual_cache)
    matched_audit = audit_matched_training_records(assets.records)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "v6_route151_smoke_preflight_pass",
                    "event_id": EXPECTED_EVENT_ID,
                    "matched_record_audit": matched_audit,
                    "group_ids": sorted(assets.group_rows),
                    "optimizer_steps_authorized": args.max_optimizer_steps,
                    "training_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("real v6 smoke requires CUDA")
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
    stance_bottleneck = MagnitudePreservingPlanningStanceBottleneck(
        model_dim=4096, hidden_dim=256
    ).cuda()
    auxiliary_modules = (
        uq_tokenizer,
        relevance_queries,
        relevance_head,
        risk_bridge,
        stance_bottleneck,
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

    before = _evaluate(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        stance_bottleneck=stance_bottleneck,
        assets=assets,
        answer_batch_size=args.answer_batch_size,
        generate_text=False,
    )
    history = []
    group_ids = sorted(assets.group_rows)
    rng = random.Random(args.seed)
    optimizer_step_count = 0
    for step in range(1, args.max_optimizer_steps + 1):
        if (step - 1) % len(group_ids) == 0:
            rng.shuffle(group_ids)
        group_id = group_ids[(step - 1) % len(group_ids)]
        group = assets.group_rows[group_id]
        anchors = assets.supervised_anchors(group_id)
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "language_nll": 0.0,
            "relevance_bce": 0.0,
            "ranking_loss": 0.0,
            "stance_ce": 0.0,
            "cross_family_preference": 0.0,
        }
        hard_stance_count = 0
        for row in anchors:
            variant = str(row["counterfactual"]["variant"])
            family = str(row["question_family"])
            baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
            relevance_target = assets.relevance[group_id].cuda(non_blocking=True)
            map_row = assets.row(group_id, "observed", "task_relevance")
            relevance = _relevance_pass(
                lm=lm,
                tokenizer=tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                baseline_vision=baseline,
                relevance_target=relevance_target,
                map_row=map_row,
                route_text=assets.route_text[group_id],
            )
            conditioned = _condition_variant(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                stance_bottleneck=stance_bottleneck,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)],
                relevance_logits=relevance["relevance_logits"],
            )
            nlls = _candidate_answer_nlls(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned["vision"],
                row=row,
                route_text=assets.route_text[group_id],
                answers=_same_variant_answers(group, row),
                micro_batch_size=args.answer_batch_size,
            )
            language_loss = nlls[0]
            preference_loss = cross_family_answer_preference_loss(
                nlls[:1], nlls[1:].unsqueeze(0), margin=preference_margin
            )
            on_uq = uq_tokenizer(
                assets.components[(group_id, "on_path_uq")].cuda(
                    non_blocking=True
                )
            )
            off_uq = uq_tokenizer(
                assets.components[(group_id, "off_path_uq")].cuda(
                    non_blocking=True
                )
            )
            on_risk = fixed_task_risk(
                on_uq.latest_scalar_uq, relevance["relevance_logits"]
            )
            off_risk = fixed_task_risk(
                off_uq.latest_scalar_uq, relevance["relevance_logits"]
            )
            ranking_loss = matched_task_risk_ranking_loss(
                on_risk, off_risk, margin=0.2
            )
            stance_loss = language_loss.new_zeros(())
            if family == "driving_implication" and variant in HARD_STANCE_VARIANTS:
                target = encode_planning_stances(
                    [_target_stance(row)], device=conditioned["stance_logits"].device
                )
                stance_loss = F.cross_entropy(conditioned["stance_logits"], target)
                hard_stance_count += 1
            total = (
                lambda_language * language_loss / len(anchors)
                + lambda_preference * preference_loss / len(anchors)
                + lambda_map * relevance["map_loss"] / len(anchors)
                + lambda_ranking * ranking_loss / len(anchors)
                + (
                    lambda_stance * stance_loss / len(HARD_STANCE_VARIANTS)
                    if family == "driving_implication"
                    and variant in HARD_STANCE_VARIANTS
                    else stance_loss
                )
            )
            total.backward()
            totals["loss"] += float(total.item())
            totals["language_nll"] += float(language_loss.item()) / len(anchors)
            totals["relevance_bce"] += float(relevance["map_loss"].item()) / len(anchors)
            totals["ranking_loss"] += float(ranking_loss.item()) / len(anchors)
            totals["stance_ce"] += (
                float(stance_loss.item()) / len(HARD_STANCE_VARIANTS)
                if family == "driving_implication" and variant in HARD_STANCE_VARIANTS
                else 0.0
            )
            totals["cross_family_preference"] += (
                float(preference_loss.item()) / len(anchors)
            )
        if hard_stance_count != len(HARD_STANCE_VARIANTS):
            raise RuntimeError("complete group did not expose all hard stances")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            lora_parameters + auxiliary_parameters, 1.0
        )
        optimizer.step()
        optimizer_step_count += 1
        item = {
            "optimizer_step": step,
            "group_id": group_id,
            "records_in_optimizer_unit": len(group),
            "hard_language_anchors": len(anchors),
            "optimizer_steps_inside_group": 0,
            "gradient_norm_before_clip": float(gradient_norm.item()),
            **totals,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LV6Smoke] " + json.dumps(item, sort_keys=True), flush=True)

    after = _evaluate(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        stance_bottleneck=stance_bottleneck,
        assets=assets,
        answer_batch_size=args.answer_batch_size,
        generate_text=True,
    )
    checks = {
        "exactly_one_optimizer_step_per_complete_group": (
            optimizer_step_count == args.max_optimizer_steps
            and all(
                row["records_in_optimizer_unit"] == 20
                and row["optimizer_steps_inside_group"] == 0
                for row in history
            )
        ),
        "language_nll_decreases": (
            after["first_group_mean_hard_language_nll"]
            < before["first_group_mean_hard_language_nll"]
        ),
        "relevance_bce_decreases": (
            after["mean_relevance_bce"] < before["mean_relevance_bce"]
        ),
        "on_path_exceeds_off_path": (
            after["mean_on_minus_off_peak_task_risk"] >= 0.2
        ),
        "hard_stance_accuracy_gte_two_thirds": (
            after["hard_stance_accuracy"] >= (2.0 / 3.0)
        ),
        "cross_family_margin_fraction_improves": (
            after["first_group_cross_family_margin_pass_fraction"]
            > before["first_group_cross_family_margin_pass_fraction"]
        ),
        "diagnostic_driving_targets_excluded": (
            matched_audit["diagnostic_only_driving_record_count"]
            == 2 * matched_audit["matched_group_count"]
        ),
        "trajectory_and_control_remain_disabled": True,
    }
    passed = all(checks.values())
    status = (
        "engineering_v6_matched_smoke_pass"
        if passed
        else "engineering_v6_matched_smoke_failed_gate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_v6_route151_smoke.pt"
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
            "stance_bottleneck": stance_bottleneck.state_dict(),
            "lora": {
                name: value.detach().cpu()
                for name, value in lm.state_dict().items()
                if "lora_" in name
            },
            "optimizer_steps": optimizer_step_count,
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
        "optimizer_steps": optimizer_step_count,
        "record_equivalent_presentations": optimizer_step_count * 20,
        "language_anchor_presentations": optimizer_step_count * 18,
        "matched_record_audit": matched_audit,
        "before": before,
        "after": after,
        "checks": checks,
        "history": history,
        "loss_weights": {
            "qa_causal_language_modeling": lambda_language,
            "dense_soft_relevance_bce": lambda_map,
            "matched_onpath_offpath_risk_ranking": lambda_ranking,
            "hard_stance_cross_entropy": lambda_stance,
            "same_variant_cross_family_answer_preference": {
                "weight": lambda_preference,
                "margin": preference_margin,
            },
            "trajectory": 0.0,
            "direct_control": 0.0,
        },
        "architecture": {
            "raw_k_magnitude_side_channel": True,
            "magnitude_layer_norm_before_classifier": False,
            "one_optimizer_step_per_20_record_group": True,
            "observed_driving_is_diagnostic_only": True,
            "view_shuffled_driving_is_diagnostic_only": True,
            "ground_truth_stance_enters_forward": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
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
            "training_protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
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
            "Route151 engineering learnability smoke only; no held-out, "
            "trajectory, closed-loop, generalization, or safety claim."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del lm
    gc.collect()
    print(json.dumps({"status": status, "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
