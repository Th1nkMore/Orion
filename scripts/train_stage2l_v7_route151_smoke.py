#!/usr/bin/env python3
"""Bounded real-ORION smoke for calibrated Stage2-L v7 objectives.

This is still an engineering learnability smoke, not formal training.  It
keeps Stage-1 U frozen and task agnostic, learns R/K/QA semantics on the VLM
side, performs exactly one optimizer step per complete 20-record group, and
does not load trajectory, control, Density UQ or a hard governor.
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
from scripts.train_stage2l_route196_bridge_smoke import (
    _generate,
    _relevance_pass as _legacy_relevance_pass,
)
from scripts.train_stage2l_v6_route151_smoke import (
    BRIDGE_TOKENS,
    CACHE_SCHEMA,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    ORION_VISUAL_TOKENS,
    SEMANTIC_TOKENS,
    SPATIAL_UQ_TOKENS,
    _condition_variant,
    _expanded_labels,
    _load_json,
    _load_orion_lm,
    _route_text,
    _target_stance,
    _training_tokens,
)
from scripts.uq_relevance_qa_factory_v3_lib import audit_v3_records
from uq_estimator.stage2l_calibrated_objective import (
    class_balanced_matched_stance_loss,
    counterfactual_answer_preference_loss,
    foreground_balanced_relevance_terms,
    geometry_normalized_task_risk_ranking_terms,
    matched_stance_metrics,
    relevance_support_metrics,
)
from uq_estimator.stage2l_matched_objective import (
    HARD_STANCE_VARIANTS,
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
    partition_complete_matched_groups,
)
from uq_estimator.stage2l_pilot import resolve_reference
from uq_estimator.stage2l_qa_contract_v3 import (
    generation_contract_metrics,
    same_family_unique_counterfactual_answers,
)
from uq_estimator.stage2l_semantic_bottleneck import PLANNING_STANCES
from uq_estimator.stage2l_semantic_bottleneck_v2 import (
    MagnitudePreservingPlanningStanceBottleneck,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)


SCHEMA = "orion.stage2l_v7_calibrated_matched_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_calibrated_training_protocol.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
EXPECTED_EVENT_ID = "route151_step218"


def _load_records(path: Path, qa_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = partition_complete_matched_groups(rows)
    if len(groups) != 5:
        raise ValueError("v7 Route151 smoke requires exactly five matched groups")
    if {str(row.get("event_id", "")) for row in rows} != {EXPECTED_EVENT_ID}:
        raise ValueError("v7 smoke is authorized only for Route151")
    if {str(row.get("split", "")) for row in rows} != {"train"}:
        raise ValueError("v7 smoke may consume only the formal train split")
    audit = audit_v3_records(rows, config=qa_config)
    if not audit["passed"] or audit["record_count"] != 100:
        raise ValueError("v7 QA records fail the calibrated contract")
    return rows


class Route151V7Assets:
    """Hash-resolved five-group assets without the historical v6 policy."""

    def __init__(
        self,
        records_path: Path,
        visual_cache_path: Path,
        qa_config: Mapping[str, Any],
    ) -> None:
        self.records_path = records_path.resolve()
        self.visual_cache_path = visual_cache_path.resolve()
        self.records = _load_records(self.records_path, qa_config)
        self.qa_audit = audit_v3_records(self.records, config=qa_config)
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
                    raise ValueError("duplicate v7 group/variant/family")
                self.rows[key] = row

        cache = torch.load(self.visual_cache_path, map_location="cpu")
        if cache.get("schema") != CACHE_SCHEMA:
            raise ValueError("unsupported multiframe visual cache")
        contexts = cache.get("contexts", {})
        if set(contexts) != set(self.group_rows):
            raise ValueError("v7 QA groups and visual-cache groups differ")
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
                with np.load(uq_path, allow_pickle=False) as archive:
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
            self.relevance[group_id] = F.adaptive_avg_pool2d(target, (10, 10))
            self.route_text[group_id] = _route_text(route_payload)

    def row(self, group_id: str, variant: str, family: str) -> Mapping[str, Any]:
        return self.rows[(str(group_id), str(variant), str(family))]

    def supervised_anchors(
        self, group_id: str
    ) -> Tuple[Mapping[str, Any], ...]:
        rows = tuple(
            row
            for row in self.group_rows[group_id]
            if row["loss_policy"]["hard_language_target"] is True
        )
        if len(rows) != 18:
            raise RuntimeError("complete v7 group must expose 18 language anchors")
        return rows


def _relevance_logits(
    *,
    lm,
    tokenizer,
    relevance_queries,
    relevance_head,
    baseline_vision,
    relevance_target,
    map_row,
    route_text,
) -> torch.Tensor:
    # The legacy helper also constructs its historical BCE, but v7 discards
    # that tensor and exclusively backpropagates the calibrated objective.
    output = _legacy_relevance_pass(
        lm=lm,
        tokenizer=tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        baseline_vision=baseline_vision,
        relevance_target=relevance_target,
        map_row=map_row,
        route_text=route_text,
    )
    return output["relevance_logits"]


def _candidate_answer_nlls_v7(
    *,
    lm,
    tokenizer,
    vision: torch.Tensor,
    row: Mapping[str, Any],
    route_text: str,
    answers: Sequence[str],
    micro_batch_size: int,
) -> torch.Tensor:
    if len(answers) < 2:
        raise ValueError("candidate set needs a target and a distinct negative")
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
                raise RuntimeError("expanded v7 supervision exceeds LM logits")
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
    assets: Route151V7Assets,
    answer_batch_size: int,
    preference_margin: float,
    required_oracle_fraction: float,
    support_fraction: float,
    calibration_bce_weight: float,
    generate_text: bool,
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
    relevance_logits_all = []
    relevance_targets_all = []
    on_uq_all = []
    off_uq_all = []
    relevance_losses = []
    stance_logits = {variant: [] for variant in HARD_STANCE_VARIANTS}
    stance_targets = {variant: [] for variant in HARD_STANCE_VARIANTS}
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
        relevance_terms = foreground_balanced_relevance_terms(
            logits,
            target,
            support_fraction_of_peak=support_fraction,
            calibration_bce_weight=calibration_bce_weight,
        )
        relevance_losses.append(float(relevance_terms.loss.item()))
        conditioned = {
            variant: _condition_variant(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                stance_bottleneck=stance_bottleneck,
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
                assets.components[(group_id, "on_path_uq")].cuda(non_blocking=True)
            ).latest_scalar_uq
        )
        off_uq_all.append(
            uq_tokenizer(
                assets.components[(group_id, "off_path_uq")].cuda(non_blocking=True)
            ).latest_scalar_uq
        )
        for variant in HARD_STANCE_VARIANTS:
            stance_logits[variant].append(conditioned[variant]["stance_logits"])
            stance_targets[variant].append(
                _target_stance(
                    assets.row(group_id, variant, "driving_implication")
                )
            )
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
    stance = matched_stance_metrics(stance_logits, stance_targets)

    target_nlls = []
    preference_passes = []
    first_group = assets.group_rows[first_group_id]
    for row in assets.supervised_anchors(first_group_id):
        variant = str(row["counterfactual"]["variant"])
        answers = same_family_unique_counterfactual_answers(first_group, row)
        nlls = _candidate_answer_nlls_v7(
            lm=lm,
            tokenizer=tokenizer,
            vision=first_conditioned[variant]["vision"],
            row=row,
            route_text=assets.route_text[first_group_id],
            answers=answers,
            micro_batch_size=answer_batch_size,
        )
        target_nlls.append(float(nlls[0].item()))
        preference_passes.extend(
            bool(value >= preference_margin)
            for value in (nlls[1:] - nlls[0]).detach().cpu().tolist()
        )

    generated = {}
    generation_metrics = {}
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
        generation_metrics = generation_contract_metrics(
            generated,
            {
                variant: _target_stance(
                    assets.row(first_group_id, variant, "driving_implication")
                )
                for variant in HARD_STANCE_VARIANTS
            },
        )
    result = {
        "mean_foreground_balanced_relevance_loss": float(
            np.mean(relevance_losses)
        ),
        "relevance_support": support_metrics,
        "ranking": {
            "learned_gap": ranking.learned_gap.detach().cpu().tolist(),
            "oracle_gap": ranking.oracle_gap.detach().cpu().tolist(),
            "attained_fraction": ranking.attained_fraction.detach().cpu().tolist(),
            "minimum_attained_fraction": float(
                ranking.attained_fraction.min().item()
            ),
            "positive_order_fraction": float(
                ranking.learned_gap.gt(0.0).float().mean().item()
            ),
        },
        "stance": stance,
        "first_group_mean_hard_language_nll": float(np.mean(target_nlls)),
        "first_group_same_family_margin_pass_fraction": float(
            np.mean(preference_passes)
        ),
        "generated_answers": generated,
        "generation_contract": generation_metrics,
    }
    for module in modules:
        module.train()
    return result


def _validate_authorization(
    *,
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    protocol_path: Path,
    amendment_path: Path,
    qa_config_path: Path,
    objective_diagnostic_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    records_path: Path,
    visual_cache_path: Path,
    output_dir: Path,
    max_optimizer_steps: int,
    answer_batch_size: int,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported v7 training protocol")
    locks = protocol.get("launch_locks", {})
    if (
        locks.get("real_orion_smoke_allowed") is not False
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
    ):
        raise ValueError("base v7 protocol does not preserve launch locks")
    diagnostic = _load_json(objective_diagnostic_path.resolve())
    if (
        diagnostic.get("status") != "objective_feasibility_pass"
        or diagnostic.get("passed") is not True
        or not all(diagnostic.get("checks", {}).values())
    ):
        raise ValueError("v7 objective feasibility prerequisite did not pass")
    expected_hashes = {
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol_sha256": sha256_file(protocol_path.resolve()),
        "qa_factory_config_sha256": sha256_file(qa_config_path.resolve()),
        "calibrated_objective_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_calibrated_objective.py"
        ),
        "qa_contract_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_qa_contract_v3.py"
        ),
        "semantic_bottleneck_v2_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_semantic_bottleneck_v2.py"
        ),
        "semantic_runtime_v2_sha256": sha256_file(
            project_root / "uq_estimator/stage2l_semantic_runtime_v2.py"
        ),
        "objective_diagnostic_sha256": sha256_file(
            objective_diagnostic_path.resolve()
        ),
        "records_sha256": sha256_file(records_path.resolve()),
        "visual_cache_sha256": sha256_file(visual_cache_path.resolve()),
        "orion_config_sha256": sha256_file(config_path.resolve()),
        "base_orion_checkpoint_sha256": sha256_file(checkpoint_path.resolve()),
    }
    authorized = amendment.get("authorized_run", {})
    amendment_locks = amendment.get("launch_locks", {})
    validated = amendment.get("validated_inputs", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment_locks.get("stage2l_v7_route151_smoke_allowed") is not True
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
        raise ValueError("Route151 v7 smoke amendment is absent, stale, or broad")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--qa-config", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--objective-diagnostic", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite v7 smoke output")
    protocol = _load_json(args.training_protocol.resolve())
    amendment = _load_json(args.launch_amendment.resolve())
    qa_config = _load_json(args.qa_config.resolve())
    _validate_authorization(
        protocol=protocol,
        amendment=amendment,
        protocol_path=args.training_protocol,
        amendment_path=args.launch_amendment,
        qa_config_path=args.qa_config,
        objective_diagnostic_path=args.objective_diagnostic,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        records_path=args.records,
        visual_cache_path=args.visual_cache,
        output_dir=args.output_dir,
        max_optimizer_steps=args.max_optimizer_steps,
        answer_batch_size=args.answer_batch_size,
    )
    losses = protocol["losses"]
    map_config = losses["foreground_balanced_relevance"]
    ranking_config = losses["geometry_normalized_on_off_ranking"]
    preference_config = losses["same_family_counterfactual_preference"]
    lambda_language = float(losses["qa_causal_language_modeling"]["weight"])
    lambda_map = float(map_config["weight"])
    support_fraction = float(map_config["support_fraction_of_peak"])
    calibration_bce_weight = float(map_config["calibration_bce_weight"])
    lambda_ranking = float(ranking_config["weight"])
    required_oracle_fraction = float(ranking_config["required_oracle_fraction"])
    lambda_stance = float(losses["class_balanced_stance"]["weight"])
    lambda_preference = float(preference_config["weight"])
    preference_margin = float(preference_config["margin"])
    if any(
        float(losses.get(name, -1.0)) != 0.0
        for name in ("trajectory", "direct_control", "observed_consequence_calibration")
    ):
        raise ValueError("v7 smoke may not train consequence/control losses")

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    assets = Route151V7Assets(args.records, args.visual_cache, qa_config)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "v7_route151_smoke_preflight_pass",
                    "event_id": EXPECTED_EVENT_ID,
                    "qa_audit": assets.qa_audit,
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
        raise SystemExit("real v7 smoke requires CUDA")

    lm, tokenizer = _load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    uq_tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    risk_bridge = TaskRiskLanguageBridge(model_dim=4096, hidden_dim=256).cuda()
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
    lora_parameters = [parameter for parameter in lm.parameters() if parameter.requires_grad]
    auxiliary_parameters = [
        parameter for module in auxiliary_modules for parameter in module.parameters()
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
        preference_margin=preference_margin,
        required_oracle_fraction=required_oracle_fraction,
        support_fraction=support_fraction,
        calibration_bce_weight=calibration_bce_weight,
        generate_text=False,
    )
    history = []
    group_ids = sorted(assets.group_rows)
    rng = random.Random(args.seed)
    for step in range(1, args.max_optimizer_steps + 1):
        if (step - 1) % len(group_ids) == 0:
            rng.shuffle(group_ids)
        group_id = group_ids[(step - 1) % len(group_ids)]
        group = assets.group_rows[group_id]
        anchors = assets.supervised_anchors(group_id)
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        relevance_target = assets.relevance[group_id].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        relevance_logits = _relevance_logits(
            lm=lm,
            tokenizer=tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            baseline_vision=baseline,
            relevance_target=relevance_target,
            map_row=assets.row(group_id, "observed", "task_relevance"),
            route_text=assets.route_text[group_id],
        )
        map_terms = foreground_balanced_relevance_terms(
            relevance_logits,
            relevance_target,
            support_fraction_of_peak=support_fraction,
            calibration_bce_weight=calibration_bce_weight,
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
            relevance_target,
            required_oracle_fraction=required_oracle_fraction,
        )
        hard_conditioned = {
            variant: _condition_variant(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                stance_bottleneck=stance_bottleneck,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=relevance_logits,
            )
            for variant in HARD_STANCE_VARIANTS
        }
        stance_loss = class_balanced_matched_stance_loss(
            {
                variant: value["stance_logits"]
                for variant, value in hard_conditioned.items()
            },
            {
                variant: _target_stance(
                    assets.row(group_id, variant, "driving_implication")
                )
                for variant in HARD_STANCE_VARIANTS
            },
        )
        auxiliary_loss = (
            lambda_map * map_terms.loss
            + lambda_ranking * ranking_terms.loss
            + lambda_stance * stance_loss
        )
        auxiliary_loss.backward()

        totals = {
            "loss": float(auxiliary_loss.item()),
            "language_nll": 0.0,
            "foreground_balanced_relevance": float(map_terms.loss.item()),
            "ranking_loss": float(ranking_terms.loss.item()),
            "minimum_attained_fraction": float(
                ranking_terms.attained_fraction.min().item()
            ),
            "class_balanced_stance": float(stance_loss.item()),
            "counterfactual_preference": 0.0,
        }
        for row in anchors:
            variant = str(row["counterfactual"]["variant"])
            current_logits = _relevance_logits(
                lm=lm,
                tokenizer=tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                baseline_vision=baseline,
                relevance_target=relevance_target,
                map_row=assets.row(group_id, "observed", "task_relevance"),
                route_text=assets.route_text[group_id],
            )
            conditioned = _condition_variant(
                uq_tokenizer=uq_tokenizer,
                risk_bridge=risk_bridge,
                stance_bottleneck=stance_bottleneck,
                baseline_vision=baseline,
                components=assets.components[(group_id, variant)].cuda(
                    non_blocking=True
                ),
                relevance_logits=current_logits,
            )
            answers = same_family_unique_counterfactual_answers(group, row)
            nlls = _candidate_answer_nlls_v7(
                lm=lm,
                tokenizer=tokenizer,
                vision=conditioned["vision"],
                row=row,
                route_text=assets.route_text[group_id],
                answers=answers,
                micro_batch_size=args.answer_batch_size,
            )
            language_loss = nlls[0]
            preference_loss = counterfactual_answer_preference_loss(
                nlls[:1], nlls[1:].unsqueeze(0), margin=preference_margin
            )
            language_objective = (
                lambda_language * language_loss
                + lambda_preference * preference_loss
            ) / len(anchors)
            language_objective.backward()
            totals["loss"] += float(language_objective.item())
            totals["language_nll"] += float(language_loss.item()) / len(anchors)
            totals["counterfactual_preference"] += (
                float(preference_loss.item()) / len(anchors)
            )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            lora_parameters + auxiliary_parameters, 1.0
        )
        optimizer.step()
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
            print("[Stage2LV7Smoke] " + json.dumps(item, sort_keys=True), flush=True)

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
        preference_margin=preference_margin,
        required_oracle_fraction=required_oracle_fraction,
        support_fraction=support_fraction,
        calibration_bce_weight=calibration_bce_weight,
        generate_text=True,
    )
    qa_gates = qa_config["generation_gates"]
    stance_gates = qa_config["stance_gates"]
    checks = {
        "exactly_one_optimizer_step_per_complete_group": all(
            row["records_in_optimizer_unit"] == 20
            and row["optimizer_steps_inside_group"] == 0
            for row in history
        ),
        "language_nll_decreases": (
            after["first_group_mean_hard_language_nll"]
            < before["first_group_mean_hard_language_nll"]
        ),
        "same_family_preference_improves_and_gte_0_8": (
            after["first_group_same_family_margin_pass_fraction"]
            > before["first_group_same_family_margin_pass_fraction"]
            and after["first_group_same_family_margin_pass_fraction"] >= 0.8
        ),
        "relevance_foreground_recall": (
            after["relevance_support"]["foreground_recall"]
            >= qa_config["task_relevance_gates"]["minimum_foreground_recall"]
        ),
        "relevance_background_fpr": (
            after["relevance_support"]["background_false_positive_rate"]
            <= qa_config["task_relevance_gates"][
                "maximum_background_false_positive_rate"
            ]
        ),
        "all_groups_positive_on_off_order": (
            after["ranking"]["positive_order_fraction"] == 1.0
        ),
        "all_groups_attain_oracle_fraction": (
            after["ranking"]["minimum_attained_fraction"]
            >= required_oracle_fraction
        ),
        "all_stance_variants_correct": all(
            value >= stance_gates["minimum_per_variant_accuracy"]
            for value in after["stance"]["per_variant_accuracy"].values()
        ),
        "all_stance_classes_recalled": all(
            value >= stance_gates["minimum_per_target_class_recall"]
            for value in after["stance"]["per_target_class_recall"].values()
        ),
        "minimum_stance_target_probability": (
            after["stance"]["minimum_target_probability"]
            >= qa_gates["minimum_hard_stance_target_probability"]
        ),
        "generated_family_tags_match": (
            after["generation_contract"]["family_tag_parse_and_accuracy"]
            >= qa_gates["family_tag_accuracy"]
        ),
        "generated_text_nonrepeating": (
            after["generation_contract"]["nonrepeating_text_fraction"] == 1.0
        ),
        "generated_driving_stances_parse": (
            after["generation_contract"]["hard_driving_stance_parse_rate"]
            >= qa_gates["hard_driving_stance_parse_rate"]
        ),
        "generated_driving_stances_agree": (
            after["generation_contract"]["hard_driving_stance_agreement"]
            >= qa_gates["hard_driving_structured_stance_agreement"]
        ),
        "diagnostic_driving_targets_excluded": (
            assets.qa_audit["hard_language_record_count"] == 90
            and assets.qa_audit["hard_stance_record_count"] == 15
        ),
        "trajectory_and_control_remain_disabled": True,
    }
    passed = all(checks.values())
    status = (
        "engineering_v7_calibrated_smoke_pass"
        if passed
        else "engineering_v7_calibrated_smoke_failed_gate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_v7_route151_smoke.pt"
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
        "record_equivalent_presentations": len(history) * 20,
        "qa_audit": assets.qa_audit,
        "before": before,
        "after": after,
        "checks": checks,
        "history": history,
        "architecture": {
            "stage1_adapter_frozen_and_task_agnostic": True,
            "task_relevance_owned_by_vlm": True,
            "raw_k_magnitude_side_channel": True,
            "one_optimizer_step_per_20_record_group": True,
            "cross_family_answers_used_as_negatives": False,
            "exact_duplicate_answers_used_as_negatives": False,
            "overall_stance_accuracy_used_as_gate": False,
            "ground_truth_stance_enters_forward": False,
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
            "objective_diagnostic": {
                "path": str(args.objective_diagnostic.resolve()),
                "sha256": sha256_file(args.objective_diagnostic.resolve()),
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
            "Route151 v7 engineering learnability smoke only; no held-out, "
            "trajectory, closed-loop, generalization or safety claim."
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
