#!/usr/bin/env python3
"""Bounded soft-gate Stage2-L semantic vertical-slice smoke.

This consumer carries the integrity-valid terminal v12.1 factorized-R
checkpoint forward without changing it.  Route and actor relevance remain
separate diagnostics; their monotonic max-union is used only by the existing
``K = U * sigmoid(R)`` language bridge.  Only ``TaskRiskLanguageBridge`` is
optimized.  Model-quality checks are reported but do not stop this first
engineering slice; lineage, matched-control, finite-value and gradient-scope
invariants remain fail-closed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Tuple

import torch

import scripts.train_stage2l_v11_identifiable_smoke as v11
from uq_estimator.stage2l_factorized_relevance_v121 import (
    COMPONENT_ORDER,
    FactorizedTaskRelevanceMapHead,
)
from uq_estimator.stage2l_factorized_runtime_v11 import (
    ContextualRelevancePassV11,
    MatchedVLMConditioningV11,
    build_matched_vlm_conditioning_v11,
)
from uq_estimator.stage2l_identifiability import REQUIRED_RISK_VARIANTS
from uq_estimator.uq_relevance_tokenizer import (
    TaskRiskLanguageBridge,
    ViewAlignedTaskRelevanceQueryTokenizer,
)


SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_launch.v1"
V121_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke.v1"
V121_STATUS = "factorized_r_gate_failed"


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


def validate_v121_terminal_validation(value: Mapping[str, Any]) -> str:
    if (
        value.get("schema")
        != "orion.stage2l_v12_1_factorized_r_validation.v1"
        or value.get("status") != "validated_failed_gate"
        or value.get("integrity_valid") is not True
        or value.get("decision")
        != "held_out_factorized_r_transfer_failed"
        or value.get("optimizer_steps") != 40
    ):
        raise ValueError("v12.1 independent terminal validation differs")
    return str(value["decision"])


def derived_union_logit(component_logits: torch.Tensor) -> torch.Tensor:
    """Return logit(max(sigmoid(route), sigmoid(actor))) exactly by monotonicity."""

    if component_logits.ndim != 5 or component_logits.shape[1] != 2:
        raise ValueError("factorized R logits must have shape [B,2,V,H,W]")
    if not component_logits.is_floating_point() or not bool(
        torch.isfinite(component_logits).all()
    ):
        raise ValueError("factorized R logits must be finite floating tensors")
    return component_logits.amax(dim=1)


def _factorized_map_logits(
    *, lm, text_tokenizer, relevance_queries, relevance_head,
    assets: v11.V11Assets, group_id: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import scripts.train_stage2l_v101_view_aligned_phase_a as v101

    baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
    features = assets.view_features[group_id].cuda(non_blocking=True)
    union_target = assets.relevance[group_id].cuda(non_blocking=True)
    row = assets.row(group_id, "observed", "task_relevance")
    prompt = v101._prompt_tokens(
        text_tokenizer, row, assets.route_text[group_id]
    ).cuda(non_blocking=True)
    attention = prompt.ne(text_tokenizer.pad_token_id or 0)
    queries = relevance_queries(features)
    vision = torch.cat((baseline, queries), dim=1)
    output = lm(
        input_ids=prompt,
        attention_mask=attention,
        images=vision,
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
    result = relevance_head(query_grid)
    if tuple(result.component_logits.shape) != (1, 2, 6, 10, 10):
        raise ValueError("v12.1 factorized R output shape differs")
    if tuple(union_target.shape) != (1, 6, 10, 10):
        raise ValueError("v12.2 union target shape differs")
    return result.component_logits, union_target


def _group_conditioning(
    *, lm, text_tokenizer, uq_tokenizer, relevance_queries, relevance_head,
    risk_bridge, assets: v11.V11Assets, group_id: str,
    protocol: Mapping[str, Any],
) -> Tuple[MatchedVLMConditioningV11, torch.Tensor]:
    target_holder: Dict[str, torch.Tensor] = {}

    def relevance_forward() -> ContextualRelevancePassV11:
        with torch.no_grad():
            component_logits, target = _factorized_map_logits(
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
            baseline_vision=baseline,
            relevance_logits=derived_union_logit(component_logits),
        )

    thresholds = protocol["controlled_u_diagnostics"]
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
            "minimum_fraction": float(thresholds["minimum_group_fraction"]),
        },
    )
    return result, target_holder["value"]


def _set_trainable(module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def _load_v121_factorized_relevance(
    *, lm, relevance_queries, relevance_head, checkpoint_path: Path,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu")
    if (
        payload.get("schema") != V121_SCHEMA
        or payload.get("status") != V121_STATUS
        or payload.get("optimizer_steps") != 40
        or payload.get("stage1_uq_loaded") is not False
        or payload.get("u_tokenizer_loaded") is not False
        or payload.get("language_training_used") is not False
        or payload.get("trajectory_or_control_loss_used") is not False
    ):
        raise ValueError("terminal v12.1 factorized-R checkpoint contract differs")
    lm_state = lm.state_dict()
    lora = payload.get("lora", {})
    if {key for key in lm_state if "lora_" in key} != set(lora):
        raise ValueError("v12.1 factorized-R LoRA keys differ")
    result = lm.load_state_dict(lora, strict=False)
    if result.unexpected_keys or any(
        "lora_" in key for key in result.missing_keys
    ):
        raise ValueError("v12.1 factorized-R LoRA load was incomplete")
    relevance_queries.load_state_dict(
        payload["view_aligned_relevance_queries"], strict=True
    )
    relevance_head.load_state_dict(
        payload["factorized_relevance_head"], strict=True
    )
    for module in (lm, relevance_queries, relevance_head):
        _set_trainable(module, False)
        module.eval()
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "optimizer_steps": payload["optimizer_steps"],
        "checkpoint_sha256": _sha256(checkpoint_path.resolve()),
        "component_order": list(COMPONENT_ORDER),
        "union_definition": "max(sigmoid(route_logit), sigmoid(actor_logit))",
        "all_parameters_frozen": all(
            not parameter.requires_grad
            for module in (lm, relevance_queries, relevance_head)
            for parameter in module.parameters()
        ),
    }


def _runtime_path() -> Path:
    return Path(__file__).resolve().parents[1] / "uq_estimator" / "stage2l_factorized_runtime_v11.py"


def _factorized_path() -> Path:
    return Path(__file__).resolve().parents[1] / "uq_estimator" / "stage2l_factorized_relevance_v121.py"


def _identifiability_path() -> Path:
    return Path(__file__).resolve().parents[1] / "uq_estimator" / "stage2l_identifiability.py"


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "runtime_sha256": _sha256(_runtime_path()),
        "factorized_relevance_sha256": _sha256(_factorized_path()),
        "identifiability_audit_sha256": _sha256(_identifiability_path()),
        "soft_gate_policy_sha256": _sha256(args.soft_gate_policy.resolve()),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "v11_records_sha256": _sha256(args.v11_records.resolve()),
        "dataset_audit_report_sha256": _sha256(args.dataset_audit_report.resolve()),
        "view_feature_cache_sha256": _sha256(args.view_feature_cache.resolve()),
        "u_tokenizer_checkpoint_sha256": _sha256(args.u_tokenizer_checkpoint.resolve()),
        "v121_checkpoint_sha256": _sha256(args.v121_checkpoint.resolve()),
        "v121_report_sha256": _sha256(args.v121_report.resolve()),
        "v121_terminal_validation_sha256": _sha256(
            args.v121_terminal_validation.resolve()
        ),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
    }


def _checkpoint_contracts(args: argparse.Namespace) -> Dict[str, Any]:
    relevance = torch.load(args.v121_checkpoint.resolve(), map_location="cpu")
    tokenizer = torch.load(
        args.u_tokenizer_checkpoint.resolve(), map_location="cpu"
    )
    validation = _read_json(args.v121_terminal_validation.resolve())
    if (
        relevance.get("schema") != V121_SCHEMA
        or relevance.get("status") != V121_STATUS
        or relevance.get("optimizer_steps") != 40
        or relevance.get("stage1_uq_loaded") is not False
        or relevance.get("u_tokenizer_loaded") is not False
        or relevance.get("language_training_used") is not False
        or relevance.get("trajectory_or_control_loss_used") is not False
    ):
        raise ValueError("preflight v12.1 factorized-R checkpoint differs")
    if (
        tokenizer.get("schema")
        != "orion.stage1_u_tokenizer_pretraining_run.v1"
        or tokenizer.get("status")
        != "bounded_task_agnostic_tokenizer_pretraining_pass"
        or tokenizer.get("task_agnostic") is not True
    ):
        raise ValueError("preflight U-tokenizer checkpoint differs")
    terminal_quality_label = validate_v121_terminal_validation(validation)
    return {
        "factorized_relevance": {
            "schema": relevance["schema"],
            "status": relevance["status"],
            "optimizer_steps": relevance["optimizer_steps"],
            "terminal_quality_label": terminal_quality_label,
            "integrity_valid": validation["integrity_valid"],
        },
        "u_tokenizer": {
            "schema": tokenizer["schema"],
            "status": tokenizer["status"],
            "task_agnostic": tokenizer["task_agnostic"],
        },
    }


def _protocol_checks(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    expected = _validated_inputs(args)
    expected.pop("trainer_sha256")
    architecture = protocol.get("architecture", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen_single_soft_gate_semantic_slice"
        or protocol.get("input_sha256") != expected
        or protocol.get("output_root") != str(args.output_dir.resolve())
        or architecture.get("only_trainable_module")
        != "TaskRiskLanguageBridge"
        or architecture.get("derived_union_operator")
        != "max(sigmoid(route_logit), sigmoid(actor_logit))"
        or any(
            architecture.get(key) is not False
            for key in (
                "stage1_trainable",
                "u_tokenizer_trainable",
                "factorized_relevance_trainable",
                "orion_lora_trainable",
                "trajectory_or_control_loss",
                "density_uq_used",
                "governor_used",
                "locked_test_read",
            )
        )
    ):
        raise ValueError("v12.2 soft-gate semantic protocol is absent or stale")
    training = protocol.get("training", {})
    if (
        int(training.get("optimizer_steps", 0)) not in range(1, 41)
        or int(training.get("anchors_per_step", 0)) not in range(1, 3)
    ):
        raise ValueError("v12.2 bounded training ceiling differs")


def _preflight(
    *, args: argparse.Namespace, protocol: Mapping[str, Any],
    assets: v11.V11Assets,
) -> Dict[str, Any]:
    sampler = v11.OneEventPerStepSampler(
        assets.event_groups["train"], seed=args.seed
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "v12_2_soft_gate_semantic_preflight_pass_training_locked",
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
        "hard_stop_conditions": list(protocol["hard_stop_conditions"]),
        "soft_diagnostics": list(protocol["soft_diagnostics"]),
    }


def _validate_launch(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    preflight = _read_json(args.trainer_preflight.resolve())
    launch = _read_json(args.launch_amendment.resolve())
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("validated_inputs") != _validated_inputs(args)
        or preflight.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or preflight.get("output_root") != str(args.output_dir.resolve())
    ):
        raise ValueError("v12.2 trainer preflight is absent or stale")
    authorized = launch.get("authorized_run", {})
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status")
        != "immutable_single_soft_gate_semantic_slice_authorization"
        or launch.get("validated_inputs") != _validated_inputs(args)
        or launch.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or launch.get("preflight_sha256")
        != _sha256(args.trainer_preflight.resolve())
        or authorized.get("output_root") != str(args.output_dir.resolve())
        or authorized.get("maximum_submissions") != 1
        or authorized.get("automatic_retry") is not False
        or authorized.get("optimizer_steps")
        != int(protocol["training"]["optimizer_steps"])
        or launch.get("locks", {}).get("formal_stage2l_allowed") is not False
        or launch.get("locks", {}).get("stage2p_allowed") is not False
        or launch.get("locks", {}).get("closed_loop_allowed") is not False
    ):
        raise ValueError("v12.2 launch amendment is absent or stale")


def _hard_factorization_checks(
    factorization: Mapping[str, Mapping[str, Any]],
) -> Dict[str, bool]:
    checks = {}
    names = (
        "shared_r_bitwise_exact",
        "zero_u_and_k_exact",
        "on_off_magnitude_matched",
        "on_off_support_spatially_distinct",
    )
    for split in ("train", "dev"):
        for name in names:
            checks["%s_%s" % (split, name)] = bool(
                factorization[split]["release_checks"][name]
            )
    return checks


@torch.no_grad()
def _spatial_artifact(
    *, lm, text_tokenizer, uq_tokenizer, relevance_queries, relevance_head,
    risk_bridge, assets: v11.V11Assets, protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": "orion.stage2l_v12_2_spatial_diagnostic.v1",
        "component_order": list(COMPONENT_ORDER),
        "camera_order": list(v11.CAMERA_ORDER),
        "union_definition": "max(sigmoid(route_logit), sigmoid(actor_logit))",
        "events": {},
    }
    for split in ("train", "dev"):
        for event_id, groups in sorted(assets.event_groups[split].items()):
            group_id = sorted(groups)[0]
            component_logits, union_target = _factorized_map_logits(
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                group_id=group_id,
            )
            conditioned, _ = _group_conditioning(
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
            payload["events"][event_id] = {
                "split": split,
                "group_id": group_id,
                "r_component_logits": component_logits.cpu(),
                "r_component_probability": component_logits.sigmoid().cpu(),
                "r_union_probability": derived_union_logit(
                    component_logits
                ).sigmoid().cpu(),
                "r_union_target": union_target.cpu(),
                "u_by_variant": {
                    key: value.cpu()
                    for key, value in conditioned.latest_scalar_uq_by_variant.items()
                },
                "k_by_variant": {
                    key: value.task_risk.cpu()
                    for key, value in conditioned.conditioning_by_variant.items()
                },
                "structured_fields": {
                    key: dict(value.deterministic_semantics.structured_fields[0])
                    for key, value in conditioned.conditioning_by_variant.items()
                },
            }
    return payload


def _rear_semantic_probe(
    factorization: Mapping[str, Mapping[str, Any]], assets: v11.V11Assets,
) -> Dict[str, Any]:
    probes = []
    for split in ("train", "dev"):
        for group_id, group in factorization[split]["per_group"].items():
            on_path = group["structured_fields"]["on_path_uq"]
            if on_path.get("risk_view") != "CAM_BACK":
                continue
            row = assets.row(group_id, "on_path_uq", "driving_implication")
            zero = assets.row(group_id, "zero_uq", "driving_implication")
            off = assets.row(group_id, "off_path_uq", "driving_implication")
            probes.append({
                "split": split,
                "group_id": group_id,
                "event_id": group["event_id"],
                "scenario_family": row.get("scenario_family"),
                "ego_speed_mps": row["model_input"]["route_context"]["payload"][
                    "ego_state"
                ]["speedometer_mps"],
                "on_path_structured_fields": on_path,
                "zero_target_answer": zero["conversation"][1]["value"],
                "off_path_target_answer": off["conversation"][1]["value"],
                "on_path_target_answer": row["conversation"][1]["value"],
            })
    families = sorted({str(value["scenario_family"]) for value in probes})
    return {
        "status": "diagnostic_only",
        "central_rear_probe_count": len(probes),
        "scenario_families": families,
        "contains_hard_brake_or_yield_context": any(
            "break" in value.lower() or "yield" in value.lower()
            for value in families
        ),
        "ordinary_forward_control": "matched zero_uq and off_path_uq answers in every listed group",
        "quality_blocks_vertical_slice": False,
        "probes": probes,
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
    parser.add_argument("--soft-gate-policy", type=Path, required=True)
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
        args.v121_checkpoint,
        args.v121_report,
        args.v121_terminal_validation,
        args.soft_gate_policy,
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v12.2 semantic-slice prerequisite is missing")
    if args.answer_batch_size < 1:
        raise ValueError("answer batch size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty v12.2 output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)

    import scripts.train_stage2l_mr1_smoke as stage2l_base
    import scripts.train_stage2l_v101_view_aligned_phase_a as v101

    v101._configure_base()
    assets = v11.V11Assets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )
    if args.preflight_only:
        if args.trainer_preflight is not None or args.launch_amendment is not None:
            raise ValueError("v12.2 preflight cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("v12.2 preflight requires a fresh output path")
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
        raise ValueError("real v12.2 run requires preflight and launch amendment")
    _validate_launch(args, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("real v12.2 semantic slice requires CUDA")

    from mmcv.utils import set_random_seed
    import scripts.train_stage2l_v10_staged_smoke as v10

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, text_tokenizer = stage2l_base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    relevance_queries = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=4096,
        image_feature_dim=1024,
        hidden_dim=256,
        grid_hw=(10, 10),
        max_views=6,
    ).cuda()
    relevance_head = FactorizedTaskRelevanceMapHead(
        model_dim=4096, hidden_dim=256
    ).cuda()
    relevance_lineage = _load_v121_factorized_relevance(
        lm=lm,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        checkpoint_path=args.v121_checkpoint,
    )
    uq_tokenizer = v10._load_frozen_u_tokenizer(
        args.u_tokenizer_checkpoint.resolve()
    ).cuda().eval()
    risk_bridge = TaskRiskLanguageBridge(
        model_dim=4096, hidden_dim=256, max_views=6
    ).cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the audited v11 scoring/training implementation with this exact
    # factorized-R consumer.  The assignment is process-local and explicit.
    v11._group_conditioning = _group_conditioning
    factorization_before = {
        split: v11._evaluate_factorization(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            protocol={
                **protocol,
                "controlled_u_gates": protocol["controlled_u_diagnostics"],
            },
        )
        for split in ("train", "dev")
    }
    hard_checks = _hard_factorization_checks(factorization_before)
    if not all(hard_checks.values()):
        raise RuntimeError(
            "v12.2 hard controlled-U integrity invariant failed: %s"
            % json.dumps(hard_checks, sort_keys=True)
        )

    compat_protocol = {
        **protocol,
        "controlled_u_gates": protocol["controlled_u_diagnostics"],
        "language_gates": protocol["language_diagnostics"],
    }
    language_before = {
        split: v11._evaluate_language(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            protocol=compat_protocol,
            answer_batch_size=args.answer_batch_size,
        )
        for split in ("train", "dev")
    }
    history = v11._train_language_bridge(
        lm=lm,
        text_tokenizer=text_tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        assets=assets,
        protocol=compat_protocol,
        seed=args.seed,
        answer_batch_size=args.answer_batch_size,
        log_interval=args.log_interval,
    )
    language_after = {
        split: v11._evaluate_language(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            uq_tokenizer=uq_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            risk_bridge=risk_bridge,
            assets=assets,
            protocol=compat_protocol,
            answer_batch_size=args.answer_batch_size,
        )
        for split in ("train", "dev")
    }
    quality_checks = v11._language_release_checks(
        before=language_before,
        after=language_after,
        factorization=factorization_before,
        protocol=compat_protocol,
    )
    spatial = _spatial_artifact(
        lm=lm,
        text_tokenizer=text_tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        assets=assets,
        protocol=compat_protocol,
    )
    torch.save(spatial, args.output_dir / "spatial_u_r_k_maps.pt")
    status = (
        "vertical_slice_semantic_quality_diagnostics_passed"
        if all(quality_checks.values())
        else "vertical_slice_semantic_completed_with_soft_quality_failures"
    )
    torch.save(
        {
            "schema": SCHEMA,
            "status": status,
            "optimizer_steps": len(history),
            "task_risk_language_bridge": {
                key: value.detach().cpu()
                for key, value in risk_bridge.state_dict().items()
            },
            "factorized_relevance_checkpoint_sha256": _sha256(
                args.v121_checkpoint.resolve()
            ),
            "u_tokenizer_checkpoint_sha256": _sha256(
                args.u_tokenizer_checkpoint.resolve()
            ),
            "formal_stage2l_ready": False,
            "stage2p_ready": False,
        },
        args.output_dir / "v122_semantic_bridge.pt",
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "engineering_vertical_slice_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "optimizer_steps": len(history),
        "factorized_relevance_lineage": relevance_lineage,
        "factorization_before": factorization_before,
        "hard_integrity_checks": hard_checks,
        "hard_integrity_passed": all(hard_checks.values()),
        "language_before": language_before,
        "language_after": language_after,
        "soft_quality_checks": quality_checks,
        "soft_quality_passed": all(quality_checks.values()),
        "history": history,
        "rear_semantic_probe": _rear_semantic_probe(
            factorization_before, assets
        ),
        "spatial_artifact": {
            "path": str((args.output_dir / "spatial_u_r_k_maps.pt").resolve()),
            "event_count": len(spatial["events"]),
            "contains_r_components": True,
            "contains_u_and_k_by_variant": True,
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
            "stage1_or_u_tokenizer_trained": False,
            "factorized_relevance_or_orion_lora_trained": False,
            "only_task_risk_language_bridge_trained": True,
            "trajectory_or_control_loss_used": False,
            "density_uq_or_governor_used": False,
            "locked_test_read": False,
            "downstream_automatically_unlocked": False,
        },
        "claim_boundary": (
            "This is a controlled-U to frozen factorized-R to K to scored-QA "
            "engineering slice. It is not a learned-U, formal language, "
            "planning, closed-loop or safety result."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "optimizer_steps": len(history),
        "soft_quality_passed": all(quality_checks.values()),
        "report": str((args.output_dir / "report.json").resolve()),
    }, sort_keys=True), flush=True)
    del lm, relevance_queries, relevance_head, uq_tokenizer, risk_bridge
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
