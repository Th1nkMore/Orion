#!/usr/bin/env python3
"""Bounded Stage2-L v10.1 Phase-A-only view-aligned engineering smoke.

This repair keeps task relevance R owned by ORION/VLM but gives each R query
the matching frozen ORION camera feature cell before global VLM fusion.  It
warm-starts the useful v10 Phase-A LoRA/query/head state, never loads Stage-1
U, and cannot run Phase B, Phase C, formal Stage2-L, Stage2-P, or closed loop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.utils import set_random_seed

import scripts.train_stage2l_mr1_smoke as base
from scripts.scenario_factory_lib import sha256_file
from scripts.train_stage2l_route196_bridge_smoke import (
    IMAGE_TOKEN_INDEX,
    _prompt_tokens,
)
from uq_estimator.stage2l_bridge_runtime import extract_relevance_query_grid
from uq_estimator.stage2l_calibrated_objective import relevance_support_metrics
from uq_estimator.stage2l_matched_objective import (
    MATCHED_VARIANTS,
    partition_complete_matched_groups,
)
from uq_estimator.stage2l_relevance_objective_v10 import (
    stage2l_relevance_objective_v10,
)
from uq_estimator.stage2l_pilot import resolve_reference
from uq_estimator.uq_relevance_tokenizer import (
    TaskRelevanceMapHead,
    ViewAlignedTaskRelevanceQueryTokenizer,
)


SCHEMA = "orion.stage2l_v101_view_aligned_phase_a.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v101_view_aligned_phase_a_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v101_view_aligned_phase_a_preflight.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
DATASET_SCHEMA = "orion.stage2l_expanded_coverage_dataset.v1"
FEATURE_CACHE_SCHEMA = "orion.stage2l_view_aligned_feature_cache.v1"
EXPECTED_EVENTS = 17
EXPECTED_TRAIN_EVENTS = 13
EXPECTED_DEV_EVENTS = 4
EXPECTED_GROUPS = 80
EXPECTED_RECORDS = 1600
ORION_VISUAL_TOKENS = 529
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


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


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _configure_base() -> None:
    base.DATASET_SCHEMA = DATASET_SCHEMA
    base.EXPECTED_EVENT_COUNT = EXPECTED_EVENTS
    base.EXPECTED_TRAIN_EVENT_COUNT = EXPECTED_TRAIN_EVENTS
    base.EXPECTED_DEV_EVENT_COUNT = EXPECTED_DEV_EVENTS
    base.EXPECTED_GROUP_COUNT = EXPECTED_GROUPS
    base.EXPECTED_RECORD_COUNT = EXPECTED_RECORDS


class PhaseAAssets:
    """Load only clean visual/route inputs and R supervision, never Stage-1 U."""

    def __init__(self, manifest_path: Path, feature_cache_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = _read_json(self.manifest_path)
        if (
            self.manifest.get("schema") != DATASET_SCHEMA
            or self.manifest.get("event_count") != EXPECTED_EVENTS
            or self.manifest.get("train_event_count") != EXPECTED_TRAIN_EVENTS
            or self.manifest.get("dev_event_count") != EXPECTED_DEV_EVENTS
            or self.manifest.get("record_count") != EXPECTED_RECORDS
            or self.manifest.get("formal_stage2l_training_allowed") is not False
            or self.manifest.get("stage2p_allowed") is not False
            or self.manifest.get("review_boundary", {}).get(
                "eligible_for_bounded_preexperiment"
            )
            is not True
        ):
            raise ValueError("v10.1 dataset scope or locks differ")
        self.records_path = Path(self.manifest["records"]["path"]).resolve()
        if (
            not self.records_path.is_file()
            or sha256_file(self.records_path)
            != self.manifest["records"]["sha256"]
        ):
            raise ValueError("v10.1 records are absent or stale")
        self.records = _read_records(self.records_path)
        audit = base.audit_records(self.records)
        groups = partition_complete_matched_groups(self.records)
        if (
            audit.get("passed") is not True
            or len(self.records) != EXPECTED_RECORDS
            or len(groups) != EXPECTED_GROUPS
        ):
            raise ValueError("v10.1 records fail V5 matched-group audit")

        self.rows = {}
        self.group_rows = {}
        self.group_event = {}
        self.group_split = {}
        for group in groups:
            group_id = str(group[0]["counterfactual"]["group_id"])
            event_ids = {str(row["event_id"]) for row in group}
            splits = {str(row["split"]) for row in group}
            if len(event_ids) != 1 or len(splits) != 1:
                raise ValueError("v10.1 group event/split identity differs")
            event_id = next(iter(event_ids))
            split = next(iter(splits))
            if split not in ("train", "dev"):
                raise ValueError("v10.1 may use only train/dev groups")
            self.group_rows[group_id] = group
            self.group_event[group_id] = event_id
            self.group_split[group_id] = split
            for row in group:
                key = (
                    group_id,
                    str(row["counterfactual"]["variant"]),
                    str(row["question_family"]),
                )
                if key in self.rows:
                    raise ValueError("duplicate v10.1 group/variant/family")
                self.rows[key] = row

        self.event_meta = {
            str(value["event_id"]): value for value in self.manifest["events"]
        }
        if set(self.event_meta) != set(self.group_event.values()):
            raise ValueError("v10.1 manifest event identities differ")
        self.event_groups = {split: {} for split in ("train", "dev")}
        for event_id, meta in sorted(self.event_meta.items()):
            split = str(meta["split"])
            event_groups = tuple(
                sorted(
                    group_id
                    for group_id, value in self.group_event.items()
                    if value == event_id
                )
            )
            if len(event_groups) != int(meta["keyframe_count"]):
                raise ValueError("v10.1 event keyframe count differs")
            self.event_groups[split][event_id] = event_groups
        if (
            len(self.event_groups["train"]) != EXPECTED_TRAIN_EVENTS
            or len(self.event_groups["dev"]) != EXPECTED_DEV_EVENTS
        ):
            raise ValueError("v10.1 event split count differs")

        self.visual_contexts = {}
        for event_id, meta in sorted(self.event_meta.items()):
            reference = meta["visual_cache"]
            path = Path(reference["path"]).resolve()
            if not path.is_file() or sha256_file(path) != reference["sha256"]:
                raise ValueError("v10.1 visual cache is absent/stale: %s" % event_id)
            cache = torch.load(path, map_location="cpu")
            if cache.get("schema") != base.CACHE_SCHEMA:
                raise ValueError("v10.1 visual cache schema differs")
            contexts = cache.get("contexts", {})
            expected = set(self.event_groups[str(meta["split"])][event_id])
            if set(contexts) != expected:
                raise ValueError("v10.1 visual cache group set differs")
            for group_id, value in contexts.items():
                if tuple(value.shape) != (1, ORION_VISUAL_TOKENS, 4096):
                    raise ValueError("v10.1 ORION visual context shape differs")
                self.visual_contexts[str(group_id)] = value.detach().float().cpu()

        feature_payload = torch.load(feature_cache_path.resolve(), map_location="cpu")
        metadata = feature_payload.get("metadata", {})
        if (
            feature_payload.get("schema") != FEATURE_CACHE_SCHEMA
            or metadata.get("camera_order") != list(CAMERA_ORDER)
            or metadata.get("event_count") != EXPECTED_EVENTS
            or metadata.get("group_count") != EXPECTED_GROUPS
            or metadata.get("stage1_uq_inputs_used") is not False
            or metadata.get("task_relevance_targets_used") is not False
            or metadata.get("qa_answers_used") is not False
            or metadata.get("trajectory_or_control_inputs_used") is not False
        ):
            raise ValueError("v10.1 view-aligned feature cache contract differs")
        contexts = feature_payload.get("contexts", {})
        if set(contexts) != set(self.group_rows):
            raise ValueError("v10.1 view-aligned cache group set differs")
        self.view_features = {}
        for group_id, value in contexts.items():
            if tuple(value.shape) != (6, 10, 10, 1024):
                raise ValueError("v10.1 view-aligned feature shape differs")
            self.view_features[str(group_id)] = value.detach().float().unsqueeze(0)

        self.relevance = {}
        self.route_text = {}
        for group_id in sorted(self.group_rows):
            row = self.row(group_id, "observed", "task_relevance")
            sidecar_ref = row["target"]["map_sidecar"]
            sidecar_path = resolve_reference(
                sidecar_ref,
                self.records_path.parent,
                "R target for %s" % group_id,
            )
            with np.load(sidecar_path, allow_pickle=False) as archive:
                relevance = archive[sidecar_ref["relevance_key"]].astype(np.float32)
            if relevance.shape != (6, 40, 40):
                raise ValueError("v10.1 task-relevance target shape differs")
            target = torch.from_numpy(relevance).unsqueeze(0)
            self.relevance[group_id] = F.adaptive_avg_pool2d(target, (10, 10))
            payload = row["model_input"]["route_context"]["payload"]
            self.route_text[group_id] = base._route_text(payload)

    def row(self, group_id: str, variant: str, family: str) -> Mapping[str, Any]:
        return self.rows[(str(group_id), str(variant), str(family))]

    def groups_for_split(self, split: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                group_id
                for group_id, value in self.group_split.items()
                if value == split
            )
        )


def _map_logits(
    *,
    lm,
    text_tokenizer,
    relevance_queries,
    relevance_head,
    assets: PhaseAAssets,
    group_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
    features = assets.view_features[group_id].cuda(non_blocking=True)
    target = assets.relevance[group_id].cuda(non_blocking=True)
    row = assets.row(group_id, "observed", "task_relevance")
    prompt = _prompt_tokens(
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
    query_grid = extract_relevance_query_grid(
        output.hidden_states[-1],
        prompt,
        image_token_index=IMAGE_TOKEN_INDEX,
        visual_token_count=ORION_VISUAL_TOKENS,
        views=6,
        grid_h=10,
        grid_w=10,
    )
    return relevance_head(query_grid), target


def _support_mask(targets: np.ndarray, support_fraction: float) -> np.ndarray:
    peaks = targets.reshape(targets.shape[0], -1).max(axis=1)
    shape = (targets.shape[0],) + (1,) * (targets.ndim - 1)
    return targets >= peaks.reshape(shape) * float(support_fraction)


def _average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    scores = scores.reshape(-1)
    truth = truth.astype(bool).reshape(-1)
    positives = int(truth.sum())
    if positives <= 0:
        raise ValueError("average precision requires positives")
    order = np.argsort(-scores, kind="mergesort")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


@torch.no_grad()
def _evaluate(
    *,
    split: str,
    lm,
    text_tokenizer,
    relevance_queries,
    relevance_head,
    assets: PhaseAAssets,
    support_fraction: float,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    lm.eval()
    relevance_queries.eval()
    relevance_head.eval()
    logits_all = []
    targets_all = []
    raw = {}
    grouped = defaultdict(list)
    losses = []
    for group_id in assets.groups_for_split(split):
        logits, target = _map_logits(
            lm=lm,
            text_tokenizer=text_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            group_id=group_id,
        )
        terms = stage2l_relevance_objective_v10(
            logits,
            target,
            phase="map_pretrain",
            support_fraction_of_peak=support_fraction,
        )
        probability = logits.sigmoid().float().cpu()
        target_cpu = target.float().cpu()
        raw[group_id] = {"probability": probability, "target": target_cpu}
        grouped[assets.group_event[group_id]].append(group_id)
        logits_all.append(logits)
        targets_all.append(target)
        losses.append(float(terms.map_loss.item()))
    logits_tensor = torch.cat(logits_all, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)
    support = relevance_support_metrics(
        logits_tensor,
        targets_tensor,
        support_fraction_of_peak=support_fraction,
    )
    probability_np = logits_tensor.sigmoid().float().cpu().numpy()
    targets_np = targets_tensor.float().cpu().numpy()
    foreground = _support_mask(targets_np, support_fraction)
    event_metrics = {}
    for event_id, group_ids in sorted(grouped.items()):
        probabilities = np.concatenate(
            [raw[group_id]["probability"].numpy() for group_id in group_ids], axis=0
        )
        targets = np.concatenate(
            [raw[group_id]["target"].numpy() for group_id in group_ids], axis=0
        )
        event_foreground = _support_mask(targets, support_fraction)
        event_metrics[event_id] = {
            "group_count": len(group_ids),
            "average_precision": _average_precision(probabilities, event_foreground),
            "foreground_prevalence": float(event_foreground.mean()),
        }
    return (
        {
            "split": split,
            "group_count": len(raw),
            "mean_map_loss": float(np.mean(losses)),
            "relevance_support": support,
            "average_precision": _average_precision(probability_np, foreground),
            "foreground_prevalence": float(foreground.mean()),
            "per_event": event_metrics,
        },
        raw,
    )


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _phase_a_gate(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
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


def _copy_v10_query_state(
    module: ViewAlignedTaskRelevanceQueryTokenizer,
    source: Mapping[str, torch.Tensor],
) -> list[str]:
    target = module.state_dict()
    copied = []
    for key, value in source.items():
        if key not in target or tuple(target[key].shape) != tuple(value.shape):
            raise ValueError("v10 query warm-start key/shape differs: %s" % key)
        target[key] = value
        copied.append(key)
    expected_new = {
        "evidence_norm.weight",
        "evidence_norm.bias",
        "evidence_projection.0.weight",
        "evidence_projection.0.bias",
        "evidence_projection.2.weight",
        "evidence_projection.2.bias",
    }
    if set(target) - set(source) != expected_new:
        raise ValueError("v10.1 query module adds an unexpected parameter set")
    module.load_state_dict(target, strict=True)
    return sorted(copied)


def _load_warm_start(
    *,
    lm,
    relevance_queries,
    relevance_head,
    checkpoint_path: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if (
        payload.get("schema") != "orion.stage2l_v10_staged_smoke.v1"
        or payload.get("status") != "phase_a_failed_gate"
        or payload.get("completed_phases") != []
        or payload.get("formal_stage2l_ready") is not False
        or payload.get("stage2p_ready") is not False
    ):
        raise ValueError("v10.1 requires the scientifically failed v10 Phase-A checkpoint")
    lm_state = lm.state_dict()
    lora = payload.get("lora", {})
    if {key for key in lm_state if "lora_" in key} != set(lora):
        raise ValueError("v10.1 LoRA warm-start keys differ")
    result = lm.load_state_dict(lora, strict=False)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("v10.1 LoRA warm-start was incomplete")
    copied = _copy_v10_query_state(
        relevance_queries, payload["relevance_queries"]
    )
    relevance_head.load_state_dict(payload["relevance_head"], strict=True)
    return {
        "source_schema": payload["schema"],
        "source_status": payload["status"],
        "copied_lora_tensor_count": len(lora),
        "copied_query_keys": copied,
        "copied_relevance_head_tensor_count": len(payload["relevance_head"]),
        "new_view_evidence_parameters_warm_started": False,
    }


def _checkpoint(
    *,
    step: int,
    status: str,
    lm,
    relevance_queries,
    relevance_head,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "optimizer_steps": int(step),
        "engineering_preexperiment_only": True,
        "phase_a_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "lora": {
            key: value.detach().cpu()
            for key, value in lm.state_dict().items()
            if "lora_" in key
        },
        "view_aligned_relevance_queries": {
            key: value.detach().cpu()
            for key, value in relevance_queries.state_dict().items()
        },
        "relevance_head": {
            key: value.detach().cpu()
            for key, value in relevance_head.state_dict().items()
        },
        "stage1_uq_loaded": False,
        "trajectory_or_control_loss_used": False,
        "provenance": dict(provenance),
    }


def _validated_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "query_module_sha256": _sha256(
            Path(__file__).resolve().parents[1]
            / "uq_estimator"
            / "uq_relevance_tokenizer.py"
        ),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "view_feature_cache_sha256": _sha256(args.view_feature_cache.resolve()),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "v10_phase_a_checkpoint_sha256": _sha256(
            args.v10_phase_a_checkpoint.resolve()
        ),
        "v10_report_sha256": _sha256(args.v10_report.resolve()),
    }


def _protocol_checks(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "frozen_phase_a_only_protocol_launch_locked"
        or protocol.get("validated_inputs") != _validated_inputs(args)
        or protocol.get("output_root") != str(args.output_dir.resolve())
        or protocol.get("camera_order") != list(CAMERA_ORDER)
        or protocol.get("maximum_optimizer_steps") not in (40, 80, 120)
    ):
        raise ValueError("v10.1 Phase-A protocol is absent or stale")
    if any(
        protocol.get("locks", {}).get(key) is not False
        for key in (
            "stage1_uq_input",
            "phase_b",
            "phase_c",
            "formal_stage2l",
            "stage2p",
            "closed_loop",
            "route203_native_glare_submission",
        )
    ):
        raise ValueError("v10.1 Phase-A protocol expands a locked scope")


def _preflight(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    assets: PhaseAAssets,
) -> dict[str, Any]:
    warm = torch.load(args.v10_phase_a_checkpoint.resolve(), map_location="cpu")
    sampler = base.EventBalancedSampler(assets.event_groups["train"], seed=args.seed)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "v101_view_aligned_phase_a_preflight_pass_training_locked",
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
        "first_two_event_balanced_units": [sampler.next(), sampler.next()],
        "warm_start_status": warm.get("status"),
        "stage1_uq_loaded": False,
        "locks": dict(protocol["locks"]),
    }


def _validate_launch(args: argparse.Namespace) -> None:
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
        raise ValueError("v10.1 Phase-A preflight is absent or stale")
    authorized = amendment.get("authorized_run", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("status") != "immutable_phase_a_only_authorization"
        or amendment.get("validated_inputs") != _validated_inputs(args)
        or amendment.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or amendment.get("preflight_sha256")
        != _sha256(args.trainer_preflight.resolve())
        or authorized.get("output_root") != str(args.output_dir.resolve())
        or authorized.get("maximum_submissions") != 1
        or authorized.get("automatic_retry") is not False
        or authorized.get("maximum_optimizer_steps")
        != int(_read_json(args.training_protocol)["maximum_optimizer_steps"])
    ):
        raise ValueError("v10.1 Phase-A launch amendment is absent or stale")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--v10-phase-a-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-report", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.dataset_manifest,
        args.view_feature_cache,
        args.v10_phase_a_checkpoint,
        args.v10_report,
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v10.1 Phase-A prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite v10.1 Phase-A output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)
    _configure_base()
    assets = PhaseAAssets(
        args.dataset_manifest.resolve(), args.view_feature_cache.resolve()
    )
    if args.preflight_only:
        if args.trainer_preflight is not None or args.launch_amendment is not None:
            raise ValueError("preflight-only mode cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("preflight-only mode requires a fresh output")
        value = _preflight(args, protocol, assets)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if (
        args.preflight_output is not None
        or args.trainer_preflight is None
        or args.launch_amendment is None
    ):
        raise ValueError("real v10.1 Phase-A requires preflight and amendment")
    _validate_launch(args)
    if not torch.cuda.is_available():
        raise RuntimeError("real v10.1 Phase-A smoke requires CUDA")

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
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    warm_start = _load_warm_start(
        lm=lm,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        checkpoint_path=args.v10_phase_a_checkpoint.resolve(),
    )

    support_fraction = float(protocol["support_fraction_of_peak"])
    milestones = tuple(map(int, protocol["evaluation_milestones"]))
    maximum_steps = int(protocol["maximum_optimizer_steps"])
    if not milestones or milestones[-1] != maximum_steps:
        raise ValueError("v10.1 evaluation milestones must end at the step ceiling")
    before = {}
    for split in ("train", "dev"):
        before[split], _ = _evaluate(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            support_fraction=support_fraction,
        )

    lora_parameters = [value for value in lm.parameters() if value.requires_grad]
    query_parameters = list(relevance_queries.parameters())
    head_parameters = list(relevance_head.parameters())
    trainable = lora_parameters + query_parameters + head_parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": float(protocol["learning_rates"]["lora"])},
            {
                "params": query_parameters + head_parameters,
                "lr": float(protocol["learning_rates"]["relevance"]),
            },
        ],
        weight_decay=float(protocol["learning_rates"]["weight_decay"]),
    )
    sampler = base.EventBalancedSampler(assets.event_groups["train"], seed=args.seed)
    history = []
    evaluations = []
    stop_reason = "maximum_steps_reached"
    completed_steps = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "preflight_sha256": _sha256(args.trainer_preflight.resolve()),
        "launch_amendment_sha256": _sha256(args.launch_amendment.resolve()),
        "warm_start": warm_start,
    }
    for step in range(1, maximum_steps + 1):
        lm.train()
        relevance_queries.train()
        relevance_head.train()
        groups = sampler.next()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for group_id in groups:
            logits, target = _map_logits(
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                group_id=group_id,
            )
            terms = stage2l_relevance_objective_v10(
                logits,
                target,
                phase="map_pretrain",
                support_fraction_of_peak=support_fraction,
            )
            objective = terms.loss / float(len(groups))
            objective.backward()
            mean_loss += float(objective.item())
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        finite = bool(torch.isfinite(norm)) and _all_finite(
            parameter.grad
            for parameter in trainable
            if parameter.grad is not None
        )
        if step <= 2 and not finite:
            raise RuntimeError("v10.1 first-two-step finite fail-fast")
        optimizer.step()
        completed_steps = step
        item = {
            "optimizer_step": step,
            "primary_group_ids": list(groups),
            "primary_event_ids": [assets.group_event[value] for value in groups],
            "loss": mean_loss,
            "gradient_norm_before_clip": float(norm.item()),
            "finite": finite,
        }
        history.append(item)
        if step == 1 or step % args.log_interval == 0:
            print("[Stage2LV101] " + json.dumps(item, sort_keys=True), flush=True)
        if step not in milestones:
            continue

        metrics = {}
        spatial_maps = {}
        for split in ("train", "dev"):
            metrics[split], raw = _evaluate(
                split=split,
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                support_fraction=support_fraction,
            )
            spatial_maps.update(raw)
        checks = _phase_a_gate(metrics, protocol["release_gates"])
        passed = all(checks.values())
        evaluations.append(
            {"optimizer_step": step, "metrics": metrics, "checks": checks, "passed": passed}
        )
        torch.save(
            _checkpoint(
                step=step,
                status="phase_a_pass" if passed else "phase_a_failed_gate",
                lm=lm,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                provenance=provenance,
            ),
            args.output_dir / ("phase_a_step%03d.pt" % step),
        )
        torch.save(spatial_maps, args.output_dir / ("spatial_maps_step%03d.pt" % step))
        print(
            "[Stage2LV101Eval] "
            + json.dumps(
                {
                    "step": step,
                    "passed": passed,
                    "train_ap": metrics["train"]["average_precision"],
                    "dev_ap": metrics["dev"]["average_precision"],
                    "checks": checks,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if passed:
            stop_reason = "phase_a_gate_passed_early"
            break
        if len(evaluations) >= 2:
            previous = evaluations[-2]["metrics"]
            current = evaluations[-1]["metrics"]
            dev_drop = (
                previous["dev"]["average_precision"]
                - current["dev"]["average_precision"]
            )
            train_gain = (
                current["train"]["average_precision"]
                - previous["train"]["average_precision"]
            )
            if (
                dev_drop >= float(protocol["early_stop"]["minimum_dev_ap_drop"])
                and train_gain
                >= float(protocol["early_stop"]["minimum_train_ap_gain"])
            ):
                stop_reason = "clear_train_dev_overfit_early_stop"
                break

    del optimizer
    final = evaluations[-1]
    report = {
        "schema": SCHEMA,
        "status": (
            "phase_a_gate_passed_engineering_only"
            if final["passed"]
            else "phase_a_stopped_without_gate_pass"
        ),
        "optimizer_steps": completed_steps,
        "stop_reason": stop_reason,
        "engineering_preexperiment_only": True,
        "phase_a_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "before": before,
        "history": history,
        "evaluations": evaluations,
        "final_metrics": final["metrics"],
        "final_checks": final["checks"],
        "warm_start": warm_start,
        "provenance": provenance,
        "locks": {
            "stage1_uq_loaded": False,
            "phase_b_run": False,
            "phase_c_run": False,
            "trajectory_or_control_loss_used": False,
            "density_uq_or_governor_used": False,
            "native_glare_used_for_training": False,
            "locked_test_read": False,
            "route203_native_glare_submission": False,
        },
        "claim_boundary": (
            "Bounded 17-event Phase-A engineering repair only; no formal "
            "Stage2-L, planning, closed-loop, generalization, or safety claim."
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
                "optimizer_steps": completed_steps,
                "stop_reason": stop_reason,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del lm, relevance_queries, relevance_head
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
