#!/usr/bin/env python3
"""One bounded Stage2-L v12.1 factorized-R-only engineering smoke."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_stage2l_v121_factorized_r_preflight import (
    component_support_statistics,
)
from scripts.scenario_factory_lib import sha256_file
from uq_estimator.stage2l_factorized_relevance_v121 import (
    COMPONENT_ORDER,
    FactorizedTaskRelevanceMapHead,
    factorized_relevance_terms_v121,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRelevanceMapHead,
    ViewAlignedTaskRelevanceQueryTokenizer,
)


SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_launch.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
EXPECTED_RECORDS = 1600
EXPECTED_GROUPS = 80
EXPECTED_EVENTS = 17


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


def _v101():
    """Import the heavy ORION/CARLA training stack only when it is needed."""

    import scripts.train_stage2l_v101_view_aligned_phase_a as module

    return module


class FactorizedAssets:
    """Extend frozen v10.1 clean assets with exact route/actor targets."""

    def __init__(self, manifest_path: Path, feature_cache_path: Path) -> None:
        v101 = _v101()
        legacy = v101.PhaseAAssets(manifest_path, feature_cache_path)
        self.__dict__.update(legacy.__dict__)
        self.component_relevance: Dict[str, torch.Tensor] = {}
        self.consumer_union_noncommutativity: Dict[str, float] = {}
        for group_id in sorted(self.group_rows):
            row = self.row(group_id, "observed", "task_relevance")
            reference = row["provenance"]["relevance_supervision"]
            path = v101.resolve_reference(
                reference,
                self.records_path.parent,
                "factorized R target for %s" % group_id,
            )
            with np.load(path, allow_pickle=False) as archive:
                union = np.asarray(archive["relevance"], dtype=np.float32)
                route = np.asarray(archive["route_corridor"], dtype=np.float32)
                actor = np.asarray(
                    archive["relevant_actor_support"], dtype=np.float32
                )
            if any(value.shape != (6, 40, 40) for value in (union, route, actor)):
                raise ValueError("factorized R raw target shape differs")
            if not np.array_equal(union, np.maximum(route, actor)):
                raise ValueError("factorized R union differs from component maximum")
            components = torch.from_numpy(np.stack((route, actor), axis=0))
            pooled = F.adaptive_avg_pool2d(components, (10, 10)).unsqueeze(0)
            pooled_raw_union = F.adaptive_avg_pool2d(
                torch.from_numpy(union).unsqueeze(0), (10, 10)
            )
            if not torch.equal(pooled_raw_union, self.relevance[group_id]):
                raise ValueError("factorized R stored consumer union differs")
            self.consumer_union_noncommutativity[group_id] = float(
                torch.max(
                    torch.abs(
                        torch.maximum(pooled[:, 0], pooled[:, 1])
                        - pooled_raw_union
                    )
                ).item()
            )
            self.component_relevance[group_id] = pooled

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


def copy_single_head_to_factorized(
    factorized: FactorizedTaskRelevanceMapHead,
    single_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Copy the v10.1 single head into two initially identical branches."""

    expected = {
        "net.0.weight",
        "net.0.bias",
        "net.1.weight",
        "net.1.bias",
        "net.3.weight",
        "net.3.bias",
    }
    if set(single_state) != expected:
        raise ValueError("single R head state differs from expected architecture")
    target = factorized.state_dict()
    mapping = {
        "shared.0.weight": "net.0.weight",
        "shared.0.bias": "net.0.bias",
        "shared.1.weight": "net.1.weight",
        "shared.1.bias": "net.1.bias",
        "route_output.weight": "net.3.weight",
        "route_output.bias": "net.3.bias",
        "actor_output.weight": "net.3.weight",
        "actor_output.bias": "net.3.bias",
    }
    if set(target) != set(mapping):
        raise ValueError("factorized R head parameter set differs")
    copied = {}
    for destination, source in mapping.items():
        value = single_state[source]
        if tuple(target[destination].shape) != tuple(value.shape):
            raise ValueError("factorized R warm-start shape differs")
        copied[destination] = value.detach().clone()
    factorized.load_state_dict(copied, strict=True)
    return {
        "source_tensor_count": len(single_state),
        "destination_tensor_count": len(copied),
        "route_actor_outputs_initially_identical": True,
    }


def _load_warm_start(
    *,
    lm,
    relevance_queries: ViewAlignedTaskRelevanceQueryTokenizer,
    relevance_head: FactorizedTaskRelevanceMapHead,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if (
        payload.get("schema") != "orion.stage2l_v101_view_aligned_phase_a.v1"
        or payload.get("status") != "phase_a_failed_gate"
        or payload.get("optimizer_steps") != 120
        or payload.get("formal_stage2l_ready") is not False
        or payload.get("stage2p_ready") is not False
        or payload.get("stage1_uq_loaded") is not False
    ):
        raise ValueError("factorized R warm-start checkpoint differs")
    lm_state = lm.state_dict()
    lora = payload.get("lora", {})
    if {key for key in lm_state if "lora_" in key} != set(lora):
        raise ValueError("factorized R LoRA warm-start keys differ")
    result = lm.load_state_dict(lora, strict=False)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("factorized R LoRA warm-start incomplete")
    relevance_queries.load_state_dict(
        payload["view_aligned_relevance_queries"], strict=True
    )
    head = copy_single_head_to_factorized(
        relevance_head, payload["relevance_head"]
    )
    return {
        "source_schema": payload["schema"],
        "source_status": payload["status"],
        "source_optimizer_steps": int(payload["optimizer_steps"]),
        "copied_lora_tensor_count": len(lora),
        "copied_query_tensor_count": len(
            payload["view_aligned_relevance_queries"]
        ),
        **head,
    }


def _map_logits(
    *,
    lm,
    text_tokenizer,
    relevance_queries,
    relevance_head,
    assets: FactorizedAssets,
    group_id: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v101 = _v101()
    baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
    features = assets.view_features[group_id].cuda(non_blocking=True)
    target = assets.component_relevance[group_id].cuda(non_blocking=True)
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
    return relevance_head(query_grid).component_logits, target


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(np.mean(values)) if values else None


def _support_truth(target: np.ndarray, fraction: float) -> np.ndarray:
    peaks = target.reshape(target.shape[0], -1).max(axis=1)
    shape = (target.shape[0],) + (1,) * (target.ndim - 1)
    return (target >= peaks.reshape(shape) * float(fraction)) & (
        peaks.reshape(shape) > 0.0
    )


def _average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    scores = np.asarray(scores).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    positives = int(truth.sum())
    if positives <= 0:
        raise ValueError("average precision requires positives")
    order = np.argsort(-scores, kind="mergesort")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def summarize_factorized_predictions(
    *,
    group_ids: Sequence[str],
    logits: torch.Tensor,
    targets: torch.Tensor,
    union_targets: torch.Tensor,
    assets: FactorizedAssets,
    support_fraction: float,
    supported_cells: Sequence[str],
) -> Dict[str, Any]:
    probability = logits.sigmoid().detach().float().cpu().numpy()
    target_np = targets.detach().float().cpu().numpy()
    union_target_np = union_targets.detach().float().cpu().numpy()
    if (
        probability.shape != target_np.shape
        or probability.ndim != 5
        or union_target_np.shape != (probability.shape[0], 6, 10, 10)
    ):
        raise ValueError("factorized evaluation tensor shape differs")
    rows: Dict[str, Dict[str, Any]] = {}
    buckets: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
        lambda: {"recall": [], "background_fpr": [], "events": [], "groups": []}
    )
    per_event: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        components = {}
        for component_index, component in enumerate(COMPONENT_ORDER):
            stats = component_support_statistics(
                target_np[index, component_index],
                probability[index, component_index],
                support_fraction=support_fraction,
            )
            components[component] = stats
            for view, value in stats.items():
                bucket = buckets[(component, view)]
                bucket["background_fpr"].append(
                    float(value["background_false_positive_rate"])
                )
                if value["positive"]:
                    bucket["recall"].append(float(value["foreground_recall"]))
                    bucket["events"].append(assets.group_event[group_id])
                    bucket["groups"].append(group_id)
        row = {
            "event_id": assets.group_event[group_id],
            "components": components,
        }
        rows[group_id] = row
        per_event[assets.group_event[group_id]].append(row)

    per_component_view: Dict[str, Dict[str, Any]] = {}
    for component in COMPONENT_ORDER:
        per_component_view[component] = {}
        for view in CAMERA_ORDER:
            value = buckets[(component, view)]
            per_component_view[component][view] = {
                "positive_group_count": len(value["groups"]),
                "positive_event_count": len(set(value["events"])),
                "mean_group_foreground_recall": _mean(value["recall"]),
                "mean_group_background_false_positive_rate": _mean(
                    value["background_fpr"]
                ),
            }

    component_ap = {}
    for component_index, component in enumerate(COMPONENT_ORDER):
        truth = _support_truth(target_np[:, component_index], support_fraction)
        component_ap[component] = _average_precision(
            probability[:, component_index], truth
        )
    union_probability = np.maximum(probability[:, 0], probability[:, 1])
    union_truth = _support_truth(union_target_np, support_fraction)

    supported = []
    for name in supported_cells:
        component, view = name.split("/", 1)
        value = per_component_view[component][view]
        if value["mean_group_foreground_recall"] is None:
            raise ValueError("supported factorized cell has no positive group")
        supported.append(float(value["mean_group_foreground_recall"]))
    actor_nonfront = [
        float(per_component_view["actor"][view]["mean_group_foreground_recall"])
        for view in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
        if "actor/%s" % view in supported_cells
    ]
    event_metrics = {}
    for event_id, event_rows in sorted(per_event.items()):
        active_recalls = []
        for row in event_rows:
            for component in COMPONENT_ORDER:
                for value in row["components"][component].values():
                    if value["positive"]:
                        active_recalls.append(float(value["foreground_recall"]))
        event_metrics[event_id] = {
            "group_count": len(event_rows),
            "mean_active_component_view_recall": _mean(active_recalls),
        }
    return {
        "group_count": len(group_ids),
        "component_average_precision": component_ap,
        "derived_union_average_precision": _average_precision(
            union_probability, union_truth
        ),
        "supported_component_view_macro_recall": _mean(supported),
        "actor_nonfront_macro_recall": _mean(actor_nonfront),
        "per_component_view": per_component_view,
        "per_event": event_metrics,
        "per_group": rows,
    }


@torch.no_grad()
def _evaluate(
    *,
    split: str,
    lm,
    text_tokenizer,
    relevance_queries,
    relevance_head,
    assets: FactorizedAssets,
    protocol: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
    lm.eval()
    relevance_queries.eval()
    relevance_head.eval()
    group_ids = assets.groups_for_split(split)
    logits_all = []
    targets_all = []
    union_targets_all = []
    losses = []
    raw = {}
    for group_id in group_ids:
        logits, target = _map_logits(
            lm=lm,
            text_tokenizer=text_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            group_id=group_id,
        )
        terms = factorized_relevance_terms_v121(
            logits,
            target,
            support_fraction_of_peak=float(protocol["support_fraction_of_peak"]),
            inactive_view_background_anchor_weight=float(
                protocol["objective"]["inactive_view_background_anchor_weight"]
            ),
            empty_component_background_anchor_weight=float(
                protocol["objective"]["empty_component_background_anchor_weight"]
            ),
            route_component_weight=float(
                protocol["objective"]["route_component_weight"]
            ),
            actor_component_weight=float(
                protocol["objective"]["actor_component_weight"]
            ),
        )
        probabilities = logits.sigmoid().float().cpu()
        target_cpu = target.float().cpu()
        union_target_cpu = assets.relevance[group_id].float().cpu()
        raw[group_id] = {
            "component_probability": probabilities,
            "component_target": target_cpu,
            "derived_union_probability": torch.maximum(
                probabilities[:, 0], probabilities[:, 1]
            ),
            "pooled_raw_union_target": union_target_cpu,
            "max_pooled_component_target": torch.maximum(
                target_cpu[:, 0], target_cpu[:, 1]
            ),
        }
        logits_all.append(logits)
        targets_all.append(target)
        union_targets_all.append(assets.relevance[group_id].cuda(non_blocking=True))
        losses.append(float(terms.loss.item()))
    logits_tensor = torch.cat(logits_all, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)
    union_targets_tensor = torch.cat(union_targets_all, dim=0)
    summary = summarize_factorized_predictions(
        group_ids=group_ids,
        logits=logits_tensor,
        targets=targets_tensor,
        union_targets=union_targets_tensor,
        assets=assets,
        support_fraction=float(protocol["support_fraction_of_peak"]),
        supported_cells=protocol["supported_component_views"],
    )
    summary.update({"split": split, "mean_factorized_loss": float(np.mean(losses))})
    return summary, raw


def factorized_gate(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> Dict[str, bool]:
    train = metrics["train"]
    dev = metrics["dev"]
    dev_actor = dev["per_component_view"]["actor"]
    nonfront_views = ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
    nonfront_recalls = [
        float(dev_actor[view]["mean_group_foreground_recall"])
        for view in nonfront_views
    ]
    background_values = []
    for component_view in gates["background_fpr_cells"]:
        component, view = component_view.split("/", 1)
        background_values.append(
            float(
                dev["per_component_view"][component][view][
                    "mean_group_background_false_positive_rate"
                ]
            )
        )
    return {
        "train_supported_macro_recall": float(
            train["supported_component_view_macro_recall"]
        )
        >= float(gates["train_min_supported_macro_recall"]),
        "dev_route_front_retained": float(
            dev["per_component_view"]["route"]["CAM_FRONT"][
                "mean_group_foreground_recall"
            ]
        )
        >= float(gates["dev_min_route_front_recall"]),
        "dev_actor_front": float(
            dev_actor["CAM_FRONT"]["mean_group_foreground_recall"]
        )
        >= float(gates["dev_min_actor_front_recall"]),
        "dev_actor_nonfront_macro": float(dev["actor_nonfront_macro_recall"])
        >= float(gates["dev_min_actor_nonfront_macro_recall"]),
        "dev_actor_nonfront_each_positive": min(nonfront_recalls)
        >= float(gates["dev_min_each_actor_nonfront_recall"]),
        "dev_background_fpr": max(background_values)
        <= float(gates["dev_max_mean_background_fpr"]),
        "dev_actor_nonfront_absolute_improvement": float(
            dev["actor_nonfront_macro_recall"]
        )
        - float(gates["baseline_dev_actor_nonfront_macro_recall"])
        >= float(gates["minimum_actor_nonfront_absolute_improvement"]),
    }


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "factorized_module_sha256": _sha256(
            PROJECT_ROOT / "uq_estimator/stage2l_factorized_relevance_v121.py"
        ),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "view_feature_cache_sha256": _sha256(args.view_feature_cache.resolve()),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "v101_checkpoint_sha256": _sha256(args.v101_checkpoint.resolve()),
        "v101_report_sha256": _sha256(args.v101_report.resolve()),
        "factorized_cpu_report_sha256": _sha256(
            args.factorized_cpu_report.resolve()
        ),
    }


def _protocol_checks(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    objective = protocol.get("objective", {})
    milestones = protocol.get("evaluation_milestones")
    maximum_steps = protocol.get("maximum_optimizer_steps")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen_bounded_r_only_protocol_launch_locked"
        or protocol.get("validated_inputs") != _validated_inputs(args)
        or protocol.get("output_root") != str(args.output_dir.resolve())
        or protocol.get("camera_order") != list(CAMERA_ORDER)
        or maximum_steps not in (20, 40, 80)
        or milestones != [20, 40]
        or milestones[-1] != maximum_steps
        or objective.get("base") != "factorized soft-target Brier"
        or float(objective.get("route_component_weight", -1.0)) != 0.5
        or float(objective.get("actor_component_weight", -1.0)) != 0.5
        or float(objective.get("derived_union_loss_weight", -1.0)) != 0.0
        or protocol.get("supported_component_views")
        != [
            "route/CAM_FRONT",
            "actor/CAM_FRONT",
            "actor/CAM_FRONT_LEFT",
            "actor/CAM_FRONT_RIGHT",
            "actor/CAM_BACK",
            "actor/CAM_BACK_LEFT",
        ]
    ):
        raise ValueError("factorized R training protocol differs")
    locks = protocol.get("locks", {})
    if any(
        locks.get(key) is not False
        for key in (
            "stage1_uq_input",
            "u_tokenizer",
            "language_training",
            "trajectory_or_control",
            "formal_stage2l",
            "stage2p",
            "closed_loop",
            "locked_test_read",
        )
    ):
        raise ValueError("factorized R protocol expands locked scope")


def _cpu_warm_start_identity(checkpoint_path: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if (
        payload.get("schema") != "orion.stage2l_v101_view_aligned_phase_a.v1"
        or payload.get("status") != "phase_a_failed_gate"
        or payload.get("optimizer_steps") != 120
    ):
        raise ValueError("factorized R CPU warm-start checkpoint differs")
    single = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256)
    single.load_state_dict(payload["relevance_head"], strict=True)
    factorized = FactorizedTaskRelevanceMapHead(model_dim=4096, hidden_dim=256)
    copy = copy_single_head_to_factorized(factorized, payload["relevance_head"])
    torch.manual_seed(0)
    grid = torch.randn(1, 2, 2, 2, 4096)
    with torch.no_grad():
        single_probability = single(grid).sigmoid()
        output = factorized(grid)
    exact = bool(
        torch.equal(single_probability, output.route_probability)
        and torch.equal(single_probability, output.actor_probability)
        and torch.equal(single_probability, output.derived_union_probability)
    )
    if not exact:
        raise ValueError("factorized R warm-start does not preserve single R")
    return {**copy, "single_to_route_actor_union_probability_bitwise_exact": exact}


def _preflight(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    assets: FactorizedAssets,
) -> Dict[str, Any]:
    v101 = _v101()
    if (
        len(assets.event_meta) != EXPECTED_EVENTS
        or len(assets.group_rows) != EXPECTED_GROUPS
        or len(assets.component_relevance) != EXPECTED_GROUPS
        or len(assets.groups_for_split("train")) != 60
        or len(assets.groups_for_split("dev")) != 20
    ):
        raise ValueError("factorized R preflight asset counts differ")
    sampler = v101.base.EventBalancedSampler(
        assets.event_groups["train"], seed=args.seed
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "factorized_r_smoke_preflight_pass_training_locked",
        "passed": True,
        "gpu_used": False,
        "training_started": False,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "output_root": str(args.output_dir.resolve()),
        "events": len(assets.event_meta),
        "groups": len(assets.group_rows),
        "train_events": sorted(assets.event_groups["train"]),
        "dev_events": sorted(assets.event_groups["dev"]),
        "first_two_event_balanced_units": [sampler.next(), sampler.next()],
        "warm_start_identity": _cpu_warm_start_identity(
            args.v101_checkpoint.resolve()
        ),
        "all_component_targets_loaded": True,
        "consumer_union_pool_max_noncommutativity": {
            "nonzero_group_count": sum(
                value > 0.0
                for value in assets.consumer_union_noncommutativity.values()
            ),
            "maximum_absolute_difference": max(
                assets.consumer_union_noncommutativity.values()
            ),
            "old_pooled_raw_union_is_training_target": False,
            "max_pooled_components_is_training_target": False,
            "both_are_report_only": True
        },
        "locks": dict(protocol["locks"]),
    }


def _validate_launch(args: argparse.Namespace) -> None:
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
        raise ValueError("factorized R trainer preflight is absent or stale")
    authorized = launch.get("authorized_run", {})
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status") != "immutable_single_factorized_r_only_authorization"
        or launch.get("validated_inputs") != _validated_inputs(args)
        or launch.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or launch.get("preflight_sha256")
        != _sha256(args.trainer_preflight.resolve())
        or authorized.get("output_root") != str(args.output_dir.resolve())
        or authorized.get("maximum_submissions") != 1
        or authorized.get("automatic_retry") is not False
        or authorized.get("maximum_optimizer_steps")
        != int(_read_json(args.training_protocol)["maximum_optimizer_steps"])
    ):
        raise ValueError("factorized R launch amendment is absent or stale")


def _checkpoint(
    *,
    step: int,
    status: str,
    lm,
    relevance_queries,
    relevance_head,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "optimizer_steps": int(step),
        "lora": {
            key: value.detach().cpu()
            for key, value in lm.state_dict().items()
            if "lora_" in key
        },
        "view_aligned_relevance_queries": {
            key: value.detach().cpu()
            for key, value in relevance_queries.state_dict().items()
        },
        "factorized_relevance_head": {
            key: value.detach().cpu()
            for key, value in relevance_head.state_dict().items()
        },
        "stage1_uq_loaded": False,
        "u_tokenizer_loaded": False,
        "language_training_used": False,
        "trajectory_or_control_loss_used": False,
        "provenance": dict(provenance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--v101-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-report", type=Path, required=True)
    parser.add_argument("--factorized-cpu-report", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.dataset_manifest,
        args.view_feature_cache,
        args.v101_checkpoint,
        args.v101_report,
        args.factorized_cpu_report,
        args.training_protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("factorized R prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite factorized R output")
    protocol = _read_json(args.training_protocol.resolve())
    _protocol_checks(args, protocol)
    v101 = _v101()
    v101._configure_base()
    assets = FactorizedAssets(
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
        raise ValueError("real factorized R smoke requires preflight and launch")
    _validate_launch(args)
    if not torch.cuda.is_available():
        raise RuntimeError("real factorized R smoke requires CUDA")

    from mmcv.utils import set_random_seed

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    lm, text_tokenizer = v101.base._load_orion_lm(
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
    warm_start = _load_warm_start(
        lm=lm,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        checkpoint_path=args.v101_checkpoint.resolve(),
    )
    before = {}
    for split in ("train", "dev"):
        before[split], _ = _evaluate(
            split=split,
            lm=lm,
            text_tokenizer=text_tokenizer,
            relevance_queries=relevance_queries,
            relevance_head=relevance_head,
            assets=assets,
            protocol=protocol,
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
    sampler = v101.base.EventBalancedSampler(
        assets.event_groups["train"], seed=args.seed
    )
    maximum_steps = int(protocol["maximum_optimizer_steps"])
    milestones = tuple(map(int, protocol["evaluation_milestones"]))
    history = []
    evaluations = []
    completed_steps = 0
    stop_reason = "maximum_steps_reached"
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
            terms = factorized_relevance_terms_v121(
                logits,
                target,
                support_fraction_of_peak=float(protocol["support_fraction_of_peak"]),
                inactive_view_background_anchor_weight=float(
                    protocol["objective"]["inactive_view_background_anchor_weight"]
                ),
                empty_component_background_anchor_weight=float(
                    protocol["objective"]["empty_component_background_anchor_weight"]
                ),
                route_component_weight=float(
                    protocol["objective"]["route_component_weight"]
                ),
                actor_component_weight=float(
                    protocol["objective"]["actor_component_weight"]
                ),
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
        if not finite:
            raise RuntimeError("factorized R non-finite gradient fail-fast")
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
            print("[Stage2LV121] " + json.dumps(item, sort_keys=True), flush=True)
        if step not in milestones:
            continue
        metrics = {}
        raw = {}
        for split in ("train", "dev"):
            metrics[split], current = _evaluate(
                split=split,
                lm=lm,
                text_tokenizer=text_tokenizer,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                assets=assets,
                protocol=protocol,
            )
            raw.update(current)
        checks = factorized_gate(metrics, protocol["engineering_gates"])
        passed = all(checks.values())
        evaluations.append(
            {"optimizer_step": step, "metrics": metrics, "checks": checks, "passed": passed}
        )
        torch.save(
            _checkpoint(
                step=step,
                status="factorized_r_gate_pass" if passed else "factorized_r_gate_failed",
                lm=lm,
                relevance_queries=relevance_queries,
                relevance_head=relevance_head,
                provenance=provenance,
            ),
            args.output_dir / ("factorized_r_step%03d.pt" % step),
        )
        torch.save(raw, args.output_dir / ("spatial_maps_step%03d.pt" % step))
        print(
            "[Stage2LV121Eval] "
            + json.dumps(
                {
                    "step": step,
                    "passed": passed,
                    "train_macro": metrics["train"]["supported_component_view_macro_recall"],
                    "dev_actor_nonfront_macro": metrics["dev"]["actor_nonfront_macro_recall"],
                    "checks": checks,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if passed:
            stop_reason = "factorized_r_engineering_gate_passed_early"
            break
        if len(evaluations) >= 2:
            previous = evaluations[-2]["metrics"]
            current = evaluations[-1]["metrics"]
            dev_drop = float(previous["dev"]["actor_nonfront_macro_recall"]) - float(
                current["dev"]["actor_nonfront_macro_recall"]
            )
            train_gain = float(current["train"]["supported_component_view_macro_recall"]) - float(
                previous["train"]["supported_component_view_macro_recall"]
            )
            if (
                dev_drop >= float(protocol["early_stop"]["minimum_dev_macro_drop"])
                and train_gain >= float(protocol["early_stop"]["minimum_train_macro_gain"])
            ):
                stop_reason = "clear_factorized_train_dev_overfit_early_stop"
                break

    del optimizer
    final = evaluations[-1]
    report = {
        "schema": SCHEMA,
        "status": (
            "factorized_r_engineering_gate_passed"
            if final["passed"]
            else "factorized_r_stopped_without_gate_pass"
        ),
        "optimizer_steps": completed_steps,
        "stop_reason": stop_reason,
        "before": before,
        "history": history,
        "evaluations": evaluations,
        "final_metrics": final["metrics"],
        "final_checks": final["checks"],
        "warm_start": warm_start,
        "provenance": provenance,
        "locks": {
            "stage1_uq_loaded": False,
            "u_tokenizer_loaded": False,
            "language_training_used": False,
            "trajectory_or_control_loss_used": False,
            "locked_test_read": False,
            "formal_stage2l_ready": False,
            "stage2p_ready": False,
        },
        "claim_boundary": "One bounded 17-event factorized-R engineering smoke; no semantic-U, language, planning, closed-loop, formal generalization or safety claim.",
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
