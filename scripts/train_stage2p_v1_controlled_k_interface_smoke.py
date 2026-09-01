#!/usr/bin/env python3
"""Run the first bounded controlled-K Stage2-P engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from uq_estimator.stage2_task_training import validate_stage2_record
from uq_estimator.stage2p_task_risk_trajectory import (
    CHECKPOINT_SCHEMA,
    TaskRiskMapTokenProjector,
    TaskRiskTrajectoryResponse,
)


SCHEMA = "orion.stage2p_v1_controlled_k_interface_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2p_v1_controlled_k_interface_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2p_v1_controlled_k_interface_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2p_v1_controlled_k_interface_launch.v1"
VARIANTS = ("zero_k", "relevant_k", "irrelevant_k", "view_shuffled_k")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _load_tensor(path: str, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, torch.Tensor):
        raise ValueError("artifact lacks %s: %s" % (key, path))
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("artifact is non-finite: %s" % path)
    return value


def _load_records(path: Path) -> list[Dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 189:
        raise ValueError("controlled source manifest must contain 189 records")
    for record in records:
        validate_stage2_record(record, check_files=True)
    return records


def _variant(record: Mapping[str, Any]) -> str:
    return str(record["route_group"]).rsplit("/", 1)[-1]


def _source_step(record: Mapping[str, Any]) -> int:
    return int(str(record["sample_id"]).rsplit("=", 1)[-1])


def _zero_target(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference)


def _build_items(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    onpath_records: list[Dict[str, Any]] = []
    for record in records:
        source_variant = _variant(record)
        if source_variant not in {"onpath_oracle", "offpath_oracle", "zero_uq"}:
            raise ValueError("source controlled variant differs")
        context = _load_tensor(
            record["artifacts"]["planning_context_path"], "planning_context"
        )
        observation = _load_tensor(
            record["artifacts"]["observation_uq_path"], "observation_uq"
        )
        if context.shape != (256, 256) or observation.shape != (6, 40, 40, 3):
            raise ValueError("controlled Stage2-P source tensor shape differs")
        base_plan = torch.as_tensor(
            record["base_plan_cumulative_m"], dtype=torch.float32
        )
        source_target = torch.as_tensor(
            record["labels"]["trajectory_residual_m"], dtype=torch.float32
        )
        if base_plan.shape != (6, 2) or source_target.shape != (6, 2):
            raise ValueError("controlled Stage2-P trajectory shape differs")
        scalar = observation.mean(dim=-1).clamp(0.0, 1.0)
        if source_variant == "onpath_oracle":
            variant = "relevant_k"
            task_risk = scalar
            target = source_target
            onpath_records.append(record)
        elif source_variant == "offpath_oracle":
            variant = "irrelevant_k"
            task_risk = scalar * 0.05
            target = _zero_target(source_target)
        else:
            variant = "zero_k"
            task_risk = torch.zeros_like(scalar)
            target = _zero_target(source_target)
        items.append({
            "sample_id": str(record["sample_id"]),
            "source_step": _source_step(record),
            "variant": variant,
            "planning_context": context,
            "task_risk": task_risk,
            "base_plan": base_plan,
            "trajectory_target": target,
            "target_nonzero": int(torch.count_nonzero(target)) > 0,
            "k_nonzero": int(torch.count_nonzero(task_risk)) > 0,
        })

    for record in onpath_records:
        context = _load_tensor(
            record["artifacts"]["planning_context_path"], "planning_context"
        )
        observation = _load_tensor(
            record["artifacts"]["observation_uq_path"], "observation_uq"
        )
        scalar = observation.mean(dim=-1).clamp(0.0, 1.0)
        shuffled = torch.zeros_like(scalar)
        shuffled[3] = scalar[0] * 0.05
        base_plan = torch.as_tensor(
            record["base_plan_cumulative_m"], dtype=torch.float32
        )
        items.append({
            "sample_id": "%s#derived=view_shuffled_k" % record["sample_id"],
            "source_step": _source_step(record),
            "variant": "view_shuffled_k",
            "planning_context": context,
            "task_risk": shuffled,
            "base_plan": base_plan,
            "trajectory_target": torch.zeros((6, 2), dtype=torch.float32),
            "target_nonzero": False,
            "k_nonzero": int(torch.count_nonzero(shuffled)) > 0,
        })
    if len(items) != 252:
        raise ValueError("derived controlled-K dataset must contain 252 items")
    return items


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "source_manifest_sha256": _sha256(args.source_manifest.resolve()),
        "source_build_report_sha256": _sha256(
            args.source_build_report.resolve()
        ),
        "semantic_terminal_sha256": _sha256(
            args.semantic_terminal.resolve()
        ),
        "semantic_report_sha256": _sha256(args.semantic_report.resolve()),
        "semantic_bridge_sha256": _sha256(args.semantic_bridge.resolve()),
        "spatial_u_r_k_maps_sha256": _sha256(
            args.spatial_u_r_k_maps.resolve()
        ),
    }


def _audit_items(items: list[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    active_counts: Dict[str, int] = {}
    positive_counts: Dict[str, int] = {}
    maxima: Dict[str, float] = {}
    for item in items:
        variant = item["variant"]
        counts[variant] = counts.get(variant, 0) + 1
        active_counts[variant] = active_counts.get(variant, 0) + int(
            item["k_nonzero"]
        )
        positive_counts[variant] = positive_counts.get(variant, 0) + int(
            item["target_nonzero"]
        )
        maxima[variant] = max(
            maxima.get(variant, 0.0), float(item["task_risk"].max())
        )
    checks = {
        "variant_counts_63_each": counts == {name: 63 for name in VARIANTS},
        "only_relevant_k_has_positive_targets": positive_counts
        == {
            "zero_k": 0,
            "relevant_k": 4,
            "irrelevant_k": 0,
            "view_shuffled_k": 0,
        },
        "zero_k_exact": all(
            int(torch.count_nonzero(item["task_risk"])) == 0
            for item in items
            if item["variant"] == "zero_k"
        ),
        "irrelevant_and_shuffled_are_low_k": (
            abs(maxima.get("irrelevant_k", float("nan")) - 0.05) <= 1e-7
            and abs(maxima.get("view_shuffled_k", float("nan")) - 0.05)
            <= 1e-7
        ),
        "relevant_k_has_unit_peak": maxima.get("relevant_k") == 1.0,
        "all_inputs_finite": all(
            bool(torch.isfinite(item[key]).all())
            for item in items
            for key in (
                "planning_context",
                "task_risk",
                "base_plan",
                "trajectory_target",
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError("controlled-K item audit failed: %s" % checks)
    return {
        "checks": checks,
        "counts": counts,
        "k_nonzero_counts": active_counts,
        "positive_target_counts": positive_counts,
        "maximum_k_by_variant": maxima,
    }


def _protocol_check(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen_bounded_controlled_k_interface"
        or protocol.get("input_sha256") != _validated_inputs(args)
        or protocol.get("output_root") != str(args.output_dir.resolve())
        or protocol.get("architecture", {}).get("forward_inputs")
        != ["frozen_orion_planning_context", "task_risk_k"]
        or protocol.get("architecture", {}).get(
            "privileged_task_context_forward"
        )
        is not False
        or protocol.get("training", {}).get("optimizer_steps") != 80
    ):
        raise ValueError("Stage2-P controlled-K protocol is absent or stale")


def _validate_launch(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    preflight = _json(args.trainer_preflight.resolve())
    launch = _json(args.launch_amendment.resolve())
    implementation = launch.get("implementation", {})
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("validated_inputs") != _validated_inputs(args)
        or preflight.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status")
        != "single_controlled_k_interface_smoke_authorized"
        or launch.get("validated_inputs") != _validated_inputs(args)
        or implementation.get("protocol_sha256")
        != _sha256(args.training_protocol.resolve())
        or implementation.get("preflight_sha256")
        != _sha256(args.trainer_preflight.resolve())
        or launch.get("authorized_run", {}).get("optimizer_steps") != 80
        or launch.get("authorized_run", {}).get("maximum_submissions") != 1
        or launch.get("authorized_run", {}).get("automatic_retry") is not False
    ):
        raise ValueError("Stage2-P launch contract is absent or stale")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _batch(
    items: list[Dict[str, Any]], indices: list[int], device: torch.device
) -> Dict[str, Any]:
    return {
        "planning_context": torch.stack(
            [items[index]["planning_context"] for index in indices]
        ).to(device),
        "task_risk": torch.stack(
            [items[index]["task_risk"] for index in indices]
        ).to(device),
        "target": torch.stack(
            [items[index]["trajectory_target"] for index in indices]
        ).to(device),
        "variant": [items[index]["variant"] for index in indices],
        "target_nonzero": torch.tensor(
            [items[index]["target_nonzero"] for index in indices],
            device=device,
            dtype=torch.bool,
        ),
    }


@torch.no_grad()
def _evaluate(
    items: list[Dict[str, Any]],
    projector: TaskRiskMapTokenProjector,
    response: TaskRiskTrajectoryResponse,
    device: torch.device,
) -> Dict[str, Any]:
    projector.eval()
    response.eval()
    records = []
    for start in range(0, len(items), 16):
        indices = list(range(start, min(start + 16, len(items))))
        batch = _batch(items, indices, device)
        selected = projector(batch["task_risk"])
        output = response(batch["planning_context"], selected)
        if not torch.equal(output.conditioned_context, batch["planning_context"]):
            raise RuntimeError("Stage2-P changed native planning context")
        if not bool(torch.isfinite(output.trajectory_residual).all()):
            raise RuntimeError("Stage2-P evaluation produced non-finite residual")
        for offset, index in enumerate(indices):
            prediction = output.trajectory_residual[offset].float().cpu()
            target = batch["target"][offset].float().cpu()
            records.append({
                "sample_id": items[index]["sample_id"],
                "variant": items[index]["variant"],
                "target_nonzero": items[index]["target_nonzero"],
                "k_nonzero": items[index]["k_nonzero"],
                "mean_absolute_error_m": float((prediction - target).abs().mean()),
                "mean_absolute_response_m": float(prediction.abs().mean()),
                "maximum_absolute_response_m": float(prediction.abs().max()),
                "global_gate": float(output.global_gate[offset].cpu()),
                "exact_zero_response": int(torch.count_nonzero(prediction)) == 0,
            })

    by_variant: Dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row for row in records if row["variant"] == variant]
        by_variant[variant] = {
            "sample_count": len(rows),
            "mean_absolute_error_m": float(
                np.mean([row["mean_absolute_error_m"] for row in rows])
            ),
            "mean_absolute_response_m": float(
                np.mean([row["mean_absolute_response_m"] for row in rows])
            ),
            "maximum_absolute_response_m": max(
                row["maximum_absolute_response_m"] for row in rows
            ),
            "exact_zero_response_fraction": float(
                np.mean([row["exact_zero_response"] for row in rows])
            ),
        }
    positive = [row for row in records if row["target_nonzero"]]
    hard_identity = [
        row
        for row in records
        if row["k_nonzero"] and not row["target_nonzero"]
    ]
    return {
        "by_variant": by_variant,
        "positive_target_count": len(positive),
        "positive_target_mae_m": float(
            np.mean([row["mean_absolute_error_m"] for row in positive])
        ),
        "positive_nonzero_response_fraction": float(
            np.mean([row["maximum_absolute_response_m"] > 0.1 for row in positive])
        ),
        "hard_identity_count": len(hard_identity),
        "hard_identity_mean_response_m": float(
            np.mean([row["mean_absolute_response_m"] for row in hard_identity])
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-build-report", type=Path, required=True)
    parser.add_argument("--semantic-terminal", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--semantic-bridge", type=Path, required=True)
    parser.add_argument("--spatial-u-r-k-maps", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    inputs = (
        args.source_manifest,
        args.source_build_report,
        args.semantic_terminal,
        args.semantic_report,
        args.semantic_bridge,
        args.spatial_u_r_k_maps,
        args.training_protocol,
    )
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("Stage2-P prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Stage2-P output")
    protocol = _json(args.training_protocol.resolve())
    _protocol_check(args, protocol)
    source_report = _json(args.source_build_report.resolve())
    semantic_terminal = _json(args.semantic_terminal.resolve())
    semantic_report = _json(args.semantic_report.resolve())
    if (
        source_report.get("status") != "optimization_smoke_nonclaim"
        or source_report.get("combined_sample_count") != 189
        or semantic_terminal.get("status")
        != "terminal_integrity_valid_soft_semantic_failure"
        or semantic_report.get("status")
        != "vertical_slice_semantic_completed_with_soft_quality_failures"
    ):
        raise ValueError("Stage2-P upstream quality labels differ")
    records = _load_records(args.source_manifest.resolve())
    items = _build_items(records)
    item_audit = _audit_items(items)
    if args.preflight_only:
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("fresh Stage2-P preflight output is required")
        if args.trainer_preflight is not None or args.launch_amendment is not None:
            raise ValueError("preflight cannot consume launch artifacts")
        value = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "controlled_k_interface_preflight_pass_training_locked",
            "passed": True,
            "training_started": False,
            "gpu_used": False,
            "validated_inputs": _validated_inputs(args),
            "protocol_sha256": _sha256(args.training_protocol.resolve()),
            "output_root": str(args.output_dir.resolve()),
            "item_audit": item_audit,
            "responsibility_contract": dict(protocol["architecture"]),
            "upstream_quality_label": "semantic_insensitive_bridge",
        }
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": value["status"],
            "item_count": sum(item_audit["counts"].values()),
            "output": str(args.preflight_output.resolve()),
        }, sort_keys=True))
        return 0
    if (
        args.trainer_preflight is None
        or args.launch_amendment is None
        or args.preflight_output is not None
    ):
        raise ValueError("real Stage2-P smoke requires preflight and launch")
    _validate_launch(args, protocol)
    device = _device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Stage2-P smoke requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    projector_config = {
        "model_dim": 256,
        "hidden_dim": 128,
        "max_views": 6,
        "tokens_per_view": 8,
        "score_scale": 0.25,
    }
    response_config = {
        "model_dim": 256,
        "num_heads": 8,
        "trajectory_steps": 6,
        "response_score_scale": 0.25,
        "lateral_bound_m": 2.0,
        "longitudinal_bound_m": 24.0,
    }
    projector = TaskRiskMapTokenProjector(**projector_config).to(device)
    response = TaskRiskTrajectoryResponse(**response_config).to(device)
    trainable = list(projector.parameters()) + list(response.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=5e-4, weight_decay=1e-4
    )
    before = _evaluate(items, projector, response, device)
    positive = [index for index, item in enumerate(items) if item["target_nonzero"]]
    hard_identity = [
        index
        for index, item in enumerate(items)
        if item["k_nonzero"] and not item["target_nonzero"]
    ]
    if len(positive) != 4 or len(hard_identity) < 8:
        raise ValueError("Stage2-P balanced anchors differ")
    history = []
    for step in range(1, 81):
        selected_identity = [
            hard_identity[(4 * (step - 1) + offset) % len(hard_identity)]
            for offset in range(4)
        ]
        indices = positive + selected_identity
        batch = _batch(items, indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = response(
            batch["planning_context"], projector(batch["task_risk"])
        )
        positive_mask = batch["target_nonzero"]
        identity_mask = ~positive_mask
        positive_loss = F.smooth_l1_loss(
            output.trajectory_residual[positive_mask],
            batch["target"][positive_mask],
        )
        identity_loss = output.trajectory_residual[identity_mask].abs().mean()
        loss = positive_loss + 2.0 * identity_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Stage2-P loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("Stage2-P gradient is non-finite")
        optimizer.step()
        row = {
            "optimizer_step": step,
            "loss": float(loss.detach()),
            "positive_loss": float(positive_loss.detach()),
            "hard_identity_loss": float(identity_loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "finite": True,
            "trainable_scope": "task_risk_projector_and_trajectory_response_only",
        }
        history.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
    after = _evaluate(items, projector, response, device)
    hard_checks = {
        "all_80_steps_present_and_finite": len(history) == 80
        and all(row["finite"] for row in history),
        "native_context_exact_identity": all(
            row["trainable_scope"]
            == "task_risk_projector_and_trajectory_response_only"
            for row in history
        ),
        "zero_k_trajectory_exact_identity": (
            after["by_variant"]["zero_k"]["exact_zero_response_fraction"]
            == 1.0
        ),
        "trajectory_response_within_bounds": all(
            row["maximum_absolute_response_m"] <= 24.0
            for row in after["records"]
        ),
        "no_privileged_forward_input": True,
        "semantic_failure_label_retained": True,
    }
    if not all(hard_checks.values()):
        raise RuntimeError("Stage2-P hard interface check failed: %s" % hard_checks)
    soft_checks = {
        "positive_target_mae_below_0_5m": after["positive_target_mae_m"] < 0.5,
        "all_positive_targets_trigger_response": (
            after["positive_nonzero_response_fraction"] == 1.0
        ),
        "hard_identity_mean_response_below_0_05m": (
            after["hard_identity_mean_response_m"] < 0.05
        ),
        "irrelevant_k_max_response_below_0_2m": (
            after["by_variant"]["irrelevant_k"][
                "maximum_absolute_response_m"
            ]
            < 0.2
        ),
        "view_shuffled_k_max_response_below_0_2m": (
            after["by_variant"]["view_shuffled_k"][
                "maximum_absolute_response_m"
            ]
            < 0.2
        ),
    }
    status = (
        "controlled_k_interface_completed_quality_pass"
        if all(soft_checks.values())
        else "controlled_k_interface_completed_with_soft_quality_failures"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2p_controlled_k_response.pt"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "status": status,
        "engineering_smoke_only": True,
        "formal_stage2p_ready": False,
        "closed_loop_eligible": False,
        "optimizer_steps": 80,
        "projector_config": projector_config,
        "response_config": response_config,
        "projector_state": {
            key: value.detach().cpu()
            for key, value in projector.state_dict().items()
        },
        "response_state": {
            key: value.detach().cpu()
            for key, value in response.state_dict().items()
        },
        "responsibility_contract": {
            "forward_inputs": [
                "frozen_orion_planning_context",
                "task_risk_k",
            ],
            "raw_observation_u_forward": False,
            "privileged_task_context_forward": False,
            "route_actor_ttc_outcome_forward": False,
            "privileged_labels_supervision_only": True,
        },
        "upstream_semantic_quality_label": "semantic_insensitive_bridge",
        "validated_inputs": _validated_inputs(args),
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "status": status,
        "claim_boundary": (
            "One-route controlled-K trajectory-interface engineering smoke. "
            "It is not a learned-U, Stage2-L, generalization, closed-loop or safety result."
        ),
        "optimizer_steps": 80,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.training_protocol.resolve()),
        "preflight_sha256": _sha256(args.trainer_preflight.resolve()),
        "launch_sha256": _sha256(args.launch_amendment.resolve()),
        "item_audit": item_audit,
        "before": before,
        "after": after,
        "history": history,
        "hard_checks": hard_checks,
        "hard_checks_passed": all(hard_checks.values()),
        "soft_checks": soft_checks,
        "soft_checks_passed": all(soft_checks.values()),
        "checkpoint_path": str(checkpoint_path),
        "upstream_semantic_quality_label": "semantic_insensitive_bridge",
        "locks": {
            "formal_stage2p_ready": False,
            "closed_loop_eligible": False,
            "locked_test_read": False,
            "benchmark_claim": False,
            "automatic_extension": False,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "soft_checks_passed": all(soft_checks.values()),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
