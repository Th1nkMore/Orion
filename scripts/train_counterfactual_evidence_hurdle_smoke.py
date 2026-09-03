#!/usr/bin/env python3
"""Run the route-only sparse hurdle-head evidence smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    CLAIM_BOUNDARY,
    EVIDENCE_COMPONENTS,
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_compaction import (  # noqa: E402
    deterministic_rademacher_projection,
    projection_sha256,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    evaluate_evidence_records,
    evaluate_hurdle_diagnostics,
    fit_train_component_scales,
    records_from_counterfactual_shard,
    run_hurdle_evidence_epoch,
    select_records,
)
from uq_estimator.observation_uq_shard import load_feature_shard  # noqa: E402


SCHEMA_VERSIONS = {
    "orion.counterfactual-evidence-hurdle-smoke/v1",
    "orion.counterfactual-evidence-high-support-hurdle-smoke/v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-shard", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--spatial-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite hurdle-head smoke")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA hurdle-head smoke requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") not in SCHEMA_VERSIONS:
        raise RuntimeError("unexpected hurdle-head smoke schema")
    high_support = config["schema_version"].endswith("high-support-hurdle-smoke/v1")
    hashes = {
        "feature_shard_sha256": _sha256(args.feature_shard),
        "target_audit_sha256": _sha256(args.target_audit),
        "spatial_audit_sha256": _sha256(args.spatial_audit),
    }
    for key, value in hashes.items():
        if value != config["inputs"][key]:
            raise RuntimeError("%s changed after hurdle smoke freeze" % key)
    target_audit = json.loads(args.target_audit.read_text(encoding="utf-8"))
    spatial_audit = json.loads(args.spatial_audit.read_text(encoding="utf-8"))
    if not target_audit.get("diagnostics_passed") or not spatial_audit.get("gate", {}).get("passed"):
        raise RuntimeError("hurdle smoke prerequisites did not pass")

    hparams = config["optimization"]
    torch.manual_seed(int(hparams["seed"]))
    device = torch.device(args.device)
    records = records_from_counterfactual_shard(load_feature_shard(args.feature_shard))
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    validation = select_records(records, ["validation"], ["local_blur", "local_dark"])
    actual_counts = {"train": len(train), "route_validation": len(validation)}
    if actual_counts != config["inputs"]["record_counts"]:
        raise RuntimeError("hurdle smoke record counts changed")

    scaling = config["target_scaling"]
    scales = fit_train_component_scales(
        train,
        device,
        batch_size=int(hparams["scale_batch_size"]),
        quantile=float(scaling["quantile"]),
        response_floor=float(scaling["response_floor"]),
    )
    expected_scales = torch.tensor(
        [scaling["component_scales"][name] for name in EVIDENCE_COMPONENTS]
    )
    if not torch.allclose(scales.cpu(), expected_scales, rtol=1e-6, atol=1e-7):
        raise RuntimeError("component scales changed after hurdle smoke freeze")
    support_thresholds = None
    if high_support:
        support_thresholds = torch.tensor(
            [scaling["scaled_responsive_q80"][name] for name in EVIDENCE_COMPONENTS],
            dtype=torch.float32,
        )
    input_projection = None
    projection_report = None
    input_quantization = None
    quantization_report = None
    if "input_projection" in config:
        projection_config = config["input_projection"]
        if projection_config.get("kind") != "frozen_rademacher_johnson_lindenstrauss":
            raise RuntimeError("unsupported hurdle input projection")
        input_projection_cpu = deterministic_rademacher_projection(
            int(projection_config["input_dim"]),
            int(projection_config["output_dim"]),
            int(projection_config["seed"]),
        )
        actual_projection_hash = projection_sha256(input_projection_cpu)
        if actual_projection_hash != projection_config["matrix_sha256"]:
            raise RuntimeError("hurdle input projection changed after freeze")
        if int(hparams["feature_dim"]) != int(projection_config["output_dim"]):
            raise RuntimeError("projected hurdle feature dimension differs")
        input_projection = input_projection_cpu.to(device=device, dtype=torch.float32)
        projection_report = {**projection_config, "matrix_sha256": actual_projection_hash}
    if "input_quantization" in config:
        quantization_config = config["input_quantization"]
        input_quantization = quantization_config.get("kind")
        if input_quantization != "dynamic_symmetric_int8_per_grid_channel":
            raise RuntimeError("unsupported hurdle input quantization")
        if input_projection is not None:
            raise RuntimeError("simultaneous hurdle projection and quantization is disabled")
        quantization_report = dict(quantization_config)

    model = ObservationEvidenceHurdleAdapter(
        feature_dim=int(hparams["feature_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        max_views=int(hparams["max_views"]),
        presence_bias=float(hparams["presence_bias"]),
        magnitude_bias=float(hparams["magnitude_bias"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    common = {
        "pair_batch_size": int(hparams["pair_batch_size"]),
        "presence_responsive_weight": float(hparams["presence_responsive_weight"]),
        "magnitude_weight": float(hparams["magnitude_weight"]),
        "ranking_weight": float(hparams["ranking_weight"]),
        "reference_weight": float(hparams["reference_weight"]),
        "response_floor": float(scaling["response_floor"]),
        "support_thresholds": support_thresholds,
        "input_projection": input_projection,
        "input_quantization": input_quantization,
    }
    history = []
    best_state = None
    best_epoch = 0
    best_value = float("inf")
    early_stopped = False
    for epoch in range(int(hparams["epochs"])):
        train_metrics = run_hurdle_evidence_epoch(
            model,
            train,
            scales,
            device,
            optimizer=optimizer,
            seed=int(hparams["seed"]) + epoch,
            **common,
        )
        validation_metrics = run_hurdle_evidence_epoch(
            model, validation, scales, device, optimizer=None, **common
        )
        history.append(
            {"epoch": epoch + 1, "train": train_metrics, "route_validation": validation_metrics}
        )
        if validation_metrics["total"] < best_value:
            best_value = validation_metrics["total"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            "[HurdleSmoke] epoch=%d/%d train=%.6f val=%.6f presence=%.6f magnitude=%.6f"
            % (
                epoch + 1,
                int(hparams["epochs"]),
                train_metrics["total"],
                validation_metrics["total"],
                validation_metrics["presence"],
                validation_metrics["magnitude"],
            ),
            flush=True,
        )
        rule = config.get("early_stop", {})
        if rule.get("enabled") and len(history) >= int(rule["minimum_epochs"]):
            recent = history[-3:]
            train_descends = all(
                recent[index]["train"]["total"]
                < recent[index - 1]["train"]["total"]
                for index in (1, 2)
            )
            validation_ascends = all(
                recent[index]["route_validation"]["total"]
                > recent[index - 1]["route_validation"]["total"]
                for index in (1, 2)
            )
            if train_descends and validation_ascends:
                early_stopped = True
                print(
                    "[HurdleSmoke] EARLY_STOP_ROUTE_OVERFIT=1 completed_epochs=%d"
                    % len(history),
                    flush=True,
                )
                break
    if best_state is None:
        raise RuntimeError("hurdle smoke checkpoint selection failed")
    model.load_state_dict(best_state)
    evaluation = evaluate_evidence_records(
        model,
        validation,
        scales,
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
        input_projection=input_projection,
        input_quantization=input_quantization,
    )
    hurdle_diagnostics = evaluate_hurdle_diagnostics(
        model,
        validation,
        scales,
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
        response_floor=float(scaling["response_floor"]),
        support_thresholds=support_thresholds,
        input_projection=input_projection,
        input_quantization=input_quantization,
    )

    thresholds = config["smoke_gate"]
    family_uplifts = {
        family: row["score_mean"] - evaluation["reference_prediction_mean"]
        for family, row in evaluation["by_family"].items()
    }
    component_pass_count = sum(
        row["patch_spearman"] >= float(thresholds["component_patch_spearman"])
        for row in evaluation["components"].values()
    )
    presence_auc_pass_count = sum(
        row["presence_auroc"]
        >= float(thresholds.get("component_high_support_presence_auroc", 0.0))
        for row in hurdle_diagnostics["components"].values()
    )
    best_validation = history[best_epoch - 1]["route_validation"]
    checks = [
        ("combined_patch_spearman", evaluation["combined_patch_spearman"], thresholds["combined_patch_spearman"]),
        ("combined_target_top20_auroc", evaluation["combined_target_top20_auroc"], thresholds["combined_target_top20_auroc"]),
        (
            "median_record_within_intervened_view_target_top20_auroc",
            evaluation["median_record_within_intervened_view_target_top20_auroc"],
            thresholds["median_record_within_intervened_view_target_top20_auroc"],
        ),
        ("component_spearman_pass_count", component_pass_count, thresholds["minimum_component_spearman_pass_count"]),
        ("minimum_train_family_uplift", min(family_uplifts.values()), thresholds["minimum_each_train_family_uplift_over_reference"]),
    ]
    if high_support:
        checks.append(
            (
                "high_support_presence_auroc_pass_count",
                presence_auc_pass_count,
                thresholds["minimum_high_support_presence_auroc_pass_count"],
            )
        )
    gate_checks = [
        {"metric": name, "value": float(value), "threshold": float(threshold), "passed": float(value) >= float(threshold)}
        for name, value, threshold in checks
    ]
    for name, value, threshold in (
        ("route_validation_ranking_loss", best_validation["ranking"], thresholds["route_validation_ranking_loss_max"]),
        ("reference_prediction_p95", evaluation["reference_prediction_p95"], thresholds["reference_prediction_p95_max"]),
    ):
        gate_checks.append(
            {"metric": name, "value": float(value), "threshold": float(threshold), "comparison": "<=", "passed": float(value) <= float(threshold)}
        )
    gate = {"passed": all(row["passed"] for row in gate_checks), "checks": gate_checks}

    checkpoint = {
        "schema_version": (
            "orion.counterfactual-evidence-high-support-hurdle-smoke-checkpoint/v1"
            if high_support
            else "orion.counterfactual-evidence-hurdle-smoke-checkpoint/v1"
        ),
        "student_state": best_state,
        "model_config": {
            "feature_dim": int(hparams["feature_dim"]),
            "hidden_dim": int(hparams["hidden_dim"]),
            "max_views": int(hparams["max_views"]),
            "presence_bias": float(hparams["presence_bias"]),
            "magnitude_bias": float(hparams["magnitude_bias"]),
        },
        "component_scales": scales.cpu(),
        "support_thresholds": support_thresholds,
        "input_projection": (
            input_projection.detach().cpu() if input_projection is not None else None
        ),
        "input_quantization": quantization_report,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "history": history,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    report = {
        "schema_version": (
            "orion.counterfactual-evidence-high-support-hurdle-smoke-report/v1"
            if high_support
            else "orion.counterfactual-evidence-hurdle-smoke-report/v1"
        ),
        "inputs": {**hashes, "config_sha256": hashlib.sha256(config_bytes).hexdigest()},
        "best_epoch": best_epoch,
        "best_route_validation_total": best_value,
        "early_stopped": early_stopped,
        "completed_epochs": len(history),
        "component_scales": {
            name: float(value) for name, value in zip(EVIDENCE_COMPONENTS, scales.cpu())
        },
        "input_projection": projection_report,
        "input_quantization": quantization_report,
        "history": history,
        "route_validation": evaluation,
        "hurdle_diagnostics": hurdle_diagnostics,
        "family_uplifts_over_reference": family_uplifts,
        "gate": gate,
        "scope_attestation": {
            "heldout_glare_read": False,
            "native_fog_read": False,
            "corruption_mask_optimizer_weight": 0.0,
            "exact_nonzero_presence_label": not high_support,
            "input_projection_used": input_projection is not None,
            "input_quantization_used": input_quantization is not None,
            "target_recomputed_after_projection": False,
            "output_compact_shard_written": False,
            "support_supervision": (
                "frozen train-responsive q80 per component"
                if high_support
                else "legacy numerical-nonzero footprint"
            ),
            "actual_target_optimizer_weight": 0.0,
            "automatic_full_training": False,
            "orion_finetuning": False,
            "stage_b": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({"output": str(args.output), "best_epoch": best_epoch, "gate": gate}, indent=2))
    print("COUNTERFACTUAL_EVIDENCE_HURDLE_SMOKE_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
