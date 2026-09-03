#!/usr/bin/env python3
"""Run a route-only anti-collapse smoke for balanced evidence regression."""

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
    ObservationEvidenceAdapter,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    evaluate_evidence_records,
    fit_train_component_scales,
    records_from_counterfactual_shard,
    run_evidence_epoch,
    select_records,
)
from uq_estimator.observation_uq_shard import load_feature_shard  # noqa: E402


SCHEMA_VERSION = "orion.counterfactual-evidence-loss-repair-smoke/v1"


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
        raise SystemExit("refusing to overwrite loss-repair smoke")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA loss-repair smoke requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected loss-repair smoke schema")
    hashes = {
        "feature_shard_sha256": _sha256(args.feature_shard),
        "target_audit_sha256": _sha256(args.target_audit),
        "spatial_audit_sha256": _sha256(args.spatial_audit),
    }
    for key, value in hashes.items():
        if value != config["inputs"][key]:
            raise RuntimeError("%s changed after smoke freeze" % key)
    target_audit = json.loads(args.target_audit.read_text(encoding="utf-8"))
    spatial_audit = json.loads(args.spatial_audit.read_text(encoding="utf-8"))
    if not target_audit.get("diagnostics_passed") or not spatial_audit.get("gate", {}).get("passed"):
        raise RuntimeError("loss-repair smoke prerequisites did not pass")

    hparams = config["optimization"]
    torch.manual_seed(int(hparams["seed"]))
    device = torch.device(args.device)
    records = records_from_counterfactual_shard(load_feature_shard(args.feature_shard))
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    validation = select_records(records, ["validation"], ["local_blur", "local_dark"])
    actual_counts = {"train": len(train), "route_validation": len(validation)}
    if actual_counts != config["inputs"]["record_counts"]:
        raise RuntimeError("loss-repair smoke record counts changed")

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
        raise RuntimeError("component scales changed after smoke freeze")

    model = ObservationEvidenceAdapter(
        feature_dim=int(hparams["feature_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        max_views=int(hparams["max_views"]),
        output_bias=float(hparams["output_bias"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    history = []
    best_state = None
    best_epoch = 0
    best_value = float("inf")
    common = {
        "pair_batch_size": int(hparams["pair_batch_size"]),
        "ranking_weight": float(hparams["ranking_weight"]),
        "reference_weight": float(hparams["reference_weight"]),
        "responsive_weight": float(hparams["responsive_weight"]),
        "response_floor": float(scaling["response_floor"]),
    }
    for epoch in range(int(hparams["epochs"])):
        train_metrics = run_evidence_epoch(
            model,
            train,
            scales,
            device,
            optimizer=optimizer,
            seed=int(hparams["seed"]) + epoch,
            **common,
        )
        validation_metrics = run_evidence_epoch(
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
            "[LossRepairSmoke] epoch=%d/%d train=%.6f val=%.6f ranking=%.6f"
            % (
                epoch + 1,
                int(hparams["epochs"]),
                train_metrics["total"],
                validation_metrics["total"],
                validation_metrics["ranking"],
            ),
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("loss-repair smoke checkpoint selection failed")
    model.load_state_dict(best_state)
    evaluation = evaluate_evidence_records(
        model,
        validation,
        scales,
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
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
        "schema_version": "orion.counterfactual-evidence-loss-repair-smoke-checkpoint/v1",
        "student_state": best_state,
        "model_config": {
            "feature_dim": int(hparams["feature_dim"]),
            "hidden_dim": int(hparams["hidden_dim"]),
            "max_views": int(hparams["max_views"]),
            "output_bias": float(hparams["output_bias"]),
        },
        "component_scales": scales.cpu(),
        "best_epoch": best_epoch,
        "history": history,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    report = {
        "schema_version": "orion.counterfactual-evidence-loss-repair-smoke-report/v1",
        "inputs": {**hashes, "config_sha256": hashlib.sha256(config_bytes).hexdigest()},
        "best_epoch": best_epoch,
        "best_route_validation_total": best_value,
        "component_scales": {
            name: float(value) for name, value in zip(EVIDENCE_COMPONENTS, scales.cpu())
        },
        "history": history,
        "route_validation": evaluation,
        "family_uplifts_over_reference": family_uplifts,
        "gate": gate,
        "scope_attestation": {
            "heldout_glare_read": False,
            "native_fog_read": False,
            "corruption_mask_optimizer_weight": 0.0,
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
    print("COUNTERFACTUAL_EVIDENCE_LOSS_REPAIR_SMOKE_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
