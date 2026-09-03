#!/usr/bin/env python3
"""Train and gate the bounded counterfactual-evidence adapter pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    COUNTERFACTUAL_TRAINING_SCHEMA_VERSION,
    evaluate_evidence_records,
    fit_train_component_scales,
    records_from_counterfactual_shard,
    run_evidence_epoch,
    select_records,
)
from uq_estimator.native_appearance_audit import (  # noqa: E402
    audit_native_appearance_score_maps,
)
from uq_estimator.native_weather_audit import (  # noqa: E402
    EXPECTED_CONDITIONS,
    validate_native_weather_payload,
)
from uq_estimator.observation_uq_shard import load_feature_shard  # noqa: E402


TRAINING_RUN_SCHEMA_VERSION = "orion.counterfactual-evidence-training-run/v1"
NATIVE_CANDIDATE = "counterfactual_evidence_adapter_total"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _native_scores(model, payload, device, batch_size):
    items = payload["items"]
    outputs = {}
    for condition in EXPECTED_CONDITIONS:
        features = payload["features_by_condition"][condition]
        key_to_index = {
            (str(item["route_id"]), int(item["sequence_index"])): index
            for index, item in enumerate(items)
        }
        previous = []
        valid = []
        for item in items:
            index = key_to_index.get(
                (str(item["route_id"]), int(item["sequence_index"]) - 1)
            )
            previous.append(
                features[index] if index is not None else torch.zeros_like(features[0])
            )
            valid.append(index is not None)
        previous = torch.stack(previous)
        chunks = []
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            prediction = model(
                features[start:stop].to(device=device, dtype=torch.float32),
                previous[start:stop].to(device=device, dtype=torch.float32),
                torch.tensor(valid[start:stop], dtype=torch.bool, device=device),
            )
            chunks.append(prediction.mean(dim=-1).cpu())
        outputs[condition] = torch.cat(chunks)
    return outputs


def _route_validation_gate(report, thresholds):
    checks = []
    for component in EVIDENCE_COMPONENTS:
        value = float(report["components"][component]["patch_spearman"])
        threshold = float(thresholds["component_patch_spearman"])
        checks.append(
            {
                "component": component,
                "metric": "patch_spearman",
                "value": value,
                "threshold": threshold,
                "passed": math.isfinite(value) and value >= threshold,
            }
        )
    for metric, threshold_key in (
        ("combined_target_top20_auroc", "combined_target_top20_auroc"),
        ("median_route_target_top20_auroc", "median_route_target_top20_auroc"),
        (
            "median_record_within_intervened_view_target_top20_auroc",
            "median_record_within_intervened_view_target_top20_auroc",
        ),
        (
            "median_route_within_intervened_view_target_top20_auroc",
            "median_route_within_intervened_view_target_top20_auroc",
        ),
    ):
        value = float(report[metric])
        threshold = float(thresholds[threshold_key])
        checks.append(
            {
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "passed": math.isfinite(value) and value >= threshold,
            }
        )
    fp_value = float(report["reference_prediction_p95"])
    fp_threshold = float(thresholds["reference_prediction_p95_max"])
    checks.append(
        {
            "metric": "reference_prediction_p95",
            "value": fp_value,
            "threshold": fp_threshold,
            "passed": math.isfinite(fp_value) and fp_value <= fp_threshold,
            "comparison": "<=",
        }
    )
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def _heldout_gate(report, thresholds):
    low = report["by_family_severity"]["local_glare/severity_1"]
    high = report["by_family_severity"]["local_glare/severity_3"]
    reference = float(report["reference_prediction_mean"])
    checks = [
        {
            "metric": "severity_1_positive_uplift",
            "value": float(low["score_mean"] - reference),
            "threshold": 0.0,
            "passed": float(low["score_mean"]) > reference,
        },
        {
            "metric": "severity_3_higher_than_severity_1",
            "value": float(high["score_mean"] - low["score_mean"]),
            "threshold": 0.0,
            "passed": float(high["score_mean"]) > float(low["score_mean"]),
        },
        {
            "metric": "combined_patch_spearman",
            "value": float(report["combined_patch_spearman"]),
            "threshold": float(thresholds["combined_patch_spearman"]),
            "passed": float(report["combined_patch_spearman"])
            >= float(thresholds["combined_patch_spearman"]),
        },
        {
            "metric": "combined_target_top20_auroc",
            "value": float(report["combined_target_top20_auroc"]),
            "threshold": float(thresholds["combined_target_top20_auroc"]),
            "passed": float(report["combined_target_top20_auroc"])
            >= float(thresholds["combined_target_top20_auroc"]),
        },
        {
            "metric": "median_record_within_intervened_view_target_top20_auroc",
            "value": float(
                report[
                    "median_record_within_intervened_view_target_top20_auroc"
                ]
            ),
            "threshold": float(
                thresholds[
                    "median_record_within_intervened_view_target_top20_auroc"
                ]
            ),
            "passed": float(
                report[
                    "median_record_within_intervened_view_target_top20_auroc"
                ]
            )
            >= float(
                thresholds[
                    "median_record_within_intervened_view_target_top20_auroc"
                ]
            ),
        },
    ]
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-shard", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--spatial-audit", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".report.json").exists():
        raise SystemExit("refusing to overwrite counterfactual adapter output")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA adapter training requested but unavailable")
    config_bytes = args.training_config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != TRAINING_RUN_SCHEMA_VERSION:
        raise RuntimeError("unexpected counterfactual training-run schema")
    feature_sha = _sha256(args.feature_shard)
    native_sha = _sha256(args.native_features)
    target_audit_sha = _sha256(args.target_audit)
    spatial_audit_sha = _sha256(args.spatial_audit)
    if feature_sha != config["inputs"]["feature_shard_sha256"]:
        raise RuntimeError("counterfactual feature shard hash changed")
    if native_sha != config["inputs"]["native_feature_sha256"]:
        raise RuntimeError("native feature shard hash changed")
    if target_audit_sha != config["inputs"]["target_audit_sha256"]:
        raise RuntimeError("target-audit artifact hash changed")
    if spatial_audit_sha != config["inputs"]["spatial_audit_sha256"]:
        raise RuntimeError("spatial-audit artifact hash changed")
    target_audit = json.loads(args.target_audit.read_text(encoding="utf-8"))
    spatial_audit = json.loads(args.spatial_audit.read_text(encoding="utf-8"))
    if not target_audit.get("diagnostics_passed"):
        raise RuntimeError("train-target distribution audit did not pass")
    if not spatial_audit.get("gate", {}).get("passed"):
        raise RuntimeError("train-target spatial-support audit did not pass")
    if spatial_audit["inputs"]["target_audit_sha256"] != target_audit_sha:
        raise RuntimeError("spatial audit references another target audit")
    hparams = config["optimization"]
    torch.manual_seed(int(hparams["seed"]))
    random_seed = int(hparams["seed"])
    device = torch.device(args.device)

    payload = load_feature_shard(args.feature_shard)
    records = records_from_counterfactual_shard(payload)
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    validation = select_records(
        records, ["validation"], ["local_blur", "local_dark"]
    )
    heldout_family = select_records(
        records, ["validation", "held_out"], ["local_glare"]
    )
    expected_counts = config["inputs"]["record_counts"]
    actual_counts = {
        "train": len(train),
        "route_validation": len(validation),
        "heldout_family": len(heldout_family),
    }
    if actual_counts != expected_counts:
        raise RuntimeError("counterfactual training record counts changed")

    scaling = config["target_scaling"]
    scales = fit_train_component_scales(
        train,
        device,
        batch_size=int(hparams["scale_batch_size"]),
        quantile=float(scaling["quantile"]),
        response_floor=float(scaling["response_floor"]),
    )
    expected_scales = torch.tensor(
        [scaling["component_scales"][name] for name in EVIDENCE_COMPONENTS],
        dtype=torch.float32,
    )
    if not torch.allclose(scales.cpu(), expected_scales, rtol=1e-6, atol=1e-7):
        raise RuntimeError("train-only component scales changed after freeze")
    model = ObservationEvidenceAdapter(
        feature_dim=int(hparams["feature_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        max_views=int(hparams["max_views"]),
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
    for epoch in range(int(hparams["epochs"])):
        train_metrics = run_evidence_epoch(
            model,
            train,
            scales,
            device,
            pair_batch_size=int(hparams["pair_batch_size"]),
            optimizer=optimizer,
            seed=random_seed + epoch,
            ranking_weight=float(hparams["ranking_weight"]),
            reference_weight=float(hparams["reference_weight"]),
        )
        validation_metrics = run_evidence_epoch(
            model,
            validation,
            scales,
            device,
            pair_batch_size=int(hparams["pair_batch_size"]),
            optimizer=None,
            ranking_weight=float(hparams["ranking_weight"]),
            reference_weight=float(hparams["reference_weight"]),
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "route_validation": validation_metrics,
            }
        )
        if validation_metrics["total"] < best_value:
            best_value = validation_metrics["total"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            "[CounterfactualEvidence] epoch=%d/%d train=%.6f val=%.6f"
            % (
                epoch + 1,
                int(hparams["epochs"]),
                train_metrics["total"],
                validation_metrics["total"],
            ),
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("counterfactual checkpoint selection failed")
    model.load_state_dict(best_state)

    route_report = evaluate_evidence_records(
        model,
        validation,
        scales,
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
    )
    heldout_report = evaluate_evidence_records(
        model,
        heldout_family,
        scales,
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
    )
    route_gate = _route_validation_gate(route_report, config["gate"]["route_validation"])
    heldout_gate = _heldout_gate(
        heldout_report, config["gate"]["heldout_family_development"]
    )

    native_payload = torch.load(args.native_features, map_location="cpu")
    validate_native_weather_payload(native_payload)
    native_scores = _native_scores(
        model,
        native_payload,
        device,
        int(hparams["evaluation_batch_size"]),
    )
    native_report = audit_native_appearance_score_maps(
        native_payload,
        {NATIVE_CANDIDATE: native_scores},
        candidate_tails={NATIVE_CANDIDATE: "positive"},
    )
    native_report["schema_version"] = "orion.counterfactual-evidence-native-audit/v1"
    native_gate = native_report["candidates"][NATIVE_CANDIDATE]["candidate_gate"]

    checkpoint = {
        "schema_version": COUNTERFACTUAL_TRAINING_SCHEMA_VERSION,
        "student_state": best_state,
        "model_config": {
            "feature_dim": int(hparams["feature_dim"]),
            "hidden_dim": int(hparams["hidden_dim"]),
            "max_views": int(hparams["max_views"]),
            "components": list(EVIDENCE_COMPONENTS),
        },
        "component_scales": scales,
        "checkpoint_selection": {
            "metric": "route_validation_total_loss",
            "best_epoch": best_epoch,
            "best_value": best_value,
            "heldout_family_used": False,
            "native_used": False,
        },
        "history": history,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "data_attestation": {
            "record_counts": actual_counts,
            "optimizer_families": ["local_blur", "local_dark"],
            "heldout_family": "local_glare",
            "corruption_mask_optimizer_weight": 0.0,
            "actual_target_optimizer_weight": 0.0,
            "orion_finetuned": False,
            "stage_b_run": False,
            "target_audit_sha256": target_audit_sha,
            "spatial_audit_sha256": spatial_audit_sha,
            "corruption_mask_read_by_optimizer": False,
        },
    }
    report = {
        key: value for key, value in checkpoint.items() if key != "student_state"
    }
    report["component_scales"] = {
        name: float(value)
        for name, value in zip(EVIDENCE_COMPONENTS, scales.cpu())
    }
    report.update(
        {
            "schema_version": "orion.counterfactual-evidence-training-report/v1",
            "inputs": {
                "feature_shard": str(args.feature_shard.resolve()),
                "feature_shard_sha256": feature_sha,
                "native_features": str(args.native_features.resolve()),
                "native_features_sha256": native_sha,
                "target_audit": str(args.target_audit.resolve()),
                "target_audit_sha256": target_audit_sha,
                "spatial_audit": str(args.spatial_audit.resolve()),
                "spatial_audit_sha256": spatial_audit_sha,
                "training_config": str(args.training_config.resolve()),
                "training_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            "evaluations": {
                "route_validation": route_report,
                "heldout_family_development": heldout_report,
                "native_development": native_report,
            },
            "gates": {
                "route_validation": route_gate,
                "heldout_family_development": heldout_gate,
                "native_development": native_gate,
                "all_passed": route_gate["passed"]
                and heldout_gate["passed"]
                and native_gate["passed"],
            },
            "continuation": {
                "adapter_integration_authorized": False,
                "orion_finetuning_authorized": False,
                "stage_b_authorized": False,
                "next_if_all_pass": "freeze and run untouched native/sensor confirmation",
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "best_epoch": best_epoch,
                "gates": report["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    print("COUNTERFACTUAL_EVIDENCE_ADAPTER_PILOT_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
