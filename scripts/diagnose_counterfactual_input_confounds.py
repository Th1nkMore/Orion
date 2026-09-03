#!/usr/bin/env python3
"""Ablate frozen adapter input branches on the fixed route validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    CounterfactualEvidenceRecord,
    _exact_quantiles_1d,
    evaluate_evidence_records,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_route_shard_records_selective,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-input-confound-diagnostic/v1"
REPORT_SCHEMA_VERSION = "orion.counterfactual-evidence-input-confound-report/v1"
DIAGNOSTIC_FAMILIES = ("local_blur", "local_dark")
VARIANTS = (
    "original",
    "no_explicit_change_scalar",
    "no_explicit_rms_scalars",
    "no_explicit_scalars",
    "no_view_embedding",
    "no_temporal_branch",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_id(row: Mapping[str, object]) -> str:
    route_ids = row.get("route_ids")
    if not isinstance(route_ids, list) or len(route_ids) != 1:
        raise RuntimeError("route shard must own exactly one route")
    return str(route_ids[0])


def _summary(values: torch.Tensor, threshold: float) -> Dict[str, float]:
    values = values.detach().cpu().float().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise RuntimeError("diagnostic values are empty or non-finite")
    q50, q90, q95, q99 = _exact_quantiles_1d(values, (0.5, 0.9, 0.95, 0.99))
    return {
        "cell_count": int(values.numel()),
        "mean": float(values.mean()),
        "p50": float(q50),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "maximum": float(values.max()),
        "fraction_above_frozen_gate_threshold": float((values > threshold).float().mean()),
    }


def _model_for_variant(
    checkpoint: Mapping[str, object], variant: str, device: torch.device
) -> ObservationEvidenceHurdleAdapter:
    if variant not in VARIANTS:
        raise RuntimeError("unknown input-confound variant")
    model = ObservationEvidenceHurdleAdapter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["student_state"])
    with torch.no_grad():
        if variant == "no_explicit_change_scalar":
            model.scalar_projection.weight[:, 0].zero_()
        elif variant == "no_explicit_rms_scalars":
            model.scalar_projection.weight[:, 1:].zero_()
        elif variant == "no_explicit_scalars":
            model.scalar_projection.weight.zero_()
        elif variant == "no_view_embedding":
            model.view_embedding.weight.zero_()
        elif variant == "no_temporal_branch":
            model.previous_projection.weight.zero_()
            if model.previous_projection.bias is not None:
                model.previous_projection.bias.zero_()
            model.previous_missing.zero_()
            model.scalar_projection.weight[:, 0].zero_()
            model.scalar_projection.weight[:, 2].zero_()
    return model.to(device).eval()


@torch.no_grad()
def _clean_distribution(
    model: ObservationEvidenceHurdleAdapter,
    records: Sequence[CounterfactualEvidenceRecord],
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:
    unique: Dict[str, CounterfactualEvidenceRecord] = {}
    for record in records:
        unique.setdefault(record.pair_id, record)
    route_values: Dict[str, List[torch.Tensor]] = {}
    all_values = []
    previous_groups: Dict[str, List[torch.Tensor]] = {"valid": [], "invalid": []}
    for record in sorted(unique.values(), key=lambda row: (row.route_id, row.frame_idx)):
        current = record.reference_current.unsqueeze(0).to(device)
        previous = record.reference_previous.unsqueeze(0).to(device)
        valid = torch.tensor([record.previous_valid], dtype=torch.bool, device=device)
        score = model(current, previous, valid)[0].detach().cpu().float().reshape(-1)
        route_values.setdefault(record.route_id, []).append(score)
        previous_groups["valid" if record.previous_valid else "invalid"].append(score)
        all_values.append(score)
    by_route = {
        route_id: {
            "frame_count": len(values),
            **_summary(torch.cat(values), threshold),
        }
        for route_id, values in sorted(route_values.items())
    }
    route_p95 = torch.tensor([row["p95"] for row in by_route.values()])
    return {
        "frame_count": len(unique),
        "overall": _summary(torch.cat(all_values), threshold),
        "by_previous_valid": {
            name: _summary(torch.cat(values), threshold)
            for name, values in previous_groups.items()
        },
        "by_route": by_route,
        "route_p95_summary": _summary(route_p95, threshold),
    }


def _compact_metrics(
    evaluation: Mapping[str, object], clean: Mapping[str, object]
) -> Dict[str, object]:
    reference_mean = float(evaluation["reference_prediction_mean"])
    family_uplifts = {
        family: float(row["score_mean"]) - reference_mean
        for family, row in evaluation["by_family"].items()
    }
    return {
        "combined_patch_spearman": float(evaluation["combined_patch_spearman"]),
        "combined_target_top20_auroc": float(
            evaluation["combined_target_top20_auroc"]
        ),
        "median_record_within_intervened_view_target_top20_auroc": float(
            evaluation[
                "median_record_within_intervened_view_target_top20_auroc"
            ]
        ),
        "reference_prediction_mean": reference_mean,
        "reference_prediction_p95": float(evaluation["reference_prediction_p95"]),
        "maximum_route_reference_p95": max(
            float(row["p95"]) for row in clean["by_route"].values()
        ),
        "minimum_family_uplift_over_reference": min(family_uplifts.values()),
        "family_uplifts_over_reference": family_uplifts,
    }


def _comparison(
    current: Mapping[str, object], baseline: Mapping[str, object]
) -> Dict[str, object]:
    p95 = float(current["reference_prediction_p95"])
    base_p95 = float(baseline["reference_prediction_p95"])
    names = (
        "combined_patch_spearman",
        "combined_target_top20_auroc",
        "median_record_within_intervened_view_target_top20_auroc",
        "reference_prediction_mean",
        "reference_prediction_p95",
        "maximum_route_reference_p95",
        "minimum_family_uplift_over_reference",
    )
    return {
        "deltas": {
            name: float(current[name]) - float(baseline[name]) for name in names
        },
        "clean_p95_reduction_fraction": (
            (base_p95 - p95) / base_p95 if base_p95 > 0 else math.nan
        ),
    }


def _diagnostic_candidate(
    metrics: Mapping[str, object],
    baseline: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> Dict[str, object]:
    checks = {
        "minimum_clean_p95_reduction_fraction": (
            (float(baseline["reference_prediction_p95"]) - float(metrics["reference_prediction_p95"]))
            / float(baseline["reference_prediction_p95"])
            >= float(thresholds["minimum_clean_p95_reduction_fraction"])
        ),
        "maximum_combined_spearman_drop": (
            float(metrics["combined_patch_spearman"])
            >= float(baseline["combined_patch_spearman"])
            - float(thresholds["maximum_combined_spearman_drop"])
        ),
        "maximum_top20_auroc_drop": (
            float(metrics["combined_target_top20_auroc"])
            >= float(baseline["combined_target_top20_auroc"])
            - float(thresholds["maximum_top20_auroc_drop"])
        ),
        "maximum_within_view_auroc_drop": (
            float(metrics["median_record_within_intervened_view_target_top20_auroc"])
            >= float(
                baseline[
                    "median_record_within_intervened_view_target_top20_auroc"
                ]
            )
            - float(thresholds["maximum_within_view_auroc_drop"])
        ),
        "positive_each_family_uplift": float(
            metrics["minimum_family_uplift_over_reference"]
        )
        > 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--clean-tail-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)

    if args.output.exists():
        raise SystemExit("refusing to overwrite input-confound diagnostic")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA input-confound diagnostic requested but unavailable")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected input-confound config schema")
    if tuple(config.get("variants", ())) != VARIANTS:
        raise RuntimeError("input-confound variant set/order changed")
    input_paths = {
        "dataset_manifest_sha256": args.dataset_manifest,
        "checkpoint_sha256": args.checkpoint,
        "training_report_sha256": args.training_report,
        "clean_tail_report_sha256": args.clean_tail_report,
    }
    hashes = {name: _sha256(path) for name, path in input_paths.items()}
    for name, value in hashes.items():
        if value != config["inputs"][name]:
            raise RuntimeError("input-confound diagnostic hash changed: %s" % name)

    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    clean_tail_report = json.loads(args.clean_tail_report.read_text(encoding="utf-8"))
    if training_report.get("schema_version") != config["inputs"]["training_report_schema"]:
        raise RuntimeError("training report schema differs")
    if clean_tail_report.get("schema_version") != config["inputs"]["clean_tail_report_schema"]:
        raise RuntimeError("clean-tail report schema differs")

    manifest = load_fp16_dataset_manifest(args.dataset_manifest, verify_shards=False)
    if (
        manifest.get("schema_version") != FP16_DIRECT_DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("expanded FP16 dataset contract differs")
    validation_rows = sorted(
        [dict(row) for row in manifest["shards"] if row.get("split") == "validation"],
        key=_route_id,
    )
    population = config["population"]
    if len(validation_rows) != int(population["route_count"]):
        raise RuntimeError("input-confound validation route count differs")
    dataset_root = args.dataset_manifest.parent
    records: List[CounterfactualEvidenceRecord] = []
    for index, row in enumerate(validation_rows, start=1):
        shard_path = dataset_root / str(row["file"])
        if not shard_path.is_file() or shard_path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError("validation shard file/size differs: %s" % shard_path)
        records.extend(
            load_fp16_route_shard_records_selective(
                shard_path, families=DIAGNOSTIC_FAMILIES
            )
        )
        print(
            "[InputConfound] mmap=%d/%d route=%s"
            % (index, len(validation_rows), _route_id(row)),
            flush=True,
        )
    if (
        len(records) != int(population["record_count"])
        or len({record.pair_id for record in records})
        != int(population["clean_frame_count"])
    ):
        raise RuntimeError("input-confound validation population differs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != config["inputs"]["checkpoint_schema"]:
        raise RuntimeError("checkpoint schema differs")
    scales = checkpoint["component_scales"].float()
    device = torch.device(args.device)
    threshold = float(config["frozen_reference_gate_threshold"])
    results = {}
    for variant in VARIANTS:
        model = _model_for_variant(checkpoint, variant, device)
        evaluation = evaluate_evidence_records(
            model,
            records,
            scales.to(device),
            device,
            batch_size=int(config["evaluation_batch_size"]),
        )
        clean = _clean_distribution(model, records, device, threshold)
        metrics = _compact_metrics(evaluation, clean)
        results[variant] = {
            "metrics": metrics,
            "clean_distribution": clean,
            "evaluation": evaluation,
        }
        print(
            "[InputConfound] variant=%s clean_p95=%.6f top20_auc=%.6f within_auc=%.6f"
            % (
                variant,
                metrics["reference_prediction_p95"],
                metrics["combined_target_top20_auroc"],
                metrics[
                    "median_record_within_intervened_view_target_top20_auroc"
                ],
            ),
            flush=True,
        )

    baseline = results["original"]["metrics"]
    tolerance = float(config["reproduction_absolute_tolerance"])
    expected_eval = training_report["route_validation"]
    expected_tail = clean_tail_report["score"]["by_route"]
    reproduction = {
        "reference_p95_absolute_difference": abs(
            float(baseline["reference_prediction_p95"])
            - float(expected_eval["reference_prediction_p95"])
        ),
        "top20_auroc_absolute_difference": abs(
            float(baseline["combined_target_top20_auroc"])
            - float(expected_eval["combined_target_top20_auroc"])
        ),
        "maximum_route_p95_absolute_difference": max(
            abs(
                float(results["original"]["clean_distribution"]["by_route"][route]["p95"])
                - float(expected_tail[route]["p95"])
            )
            for route in expected_tail
        ),
    }
    reproduction["passed"] = all(
        float(value) <= tolerance for value in reproduction.values()
    )

    comparisons = {
        variant: _comparison(results[variant]["metrics"], baseline)
        for variant in VARIANTS
        if variant != "original"
    }
    candidates = {
        variant: _diagnostic_candidate(
            results[variant]["metrics"],
            baseline,
            config["diagnostic_candidate"],
        )
        for variant in VARIANTS
        if variant != "original"
    }
    passed = [variant for variant, row in candidates.items() if row["passed"]]
    if not reproduction["passed"]:
        decision = "invalid_diagnostic_original_metrics_not_reproduced"
        selected = None
    elif passed:
        selected = max(
            passed,
            key=lambda name: comparisons[name]["clean_p95_reduction_fraction"],
        )
        decision = "input_branch_candidate_%s_requires_bounded_retraining" % selected
    else:
        selected = None
        decision = "no_input_ablation_pareto_candidate_use_tail_or_invariance_loss_repair"

    scalar_weights = checkpoint["student_state"]["scalar_projection.weight"].float()
    view_weights = checkpoint["student_state"]["view_embedding.weight"].float()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inputs": {
            **hashes,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        },
        "population": {
            "route_count": len(validation_rows),
            "record_count": len(records),
            "clean_frame_count": len({record.pair_id for record in records}),
            "families": list(DIAGNOSTIC_FAMILIES),
        },
        "reproduction": reproduction,
        "frozen_weight_diagnostics": {
            "scalar_projection_input_channel_l2": {
                "temporal_change": float(scalar_weights[:, 0].norm()),
                "current_log_rms": float(scalar_weights[:, 1].norm()),
                "previous_log_rms": float(scalar_weights[:, 2].norm()),
            },
            "view_embedding_row_l2": [
                float(row.norm()) for row in view_weights
            ],
        },
        "variants": results,
        "comparisons_to_original": comparisons,
        "diagnostic_candidates": candidates,
        "selected_input_branch_candidate": selected,
        "decision": decision,
        "interpretation_boundary": (
            "Weight-zeroing is a causal checkpoint diagnostic, not a deployable model. "
            "Any selected branch removal must be retrained and pass the unchanged gate."
        ),
        "frozen_gate": {
            "reference_prediction_p95_threshold": threshold,
            "threshold_changed": False,
            "original_status": "failed",
            "amendment_made": False,
        },
        "scope_attestation": {
            "training_performed": False,
            "checkpoint_file_changed": False,
            "validation_co_sharded_glare_metadata_visible": True,
            "validation_co_sharded_glare_tensor_values_accessed": False,
            "heldout_split_tensor_values_accessed": False,
            "native_weather_read": False,
            "orion_finetuning": False,
            "stage_b": False,
            "automatic_followup": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reproduction": reproduction,
                "decision": decision,
                "selected_input_branch_candidate": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("COUNTERFACTUAL_EVIDENCE_INPUT_CONFOUND_DIAGNOSTIC_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
