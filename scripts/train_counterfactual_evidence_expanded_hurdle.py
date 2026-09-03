#!/usr/bin/env python3
"""Train one bounded hurdle adapter from audited route-local FP16 shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    CLAIM_BOUNDARY,
    EVIDENCE_COMPONENTS,
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    CounterfactualEvidenceRecord,
    evaluate_evidence_records,
    evaluate_hurdle_diagnostics,
    run_hurdle_evidence_epoch,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_route_shard_records_selective,
)


EXPANDED_SCHEMA_VERSION = "orion.counterfactual-evidence-expanded-hurdle-training/v1"
NO_VIEW_REPAIR_SCHEMA_VERSION = "orion.counterfactual-evidence-no-view-repair-training/v1"
RUN_CONTRACTS = {
    EXPANDED_SCHEMA_VERSION: {
        "kind": "expanded",
        "checkpoint_schema": "orion.counterfactual-evidence-expanded-hurdle-checkpoint/v1",
        "report_schema": "orion.counterfactual-evidence-expanded-hurdle-report/v1",
        "training_state_schema": "orion.counterfactual-evidence-expanded-hurdle-training-state/v1",
    },
    NO_VIEW_REPAIR_SCHEMA_VERSION: {
        "kind": "no_view_repair",
        "checkpoint_schema": "orion.counterfactual-evidence-no-view-repair-checkpoint/v1",
        "report_schema": "orion.counterfactual-evidence-no-view-repair-report/v1",
        "training_state_schema": "orion.counterfactual-evidence-no-view-repair-training-state/v1",
    },
}
OPTIMIZER_FAMILIES = ("local_blur", "local_dark")


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


def _selected_rows(manifest: Mapping[str, object], split: str) -> List[dict]:
    rows = [dict(row) for row in manifest["shards"] if row.get("split") == split]
    return sorted(rows, key=_route_id)


def _check_row_file(dataset_root: Path, row: Mapping[str, object]) -> Path:
    path = dataset_root / str(row["file"])
    if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
        raise RuntimeError("route shard file/size differs: %s" % path)
    if _sha256(path) != str(row["sha256"]):
        raise RuntimeError("route shard SHA256 differs: %s" % path)
    return path


def _load_optimizer_records(
    dataset_root: Path,
    row: Mapping[str, object],
    expected_split: str,
) -> List[CounterfactualEvidenceRecord]:
    records = load_fp16_route_shard_records_selective(
        dataset_root / str(row["file"]), families=OPTIMIZER_FAMILIES
    )
    route_id = _route_id(row)
    if (
        len(records) != 64
        or {record.route_id for record in records} != {route_id}
        or {record.split for record in records} != {expected_split}
        or {record.family for record in records} != set(OPTIMIZER_FAMILIES)
        or {record.severity for record in records} != {1.0, 3.0}
    ):
        raise RuntimeError("optimizer-family route population differs: %s" % route_id)
    return records


def _weighted_metrics(
    rows: Iterable[Tuple[Mapping[str, float], int]]
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for metrics, weight in rows:
        count += int(weight)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * int(weight)
    if count <= 0:
        raise RuntimeError("weighted metric population is empty")
    return {name: value / count for name, value in totals.items()}


def _route_validation_gate(
    evaluation: Mapping[str, object],
    hurdle: Mapping[str, object],
    best_validation: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> Dict[str, object]:
    family_uplifts = {
        family: float(row["score_mean"])
        - float(evaluation["reference_prediction_mean"])
        for family, row in evaluation["by_family"].items()
    }
    component_pass_count = sum(
        float(row["patch_spearman"])
        >= float(thresholds["component_patch_spearman"])
        for row in evaluation["components"].values()
    )
    presence_pass_count = sum(
        float(row["presence_auroc"])
        >= float(thresholds["component_high_support_presence_auroc"])
        for row in hurdle["components"].values()
    )
    checks = [
        (
            "combined_patch_spearman",
            evaluation["combined_patch_spearman"],
            thresholds["combined_patch_spearman"],
            ">=",
        ),
        (
            "combined_target_top20_auroc",
            evaluation["combined_target_top20_auroc"],
            thresholds["combined_target_top20_auroc"],
            ">=",
        ),
        (
            "median_record_within_intervened_view_target_top20_auroc",
            evaluation[
                "median_record_within_intervened_view_target_top20_auroc"
            ],
            thresholds[
                "median_record_within_intervened_view_target_top20_auroc"
            ],
            ">=",
        ),
        (
            "component_spearman_pass_count",
            component_pass_count,
            thresholds["minimum_component_spearman_pass_count"],
            ">=",
        ),
        (
            "high_support_presence_auroc_pass_count",
            presence_pass_count,
            thresholds["minimum_high_support_presence_auroc_pass_count"],
            ">=",
        ),
        (
            "minimum_train_family_uplift",
            min(family_uplifts.values()),
            thresholds["minimum_each_train_family_uplift_over_reference"],
            ">=",
        ),
        (
            "route_validation_ranking_loss",
            best_validation["ranking"],
            thresholds["route_validation_ranking_loss_max"],
            "<=",
        ),
        (
            "reference_prediction_p95",
            evaluation["reference_prediction_p95"],
            thresholds["reference_prediction_p95_max"],
            "<=",
        ),
    ]
    gate_checks = []
    for name, value, threshold, comparison in checks:
        value = float(value)
        threshold = float(threshold)
        passed = math.isfinite(value) and (
            value >= threshold if comparison == ">=" else value <= threshold
        )
        gate_checks.append(
            {
                "metric": name,
                "value": value,
                "threshold": threshold,
                "comparison": comparison,
                "passed": passed,
            }
        )
    return {
        "passed": all(row["passed"] for row in gate_checks),
        "checks": gate_checks,
        "family_uplifts_over_reference": family_uplifts,
    }


def _comparison(
    evaluation: Mapping[str, object],
    baseline: Mapping[str, float],
    *,
    baseline_label: str,
    current_label: str,
) -> Dict[str, object]:
    current = {
        "combined_patch_spearman": float(evaluation["combined_patch_spearman"]),
        "combined_target_top20_auroc": float(
            evaluation["combined_target_top20_auroc"]
        ),
        "median_record_within_intervened_view_target_top20_auroc": float(
            evaluation[
                "median_record_within_intervened_view_target_top20_auroc"
            ]
        ),
        "reference_prediction_p95": float(evaluation["reference_prediction_p95"]),
    }
    return {
        name: {
            baseline_label: float(baseline[name]),
            current_label: value,
            "delta": value - float(baseline[name]),
        }
        for name, value in current.items()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--spatial-audit", type=Path, required=True)
    parser.add_argument("--input-confound-report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)

    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite expanded hurdle run")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA expanded hurdle run requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    schema_version = config.get("schema_version")
    if schema_version not in RUN_CONTRACTS:
        raise RuntimeError("unexpected expanded hurdle config schema")
    run_contract = RUN_CONTRACTS[schema_version]
    run_kind = str(run_contract["kind"])
    hashes = {
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "target_audit_sha256": _sha256(args.target_audit),
        "spatial_audit_sha256": _sha256(args.spatial_audit),
    }
    input_confound_report = None
    if run_kind == "no_view_repair":
        if args.input_confound_report is None:
            raise RuntimeError("no-view repair requires its frozen diagnostic report")
        hashes["input_confound_report_sha256"] = _sha256(
            args.input_confound_report
        )
        input_confound_report = json.loads(
            args.input_confound_report.read_text(encoding="utf-8")
        )
        if (
            not input_confound_report.get("reproduction", {}).get("passed")
            or input_confound_report.get("selected_input_branch_candidate")
            != "no_view_embedding"
            or input_confound_report.get("decision")
            != "input_branch_candidate_no_view_embedding_requires_bounded_retraining"
        ):
            raise RuntimeError("no-view repair diagnostic contract differs")
    elif args.input_confound_report is not None:
        raise RuntimeError("expanded baseline does not accept a repair diagnostic")
    for name, value in hashes.items():
        if value != config["inputs"][name]:
            raise RuntimeError("expanded hurdle input hash changed: %s" % name)

    target_audit = json.loads(args.target_audit.read_text(encoding="utf-8"))
    spatial_audit = json.loads(args.spatial_audit.read_text(encoding="utf-8"))
    if not target_audit.get("diagnostics_passed"):
        raise RuntimeError("expanded target audit did not pass")
    if not spatial_audit.get("gate", {}).get("passed"):
        raise RuntimeError("expanded spatial audit did not pass")
    if spatial_audit["inputs"]["target_audit_sha256"] != hashes["target_audit_sha256"]:
        raise RuntimeError("expanded spatial audit references another target audit")

    manifest = load_fp16_dataset_manifest(args.dataset_manifest, verify_shards=False)
    if (
        manifest.get("schema_version") != FP16_DIRECT_DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("written_route_count") != 90
        or manifest.get("written_clean_count") != 1440
        or manifest.get("written_observed_count") != 5760
    ):
        raise RuntimeError("expanded FP16 manifest contract differs")
    train_rows = _selected_rows(manifest, "train")
    validation_rows = _selected_rows(manifest, "validation")
    if len(train_rows) != 70 or len(validation_rows) != 10:
        raise RuntimeError("expanded route split counts differ")
    expected = config["inputs"]["record_counts"]
    if expected != {
        "train": 4480,
        "route_validation": 640,
        "train_routes": 70,
        "route_validation_routes": 10,
    }:
        raise RuntimeError("expanded config record counts differ")

    validation_routes = {_route_id(row) for row in validation_rows}
    strata = config["validation_strata"]
    legacy_routes = set(strata["legacy_validation_5"])
    added_routes = set(strata["added_validation_5"])
    if (
        legacy_routes & added_routes
        or legacy_routes | added_routes != validation_routes
        or len(legacy_routes) != 5
        or len(added_routes) != 5
    ):
        raise RuntimeError("expanded validation strata differ")

    scaling = config["target_scaling"]
    scales = torch.tensor(
        [scaling["component_scales"][name] for name in EVIDENCE_COMPONENTS],
        dtype=torch.float32,
    )
    support_thresholds = torch.tensor(
        [scaling["scaled_responsive_q80"][name] for name in EVIDENCE_COMPONENTS],
        dtype=torch.float32,
    )
    for index, name in enumerate(EVIDENCE_COMPONENTS):
        audited = float(target_audit["audit"]["components"][name]["selected_scale"])
        audited_q80 = float(
            target_audit["audit"]["components"][name]["responsive_quantiles"]["q80"]
        )
        if not math.isclose(float(scales[index]), audited, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError("expanded component scale changed: %s" % name)
        expected_support = audited_q80 / audited
        if not math.isclose(
            float(support_thresholds[index]),
            expected_support,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise RuntimeError("expanded support threshold changed: %s" % name)

    dataset_root = args.dataset_manifest.parent
    for index, row in enumerate(train_rows + validation_rows, start=1):
        _check_row_file(dataset_root, row)
        if index % 10 == 0 or index == len(train_rows) + len(validation_rows):
            print(
                "[ExpandedHurdle] integrity=%d/%d"
                % (index, len(train_rows) + len(validation_rows)),
                flush=True,
            )

    validation_records: List[CounterfactualEvidenceRecord] = []
    for index, row in enumerate(validation_rows, start=1):
        validation_records.extend(
            _load_optimizer_records(dataset_root, row, "validation")
        )
        print(
            "[ExpandedHurdle] validation_mmap=%d/%d route=%s"
            % (index, len(validation_rows), _route_id(row)),
            flush=True,
        )
    if len(validation_records) != int(expected["route_validation"]):
        raise RuntimeError("expanded validation record count differs")

    hparams = config["optimization"]
    use_view_embedding = hparams.get("use_view_embedding", True)
    if not isinstance(use_view_embedding, bool) or (
        run_kind == "no_view_repair" and use_view_embedding is not False
    ) or (run_kind == "expanded" and use_view_embedding is not True):
        raise RuntimeError("view-embedding architecture differs from run contract")
    seed = int(hparams["seed"])
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = ObservationEvidenceHurdleAdapter(
        feature_dim=int(hparams["feature_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        max_views=int(hparams["max_views"]),
        presence_bias=float(hparams["presence_bias"]),
        magnitude_bias=float(hparams["magnitude_bias"]),
        use_view_embedding=use_view_embedding,
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
        "support_thresholds": support_thresholds.to(device),
    }

    history = []
    best_state = None
    best_epoch = 0
    best_value = float("inf")
    early_stopped = False
    for epoch_index in range(int(hparams["epochs"])):
        epoch_rows = list(train_rows)
        random.Random(seed + epoch_index).shuffle(epoch_rows)
        route_metrics = []
        for route_index, row in enumerate(epoch_rows, start=1):
            records = _load_optimizer_records(dataset_root, row, "train")
            metrics = run_hurdle_evidence_epoch(
                model,
                records,
                scales.to(device),
                device,
                optimizer=optimizer,
                seed=seed + epoch_index * 1000 + route_index,
                **common,
            )
            route_metrics.append((metrics, len(records)))
            del records
            gc.collect()
            if route_index % 5 == 0 or route_index == len(epoch_rows):
                print(
                    "[ExpandedHurdle] epoch=%d train_routes=%d/%d"
                    % (epoch_index + 1, route_index, len(epoch_rows)),
                    flush=True,
                )
        train_metrics = _weighted_metrics(route_metrics)
        validation_metrics = run_hurdle_evidence_epoch(
            model,
            validation_records,
            scales.to(device),
            device,
            optimizer=None,
            **common,
        )
        history.append(
            {
                "epoch": epoch_index + 1,
                "train": train_metrics,
                "route_validation": validation_metrics,
            }
        )
        if validation_metrics["total"] < best_value:
            best_value = validation_metrics["total"]
            best_epoch = epoch_index + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            "[ExpandedHurdle] epoch=%d/%d train=%.6f val=%.6f presence=%.6f magnitude=%.6f"
            % (
                epoch_index + 1,
                int(hparams["epochs"]),
                train_metrics["total"],
                validation_metrics["total"],
                validation_metrics["presence"],
                validation_metrics["magnitude"],
            ),
            flush=True,
        )
        rule = config["early_stop"]
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
                    "[ExpandedHurdle] EARLY_STOP_ROUTE_OVERFIT=1 completed_epochs=%d"
                    % len(history),
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("expanded hurdle checkpoint selection failed")
    training_state_path = args.output.with_suffix(".training_state.pt")
    torch.save(
        {
            "schema_version": run_contract["training_state_schema"],
            "student_state": best_state,
            "best_epoch": best_epoch,
            "best_route_validation_total": best_value,
            "early_stopped": early_stopped,
            "history": history,
            "component_scales": scales,
            "support_thresholds": support_thresholds,
            "inputs": {
                **hashes,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
        },
        training_state_path,
    )
    print(
        "[ExpandedHurdle] TRAINING_STATE_SAVED=%s" % training_state_path,
        flush=True,
    )
    model.load_state_dict(best_state)
    evaluation = evaluate_evidence_records(
        model,
        validation_records,
        scales.to(device),
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
    )
    hurdle = evaluate_hurdle_diagnostics(
        model,
        validation_records,
        scales.to(device),
        device,
        batch_size=int(hparams["evaluation_batch_size"]),
        response_floor=float(scaling["response_floor"]),
        support_thresholds=support_thresholds.to(device),
    )
    validation_by_stratum = {}
    for name, route_ids in (
        ("legacy_validation_5", legacy_routes),
        ("added_validation_5", added_routes),
    ):
        records = [
            record for record in validation_records if record.route_id in route_ids
        ]
        if len(records) != 320:
            raise RuntimeError("validation stratum record count differs: %s" % name)
        validation_by_stratum[name] = evaluate_evidence_records(
            model,
            records,
            scales.to(device),
            device,
            batch_size=int(hparams["evaluation_batch_size"]),
        )

    best_validation = history[best_epoch - 1]["route_validation"]
    gate = _route_validation_gate(
        evaluation, hurdle, best_validation, config["smoke_gate"]
    )
    comparison = _comparison(
        evaluation,
        config["comparison_baseline_35_route"],
        baseline_label="baseline_35_route",
        current_label="current_70_route",
    )
    comparison_to_expanded = None
    if "comparison_baseline_expanded_70_route" in config:
        comparison_to_expanded = _comparison(
            evaluation,
            config["comparison_baseline_expanded_70_route"],
            baseline_label="baseline_expanded_70_route",
            current_label="current_no_view_repair",
        )
    checkpoint = {
        "schema_version": run_contract["checkpoint_schema"],
        "student_state": best_state,
        "model_config": {
            "feature_dim": int(hparams["feature_dim"]),
            "hidden_dim": int(hparams["hidden_dim"]),
            "max_views": int(hparams["max_views"]),
            "presence_bias": float(hparams["presence_bias"]),
            "magnitude_bias": float(hparams["magnitude_bias"]),
            "use_view_embedding": use_view_embedding,
        },
        "component_scales": scales,
        "support_thresholds": support_thresholds,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "history": history,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    report = {
        "schema_version": run_contract["report_schema"],
        "inputs": {
            **hashes,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture_variant": run_kind,
        "use_view_embedding": use_view_embedding,
        "best_epoch": best_epoch,
        "best_route_validation_total": best_value,
        "early_stopped": early_stopped,
        "completed_epochs": len(history),
        "component_scales": {
            name: float(value) for name, value in zip(EVIDENCE_COMPONENTS, scales)
        },
        "support_thresholds": {
            name: float(value)
            for name, value in zip(EVIDENCE_COMPONENTS, support_thresholds)
        },
        "history": history,
        "route_validation": evaluation,
        "route_validation_by_stratum": validation_by_stratum,
        "hurdle_diagnostics": hurdle,
        "comparison_to_35_route_hidden128_epoch1": comparison,
        "comparison_to_expanded_70_route_hidden128": comparison_to_expanded,
        "gate": gate,
        "decision": (
            (
                "no_view_repair_gate_passed_review_before_heldout"
                if gate["passed"]
                else "no_view_repair_gate_failed_stop_before_heldout"
            )
            if run_kind == "no_view_repair"
            else (
                "expanded_adapter_gate_passed_review_before_heldout"
                if gate["passed"]
                else "expanded_adapter_gate_failed_stop_same_objective_expansion"
            )
        ),
        "scope_attestation": {
            "optimizer_families": list(OPTIMIZER_FAMILIES),
            "optimizer_train_routes": 70,
            "checkpoint_selection_routes": 10,
            "use_view_embedding": use_view_embedding,
            "input_confound_diagnostic_read": input_confound_report is not None,
            "route_shards_mmap_streamed": True,
            "dataset_copy_written": False,
            "validation_co_sharded_glare_metadata_visible": True,
            "validation_co_sharded_glare_tensor_values_accessed": False,
            "held_out_split_tensor_values_accessed": False,
            "corruption_mask_optimizer_weight": 0.0,
            "actual_target_optimizer_weight": 0.0,
            "native_weather_read": False,
            "orion_finetuning": False,
            "stage_b": False,
            "automatic_followup": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "best_epoch": best_epoch,
                "completed_epochs": len(history),
                "early_stopped": early_stopped,
                "gate": gate,
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("COUNTERFACTUAL_EVIDENCE_EXPANDED_HURDLE_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
