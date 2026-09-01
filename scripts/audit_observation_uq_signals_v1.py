#!/usr/bin/env python3
"""Compare clean-calibrated UQ candidates on an immutable feature shard."""

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

from uq_estimator.observation_uq_shard import (  # noqa: E402
    examples_from_feature_shard,
    load_feature_shard,
)
from uq_estimator.observation_uq_signal_audit import (  # noqa: E402
    SIGNAL_AUDIT_SCHEMA_VERSION,
    apply_calibrator,
    attach_route_shift_diagnostics,
    evaluate_detailed_score_maps,
    evaluate_score_maps,
    feature_rms,
    fit_clean_position_calibrator,
    paired_clean_delta_maps,
    spatial_neighbor_residual,
    temporal_cosine_residual,
)
from uq_estimator.observation_uq_v3 import (  # noqa: E402
    CleanConditionalTeacher,
    ObservationUQError,
    _batches,
    _collate,
    conditional_surprise,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_teachers(checkpoint, device):
    config = checkpoint["model_config"]
    teachers = []
    for state in checkpoint["teacher_states"]:
        teacher = CleanConditionalTeacher(
            feature_dim=int(config["feature_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            max_views=int(config["max_views"]),
            mask_block_size=int(config["mask_block_size"]),
            mask_halo=int(config["mask_halo"]),
        ).to(device)
        teacher.load_state_dict(state)
        teacher.eval()
        teachers.append(teacher)
    if not teachers:
        raise ObservationUQError("Teacher checkpoint has no ensemble states")
    return teachers


@torch.no_grad()
def _compute_raw_maps(examples, teachers, checkpoint, batch_size, device):
    result = {
        "teacher_raw": {},
        "temporal_raw": {},
        "spatial_raw": {},
        "feature_rms_raw": {},
    }
    disagreement_weight = float(
        checkpoint["training_config"]["disagreement_weight"]
    )
    clean_scale = float(checkpoint["clean_scale"])
    for batch in _batches(examples, batch_size, False, 0):
        current, previous, valid = _collate(batch, device)
        score_batch = {
            "teacher_raw": conditional_surprise(
                teachers, current, previous, valid, disagreement_weight
            )
            / clean_scale,
            "temporal_raw": temporal_cosine_residual(current, previous, valid),
            "spatial_raw": spatial_neighbor_residual(current),
            "feature_rms_raw": feature_rms(current),
        }
        for signal, scores in score_batch.items():
            for index, item in enumerate(batch):
                result[signal][item.sample_id] = scores[index].cpu().float()
    return result


def _gate_candidate(validation, heldout):
    checks = []
    for split, evaluation in (
        ("validation_heldout_family", validation),
        ("heldout_route_and_family", heldout),
    ):
        auc = float(evaluation["corruption_mask_patch_auroc_diagnostic_only"])
        checks.append(
            {
                "split": split,
                "metric": "mask_auroc",
                "value": auc,
                "threshold": 0.55,
                "passed": auc >= 0.55,
            }
        )
        for family, row in evaluation["by_family"].items():
            if family == "clean":
                continue
            rho = float(row["severity_score_spearman"])
            uplift = float(row["score_uplift_over_clean"])
            ratio = float(row["uplift_to_clean_route_shift_ratio"])
            checks.extend(
                [
                    {
                        "split": split,
                        "family": family,
                        "metric": "positive_uplift",
                        "value": uplift,
                        "threshold": 0.0,
                        "passed": uplift > 0.0,
                    },
                    {
                        "split": split,
                        "family": family,
                        "metric": "severity_spearman",
                        "value": rho,
                        "threshold": 0.10,
                        "passed": rho >= 0.10,
                    },
                    {
                        "split": split,
                        "family": family,
                        "metric": "uplift_to_route_shift_ratio",
                        "value": ratio,
                        "threshold": 0.25,
                        "passed": ratio >= 0.25,
                    },
                ]
            )
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def _temporal_followup_gate(detailed):
    checks = []
    for split in ("validation_heldout_family", "heldout_route_and_family"):
        row = detailed[split]
        route_aucs = sorted(
            float(value["corruption_mask_patch_auroc_diagnostic_only"])
            for value in row["by_route"].values()
        )
        passing_routes = sum(value >= 0.55 for value in route_aucs)
        required_routes = max(1, int(math.ceil(0.8 * len(route_aucs))))
        median_auc = route_aucs[len(route_aucs) // 2]
        valid_auc = float(
            row["previous_valid_only"][
                "corruption_mask_patch_auroc_diagnostic_only"
            ]
        )
        checks.extend(
            [
                {
                    "split": split,
                    "metric": "routes_with_auc_at_least_0.55",
                    "value": passing_routes,
                    "threshold": required_routes,
                    "route_count": len(route_aucs),
                    "passed": passing_routes >= required_routes,
                },
                {
                    "split": split,
                    "metric": "median_route_auc",
                    "value": median_auc,
                    "threshold": 0.60,
                    "passed": median_auc >= 0.60,
                },
                {
                    "split": split,
                    "metric": "previous_valid_only_auc",
                    "value": valid_auc,
                    "threshold": 0.60,
                    "passed": valid_auc >= 0.60,
                },
            ]
        )
        for family, severities in row["by_severity"].items():
            ordered = sorted(
                (float(severity), value) for severity, value in severities.items()
            )
            for severity, value in ordered:
                gap = float(value["inside_minus_outside"])
                checks.append(
                    {
                        "split": split,
                        "family": family,
                        "severity": severity,
                        "metric": "mask_inside_minus_outside",
                        "value": gap,
                        "threshold": 0.0,
                        "passed": gap > 0.0,
                    }
                )
            if len(ordered) >= 2:
                low = float(ordered[0][1]["example_score_mean"])
                high = float(ordered[-1][1]["example_score_mean"])
                checks.append(
                    {
                        "split": split,
                        "family": family,
                        "metric": "higher_severity_higher_mean",
                        "value": high - low,
                        "threshold": 0.0,
                        "passed": high > low,
                    }
                )
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)

    payload = load_feature_shard(args.shard)
    examples = examples_from_feature_shard(payload)
    checkpoint = torch.load(args.teacher, map_location="cpu")
    teachers = _load_teachers(checkpoint, device)
    clean_train = [
        item for item in examples if item.split == "train" and item.family == "clean"
    ]
    clean_validation = [
        item
        for item in examples
        if item.split == "validation" and item.family == "clean"
    ]
    clean_heldout = [
        item
        for item in examples
        if item.split == "held_out" and item.family == "clean"
    ]
    validation = [item for item in examples if item.split == "validation"]
    heldout = [item for item in examples if item.split == "held_out"]
    if not all((clean_train, clean_validation, clean_heldout, validation, heldout)):
        raise ObservationUQError("signal audit requires all clean/evaluation splits")

    raw = _compute_raw_maps(
        examples, teachers, checkpoint, args.batch_size, device
    )
    candidates = dict(raw)
    calibration = {}
    for source, tail in (
        ("teacher_raw", "positive"),
        ("temporal_raw", "positive"),
        ("spatial_raw", "positive"),
        ("feature_rms_raw", "absolute"),
    ):
        calibrator = fit_clean_position_calibrator(
            raw[source], clean_train, tail=tail
        )
        name = source.replace("_raw", "_viewpos_z")
        candidates[name] = apply_calibrator(raw[source], calibrator)
        calibration[name] = {
            "source": source,
            "tail": tail,
            "example_count": calibrator.example_count,
            "families": ["clean"],
            "route_count": len({item.route_id for item in clean_train}),
        }
    candidates["paired_clean_delta_oracle"] = paired_clean_delta_maps(payload)

    evaluations = {}
    gates = {}
    clean_train_evaluations = {}
    for name, score_maps in candidates.items():
        train_result = evaluate_score_maps(clean_train, score_maps)
        validation_result = attach_route_shift_diagnostics(
            train_result, evaluate_score_maps(validation, score_maps)
        )
        heldout_result = attach_route_shift_diagnostics(
            train_result, evaluate_score_maps(heldout, score_maps)
        )
        clean_train_evaluations[name] = train_result
        evaluations[name] = {
            "validation_heldout_family": validation_result,
            "heldout_route_and_family": heldout_result,
        }
        if name != "paired_clean_delta_oracle":
            gates[name] = _gate_candidate(validation_result, heldout_result)
        print(
            "[ObservationUQSignalAudit] signal=%s val_auc=%.6f heldout_auc=%.6f gate=%s"
            % (
                name,
                validation_result["corruption_mask_patch_auroc_diagnostic_only"],
                heldout_result["corruption_mask_patch_auroc_diagnostic_only"],
                gates.get(name, {}).get("passed", "oracle_only"),
            ),
            flush=True,
        )

    detailed_evaluations = {}
    for name in ("temporal_viewpos_z", "paired_clean_delta_oracle"):
        detailed_evaluations[name] = {
            "validation_heldout_family": evaluate_detailed_score_maps(
                validation, candidates[name]
            ),
            "heldout_route_and_family": evaluate_detailed_score_maps(
                heldout, candidates[name]
            ),
        }
    temporal_followup_gate = _temporal_followup_gate(
        detailed_evaluations["temporal_viewpos_z"]
    )

    report = {
        "schema_version": SIGNAL_AUDIT_SCHEMA_VERSION,
        "inputs": {
            "shard": str(args.shard.resolve()),
            "shard_sha256": _sha256_file(args.shard),
            "teacher": str(args.teacher.resolve()),
            "teacher_sha256": _sha256_file(args.teacher),
        },
        "data_attestation": {
            "calibrator_example_count": len(clean_train),
            "calibrator_route_count": len({item.route_id for item in clean_train}),
            "calibrator_families": ["clean"],
            "corruption_metadata_used_for_score_or_calibration": False,
            "corruption_metadata_used_for_evaluation_only": True,
            "actual_target_read": False,
            "adapter_trained": False,
            "paired_clean_delta_is_deployable": False,
        },
        "calibration": calibration,
        "clean_train_evaluations": clean_train_evaluations,
        "evaluations": evaluations,
        "detailed_evaluations": detailed_evaluations,
        "candidate_gates": gates,
        "temporal_followup_gate": temporal_followup_gate,
        "gate_policy": {
            "mask_auroc": 0.55,
            "positive_uplift": True,
            "severity_spearman": 0.10,
            "uplift_to_clean_route_shift_ratio": 0.25,
            "all_checks_must_pass_both_splits": True,
            "adapter_authorized": False,
            "stage_b_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_passes": {
            name: row["passed"] for name, row in gates.items()
        },
        "temporal_followup_passed": temporal_followup_gate["passed"],
        "adapter_authorized": False,
        "stage_b_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
