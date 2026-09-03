#!/usr/bin/env python3
"""Evaluate a frozen pairwise adapter on untouched local_glare shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_evidence_heldout import (  # noqa: E402
    CounterfactualHeldoutError,
    heldout_family_transfer_gate,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    evaluate_evidence_records,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_route_shard_records_selective,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-pairwise-glare-report/v1"
CHECKPOINT_SCHEMA = "orion.counterfactual-evidence-pairwise-native-checkpoint/v1"
TRAINING_REPORT_SCHEMA = "orion.counterfactual-evidence-pairwise-native-report/v1"
PROTOCOL_SCHEMA = "orion.counterfactual-evidence-pairwise-native-protocol/v4"
FAMILY = "local_glare"
SPLITS = ("validation", "held_out")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_id(row: Mapping[str, Any]) -> str:
    route_ids = row.get("route_ids")
    if not isinstance(route_ids, list) or len(route_ids) != 1:
        raise CounterfactualHeldoutError("route shard must own exactly one route")
    return str(route_ids[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite pairwise glare report")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA pairwise glare evaluation requested but unavailable")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise CounterfactualHeldoutError("pairwise protocol differs")
    if _sha256(args.dataset_manifest) != protocol["frozen_inputs"][
        "synthetic_dataset_manifest_sha256"
    ]:
        raise CounterfactualHeldoutError("pairwise glare dataset hash differs")
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    if (
        training_report.get("schema_version") != TRAINING_REPORT_SCHEMA
        or training_report.get("decision")
        != "freeze_checkpoint_then_run_unchanged_glare_and_native_heldout_gates"
        or training_report.get("scope_attestation", {}).get(
            "local_glare_tensor_values_read"
        )
        is not False
    ):
        raise CounterfactualHeldoutError("pairwise training report contract differs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("requires_inference_baseline_calibration") is not True
        or checkpoint.get("best_epoch") != training_report.get("best_epoch")
    ):
        raise CounterfactualHeldoutError("pairwise checkpoint contract differs")
    model_config = checkpoint.get("model_config", {})
    if model_config.get("use_view_embedding") is not False:
        raise CounterfactualHeldoutError("pairwise checkpoint is not view-equivariant")
    device = torch.device(args.device)
    model = ObservationEvidenceHurdleAdapter(**model_config).to(device)
    model.load_state_dict(checkpoint["student_state"])
    model.eval()
    scales = checkpoint["component_scales"].to(device=device, dtype=torch.float32)

    manifest = load_fp16_dataset_manifest(args.dataset_manifest, verify_shards=False)
    if (
        manifest.get("schema_version") != FP16_DIRECT_DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("written_route_count") != 90
    ):
        raise CounterfactualHeldoutError("pairwise glare manifest contract differs")
    dataset_root = args.dataset_manifest.parent
    evaluations = {}
    gates = {}
    population = {}
    thresholds = protocol["unchanged_evaluation_gates"]["local_glare"]
    gate_thresholds = {
        "combined_patch_spearman_min": thresholds[
            "combined_patch_spearman_min"
        ],
        "combined_target_top20_auroc_min": thresholds[
            "combined_target_top20_auroc_min"
        ],
        "median_record_within_intervened_view_target_top20_auroc_min": thresholds[
            "median_record_within_intervened_view_target_top20_auroc_min"
        ],
        "severity_1_uplift_over_reference_min": 0.0,
        "severity_3_uplift_over_reference_min": 0.0,
        "severity_3_minus_severity_1_min": 0.0,
    }
    for split in SPLITS:
        rows = sorted(
            [dict(row) for row in manifest["shards"] if row.get("split") == split],
            key=_route_id,
        )
        if len(rows) != 10:
            raise CounterfactualHeldoutError(
                "%s pairwise glare split needs ten routes" % split
            )
        records = []
        for index, row in enumerate(rows, start=1):
            shard_path = dataset_root / str(row["file"])
            if not shard_path.is_file() or _sha256(shard_path) != str(row["sha256"]):
                raise CounterfactualHeldoutError("pairwise glare shard hash differs")
            route_records = load_fp16_route_shard_records_selective(
                shard_path, families=(FAMILY,)
            )
            if (
                len(route_records) != 32
                or {record.split for record in route_records} != {split}
                or {record.family for record in route_records} != {FAMILY}
                or {record.severity for record in route_records} != {1.0, 3.0}
            ):
                raise CounterfactualHeldoutError(
                    "%s pairwise glare route population differs" % split
                )
            records.extend(route_records)
            print(
                "[PairwiseGlare] split=%s routes=%d/%d route=%s"
                % (split, index, len(rows), _route_id(row)),
                flush=True,
            )
        evaluation = evaluate_evidence_records(
            model, records, scales, device, batch_size=args.batch_size
        )
        gate = heldout_family_transfer_gate(evaluation, gate_thresholds)
        evaluations[split] = evaluation
        gates[split] = gate
        population[split] = {
            "route_count": len(rows),
            "record_count": len(records),
            "route_ids": [_route_id(row) for row in rows],
        }

    both_passed = all(gates[split]["passed"] for split in SPLITS)
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "dataset_manifest_sha256": _sha256(args.dataset_manifest),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "training_report_sha256": _sha256(args.training_report),
            "protocol_sha256": _sha256(args.protocol),
        },
        "population": population,
        "evaluations": evaluations,
        "gates": {**gates, "both_splits_passed": both_passed},
        "scope_attestation": {
            "evaluated_family": FAMILY,
            "checkpoint_updated": False,
            "adapter_trained": False,
            "native_final_heldout_read": False,
            "paired_reference_used_for_model_score": False,
            "paired_reference_used_for_evaluation_metrics_only": True,
            "corruption_mask_used_for_scoring": False,
            "orion_finetuned": False,
            "stage_b": False,
        },
        "decision": (
            "proceed_to_frozen_pairwise_native_heldout_evaluation"
            if both_passed
            else "stop_pairwise_repair_for_glare_regression"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "both_splits_passed": both_passed,
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
