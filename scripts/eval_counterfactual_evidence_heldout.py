#!/usr/bin/env python3
"""Evaluate one frozen hurdle adapter on untouched local_glare route shards."""

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
    amended_training_gate,
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


SCHEMA_VERSION = "orion.counterfactual-evidence-heldout-family-report/v1"
CHECKPOINT_SCHEMA = "orion.counterfactual-evidence-no-view-repair-checkpoint/v1"
AMENDMENT_SCHEMA = "orion.counterfactual-evidence-reference-semantics-amendment/v1"
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


def _check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != str(expected):
        raise CounterfactualHeldoutError("%s hash differs" % label)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite held-out report")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA held-out evaluation requested but unavailable")

    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if amendment.get("schema_version") != AMENDMENT_SCHEMA:
        raise CounterfactualHeldoutError("reference-semantics amendment differs")
    frozen = amendment["frozen_inputs"]
    _check_hash(args.dataset_manifest, frozen["dataset_manifest_sha256"], "dataset")
    _check_hash(args.checkpoint, frozen["no_view_checkpoint_sha256"], "checkpoint")
    _check_hash(
        args.training_report,
        frozen["no_view_training_report_sha256"],
        "training report",
    )
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    entry_gate = amended_training_gate(training_report["gate"])
    if entry_gate["relative_core_passed"] is not True:
        raise CounterfactualHeldoutError("seven-metric relative training core failed")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CounterfactualHeldoutError("no-view checkpoint schema differs")
    model_config = checkpoint.get("model_config", {})
    if model_config.get("use_view_embedding") is not False:
        raise CounterfactualHeldoutError("checkpoint is not the frozen no-view model")
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
        raise CounterfactualHeldoutError("expanded dataset manifest contract differs")
    dataset_root = args.dataset_manifest.parent
    evaluations = {}
    gates = {}
    population = {}
    for split in SPLITS:
        rows = sorted(
            [dict(row) for row in manifest["shards"] if row.get("split") == split],
            key=_route_id,
        )
        if len(rows) != 10:
            raise CounterfactualHeldoutError("%s must contain ten routes" % split)
        records = []
        for index, row in enumerate(rows, start=1):
            shard_path = dataset_root / str(row["file"])
            _check_hash(shard_path, str(row["sha256"]), "%s route shard" % split)
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
                    "%s local_glare route population differs" % split
                )
            records.extend(route_records)
            print(
                "[HeldoutFamily] split=%s routes=%d/%d route=%s"
                % (split, index, len(rows), _route_id(row)),
                flush=True,
            )
        evaluation = evaluate_evidence_records(
            model, records, scales, device, batch_size=args.batch_size
        )
        gate = heldout_family_transfer_gate(
            evaluation, amendment["transfer_gate"]
        )
        evaluations[split] = evaluation
        gates[split] = gate
        population[split] = {
            "route_count": len(rows),
            "record_count": len(records),
            "route_ids": [_route_id(row) for row in rows],
        }
        print(
            "[HeldoutFamily] split=%s spearman=%.6f auroc=%.6f pass=%s"
            % (
                split,
                evaluation["combined_patch_spearman"],
                evaluation["combined_target_top20_auroc"],
                gate["passed"],
            ),
            flush=True,
        )

    both_passed = all(gates[split]["passed"] for split in SPLITS)
    if both_passed:
        decision = "proceed_to_separately_frozen_native_weather_evaluation"
    elif gates["validation"]["passed"] and not gates["held_out"]["passed"]:
        decision = "stop_for_route_domain_shift_diagnosis"
    else:
        decision = "stop_for_intervention_family_transfer_failure"
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "dataset_manifest_sha256": _sha256(args.dataset_manifest),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "training_report": str(args.training_report.resolve()),
            "training_report_sha256": _sha256(args.training_report),
            "amendment": str(args.amendment.resolve()),
            "amendment_sha256": _sha256(args.amendment),
        },
        "quantity": amendment["quantity"],
        "entry_gate": entry_gate,
        "population": population,
        "evaluations": evaluations,
        "gates": {**gates, "both_splits_passed": both_passed},
        "scope_attestation": {
            "evaluated_family": FAMILY,
            "optimizer_families_read": False,
            "checkpoint_updated": False,
            "adapter_trained": False,
            "native_weather_read": False,
            "orion_finetuned": False,
            "closed_loop_stage_b_run": False,
            "corruption_mask_used_for_scoring": False,
            "synthetic_transfer_is_external_validity_evidence": False,
        },
        "decision": decision,
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
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
