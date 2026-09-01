#!/usr/bin/env python3
"""Train and gate the v3 clean-conditional Teacher before adapter training."""

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

from uq_estimator.observation_uq_shard import (  # noqa: E402
    examples_from_feature_shard,
    load_feature_shard,
)
from uq_estimator.observation_uq_v3 import (  # noqa: E402
    run_clean_only_adapter_training,
    run_teacher_viability_training,
)


CONTINUATION_SCHEMA_VERSION = "orion.observation-uq-continuation/v3.1"


def _adapter_continuation_decision(checkpoint, config):
    thresholds = config.get("teacher_gate", {})
    min_auc = float(thresholds.get("min_mask_auroc", 0.55))
    min_uplift = float(thresholds.get("min_score_uplift", 0.005))
    min_spearman = float(thresholds.get("min_severity_spearman", 0.10))
    checks = []
    for split in (
        "validation_heldout_family",
        "heldout_route_and_family",
    ):
        evaluation = checkpoint["evaluations"][split]
        auc = float(evaluation["corruption_mask_patch_auroc_diagnostic_only"])
        checks.append(
            {
                "split": split,
                "metric": "corruption_mask_patch_auroc_diagnostic_only",
                "value": auc,
                "threshold": min_auc,
                "passed": auc >= min_auc,
            }
        )
        for family in checkpoint["training_config"]["heldout_families"]:
            row = evaluation["by_family"][family]
            uplift = float(row["teacher_score_uplift_over_clean"])
            spearman = float(row["severity_teacher_score_spearman"])
            checks.extend(
                [
                    {
                        "split": split,
                        "family": family,
                        "metric": "teacher_score_uplift_over_clean",
                        "value": uplift,
                        "threshold": min_uplift,
                        "passed": uplift >= min_uplift,
                    },
                    {
                        "split": split,
                        "family": family,
                        "metric": "severity_teacher_score_spearman",
                        "value": spearman,
                        "threshold": min_spearman,
                        "passed": spearman >= min_spearman,
                    },
                ]
            )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "policy": "all pre-registered development checks must pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-family", action="append", default=[])
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--teacher-members", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--disagreement-weight", type=float, default=0.25)
    parser.add_argument("--mask-block-size", type=int, default=4)
    parser.add_argument("--mask-halo", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=4)
    parser.add_argument(
        "--resume",
        type=Path,
        help="optional v3.1 progress checkpoint; teacher-epochs remains the total",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    heldout_families = args.heldout_family or ["local_glare"]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    payload = load_feature_shard(args.shard)
    examples = examples_from_feature_shard(payload)
    checkpoint = run_teacher_viability_training(
        examples=examples,
        heldout_families=heldout_families,
        output_path=args.output,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        teacher_members=args.teacher_members,
        teacher_epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        disagreement_weight=args.disagreement_weight,
        mask_block_size=args.mask_block_size,
        mask_halo=args.mask_halo,
        validation_interval=args.validation_interval,
        resume_path=args.resume,
        seed=args.seed,
        device=device,
    )
    summary = {
        "checkpoint": str(args.output.resolve()),
        "report": str(args.output.with_suffix(".report.json").resolve()),
        "schema_version": checkpoint["schema_version"],
        "data_attestation": checkpoint["data_attestation"],
        "teacher_first_loss": checkpoint["history"]["teacher_train"][0]["loss"],
        "teacher_last_loss": checkpoint["history"]["teacher_train"][-1]["loss"],
        "checkpoint_selection": checkpoint["checkpoint_selection"],
        "gate_inputs": checkpoint["gate_inputs"],
        "evaluations": checkpoint["evaluations"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))

    continuation_path = args.output.parent / "continue_adapter_on_pass.json"
    if continuation_path.exists():
        raw_config = continuation_path.read_bytes()
        continuation = json.loads(raw_config.decode("utf-8"))
        if continuation.get("schema_version") != CONTINUATION_SCHEMA_VERSION:
            raise SystemExit("unsupported continuation config %s" % continuation_path)
        decision = _adapter_continuation_decision(checkpoint, continuation)
        decision.update(
            {
                "schema_version": "orion.observation-uq-continuation-decision/v3.1",
                "config": str(continuation_path.resolve()),
                "config_sha256": hashlib.sha256(raw_config).hexdigest(),
                "teacher_checkpoint": str(args.output.resolve()),
                "adapter_authorized_by_config": bool(
                    continuation.get("run_clean_only_adapter_on_pass", False)
                ),
                "actual_target_training_authorized": False,
                "stage_b_authorized": False,
            }
        )
        decision_path = args.output.parent / "adapter_continuation_decision.json"
        decision_path.write_text(
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        print(
            "[ObservationUQContinuation] teacher_gate_passed=%s decision=%s"
            % (decision["passed"], decision_path),
            flush=True,
        )
        if decision["passed"] and decision["adapter_authorized_by_config"]:
            adapter_output = args.output.parent / "adapter_v31.pt"
            adapter_config = continuation.get("adapter", {})
            adapter_checkpoint = run_clean_only_adapter_training(
                examples=examples,
                teacher_checkpoint=checkpoint,
                output_path=adapter_output,
                adapter_epochs=int(adapter_config.get("epochs", 24)),
                batch_size=int(adapter_config.get("batch_size", args.batch_size)),
                learning_rate=float(
                    adapter_config.get("learning_rate", args.learning_rate)
                ),
                seed=int(adapter_config.get("seed", args.seed + 10000)),
                device=device,
            )
            print(
                json.dumps(
                    {
                        "adapter_checkpoint": str(adapter_output.resolve()),
                        "adapter_report": str(
                            adapter_output.with_suffix(".report.json").resolve()
                        ),
                        "schema_version": adapter_checkpoint["schema_version"],
                        "checkpoint_selection": adapter_checkpoint[
                            "checkpoint_selection"
                        ],
                        "data_attestation": adapter_checkpoint["data_attestation"],
                        "evaluations": adapter_checkpoint["evaluations"],
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=True,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
