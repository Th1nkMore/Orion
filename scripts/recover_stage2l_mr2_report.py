#!/usr/bin/env python3
"""Recover the MR2 report from the checkpoint of the completed 40-step run.

The original bounded job completed training and after-evaluation, saved its
checkpoint, and then failed while serializing the report because the MR2
protocol calls its coverage section ``coverage_change`` while the reused MR1
report builder required ``known_coverage_gaps``.  This recovery entry point
performs no optimization.  It deterministically reconstructs the initial
evaluation, loads the already-saved trainable state, recomputes the final
evaluation, and writes a new report with explicit recovery provenance.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from mmcv.utils import set_random_seed

import scripts.train_stage2l_mr1_smoke as base
import scripts.train_stage2l_mr2_coverage_smoke as mr2
from scripts.scenario_factory_lib import sha256_file
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    TaskRelevanceMapHead,
    TaskRiskLanguageBridge,
    UQComponentTokenizer,
)
from uq_estimator.stage2l_structured_field_head import VLMTaskSemanticFieldHead


SCHEMA = "orion.stage2l_mr2_report_recovery.v1"
EXPECTED_FAILURE = "KeyError: 'known_coverage_gaps'"
CHECKPOINT_NAME = "stage2l_mr1_multiroute_smoke.pt"
TRAINABLE_STATE_KEYS = (
    "uq_tokenizer",
    "relevance_queries",
    "relevance_head",
    "risk_bridge",
    "task_field_head",
    "lora",
)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def parse_history(log_path: Path) -> list[Dict[str, Any]]:
    prefix = "[Stage2LMR1] "
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            if not isinstance(value, dict):
                raise ValueError("history line is not a JSON object")
            rows.append(value)
    return rows


def _all_json_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_json_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_json_numbers_finite(item) for item in value)
    return False


def validate_history(
    history: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int,
    expected_events: Iterable[str],
) -> Dict[str, int]:
    event_ids = set(expected_events)
    if [int(row.get("optimizer_step", -1)) for row in history] != list(
        range(1, expected_steps + 1)
    ):
        raise ValueError("failed log does not contain exactly the expected steps")
    presentations = {event: 0 for event in event_ids}
    for row in history:
        current = list(row.get("primary_event_ids", []))
        if (
            int(row.get("primary_group_count", -1)) != len(event_ids)
            or len(current) != len(event_ids)
            or set(current) != event_ids
        ):
            raise ValueError("a recovered history step is not event balanced")
        if not all(
            row.get(flag) is True
            for flag in ("finite_loss", "finite_gradient_norm", "finite_gradients")
        ):
            raise ValueError("a recovered history step is non-finite")
        if not _all_json_numbers_finite(row):
            raise ValueError("a recovered history step contains NaN or infinity")
        for event in current:
            presentations[event] += 1
    if set(presentations.values()) != {expected_steps}:
        raise ValueError("per-event presentation counts differ from the frozen run")
    return dict(sorted(presentations.items()))


def _checks(
    *,
    after: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    train_events: Iterable[str],
    group_split: Mapping[str, str],
    gates: Mapping[str, float],
) -> Dict[str, bool]:
    train = after["train"]
    dev = after["dev"]
    expected_events = set(train_events)
    return {
        "runtime_first_two_steps_finite": all(
            all(
                row.get(flag) is True
                for flag in (
                    "finite_loss",
                    "finite_gradient_norm",
                    "finite_gradients",
                )
            )
            for row in history[:2]
        ),
        "every_step_covers_all_13_train_events": all(
            row["primary_group_count"] == len(expected_events)
            and len(set(row["primary_event_ids"])) == len(expected_events)
            and set(row["primary_event_ids"]) == expected_events
            for row in history
        ),
        "dev_labels_never_enter_optimizer": all(
            all(group_split[value] == "train" for value in row["primary_group_ids"])
            and group_split[row["language_group_id"]] == "train"
            for row in history
        ),
        "train_relevance_foreground_recall": (
            train["relevance_support"]["foreground_recall"]
            >= gates["train_min_foreground_recall"]
        ),
        "train_relevance_background_fpr": (
            train["relevance_support"]["background_false_positive_rate"]
            <= gates["train_max_background_fpr"]
        ),
        "train_all_groups_positive_order": (
            train["ranking"]["positive_order_fraction"] == 1.0
        ),
        "train_all_groups_attain_margin": (
            train["ranking"]["minimum_attained_fraction"]
            >= gates["train_min_oracle_fraction"]
        ),
        "train_task_field_accuracy": (
            train["task_fields"]["overall_accuracy"]
            >= gates["train_min_task_field_accuracy"]
        ),
        "dev_relevance_foreground_recall": (
            dev["relevance_support"]["foreground_recall"]
            >= gates["dev_min_foreground_recall"]
        ),
        "dev_relevance_background_fpr": (
            dev["relevance_support"]["background_false_positive_rate"]
            <= gates["dev_max_background_fpr"]
        ),
        "dev_all_groups_positive_order": (
            dev["ranking"]["positive_order_fraction"] == 1.0
        ),
        "dev_all_groups_attain_margin": (
            dev["ranking"]["minimum_attained_fraction"]
            >= gates["dev_min_oracle_fraction"]
        ),
        "dev_task_field_accuracy": (
            dev["task_fields"]["overall_accuracy"]
            >= gates["dev_min_task_field_accuracy"]
        ),
        "dev_supported_class_macro_recall": (
            dev["task_fields"]["supported_class_macro_recall"]
            >= gates["dev_min_supported_class_macro_recall"]
        ),
        "dev_zero_uq_absence_semantics": (
            dev["task_fields"]["zero_uq_complete_field_accuracy"]
            >= gates["dev_min_zero_uq_complete_field_accuracy"]
        ),
        "dev_stance_accuracy": (
            dev["task_fields"]["per_field_accuracy"]["stance"]
            >= gates["dev_min_stance_accuracy"]
        ),
        "deterministic_render_parse": (
            dev["deterministic_render"]["semantic_parse_rate"] == 1.0
        ),
        "deterministic_render_fields": (
            dev["deterministic_render"]["semantic_field_accuracy"]
            >= gates["dev_min_render_field_accuracy"]
        ),
        "trajectory_control_density_and_governor_disabled": True,
    }


def _validate_recovery_amendment(
    amendment: Mapping[str, Any],
    *,
    amendment_path: Path,
    recovery_script: Path,
    failed_log: Path,
    trained_checkpoint: Path,
    output_report: Path,
) -> None:
    inputs = amendment.get("validated_inputs", {})
    run = amendment.get("authorized_recovery", {})
    locks = amendment.get("launch_locks", {})
    expected = {
        "recovery_script_sha256": sha256_file(recovery_script),
        "failed_log_sha256": sha256_file(failed_log),
        "trained_checkpoint_sha256": sha256_file(trained_checkpoint),
    }
    if (
        amendment.get("schema") != "orion.stage2l_mr2_recovery_amendment.v1"
        or inputs != expected
        or run.get("optimizer_steps") != 0
        or run.get("may_update_weights") is not False
        or run.get("may_submit_training") is not False
        or int(run.get("maximum_recovery_submissions", 0)) != 1
        or Path(str(run.get("output_report", ""))).resolve()
        != output_report.resolve()
        or locks.get("formal_stage2l_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("route203_glare_allowed") is not False
        or amendment.get("amendment_path") != str(amendment_path.resolve())
    ):
        raise ValueError("MR2 recovery amendment is absent, stale, or too broad")


def _load_trained_state(
    checkpoint: Mapping[str, Any],
    *,
    lm,
    uq_tokenizer,
    relevance_queries,
    relevance_head,
    risk_bridge,
    field_head,
) -> None:
    missing_keys = set(TRAINABLE_STATE_KEYS) - set(checkpoint)
    if missing_keys:
        raise ValueError("trained checkpoint is incomplete: %s" % sorted(missing_keys))
    uq_tokenizer.load_state_dict(checkpoint["uq_tokenizer"], strict=True)
    relevance_queries.load_state_dict(checkpoint["relevance_queries"], strict=True)
    relevance_head.load_state_dict(checkpoint["relevance_head"], strict=True)
    risk_bridge.load_state_dict(checkpoint["risk_bridge"], strict=True)
    field_head.load_state_dict(checkpoint["task_field_head"], strict=True)
    result = lm.load_state_dict(checkpoint["lora"], strict=False)
    unexpected = list(result.unexpected_keys)
    missing_lora = [name for name in result.missing_keys if "lora_" in name]
    if unexpected or missing_lora:
        raise ValueError(
            "LoRA state mismatch: unexpected=%s missing_lora=%s"
            % (unexpected, missing_lora)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--original-trainer", type=Path, required=True)
    parser.add_argument("--original-base-trainer", type=Path, required=True)
    parser.add_argument("--failed-log", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-amendment", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    paths = (
        args.config,
        args.base_checkpoint,
        args.dataset_manifest,
        args.training_protocol,
        args.trainer_preflight,
        args.launch_amendment,
        args.original_trainer,
        args.original_base_trainer,
        args.failed_log,
        args.trained_checkpoint,
        args.recovery_amendment,
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more recovery inputs are missing")
    if args.output_report.exists():
        raise FileExistsError("refusing to overwrite recovered report")
    if args.answer_batch_size != 2 or args.seed != 20260831:
        raise ValueError("recovery must preserve the frozen MR2 batch size and seed")
    if args.trained_checkpoint.name != CHECKPOINT_NAME:
        raise ValueError("unexpected MR2 checkpoint filename")

    failed_text = args.failed_log.read_text(encoding="utf-8", errors="replace")
    if EXPECTED_FAILURE not in failed_text or failed_text.count("Traceback (most recent call last):") != 1:
        raise ValueError("failed log does not have the attested terminal schema error")

    mr2._configure_base()
    protocol_path = args.training_protocol.resolve()
    protocol = _read_json(protocol_path)
    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    assets = base.MultiRouteAssets(args.dataset_manifest)
    base._validate_protocol(
        protocol=protocol,
        protocol_path=protocol_path,
        project_root=Path(__file__).resolve().parents[1],
        assets=assets,
        max_optimizer_steps=40,
        language_anchors_per_step=6,
    )
    if sha256_file(args.original_trainer) != protocol["implementation_sources"][
        "scripts/train_stage2l_mr2_coverage_smoke.py"
    ]:
        raise ValueError("original MR2 trainer differs from frozen protocol")
    if sha256_file(args.original_base_trainer) != protocol["implementation_sources"][
        "scripts/train_stage2l_mr1_smoke.py"
    ]:
        raise ValueError("original MR1 base trainer differs from frozen protocol")

    mr2._patch_validated_input_lineage()
    expected_hashes = base._validated_input_hashes(
        assets=assets,
        protocol_path=protocol_path,
        trainer_preflight_path=args.trainer_preflight.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.base_checkpoint.resolve(),
    )
    preflight = _read_json(args.trainer_preflight)
    if (
        preflight.get("schema") != mr2.PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("trainer", {}).get("sha256")
        != sha256_file(args.original_base_trainer)
        or preflight.get("training_protocol", {}).get("sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError("original MR2 preflight is absent or stale")
    base._validate_amendment(
        amendment=_read_json(args.launch_amendment),
        expected_hashes=expected_hashes,
        output_dir=args.trained_checkpoint.parent.resolve(),
        max_optimizer_steps=40,
        answer_batch_size=args.answer_batch_size,
        language_anchors_per_step=6,
    )
    _validate_recovery_amendment(
        _read_json(args.recovery_amendment),
        amendment_path=args.recovery_amendment,
        recovery_script=Path(__file__).resolve(),
        failed_log=args.failed_log,
        trained_checkpoint=args.trained_checkpoint,
        output_report=args.output_report,
    )

    history = parse_history(args.failed_log)
    presentations = validate_history(
        history,
        expected_steps=40,
        expected_events=assets.event_groups["train"],
    )
    if not torch.cuda.is_available():
        raise SystemExit("MR2 report recovery requires CUDA for exact ORION evaluation")

    losses = protocol["losses"]
    required_oracle_fraction = float(
        losses["on_off_ranking"]["required_oracle_fraction"]
    )
    evaluation_args = {
        "assets": assets,
        "answer_batch_size": args.answer_batch_size,
        "required_oracle_fraction": required_oracle_fraction,
        "support_fraction": float(losses["dense_relevance"]["support_fraction"]),
        "calibration_bce_weight": float(
            losses["dense_relevance"]["calibration_bce_weight"]
        ),
        "background_support_weight": float(
            losses["dense_relevance"]["background_support_weight"]
        ),
        "background_probability_margin": float(
            losses["dense_relevance"]["background_probability_margin"]
        ),
    }

    lm, tokenizer = base._load_orion_lm(
        args.config.resolve(), args.base_checkpoint.resolve()
    )
    uq_tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_head = TaskRelevanceMapHead(
        model_dim=4096, hidden_dim=256
    ).cuda()
    risk_bridge = TaskRiskLanguageBridge(model_dim=4096, hidden_dim=256).cuda()
    field_head = VLMTaskSemanticFieldHead(model_dim=4096, hidden_dim=256).cuda()
    evaluation_args.update(
        {
            "lm": lm,
            "tokenizer": tokenizer,
            "uq_tokenizer": uq_tokenizer,
            "relevance_queries": relevance_queries,
            "relevance_head": relevance_head,
            "risk_bridge": risk_bridge,
            "field_head": field_head,
        }
    )
    before = {
        split: base._evaluate_split(
            **evaluation_args, split=split, generate_text=False
        )
        for split in ("train", "dev")
    }
    trained = torch.load(args.trained_checkpoint, map_location="cpu")
    if (
        trained.get("schema") != mr2.SCHEMA
        or int(trained.get("optimizer_steps", -1)) != 40
        or trained.get("engineering_preexperiment_only") is not True
        or trained.get("formal_training_ready") is not False
        or trained.get("stage2p_ready") is not False
    ):
        raise ValueError("saved MR2 checkpoint identity or locks differ")
    _load_trained_state(
        trained,
        lm=lm,
        uq_tokenizer=uq_tokenizer,
        relevance_queries=relevance_queries,
        relevance_head=relevance_head,
        risk_bridge=risk_bridge,
        field_head=field_head,
    )
    after = {
        split: base._evaluate_split(
            **evaluation_args, split=split, generate_text=(split == "dev")
        )
        for split in ("train", "dev")
    }
    checks = _checks(
        after=after,
        history=history,
        train_events=assets.event_groups["train"],
        group_split=assets.group_split,
        gates=protocol["release_gates"],
    )
    diagnostics = {
        "train_auxiliary_language_nll_decreases": (
            after["train"]["diagnostic_mean_auxiliary_language_nll"]
            < before["train"]["diagnostic_mean_auxiliary_language_nll"]
        ),
        "dev_auxiliary_language_nll_decreases": (
            after["dev"]["diagnostic_mean_auxiliary_language_nll"]
            < before["dev"]["diagnostic_mean_auxiliary_language_nll"]
        ),
        "free_generation_is_release_evidence": False,
        "unsupported_spatial_classes_are_release_gates": False,
        "formal_human_per_frame_review_complete": False,
    }
    passed = all(checks.values())
    status = (
        "engineering_mr1_multiroute_smoke_pass"
        if passed
        else "engineering_mr1_multiroute_smoke_failed_gate"
    )
    report = {
        "schema": mr2.SCHEMA,
        "status": status,
        "engineering_preexperiment_only": True,
        "formal_training_ready": False,
        "stage2p_ready": False,
        "optimizer_steps": len(history),
        "primary_group_presentations": len(history) * mr2.EXPECTED_TRAIN_EVENT_COUNT,
        "language_anchor_presentations": len(history) * 6,
        "train_events": sorted(assets.event_groups["train"]),
        "dev_events": sorted(assets.event_groups["dev"]),
        "before": before,
        "after": after,
        "checks": checks,
        "diagnostics": diagnostics,
        "history": history,
        "architecture": {
            "stage1_adapter_frozen_and_task_agnostic": True,
            "task_relevance_owned_by_vlm": True,
            "fixed_k_equals_u_times_sigmoid_r": True,
            "field_head_reads_u_and_k": True,
            "task_field_gradient_to_relevance_logits": False,
            "qa_language_gradient_to_relevance_logits": False,
            "dev_labels_enter_optimizer": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
            "trajectory_or_control_loss": False,
            "free_language_is_release_evidence": False,
        },
        "known_coverage_gaps": protocol["coverage_change"],
        "diagnostic_identity": {
            "name": "MR2 expanded event/class coverage smoke",
            "intended_change_from_mr1_40": (
                "8 events (6/2) and 37 groups -> 17 events (13/4) and 80 "
                "groups; optimizer steps, per-event primary exposure, "
                "architecture, losses, seed and release thresholds remain fixed"
            ),
            "not_a_duration_extension": True,
            "not_formal_training": True,
        },
        "recovery": {
            "schema": SCHEMA,
            "reason": "terminal report schema mismatch after checkpoint save",
            "terminal_error": EXPECTED_FAILURE,
            "source_slurm_job_id": "1109479",
            "source_slurm_state": "FAILED",
            "optimizer_steps_replayed": 0,
            "weights_updated_during_recovery": False,
            "initial_evaluation_reconstructed_from_frozen_seed_and_sources": True,
            "final_evaluation_recomputed_from_saved_checkpoint": True,
            "history_event_presentations": presentations,
        },
        "provenance": {
            "validated_inputs": expected_hashes,
            "dataset_manifest": str(assets.manifest_path),
            "training_protocol": str(protocol_path),
            "trainer_preflight": str(args.trainer_preflight.resolve()),
            "launch_amendment": str(args.launch_amendment.resolve()),
            "original_failed_log": {
                "path": str(args.failed_log.resolve()),
                "sha256": sha256_file(args.failed_log),
            },
            "output_checkpoint": {
                "path": str(args.trained_checkpoint.resolve()),
                "sha256": sha256_file(args.trained_checkpoint),
            },
            "recovery_entrypoint": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "recovery_amendment": {
                "path": str(args.recovery_amendment.resolve()),
                "sha256": sha256_file(args.recovery_amendment),
            },
        },
        "claim_boundary": (
            "One bounded expanded-coverage engineering diagnostic recovered "
            "without optimizer updates. It does not authorize formal Stage2-L, "
            "Stage2-P, locked-test reading, planning, closed-loop, "
            "generalization, or safety claims."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "report": str(args.output_report.resolve()),
                "checkpoint": str(args.trained_checkpoint.resolve()),
                "optimizer_steps_replayed": 0,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    del lm, tokenizer, uq_tokenizer, relevance_queries, relevance_head, risk_bridge
    del field_head, trained
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
