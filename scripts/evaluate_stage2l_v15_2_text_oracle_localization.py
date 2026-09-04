#!/usr/bin/env python3
"""Localize the Stage2-L U bottleneck with an exact natural-language oracle.

All 120 frozen dev U states are rendered as task-free natural-language facts.
The continuous U-token channel is absent: ORION receives only its original 529
visual tokens plus the oracle text in the ordinary prompt.  The original ORION
checkpoint and the completed v15 LoRA are evaluated sequentially in one model
load.  No optimizer, bridge, R, task, planning, trajectory, control, or
closed-loop component is created.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import scripts.train_stage2l_mr1_smoke as base
import scripts.train_stage2l_v14_u_concept_lora_smoke as v14
from scripts.train_stage2l_route196_bridge_smoke import _generate
from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
)
from uq_estimator.stage2l_u_concept_qa_v14 import TAG_ORDER, U_VARIANTS
from uq_estimator.stage2l_u_text_oracle_v15_2 import (
    SCHEMA as PROMPT_SCHEMA,
    decode_candidate_nlls,
    parse_text_oracle_answer,
    text_oracle_candidates,
    text_oracle_field_row,
    text_oracle_full_row,
)


SCHEMA = "orion.stage2l-v15-2-text-oracle-localization/v1"
PREFLIGHT_SCHEMA = "orion.stage2l-v15-2-text-oracle-preflight/v1"
PROTOCOL_SCHEMA = "orion.stage2l-v15-2-text-oracle-protocol/v1"
V15_CHECKPOINT_SCHEMA = "orion.stage2l-v15-u-language-alignment-pilot/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validated_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "v11_records_sha256": _sha256(args.v11_records.resolve()),
        "dataset_audit_report_sha256": _sha256(
            args.dataset_audit_report.resolve()
        ),
        "view_feature_cache_sha256": _sha256(args.view_feature_cache.resolve()),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "v15_checkpoint_sha256": _sha256(args.v15_checkpoint.resolve()),
        "v15_report_sha256": _sha256(args.v15_report.resolve()),
    }


def _implementation_hashes() -> dict[str, str]:
    root = _project_root()
    relative_paths = (
        "scripts/evaluate_stage2l_v15_2_text_oracle_localization.py",
        "uq_estimator/stage2l_u_text_oracle_v15_2.py",
        "scripts/train_stage2l_mr1_smoke.py",
        "scripts/train_stage2l_v14_u_concept_lora_smoke.py",
        "scripts/train_stage2l_route196_bridge_smoke.py",
        "scripts/train_stage2l_v7_route151_smoke.py",
        "uq_estimator/stage2l_u_concept_qa_v14.py",
    )
    return {relative: _sha256(root / relative) for relative in relative_paths}


def _validate_protocol(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    architecture = protocol.get("architecture", {})
    evaluation = protocol.get("evaluation", {})
    output_key = "preflight_output" if args.preflight_only else "output"
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "bounded_text_oracle_localization"
        or protocol.get("prompt_schema") != PROMPT_SCHEMA
        or protocol.get("input_sha256") != _validated_inputs(args)
        or protocol.get("implementation_sha256") != _implementation_hashes()
        or protocol.get("training_performed") is not False
        or protocol.get("optimizer_steps") != 0
        or architecture.get("continuous_u_tokens_present") is not False
        or architecture.get("text_oracle_is_only_u_input") is not True
        or architecture.get("orion_visual_tokens") != 529
        or architecture.get("r_bridge_task_route_risk_action_present") is not False
        or architecture.get("trajectory_control_closed_loop_present") is not False
        or evaluation.get("split") != "dev"
        or evaluation.get("dev_groups") != 20
        or evaluation.get("u_states") != 120
        or evaluation.get("all_six_fields") is not True
        or evaluation.get("primary_decode") != "all_candidate_nll"
        or evaluation.get("model_controls")
        != ["original_orion", "v15_lora"]
        or evaluation.get("answer_micro_batch_size") != args.answer_batch_size
        or protocol.get("automatic_retry") is not False
        or Path(str(protocol.get(output_key, ""))).resolve()
        != args.output.resolve()
    ):
        raise ValueError("v15.2 text-oracle protocol is absent or stale")


def _sequence_audit(args: argparse.Namespace, assets: v14.UConceptAssets) -> dict:
    from mmcv.utils import Config
    from transformers import AutoTokenizer

    cfg = Config.fromfile(str(args.config.resolve()))
    tokenizer_path = Path(str(cfg.model.tokenizer))
    if not tokenizer_path.is_absolute():
        tokenizer_path = (args.config.resolve().parents[3] / tokenizer_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    maximum = 0
    longest = None
    dev_groups = assets.groups_for_split("dev")
    for group_id in dev_groups:
        for variant in U_VARIANTS:
            summary = assets.summaries[(group_id, variant)]
            for tag in TAG_ORDER:
                row = text_oracle_field_row(summary, tag)
                for answer in text_oracle_candidates(tag):
                    ids, _ = base._training_tokens(
                        tokenizer, row, "", answer=answer
                    )
                    expanded = int(ids.numel()) - 1 + 529
                    if expanded > maximum:
                        maximum = expanded
                        longest = {
                            "group_id": group_id,
                            "variant": variant,
                            "tag": tag,
                            "answer": answer,
                            "expanded_tokens": expanded,
                        }
    return {
        "model_max_length": 2048,
        "conditioning_tokens": 529,
        "maximum_expanded_tokens": maximum,
        "longest": longest,
        "passed": maximum <= 2048,
    }


def _preflight(
    args: argparse.Namespace,
    assets: v14.UConceptAssets,
) -> dict[str, Any]:
    dataset = v14._dataset_audit(assets)
    sequences = _sequence_audit(args, assets)
    checks = {
        "dataset_audit_passed": dataset["passed"],
        "exactly_twenty_dev_groups": len(assets.groups_for_split("dev")) == 20,
        "exactly_one_hundred_twenty_dev_states": (
            len(assets.groups_for_split("dev")) * len(U_VARIANTS) == 120
        ),
        "sequence_budget_passed": sequences["passed"],
        "all_field_vocabularies_finite": all(
            len(FIELD_VOCABULARIES[tag]) >= 2 for tag in TAG_ORDER
        ),
    }
    if not all(checks.values()):
        raise ValueError("v15.2 text-oracle preflight failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "text_oracle_localization_preflight_pass",
        "passed": True,
        "training_started": False,
        "gpu_used": False,
        "checks": checks,
        "validated_inputs": _validated_inputs(args),
        "implementation_sha256": _implementation_hashes(),
        "protocol_sha256": _sha256(args.protocol.resolve()),
        "dataset_audit": dataset,
        "sequence_audit": sequences,
        "evaluation_contract": {
            "model_controls": ["original_orion", "v15_lora"],
            "dev_groups": 20,
            "u_states": 120,
            "field_decisions_per_model": 720,
            "continuous_u_tokens_present": False,
            "training_performed": False,
        },
    }


def _validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight is None:
        raise ValueError("runtime requires a frozen preflight")
    value = _read_json(args.preflight.resolve())
    if (
        value.get("schema") != PREFLIGHT_SCHEMA
        or value.get("passed") is not True
        or value.get("training_started") is not False
        or value.get("validated_inputs") != _validated_inputs(args)
        or value.get("implementation_sha256") != _implementation_hashes()
        or value.get("protocol_sha256") != _sha256(args.protocol.resolve())
    ):
        raise ValueError("v15.2 text-oracle preflight is absent or stale")
    return value


def _validate_v15_lineage(
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(args.v15_checkpoint.resolve(), map_location="cpu")
    report = _read_json(args.v15_report.resolve())
    checkpoint_hash = _sha256(args.v15_checkpoint.resolve())
    if (
        payload.get("schema") != V15_CHECKPOINT_SCHEMA
        or payload.get("status") != "bounded_u_language_alignment_complete"
        or payload.get("optimizer_steps") != 720
        or payload.get("task_free_u_language_alignment") is not True
        or payload.get("r_bridge_route_task_risk_action_absent") is not True
        or payload.get("formal_stage2l_ready") is not False
        or payload.get("stage2p_ready") is not False
        or payload.get("closed_loop_eligible") is not False
        or report.get("schema") != V15_CHECKPOINT_SCHEMA
        or report.get("optimizer_steps") != 720
        or report.get("provenance", {}).get("checkpoint_sha256")
        != checkpoint_hash
    ):
        raise ValueError("completed v15 checkpoint/report lineage differs")
    state = payload.get("orion_lora")
    if not isinstance(state, dict) or not state:
        raise ValueError("v15 checkpoint has no ORION LoRA state")
    continuous = report.get("after", {}).get("dev", {})
    required = (
        "nonzero_accuracy_excluding_presence",
        "balanced_accuracy_by_tag",
        "counterfactual",
    )
    if any(key not in continuous for key in required):
        raise ValueError("v15 report lacks the frozen continuous-U comparison")
    return state, {
        "checkpoint_schema": payload["schema"],
        "checkpoint_status": payload["status"],
        "optimizer_steps": payload["optimizer_steps"],
        "lora_tensor_count": len(state),
        "lora_parameter_count": sum(value.numel() for value in state.values()),
        "continuous_u_after": {
            "nonzero_accuracy_excluding_presence": continuous[
                "nonzero_accuracy_excluding_presence"
            ],
            "balanced_accuracy_by_tag": continuous["balanced_accuracy_by_tag"],
            "counterfactual": continuous["counterfactual"],
        },
    }


def _load_v15_lora(lm, state: Mapping[str, torch.Tensor]) -> None:
    expected = {name for name in lm.state_dict() if "lora_" in name}
    if set(state) != expected:
        raise ValueError("v15 LoRA keys differ from the loaded ORION model")
    result = lm.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        "lora_" in name for name in result.missing_keys
    ):
        raise ValueError("v15 LoRA load was incomplete")
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad = False


@torch.no_grad()
def _decode_field(
    *, lm, tokenizer, vision: torch.Tensor, summary, tag: str, batch_size: int
) -> dict[str, Any]:
    row = text_oracle_field_row(summary, tag)
    answers = text_oracle_candidates(tag)
    nll_tensor = base._answer_nlls_mr1(
        lm=lm,
        tokenizer=tokenizer,
        vision=vision,
        row=row,
        route_text="",
        answers=answers,
        micro_batch_size=batch_size,
    )
    nlls = [float(value) for value in nll_tensor.detach().cpu().tolist()]
    expected = summary.fields()[tag]
    predicted = decode_candidate_nlls(tag, nlls)
    target_index = answers.index(expected)
    wrong = [value for index, value in enumerate(nlls) if index != target_index]
    return {
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "target_margin": float(min(wrong) - nlls[target_index]),
        "candidate_nlls": dict(zip(answers, nlls)),
    }


def _balanced_accuracy(entries: Sequence[Mapping[str, Any]], tag: str) -> float:
    recalls = []
    for value in FIELD_VOCABULARIES[tag]:
        supported = [entry for entry in entries if entry["expected"] == value]
        if supported:
            recalls.append(float(np.mean([entry["correct"] for entry in supported])))
    if not recalls:
        raise ValueError("field has no supported class")
    return float(np.mean(recalls))


def _counterfactual_metrics(records: Mapping[str, Mapping[str, Any]]) -> dict:
    changed = []
    unchanged = []
    unchanged_correct = []
    group_ids = sorted({key.split("::", 1)[0] for key in records})
    for group_id in group_ids:
        for left_variant, right_variant in combinations(U_VARIANTS, 2):
            left = records["%s::%s" % (group_id, left_variant)]
            right = records["%s::%s" % (group_id, right_variant)]
            for tag in TAG_ORDER:
                left_field = left["fields"][tag]
                right_field = right["fields"][tag]
                if left_field["expected"] != right_field["expected"]:
                    changed.append(left_field["correct"] and right_field["correct"])
                else:
                    invariant = left_field["predicted"] == right_field["predicted"]
                    unchanged.append(invariant)
                    unchanged_correct.append(
                        invariant
                        and left_field["correct"]
                        and right_field["correct"]
                    )
    return {
        "changed_field_pair_count": len(changed),
        "changed_field_exact_response_fraction": float(np.mean(changed)),
        "unchanged_field_pair_count": len(unchanged),
        "unchanged_field_invariance_fraction": float(np.mean(unchanged)),
        "unchanged_field_correct_invariance_fraction": float(
            np.mean(unchanged_correct)
        ),
    }


@torch.no_grad()
def _evaluate_model(
    *, lm, tokenizer, assets: v14.UConceptAssets, batch_size: int
) -> dict[str, Any]:
    lm.eval()
    dev_groups = assets.groups_for_split("dev")
    if len(dev_groups) != 20:
        raise ValueError("text oracle requires all twenty dev groups")
    representative_groups = {
        groups[0] for _, groups in sorted(assets.event_groups["dev"].items())
    }
    records = {}
    by_tag = defaultdict(list)
    free = []
    for group_id in dev_groups:
        vision = assets.visual_contexts[group_id].cuda(non_blocking=True)
        if tuple(vision.shape) != (1, 529, 4096):
            raise RuntimeError("original ORION visual context shape differs")
        for variant in U_VARIANTS:
            summary = assets.summaries[(group_id, variant)]
            fields = {}
            for tag in TAG_ORDER:
                result = _decode_field(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=vision,
                    summary=summary,
                    tag=tag,
                    batch_size=batch_size,
                )
                fields[tag] = result
                by_tag[tag].append(result)
            record = {
                "event_id": assets.group_event[group_id],
                "group_id": group_id,
                "variant": variant,
                "fields": fields,
            }
            if group_id in representative_groups:
                row = text_oracle_full_row(summary)
                text = _generate(
                    lm=lm,
                    tokenizer=tokenizer,
                    vision=vision,
                    row=row,
                    route_text="",
                )
                try:
                    parsed = parse_text_oracle_answer(text)
                    parse_error = None
                except ValueError as error:
                    parsed = {}
                    parse_error = str(error)
                expected = dict(summary.fields())
                free.append(
                    {
                        "group_id": group_id,
                        "variant": variant,
                        "text": text,
                        "strictly_parseable": parse_error is None,
                        "parse_error": parse_error,
                        "field_accuracy": float(
                            np.mean(
                                [parsed.get(tag) == expected[tag] for tag in TAG_ORDER]
                            )
                        ),
                        "all_fields_exact": parsed == expected,
                    }
                )
            records["%s::%s" % (group_id, variant)] = record
    nonzero = [
        entry
        for key, record in records.items()
        if not key.endswith("::zero_u")
        for tag, entry in record["fields"].items()
        if tag != "U_PRESENT"
    ]
    all_entries = [entry for entries in by_tag.values() for entry in entries]
    result = {
        "split": "dev",
        "dev_group_count": len(dev_groups),
        "u_state_count": len(records),
        "field_decision_count": len(all_entries),
        "accuracy_by_tag": {
            tag: float(np.mean([entry["correct"] for entry in by_tag[tag]]))
            for tag in TAG_ORDER
        },
        "balanced_accuracy_by_tag": {
            tag: _balanced_accuracy(by_tag[tag], tag) for tag in TAG_ORDER
        },
        "mean_target_margin_by_tag": {
            tag: float(np.mean([entry["target_margin"] for entry in by_tag[tag]]))
            for tag in TAG_ORDER
        },
        "positive_target_margin_fraction_by_tag": {
            tag: float(
                np.mean([entry["target_margin"] > 0.0 for entry in by_tag[tag]])
            )
            for tag in TAG_ORDER
        },
        "nonzero_accuracy_excluding_presence": float(
            np.mean([entry["correct"] for entry in nonzero])
        ),
        "counterfactual": _counterfactual_metrics(records),
        "free_generation": {
            "sample_count": len(free),
            "strict_parseable_fraction": float(
                np.mean([entry["strictly_parseable"] for entry in free])
            ),
            "mean_field_accuracy": float(
                np.mean([entry["field_accuracy"] for entry in free])
            ),
            "exact_all_fields_fraction": float(
                np.mean([entry["all_fields_exact"] for entry in free])
            ),
            "records": free,
        },
        "records": records,
    }
    if result["u_state_count"] != 120 or result["field_decision_count"] != 720:
        raise RuntimeError("text-oracle evaluation coverage differs")
    return result


def _localization_summary(
    original: Mapping[str, Any],
    v15_text: Mapping[str, Any],
    v15_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    continuous = v15_lineage["continuous_u_after"]
    text_score = float(v15_text["nonzero_accuracy_excluding_presence"])
    continuous_score = float(continuous["nonzero_accuracy_excluding_presence"])
    original_score = float(original["nonzero_accuracy_excluding_presence"])
    text_counterfactual = float(
        v15_text["counterfactual"]["changed_field_exact_response_fraction"]
    )
    continuous_counterfactual = float(
        continuous["counterfactual"]["changed_field_exact_response_fraction"]
    )
    if text_score >= 0.8 and text_counterfactual >= 0.7 and continuous_score < 0.5:
        verdict = "continuous_u_interface_is_primary_bottleneck_supported"
    elif text_score < 0.5:
        verdict = "language_instruction_or_prompt_bottleneck_not_excluded"
    else:
        verdict = "mixed_interface_and_language_limitations"
    return {
        "verdict": verdict,
        "thresholds_are_localization_aids_not_training_gates": True,
        "v15_text_oracle_nonzero_accuracy_excluding_presence": text_score,
        "original_text_oracle_nonzero_accuracy_excluding_presence": original_score,
        "frozen_v15_continuous_u_nonzero_accuracy_excluding_presence": continuous_score,
        "text_minus_continuous_u_nonzero_accuracy": text_score - continuous_score,
        "v15_text_oracle_changed_field_response": text_counterfactual,
        "frozen_v15_continuous_u_changed_field_response": continuous_counterfactual,
        "text_minus_continuous_u_changed_field_response": (
            text_counterfactual - continuous_counterfactual
        ),
        "interpretation": (
            "High text-oracle performance with the same frozen ORION visual "
            "context supports an interface/alignment bottleneck for continuous "
            "U tokens. Low text-oracle performance means language instruction "
            "or QA design remains a competing explanation. This diagnostic "
            "does not validate a future U-QFormer or any driving behavior."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v15-checkpoint", type=Path, required=True)
    parser.add_argument("--v15-report", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-batch-size", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.v15_checkpoint,
        args.v15_report,
        args.dataset_manifest,
        args.v11_records,
        args.dataset_audit_report,
        args.view_feature_cache,
        args.protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v15.2 text-oracle prerequisite is missing")
    if args.answer_batch_size != 10:
        raise ValueError("v15.2 text oracle freezes answer batch size at ten")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v15.2 text-oracle output")
    protocol = _read_json(args.protocol.resolve())
    _validate_protocol(args, protocol)

    v14.v121._v101()._configure_base()
    assets = v14.UConceptAssets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )
    if args.preflight_only:
        if args.preflight is not None:
            raise ValueError("preflight cannot consume itself")
        value = _preflight(args, assets)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": value["status"], "output": str(args.output)}))
        return 0

    preflight = _validate_preflight(args)
    if not torch.cuda.is_available():
        raise RuntimeError("v15.2 text-oracle diagnostic requires CUDA")
    v15_state, v15_lineage = _validate_v15_lineage(args)
    lm, tokenizer = base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad = False
    original = _evaluate_model(
        lm=lm,
        tokenizer=tokenizer,
        assets=assets,
        batch_size=args.answer_batch_size,
    )
    _load_v15_lora(lm, v15_state)
    v15_text = _evaluate_model(
        lm=lm,
        tokenizer=tokenizer,
        assets=assets,
        batch_size=args.answer_batch_size,
    )
    if any(parameter.requires_grad for parameter in lm.parameters()):
        raise RuntimeError("text-oracle diagnostic changed frozen parameter scope")
    report = {
        "schema": SCHEMA,
        "status": "text_oracle_localization_complete",
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "training_performed": False,
        "optimizer_steps": 0,
        "continuous_u_tokens_present": False,
        "orion_visual_tokens_per_state": 529,
        "prompt_schema": PROMPT_SCHEMA,
        "validated_inputs": _validated_inputs(args),
        "implementation_sha256": _implementation_hashes(),
        "protocol_sha256": _sha256(args.protocol.resolve()),
        "preflight_sha256": _sha256(args.preflight.resolve()),
        "preflight": preflight,
        "v15_lineage": v15_lineage,
        "model_controls": {
            "original_orion": original,
            "v15_lora": v15_text,
        },
        "localization": _localization_summary(original, v15_text, v15_lineage),
        "locks": {
            "u_qformer_validated": False,
            "stage2l_unlocked": False,
            "stage2p_unlocked": False,
            "closed_loop_unlocked": False,
            "automatic_retry": False,
        },
        "claim_boundary": (
            "This is a text-oracle localization diagnostic. It can distinguish "
            "a continuous-U interface bottleneck from a broader language/QA "
            "failure, but it does not validate U-QFormer, task relevance, "
            "planning, closed-loop safety, or external generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["localization"]["verdict"],
                "original_text_oracle_nonzero_accuracy": original[
                    "nonzero_accuracy_excluding_presence"
                ],
                "v15_text_oracle_nonzero_accuracy": v15_text[
                    "nonzero_accuracy_excluding_presence"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
