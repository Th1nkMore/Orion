#!/usr/bin/env python3
"""Diagnose v14.1 U semantics separately from free-generation rendering.

The completed v14.1 LoRA and every upstream representation remain frozen.  On
one representative group per held-out dev event, this script evaluates:

1. deterministic free generation under a literal six-line schema; and
2. constrained per-field decoding by scoring every legal canonical answer.

No optimizer, route context, relevance, risk, action, trajectory, control, or
closed-loop component is constructed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import scripts.train_stage2l_v14_u_concept_lora_smoke as v14
import scripts.train_stage2l_mr1_smoke as base
import scripts.train_stage2l_v10_staged_smoke as v10
from scripts.train_stage2l_route196_bridge_smoke import _generate
from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
    SCHEMA as PROMPT_SCHEMA,
    build_explicit_u_qa_row,
    candidate_answers,
    decode_candidate_nlls,
    parse_strict_u_answer,
)
from uq_estimator.stage2l_u_concept_qa_v14 import TAG_ORDER, U_VARIANTS


SCHEMA = "orion.stage2l-v14-2-explicit-u-output-diagnostic/v1"
PROTOCOL_SCHEMA = "orion.stage2l-v14-2-explicit-u-output-protocol/v1"
V14_CHECKPOINT_SCHEMA = "orion.stage2l-v14-u-concept-lora-smoke/v1"


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


def _validated_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "v11_records_sha256": _sha256(args.v11_records.resolve()),
        "dataset_audit_report_sha256": _sha256(
            args.dataset_audit_report.resolve()
        ),
        "view_feature_cache_sha256": _sha256(args.view_feature_cache.resolve()),
        "u_tokenizer_checkpoint_sha256": _sha256(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "trained_v14_1_checkpoint_sha256": _sha256(
            args.trained_checkpoint.resolve()
        ),
    }


def _validate_protocol(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> None:
    output_key = "preflight_output" if args.preflight_only else "output"
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen_checkpoint_output_diagnostic"
        or protocol.get("input_sha256") != _validated_inputs(args)
        or protocol.get("prompt_schema") != PROMPT_SCHEMA
        or protocol.get("training_performed") is not False
        or protocol.get("optimizer_steps") != 0
        or protocol.get("split") != "dev"
        or protocol.get("diagnostic_groups") != 4
        or protocol.get("u_variants") != list(U_VARIANTS)
        or protocol.get("automatic_retry") is not False
        or Path(str(protocol.get(output_key, ""))).resolve()
        != args.output.resolve()
    ):
        raise ValueError("v14.2 explicit-output protocol is absent or stale")


def _load_trained_lora(lm, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu")
    if (
        payload.get("schema") != V14_CHECKPOINT_SCHEMA
        or payload.get("status") != "bounded_u_concept_lora_complete"
        or payload.get("optimizer_steps") != 200
        or payload.get("stage1_and_u_tokenizer_frozen") is not True
        or payload.get("r_k_route_risk_action_absent") is not True
        or payload.get("formal_stage2l_ready") is not False
        or payload.get("stage2p_ready") is not False
        or payload.get("closed_loop_eligible") is not False
    ):
        raise ValueError("v14.1 trained checkpoint contract differs")
    state = payload.get("orion_lora", {})
    expected = {name for name in lm.state_dict() if "lora_" in name}
    if set(state) != expected:
        raise ValueError("v14.1 LoRA tensor keys differ")
    result = lm.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        "lora_" in name for name in result.missing_keys
    ):
        raise ValueError("v14.1 LoRA load was incomplete")
    for parameter in lm.parameters():
        parameter.requires_grad = False
    lm.eval()
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "optimizer_steps": payload["optimizer_steps"],
        "lora_tensor_count": len(state),
        "lora_parameter_count": sum(value.numel() for value in state.values()),
        "all_lm_parameters_frozen": all(
            not parameter.requires_grad for parameter in lm.parameters()
        ),
    }


@torch.no_grad()
def _candidate_decode(
    *, lm, tokenizer, vision: torch.Tensor, summary
) -> dict[str, Any]:
    fields = summary.fields()
    predicted = {}
    target_margins = {}
    candidate_nlls = {}
    for tag in TAG_ORDER:
        row = build_explicit_u_qa_row(summary, tag)
        answers = candidate_answers(tag)
        nll_tensor = base._answer_nlls_mr1(
            lm=lm,
            tokenizer=tokenizer,
            vision=vision,
            row=row,
            route_text="",
            answers=answers,
            micro_batch_size=len(answers),
        )
        nlls = [float(value) for value in nll_tensor.detach().cpu().tolist()]
        prediction = decode_candidate_nlls(tag, nlls)
        target_index = FIELD_VOCABULARIES[tag].index(fields[tag])
        wrong = [
            value for index, value in enumerate(nlls) if index != target_index
        ]
        predicted[tag] = prediction
        candidate_nlls[tag] = {
            value: nll for value, nll in zip(FIELD_VOCABULARIES[tag], nlls)
        }
        target_margins[tag] = float(min(wrong) - nlls[target_index])
    correctness = {tag: predicted[tag] == fields[tag] for tag in TAG_ORDER}
    return {
        "predicted": predicted,
        "expected": dict(fields),
        "correct_by_field": correctness,
        "field_accuracy": float(np.mean(list(correctness.values()))),
        "all_fields_exact": all(correctness.values()),
        "target_margin_by_field": target_margins,
        "mean_target_margin": float(np.mean(list(target_margins.values()))),
        "candidate_nlls": candidate_nlls,
    }


@torch.no_grad()
def _evaluate(
    *, lm, tokenizer, uq_tokenizer, assets: v14.UConceptAssets
) -> dict[str, Any]:
    event_groups = assets.event_groups["dev"]
    diagnostic_groups = [values[0] for _, values in sorted(event_groups.items())]
    if len(diagnostic_groups) != 4:
        raise ValueError("v14.2 requires one group from each of four dev events")
    samples = {}
    constrained_correct = defaultdict(list)
    constrained_by_variant = defaultdict(list)
    constrained_margins = defaultdict(list)
    free_field_scores = []
    free_exact = []
    free_parseable = []
    for group_id in diagnostic_groups:
        baseline = assets.visual_contexts[group_id].cuda(non_blocking=True)
        for variant in U_VARIANTS:
            summary = assets.summaries[(group_id, variant)]
            vision = v14._condition(
                baseline=baseline,
                components=assets.variants[(group_id, variant)],
                uq_tokenizer=uq_tokenizer,
            )
            constrained = _candidate_decode(
                lm=lm,
                tokenizer=tokenizer,
                vision=vision,
                summary=summary,
            )
            for tag, correct in constrained["correct_by_field"].items():
                constrained_correct[tag].append(correct)
            constrained_by_variant[variant].append(
                constrained["field_accuracy"]
            )
            for tag, margin in constrained["target_margin_by_field"].items():
                constrained_margins[tag].append(margin)

            row = build_explicit_u_qa_row(summary)
            free_text = _generate(
                lm=lm,
                tokenizer=tokenizer,
                vision=vision,
                row=row,
                route_text="",
            )
            parse_error = None
            try:
                parsed = parse_strict_u_answer(free_text)
            except ValueError as error:
                parsed = {}
                parse_error = str(error)
            expected = dict(summary.fields())
            field_accuracy = float(
                np.mean([parsed.get(tag) == expected[tag] for tag in TAG_ORDER])
            )
            exact = parsed == expected
            free_parseable.append(parse_error is None)
            free_field_scores.append(field_accuracy)
            free_exact.append(exact)
            samples["%s::%s" % (group_id, variant)] = {
                "free_generation": {
                    "text": free_text,
                    "strictly_parseable": parse_error is None,
                    "parse_error": parse_error,
                    "parsed": parsed,
                    "expected": expected,
                    "field_accuracy": field_accuracy,
                    "all_fields_exact": exact,
                },
                "constrained_decode": constrained,
            }
    all_constrained = [
        value
        for values in constrained_correct.values()
        for value in values
    ]
    return {
        "split": "dev",
        "diagnostic_groups": diagnostic_groups,
        "sample_count": len(samples),
        "free_generation": {
            "strict_parseable_fraction": float(np.mean(free_parseable)),
            "mean_field_accuracy": float(np.mean(free_field_scores)),
            "exact_all_fields_fraction": float(np.mean(free_exact)),
        },
        "constrained_decode": {
            "mean_field_accuracy": float(np.mean(all_constrained)),
            "exact_field_accuracy_by_tag": {
                tag: float(np.mean(constrained_correct[tag])) for tag in TAG_ORDER
            },
            "mean_field_accuracy_by_variant": {
                variant: float(np.mean(constrained_by_variant[variant]))
                for variant in U_VARIANTS
            },
            "mean_target_margin_by_tag": {
                tag: float(np.mean(constrained_margins[tag])) for tag in TAG_ORDER
            },
            "positive_target_margin_fraction": float(
                np.mean(
                    [
                        margin > 0.0
                        for values in constrained_margins.values()
                        for margin in values
                    ]
                )
            ),
        },
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.trained_checkpoint,
        args.dataset_manifest,
        args.v11_records,
        args.dataset_audit_report,
        args.view_feature_cache,
        args.u_tokenizer_checkpoint,
        args.protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("v14.2 explicit-output prerequisite is missing")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v14.2 output")
    protocol = _read_json(args.protocol.resolve())
    _validate_protocol(args, protocol)
    if args.preflight_only:
        value = {
            "schema": SCHEMA,
            "status": "explicit_output_preflight_pass",
            "passed": True,
            "training_performed": False,
            "gpu_used": False,
            "validated_inputs": _validated_inputs(args),
            "protocol_sha256": _sha256(args.protocol.resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "prompt_module_sha256": _sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "uq_estimator"
                    / "stage2l_u_concept_explicit_schema_v14_2.py"
                ).resolve()
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("v14.2 explicit-output diagnostic requires CUDA")

    v14.v121._v101()._configure_base()
    assets = v14.UConceptAssets(
        args.dataset_manifest,
        args.view_feature_cache,
        args.v11_records,
        args.dataset_audit_report,
    )
    lm, tokenizer = base._load_orion_lm(
        args.config.resolve(), args.checkpoint.resolve()
    )
    trained = _load_trained_lora(lm, args.trained_checkpoint)
    uq_tokenizer = v10._load_frozen_u_tokenizer(
        args.u_tokenizer_checkpoint.resolve()
    ).cuda().eval()
    for parameter in uq_tokenizer.parameters():
        parameter.requires_grad = False
    evaluation = _evaluate(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        assets=assets,
    )
    if any(parameter.requires_grad for parameter in lm.parameters()) or any(
        parameter.requires_grad for parameter in uq_tokenizer.parameters()
    ):
        raise RuntimeError("v14.2 changed a frozen parameter scope")
    report = {
        "schema": SCHEMA,
        "status": "explicit_output_diagnostic_complete",
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "training_performed": False,
        "optimizer_steps": 0,
        "prompt_schema": PROMPT_SCHEMA,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.protocol.resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "trained_checkpoint": trained,
        "evaluation": evaluation,
        "claim_boundary": (
            "This diagnostic separates constrained recognition of frozen U "
            "fields from free-generation rendering. It is not task relevance, "
            "planning, closed-loop, safety, or external-generalization evidence."
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
                "free_parseable_fraction": evaluation["free_generation"][
                    "strict_parseable_fraction"
                ],
                "free_field_accuracy": evaluation["free_generation"][
                    "mean_field_accuracy"
                ],
                "constrained_field_accuracy": evaluation[
                    "constrained_decode"
                ]["mean_field_accuracy"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
