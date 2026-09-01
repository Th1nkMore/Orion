#!/usr/bin/env python3
"""Audit Stage2-L v8 teacher-forcing and generation prompt token alignment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from transformers import AutoTokenizer

from mmcv.datasets.data_utils import conversation as conversation_lib
from mmcv.datasets.data_utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from mmcv.datasets.data_utils.data_utils import preprocess, tokenizer_image_token
from uq_estimator.stage2l_qa_contract_v4 import render_structured_answer


SCHEMA = "orion.stage2l_v8_generation_prompt_alignment.v1"


def _route_text(route_context: Mapping[str, Any]) -> str:
    command_names = {
        0: "LEFT",
        1: "RIGHT",
        2: "STRAIGHT",
        3: "FOLLOW_LANE",
        4: "CHANGE_LEFT",
        5: "CHANGE_RIGHT",
    }
    command = route_context.get("command")
    command_text = command_names.get(int(command) - 1, str(command))
    points = route_context["orion_unmodified_plan_right_forward_m"]
    rendered = "; ".join(
        "(%.2f, %.2f)" % (float(x), float(y)) for x, y in points
    )
    return (
        "Route command: %s. Frozen ORION future path points "
        "(right_m, forward_m): %s."
    ) % (command_text, rendered)


def _question(row: Mapping[str, Any], route_text: str) -> str:
    return "%s\n%s\n%s" % (
        DEFAULT_IMAGE_TOKEN,
        route_text,
        row["conversation"][0]["value"],
    )


def _training_tokens(tokenizer, row: Mapping[str, Any], route_text: str):
    target = render_structured_answer(
        str(row["question_family"]), row["target"]["structured_summary"]
    )
    converted = preprocess(
        [[
            {"from": "human", "value": _question(row, route_text)},
            {"from": "gpt", "value": target},
        ]],
        tokenizer,
        has_image=True,
    )
    return converted["input_ids"][0], converted["labels"][0], target


def _prompt_tokens(tokenizer, row: Mapping[str, Any], route_text: str):
    conversation = conversation_lib.default_conversation.copy()
    conversation.append_message(
        conversation.roles[0], _question(row, route_text)
    )
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    ids = tokenizer_image_token(
        prompt, tokenizer, return_tensors="pt"
    )
    return ids, prompt


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if int(left_value) != int(right_value):
            break
        count += 1
    return count


def audit(records_path: Path, tokenizer_path: Path) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anchors = [
        row
        for row in rows
        if row.get("loss_policy", {}).get("hard_language_target") is True
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    diagnostics = []
    for row in anchors:
        route_text = _route_text(row["model_input"]["route_context"]["payload"])
        full_ids, labels, target = _training_tokens(tokenizer, row, route_text)
        prompt_ids, prompt_text = _prompt_tokens(tokenizer, row, route_text)
        full_values = [int(value) for value in full_ids.tolist()]
        prompt_values = [int(value) for value in prompt_ids.tolist()]
        label_values = [int(value) for value in labels.tolist()]
        supervised = [
            index for index, value in enumerate(label_values)
            if value != IGNORE_INDEX
        ]
        first_supervised = supervised[0] if supervised else -1
        contiguous_supervision = bool(supervised) and supervised == list(
            range(supervised[0], supervised[-1] + 1)
        )
        common_prefix = _longest_common_prefix(full_values, prompt_values)
        decoded_supervision = tokenizer.decode(
            [label_values[index] for index in supervised],
            skip_special_tokens=True,
        )
        diagnostics.append(
            {
                "sample_id": row.get("sample_id"),
                "group_id": row["counterfactual"]["group_id"],
                "variant": row["counterfactual"]["variant"],
                "question_family": row["question_family"],
                "full_token_count": len(full_values),
                "prompt_token_count": len(prompt_values),
                "first_supervised_index": first_supervised,
                "supervised_token_count": len(supervised),
                "supervision_contiguous": contiguous_supervision,
                "prompt_is_full_sequence_prefix": (
                    len(prompt_values) <= len(full_values)
                    and common_prefix == len(prompt_values)
                ),
                "prompt_and_training_common_prefix": common_prefix,
                "prompt_minus_first_supervised": (
                    len(prompt_values) - first_supervised
                    if first_supervised >= 0 else None
                ),
                "target": target,
                "decoded_supervision": decoded_supervision,
                "target_present_in_decoded_supervision": (
                    target in decoded_supervision
                ),
                "prompt_tail": prompt_text[-160:],
            }
        )
    delta_counts = Counter(
        row["prompt_minus_first_supervised"] for row in diagnostics
    )
    checks = {
        "exactly_90_hard_language_anchors": len(diagnostics) == 90,
        "all_records_have_supervised_tokens": all(
            row["supervised_token_count"] > 0 for row in diagnostics
        ),
        "all_supervision_is_contiguous": all(
            row["supervision_contiguous"] for row in diagnostics
        ),
        "generation_prompt_is_training_prefix": all(
            row["prompt_is_full_sequence_prefix"] for row in diagnostics
        ),
        "target_is_present_in_supervised_decode": all(
            row["target_present_in_decoded_supervision"]
            for row in diagnostics
        ),
    }
    return {
        "schema": SCHEMA,
        "status": "alignment_pass" if all(checks.values()) else "alignment_failed",
        "records_path": str(records_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "anchor_count": len(diagnostics),
        "checks": checks,
        "prompt_minus_first_supervised_histogram": {
            str(key): value for key, value in sorted(delta_counts.items())
        },
        "failures": [
            row for row in diagnostics
            if not (
                row["supervised_token_count"] > 0
                and row["supervision_contiguous"]
                and row["prompt_is_full_sequence_prefix"]
                and row["target_present_in_decoded_supervision"]
            )
        ],
        "examples": diagnostics[:4],
        "claim_boundary": (
            "Token-alignment diagnostic only; it does not test learned "
            "semantics, free generation, generalization, planning, or safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.records.resolve(), args.tokenizer.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "alignment_pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
