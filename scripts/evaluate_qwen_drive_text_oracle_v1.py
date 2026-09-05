#!/usr/bin/env python3
"""Compare Qwen-Drive with ORION on the frozen v15.2 text oracle.

This diagnostic is deliberately text-only.  It reads the 120 authoritative U
states and their exact counterfactual grouping from the completed ORION v15.2
report, scores every legal answer with the Qwen-Drive VLM, and reproduces the
same field and changed-field metrics.  It does not load a planning expert,
train a parameter, process an image, or run CARLA.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "orion.qwen-drive-text-oracle/v1"
ORION_REPORT_SCHEMA = "orion.stage2l-v15-2-text-oracle-localization/v1"
TAG_ORDER = (
    "U_PRESENT",
    "U_VIEW",
    "U_REGION",
    "U_LEVEL",
    "U_TREND",
    "U_COMPONENT",
)
U_VARIANTS = (
    "zero_u",
    "observed_u",
    "view_shifted_u",
    "spatial_shifted_u",
    "component_shifted_u",
    "temporal_reversed_u",
)
FIELD_VOCABULARIES: Mapping[str, tuple[str, ...]] = {
    "U_PRESENT": ("yes", "no"),
    "U_VIEW": (
        "front",
        "front_left",
        "front_right",
        "rear",
        "rear_left",
        "rear_right",
        "none",
    ),
    "U_REGION": (
        "upper_left",
        "upper_center",
        "upper_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "lower_left",
        "lower_center",
        "lower_right",
        "none",
    ),
    "U_LEVEL": ("low", "medium", "high", "none"),
    "U_TREND": ("rising", "stable", "falling"),
    "U_COMPONENT": (
        "persistent_direction",
        "persistent_magnitude",
        "transient_inconsistency",
        "mixed",
        "none",
    ),
}
FIELD_QUESTIONS: Mapping[str, str] = {
    "U_PRESENT": "Is observation uncertainty present?",
    "U_VIEW": "Which camera view contains the strongest observation uncertainty?",
    "U_REGION": "Which image region contains its strongest location?",
    "U_LEVEL": "What is its overall uncertainty level?",
    "U_TREND": "What is its temporal trend from the first frame to the latest frame?",
    "U_COMPONENT": "Which uncertainty component is dominant?",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return value


def _words(value: str) -> str:
    return str(value).replace("_", "-")


def render_text_oracle_summary(fields: Mapping[str, str]) -> str:
    """Render exactly the natural-language facts used by ORION v15.2."""

    if fields["U_PRESENT"] == "no":
        return (
            "No observation uncertainty is present. Consequently no camera "
            "view, no image region, no uncertainty level, and no uncertainty "
            "component apply. The temporal trend is stable."
        )
    return (
        "Observation uncertainty is present. Its strongest location is in the "
        f"{_words(fields['U_VIEW'])} camera view and the "
        f"{_words(fields['U_REGION'])} image region. Its overall uncertainty "
        f"level is {fields['U_LEVEL']}. From the first frame to the latest "
        f"frame, its temporal trend is {fields['U_TREND']}. Its dominant "
        f"uncertainty component is {_words(fields['U_COMPONENT'])}."
    )


def field_prompt(fields: Mapping[str, str], tag: str) -> str:
    if tag not in TAG_ORDER:
        raise ValueError("unknown U field")
    choices = ", ".join(FIELD_VOCABULARIES[tag])
    return (
        "The following is an exact, authoritative, task-free description of "
        "observation uncertainty. Read the description literally. Do not infer "
        "task relevance, driving risk, an action, a trajectory, or control.\n"
        f"Observation-uncertainty description: {render_text_oracle_summary(fields)}\n"
        f"Question: {FIELD_QUESTIONS[tag]}\n"
        f"Answer with exactly one canonical value from this list: {choices}. "
        "Output only that value, with no explanation or punctuation."
    )


def _validate_fields(fields: Mapping[str, str]) -> dict[str, str]:
    if set(fields) != set(TAG_ORDER):
        raise ValueError("text-oracle state does not contain exactly six fields")
    normalized = {tag: str(fields[tag]) for tag in TAG_ORDER}
    for tag, value in normalized.items():
        if value not in FIELD_VOCABULARIES[tag]:
            raise ValueError("text-oracle field value is outside its vocabulary")
    return normalized


def load_frozen_states(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract model-independent expected states from a completed v15.2 report."""

    if (
        report.get("schema") != ORION_REPORT_SCHEMA
        or report.get("status") != "text_oracle_localization_complete"
        or report.get("training_performed") is not False
        or report.get("continuous_u_tokens_present") is not False
    ):
        raise ValueError("ORION v15.2 report lineage is invalid")
    controls = report.get("model_controls", {})
    original = controls.get("original_orion", {}).get("records", {})
    v15 = controls.get("v15_lora", {}).get("records", {})
    if set(original) != set(v15) or len(v15) != 120:
        raise ValueError("ORION v15.2 report does not contain the frozen 120 states")
    states: dict[str, dict[str, Any]] = {}
    for key in sorted(v15):
        current = v15[key]
        reference = original[key]
        group_id = str(current.get("group_id"))
        variant = str(current.get("variant"))
        if key != "%s::%s" % (group_id, variant) or variant not in U_VARIANTS:
            raise ValueError("text-oracle counterfactual identity is malformed")
        current_fields = {
            tag: current.get("fields", {}).get(tag, {}).get("expected")
            for tag in TAG_ORDER
        }
        reference_fields = {
            tag: reference.get("fields", {}).get(tag, {}).get("expected")
            for tag in TAG_ORDER
        }
        fields = _validate_fields(current_fields)
        if fields != _validate_fields(reference_fields):
            raise ValueError("ORION controls disagree on a frozen expected state")
        states[key] = {
            "group_id": group_id,
            "variant": variant,
            "fields": fields,
        }
    groups = {state["group_id"] for state in states.values()}
    if len(groups) != 20:
        raise ValueError("text-oracle states do not cover exactly twenty dev groups")
    for group_id in groups:
        variants = {
            state["variant"]
            for state in states.values()
            if state["group_id"] == group_id
        }
        if variants != set(U_VARIANTS):
            raise ValueError("a dev group is missing a counterfactual variant")
    return states


def _balanced_accuracy(entries: Sequence[Mapping[str, Any]], tag: str) -> float:
    recalls = []
    for value in FIELD_VOCABULARIES[tag]:
        supported = [entry for entry in entries if entry["expected"] == value]
        if supported:
            recalls.append(
                sum(bool(entry["correct"]) for entry in supported) / len(supported)
            )
    if not recalls:
        raise ValueError("field has no supported class")
    return sum(recalls) / len(recalls)


def counterfactual_metrics(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    changed: list[bool] = []
    unchanged: list[bool] = []
    unchanged_correct: list[bool] = []
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
        "changed_field_exact_response_fraction": sum(changed) / len(changed),
        "unchanged_field_pair_count": len(unchanged),
        "unchanged_field_invariance_fraction": sum(unchanged) / len(unchanged),
        "unchanged_field_correct_invariance_fraction": (
            sum(unchanged_correct) / len(unchanged_correct)
        ),
    }


def aggregate_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_tag: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    nonzero_without_presence: list[bool] = []
    all_fields: list[bool] = []
    for record in records.values():
        for tag in TAG_ORDER:
            field = record["fields"][tag]
            by_tag[tag].append(field)
            all_fields.append(bool(field["correct"]))
            if record["variant"] != "zero_u" and tag != "U_PRESENT":
                nonzero_without_presence.append(bool(field["correct"]))
    return {
        "dev_group_count": len({value["group_id"] for value in records.values()}),
        "state_count": len(records),
        "field_decision_count": len(all_fields),
        "accuracy": sum(all_fields) / len(all_fields),
        "accuracy_by_tag": {
            tag: sum(bool(entry["correct"]) for entry in by_tag[tag])
            / len(by_tag[tag])
            for tag in TAG_ORDER
        },
        "balanced_accuracy_by_tag": {
            tag: _balanced_accuracy(by_tag[tag], tag) for tag in TAG_ORDER
        },
        "nonzero_accuracy_excluding_presence": (
            sum(nonzero_without_presence) / len(nonzero_without_presence)
        ),
        "counterfactual": counterfactual_metrics(records),
        "records": dict(records),
    }


class QwenTextScorer:
    """Length-normalized candidate NLL under the Qwen-Drive VLM."""

    def __init__(self, model_path: Path, device: str, dtype_name: str) -> None:
        import torch
        import transformers
        from qwen_drive import QwenDriveForPlanning
        from transformers.models.qwen3_5 import modeling_qwen3_5

        if modeling_qwen3_5.chunk_gated_delta_rule is None:
            raise RuntimeError(
                "flash-linear-attention is required; refusing the pathological "
                "Qwen3.5 PyTorch fallback"
            )

        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtypes:
            raise ValueError("unsupported dtype")
        self.torch = torch
        self.model = QwenDriveForPlanning.from_pretrained(
            str(model_path),
            dtype=dtypes[dtype_name],
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device).eval()
        self.processor = self.model.processor
        self.tokenizer = self.processor.tokenizer
        self.device = device
        self.im_start_id = self.processor.im_start_id
        self.im_end_id = self.processor.im_end_id
        self.newline_ids = list(self.processor.newline_ids)
        self.runtime = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "flash_linear_attention_version": importlib.metadata.version(
                "flash-linear-attention"
            ),
            "fla_core_version": importlib.metadata.version("fla-core"),
            "einops_version": importlib.metadata.version("einops"),
            "linear_attention_backend": "flash-linear-attention",
            "causal_conv_backend": (
                "causal-conv1d"
                if modeling_qwen3_5.causal_conv1d_fn is not None
                else "transformers_torch_fallback"
            ),
        }

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prompt_ids(self, question: str) -> list[int]:
        return (
            [self.im_start_id]
            + self._encode("user")
            + self.newline_ids
            + self._encode(question)
            + [self.im_end_id]
            + self.newline_ids
            + [self.im_start_id]
            + self._encode("assistant")
            + self.newline_ids
        )

    def score(self, question: str, candidates: Sequence[str]) -> list[float]:
        torch = self.torch
        prompt = self._prompt_ids(question)
        answers = [self._encode(candidate) for candidate in candidates]
        if not candidates or any(not answer for answer in answers):
            raise ValueError("candidate answer tokenization is empty")
        lengths = [len(prompt) + len(answer) for answer in answers]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Qwen tokenizer has neither pad nor EOS token")
        input_ids = torch.full(
            (len(answers), max(lengths)), int(pad_id), dtype=torch.long
        )
        attention = torch.zeros_like(input_ids)
        for index, answer in enumerate(answers):
            sequence = prompt + answer
            input_ids[index, : len(sequence)] = torch.tensor(sequence)
            attention[index, : len(sequence)] = 1
        input_ids = input_ids.to(self.device)
        attention = attention.to(self.device)
        with torch.inference_mode():
            output = self.model.vlm(
                input_ids=input_ids,
                attention_mask=attention,
                mm_token_type_ids=self.model._modality_ids(input_ids),
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits.float()
        start = len(prompt)
        nlls = []
        for index, answer in enumerate(answers):
            prediction = logits[index, start - 1 : start + len(answer) - 1]
            target = input_ids[index, start : start + len(answer)]
            nll = torch.nn.functional.cross_entropy(
                prediction, target, reduction="mean"
            )
            nlls.append(float(nll.item()))
        return nlls


def evaluate(
    scorer: QwenTextScorer,
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for state_index, (key, state) in enumerate(sorted(states.items()), start=1):
        fields = {}
        expected_fields = state["fields"]
        for tag in TAG_ORDER:
            candidates = FIELD_VOCABULARIES[tag]
            nlls = scorer.score(field_prompt(expected_fields, tag), candidates)
            expected = expected_fields[tag]
            predicted = candidates[min(range(len(nlls)), key=nlls.__getitem__)]
            target_index = candidates.index(expected)
            wrong = [value for index, value in enumerate(nlls) if index != target_index]
            fields[tag] = {
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "target_margin": min(wrong) - nlls[target_index],
                "candidate_nlls": dict(zip(candidates, nlls)),
            }
        records[key] = {
            "group_id": state["group_id"],
            "variant": state["variant"],
            "fields": fields,
        }
        if state_index % 10 == 0 or state_index == len(states):
            print(
                json.dumps({"completed_states": state_index, "total_states": len(states)}),
                flush=True,
            )
    return aggregate_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--orion-report", type=Path, required=True)
    parser.add_argument("--expected-orion-report-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--qwen-code-revision", required=True)
    parser.add_argument("--invalidated-job-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    args = parser.parse_args()
    model_weights = args.model / "model.safetensors"
    prerequisites = (args.model / "config.json", model_weights, args.orion_report)
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("Qwen text-oracle prerequisite is missing")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a Qwen text-oracle report")
    orion_hash = _sha256(args.orion_report.resolve())
    if orion_hash != args.expected_orion_report_sha256:
        raise ValueError("ORION v15.2 report hash differs")
    model_hash = _sha256(model_weights.resolve())
    if model_hash != args.expected_model_sha256:
        raise ValueError("Qwen-Drive model hash differs")

    orion_report = _read_json(args.orion_report.resolve())
    states = load_frozen_states(orion_report)
    scorer = QwenTextScorer(args.model.resolve(), args.device, args.dtype)
    qwen = evaluate(scorer, states)
    original = orion_report["model_controls"]["original_orion"]
    v15 = orion_report["model_controls"]["v15_lora"]
    qwen_changed = qwen["counterfactual"]["changed_field_exact_response_fraction"]
    qwen_nonzero = qwen["nonzero_accuracy_excluding_presence"]
    thresholds = {
        "minimum_nonzero_accuracy_excluding_presence": 0.8,
        "minimum_changed_field_exact_response": 0.7,
    }
    supports_backbone_hypothesis = (
        qwen_nonzero >= thresholds["minimum_nonzero_accuracy_excluding_presence"]
        and qwen_changed >= thresholds["minimum_changed_field_exact_response"]
    )
    report = {
        "schema": SCHEMA,
        "status": "qwen_drive_text_oracle_complete",
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "training_performed": False,
        "optimizer_steps": 0,
        "images_present": False,
        "planning_expert_weights_loaded": False,
        "planning_expert_used": False,
        "continuous_u_tokens_present": False,
        "prompt_contract": "exact_orion_v15_2_text_oracle",
        "runtime": scorer.runtime,
        "invalidated_attempts": [
            {
                "slurm_job_id": job_id,
                "reason": "missing_flash_linear_attention_backend",
                "report_written": False,
            }
            for job_id in args.invalidated_job_id
        ],
        "input_lineage": {
            "orion_report": str(args.orion_report.resolve()),
            "orion_report_sha256": orion_hash,
            "model": str(args.model.resolve()),
            "model_safetensors_sha256": model_hash,
            "model_revision": args.model_revision,
            "qwen_code_revision": args.qwen_code_revision,
        },
        "thresholds": thresholds,
        "comparators": {
            "original_orion": {
                "nonzero_accuracy_excluding_presence": original[
                    "nonzero_accuracy_excluding_presence"
                ],
                "changed_field_exact_response": original["counterfactual"][
                    "changed_field_exact_response_fraction"
                ],
            },
            "v15_lora": {
                "nonzero_accuracy_excluding_presence": v15[
                    "nonzero_accuracy_excluding_presence"
                ],
                "changed_field_exact_response": v15["counterfactual"][
                    "changed_field_exact_response_fraction"
                ],
            },
        },
        "qwen_drive": qwen,
        "decision": {
            "supports_qwen_backbone_hypothesis": supports_backbone_hypothesis,
            "verdict": (
                "qwen_text_instruction_path_is_materially_stronger"
                if supports_backbone_hypothesis
                else "qwen_text_instruction_advantage_not_established"
            ),
            "interpretation": (
                "A pass localizes the current ORION failure partly to its language or "
                "instruction path. It does not yet validate Qwen multimodal U tokens, "
                "planning, or closed-loop behavior."
            ),
        },
        "claim_boundary": (
            "Frozen text-only backbone diagnostic on the ORION v15.2 dev states. "
            "No image, U-token, task relevance, trajectory, control, or closed-loop "
            "claim is supported."
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
                "nonzero_accuracy_excluding_presence": qwen_nonzero,
                "changed_field_exact_response": qwen_changed,
                "verdict": report["decision"]["verdict"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
