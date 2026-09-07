#!/usr/bin/env python3
"""Evaluate Qwen's text/numeric upper bound on frozen V1k route queries."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


SCHEMA = "orion.qwen-visibility-numeric-text-upper-bound/v1"
CANDIDATES = ("ON_ROUTE", "OFF_ROUTE")
SYSTEM_PROMPT = (
    "You are validating an authoritative metric visibility-belief record. "
    "G00-G15 are global scene tokens and F00-F31 are frame-local visibility "
    "frontiers. Each F token describes a boundary between observed space and "
    "occluded unknown space in the current frame. Ego coordinates use x "
    "forward and y left. route_weight_mean is a scalar in [0,1] measuring "
    "overlap with the planned route; values at least 0.2 mean ON_ROUTE. "
    "The record does not assert that a hidden actor exists."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def render_question(frontier: str, value: float) -> str:
    """Expose the exact scalar in language; no continuous U token is present."""

    return (
        "Authoritative visibility-belief record for this sample:\n"
        "%s.route_weight_mean = %.6f\n"
        "Read that value literally. Is %s.route_weight_mean at least 0.2? "
        "Reply with exactly ON_ROUTE or OFF_ROUTE, with no explanation."
        % (frontier, float(value), frontier)
    )


def route_value(record: Dict[str, Any]) -> float:
    token_path = Path(str(record["token_artifact"]))
    with np.load(token_path, allow_pickle=False) as artifact:
        names = tuple(
            str(value)
            for value in artifact["visibility_tokens_feature_names"].tolist()
        )
        rows = np.asarray(artifact["visibility_tokens_frontier"], dtype=np.float32)
        mask = np.asarray(artifact["visibility_tokens_frontier_mask"], dtype=bool)
    row_index = int(str(record["query_frontier"])[1:])
    if not 0 <= row_index < len(mask) or not bool(mask[row_index]):
        raise ValueError("query frontier is not valid: %s" % record["sample_id"])
    return float(rows[row_index, names.index("route_weight_mean")])


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    result: Dict[str, Any] = {}
    for mode in ("text_only", "rgb_plus_text"):
        selected = [row for row in rows if row["mode"] == mode]
        correct = sum(bool(row["correct"]) for row in selected)
        by_split = {}
        for split in ("validation", "held_out"):
            split_rows = [row for row in selected if row["split"] == split]
            by_label = {}
            for label in CANDIDATES:
                label_rows = [row for row in split_rows if row["expected"] == label]
                by_label[label] = {
                    "count": len(label_rows),
                    "correct": sum(bool(row["correct"]) for row in label_rows),
                }
            by_split[split] = {
                "count": len(split_rows),
                "correct": sum(bool(row["correct"]) for row in split_rows),
                "accuracy": (
                    sum(bool(row["correct"]) for row in split_rows) / len(split_rows)
                    if split_rows
                    else None
                ),
                "by_label": by_label,
            }
        result[mode] = {
            "count": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else None,
            "prediction_counts": dict(
                sorted(Counter(row["predicted"] for row in selected).items())
            ),
            "by_split": by_split,
        }
    by_key = {(row["sample_id"], row["mode"]): row for row in rows}
    paired = [
        by_key[(sample_id, "text_only")]["predicted"]
        == by_key[(sample_id, "rgb_plus_text")]["predicted"]
        for sample_id in sorted({row["sample_id"] for row in rows})
    ]
    result["mode_prediction_agreement"] = sum(paired) / len(paired) if paired else None
    return result


class QwenUpperBoundScorer:
    def __init__(self, model_path: Path, device: str, dtype_name: str) -> None:
        import torch
        import transformers
        from qwen_drive import QwenDriveForPlanning
        from transformers.models.qwen3_5 import modeling_qwen3_5

        from uq_estimator import qwen_visibility_training as training
        from uq_estimator import qwen_visibility_vlm as visibility_vlm

        if modeling_qwen3_5.chunk_gated_delta_rule is None:
            raise RuntimeError("flash-linear-attention is required")
        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch = torch
        self.training = training
        self.visibility_vlm = visibility_vlm
        self.device = device
        self.model = QwenDriveForPlanning.from_pretrained(
            str(model_path),
            dtype=dtypes[dtype_name],
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device).eval()
        self.processor = self.model.processor
        self.tokenizer = self.processor.tokenizer
        self.runtime = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "flash_linear_attention_version": importlib.metadata.version(
                "flash-linear-attention"
            ),
            "fla_core_version": importlib.metadata.version("fla-core"),
        }

    def _encode(self, value: str) -> List[int]:
        return list(self.tokenizer.encode(value, add_special_tokens=False))

    def _text_prompt_ids(self, question: str) -> List[int]:
        processor = self.processor
        return (
            [int(processor.im_start_id)]
            + self._encode("system")
            + list(processor.newline_ids)
            + self._encode(SYSTEM_PROMPT)
            + [int(processor.im_end_id)]
            + list(processor.newline_ids)
            + [int(processor.im_start_id)]
            + self._encode("user")
            + list(processor.newline_ids)
            + self._encode(question)
            + [int(processor.im_end_id)]
            + list(processor.newline_ids)
            + [int(processor.im_start_id)]
            + self._encode("assistant")
            + list(processor.newline_ids)
        )

    def _candidate_ids(self, candidate: str):
        return self.training.encode_grounding_answer(
            self.processor, candidate, self.device
        )

    def score_text(self, question: str, candidates: Sequence[str]) -> List[float]:
        torch = self.torch
        prompt = self._text_prompt_ids(question)
        nlls = []
        for candidate in candidates:
            answer = self._candidate_ids(candidate)
            ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            full_ids = torch.cat([ids, answer.unsqueeze(0)], dim=1)
            with torch.inference_mode():
                outputs = self.model.vlm(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    mm_token_type_ids=self.model._modality_ids(full_ids),
                    use_cache=False,
                    return_dict=True,
                )
            start = len(prompt)
            logits = outputs.logits[:, start - 1 : start + len(answer) - 1].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), answer.reshape(-1)
            )
            nlls.append(float(loss.item()))
        return nlls

    def score_rgb(
        self, images: Sequence[str], question: str, candidates: Sequence[str]
    ) -> List[float]:
        torch = self.torch
        inputs = self.processor.encode_vqa(
            list(images), question, system=SYSTEM_PROMPT, device="cpu"
        )
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            prompt_embeddings = self.visibility_vlm._official_multimodal_embeddings(
                self.model, inputs
            )
        nlls = []
        for candidate in candidates:
            answer = self._candidate_ids(candidate)
            answer_embeddings = self.model.vlm.get_input_embeddings()(
                answer.unsqueeze(0)
            )
            full_embeddings = torch.cat([prompt_embeddings, answer_embeddings], dim=1)
            full_ids = torch.cat([inputs["input_ids"], answer.unsqueeze(0)], dim=1)
            positions = self.model._rope_positions(full_ids, inputs["image_grid_thw"])
            with torch.inference_mode():
                outputs = self.model.vlm.model.language_model(
                    input_ids=None,
                    inputs_embeds=full_embeddings,
                    position_ids=positions,
                    use_cache=False,
                )
                start = int(prompt_embeddings.shape[1])
                hidden = outputs.last_hidden_state[
                    :, start - 1 : start + len(answer) - 1
                ]
                logits = self.model.vlm.lm_head(hidden).float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), answer.reshape(-1)
                )
            nlls.append(float(loss.item()))
        return nlls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-curriculum-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite upper-bound report")
    checks = {
        args.model / "model.safetensors": args.expected_model_sha256,
        args.manifest: args.expected_manifest_sha256,
        args.curriculum: args.expected_curriculum_sha256,
    }
    for path, expected in checks.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError("missing or changed input: %s" % path)

    manifest = _read_json(args.manifest)
    curriculum = _read_json(args.curriculum)
    records = {row["sample_id"]: row for row in manifest["records"]}
    examples = {
        row["example_id"]: row for row in curriculum["examples"]
    }
    evaluation_ids = list(curriculum["evaluation_example_ids"])
    if len(evaluation_ids) != 20:
        raise ValueError("upper bound requires the frozen 20 evaluation routes")

    load_start = time.monotonic()
    scorer = QwenUpperBoundScorer(args.model, args.device, args.dtype)
    load_seconds = time.monotonic() - load_start
    rows = []
    evaluation_start = time.monotonic()
    for index, example_id in enumerate(evaluation_ids, start=1):
        example = examples[example_id]
        record = records[example["sample_id"]]
        value = route_value(record)
        expected = "ON_ROUTE" if value >= 0.2 else "OFF_ROUTE"
        if expected != example["expected_answer"]:
            raise ValueError("text upper-bound target disagrees with frozen example")
        question = render_question(example["query_frontier"], value)
        for mode, nlls in (
            ("text_only", scorer.score_text(question, CANDIDATES)),
            (
                "rgb_plus_text",
                scorer.score_rgb(record["camera_images"], question, CANDIDATES),
            ),
        ):
            predicted = CANDIDATES[min(range(len(nlls)), key=nlls.__getitem__)]
            expected_index = CANDIDATES.index(expected)
            rows.append(
                {
                    "mode": mode,
                    "example_id": example_id,
                    "sample_id": example["sample_id"],
                    "split": example["split"],
                    "frontier": example["query_frontier"],
                    "route_weight_mean": value,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "target_margin": nlls[1 - expected_index] - nlls[expected_index],
                    "candidate_nlls": dict(zip(CANDIDATES, nlls)),
                }
            )
        print(json.dumps({"completed": index, "total": len(evaluation_ids)}), flush=True)
    metrics = summarize(rows)
    thresholds = {
        "minimum_overall_accuracy": 0.9,
        "minimum_each_split_and_label_accuracy": 0.8,
    }
    passed_by_mode = {}
    for mode in ("text_only", "rgb_plus_text"):
        metric = metrics[mode]
        slices = [
            values
            for split in metric["by_split"].values()
            for values in split["by_label"].values()
        ]
        passed_by_mode[mode] = bool(
            metric["accuracy"] >= thresholds["minimum_overall_accuracy"]
            and all(
                row["count"] > 0
                and row["correct"] / row["count"]
                >= thresholds["minimum_each_split_and_label_accuracy"]
                for row in slices
            )
        )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "training_performed": False,
        "optimizer_steps": 0,
        "continuous_u_tokens_present": False,
        "planning_expert_loaded": False,
        "forced_choice_scoring": "minimum mean answer-token negative log likelihood",
        "system_prompt": SYSTEM_PROMPT,
        "thresholds": thresholds,
        "passed_by_mode": passed_by_mode,
        "metrics": metrics,
        "rows": rows,
        "timing": {
            "load_seconds": load_seconds,
            "evaluation_seconds": time.monotonic() - evaluation_start,
        },
        "runtime": scorer.runtime,
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "inputs": {
            "model": str(args.model.resolve()),
            "model_sha256": args.expected_model_sha256,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": args.expected_manifest_sha256,
            "curriculum": str(args.curriculum.resolve()),
            "curriculum_sha256": args.expected_curriculum_sha256,
        },
        "claim_boundary": {
            "language_and_numeric_upper_bound_only": True,
            "continuous_u_alignment_claim_allowed": False,
            "visual_spatial_alignment_claim_allowed": False,
            "planning_or_safety_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed_by_mode": passed_by_mode, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
