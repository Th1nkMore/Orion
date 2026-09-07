#!/usr/bin/env python3
"""Independently audit a Qwen numeric-text upper-bound report and its inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from evaluate_qwen_visibility_numeric_text_upper_bound import (
    CANDIDATES,
    SCHEMA,
    SYSTEM_PROMPT,
    route_value,
    summarize,
)


AUDIT_SCHEMA = "orion.qwen-visibility-numeric-text-upper-bound-audit/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def audit(report: Dict[str, Any], report_path: Path) -> Dict[str, Any]:
    failures: List[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(report.get("schema") == SCHEMA, "unexpected report schema")
    require(report.get("status") == "complete", "report is not complete")
    require(report.get("training_performed") is False, "training was performed")
    require(report.get("optimizer_steps") == 0, "optimizer steps are nonzero")
    require(
        report.get("continuous_u_tokens_present") is False,
        "continuous U tokens were present",
    )
    require(
        report.get("planning_expert_loaded") is False,
        "Planning Expert was loaded",
    )
    require(report.get("system_prompt") == SYSTEM_PROMPT, "system prompt changed")

    inputs = report.get("inputs", {})
    for name in ("model", "manifest", "curriculum"):
        path = Path(str(inputs.get(name, "")))
        if name == "model":
            path = path / "model.safetensors"
        expected = inputs.get(name + "_sha256")
        require(path.is_file(), "missing input file: %s" % name)
        if path.is_file():
            require(sha256(path) == expected, "input hash mismatch: %s" % name)

    manifest_path = Path(str(inputs.get("manifest", "")))
    curriculum_path = Path(str(inputs.get("curriculum", "")))
    if not manifest_path.is_file() or not curriculum_path.is_file():
        manifest, curriculum = {}, {}
    else:
        manifest = read_object(manifest_path)
        curriculum = read_object(curriculum_path)
    records = {row["sample_id"]: row for row in manifest.get("records", [])}
    examples = {
        row["example_id"]: row for row in curriculum.get("examples", [])
    }
    evaluation_ids = list(curriculum.get("evaluation_example_ids", []))
    require(len(evaluation_ids) == 20, "evaluation set is not the frozen 20 examples")

    rows = list(report.get("rows", []))
    require(len(rows) == 40, "report does not contain 20 examples x 2 modes")
    seen = set()
    for row in rows:
        key = (row.get("example_id"), row.get("mode"))
        require(key not in seen, "duplicate example/mode row: %r" % (key,))
        seen.add(key)
        require(row.get("mode") in ("text_only", "rgb_plus_text"), "unknown mode")
        example = examples.get(row.get("example_id"))
        require(example is not None, "row is absent from frozen curriculum")
        if example is None:
            continue
        require(row.get("sample_id") == example.get("sample_id"), "sample mismatch")
        require(row.get("split") == example.get("split"), "split mismatch")
        require(
            row.get("frontier") == example.get("query_frontier"),
            "frontier mismatch",
        )
        record = records.get(example.get("sample_id"))
        require(record is not None, "sample is absent from manifest")
        if record is None:
            continue
        value = route_value(record)
        expected = "ON_ROUTE" if value >= 0.2 else "OFF_ROUTE"
        require(math.isclose(float(row.get("route_weight_mean")), value), "value mismatch")
        require(row.get("expected") == expected, "threshold target mismatch")
        nlls = row.get("candidate_nlls", {})
        require(set(nlls) == set(CANDIDATES), "candidate NLL keys changed")
        if set(nlls) != set(CANDIDATES):
            continue
        require(all(math.isfinite(float(nlls[x])) for x in CANDIDATES), "non-finite NLL")
        predicted = min(CANDIDATES, key=lambda candidate: float(nlls[candidate]))
        require(row.get("predicted") == predicted, "prediction is not NLL argmin")
        require(row.get("correct") is (predicted == expected), "correct flag mismatch")
        other = CANDIDATES[1 - CANDIDATES.index(expected)]
        margin = float(nlls[other]) - float(nlls[expected])
        require(math.isclose(float(row.get("target_margin")), margin), "margin mismatch")

    for example_id in evaluation_ids:
        for mode in ("text_only", "rgb_plus_text"):
            require((example_id, mode) in seen, "missing frozen example/mode pair")

    recomputed = summarize(rows) if rows else {}
    require(report.get("metrics") == recomputed, "summary metrics mismatch")
    thresholds = report.get("thresholds", {})
    passed_by_mode = {}
    for mode in ("text_only", "rgb_plus_text"):
        metric = recomputed.get(mode, {})
        slices = [
            label
            for split in metric.get("by_split", {}).values()
            for label in split.get("by_label", {}).values()
        ]
        passed_by_mode[mode] = bool(
            metric
            and metric.get("accuracy", -1)
            >= thresholds.get("minimum_overall_accuracy", math.inf)
            and all(
                item.get("count", 0) > 0
                and item.get("correct", 0) / item["count"]
                >= thresholds.get("minimum_each_split_and_label_accuracy", math.inf)
                for item in slices
            )
        )
    require(report.get("passed_by_mode") == passed_by_mode, "pass decision mismatch")

    boundary = report.get("claim_boundary", {})
    require(
        boundary
        == {
            "language_and_numeric_upper_bound_only": True,
            "continuous_u_alignment_claim_allowed": False,
            "visual_spatial_alignment_claim_allowed": False,
            "planning_or_safety_claim_allowed": False,
        },
        "claim boundary changed",
    )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
        "row_count": len(rows),
        "evaluation_example_count": len(evaluation_ids),
        "recomputed_metrics": recomputed,
        "recomputed_passed_by_mode": passed_by_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit report")
    result = audit(read_object(args.report), args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
