#!/usr/bin/env python3
"""Verify registered hard closed-loop facts against local evaluator JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EVALUATOR_FIELDS = {
    "eligible",
    "route_completion_percent",
    "pedestrian_collisions",
    "vehicle_collisions",
    "layout_collisions",
    "collisions",
    "score_penalty",
}


def _find_eval(results_root: Path, job_id: int) -> Path:
    candidates = sorted(results_root.glob(f"**/*{job_id}/eval*.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"job {job_id}: expected one evaluator JSON, found {len(candidates)}"
        )
    return candidates[0]


def _observed(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    records = payload.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError(f"{eval_path}: expected one evaluator record")
    record = records[0]
    infractions = record.get("infractions", {})

    def count(name):
        value = infractions.get(name, [])
        return len(value) if isinstance(value, list) else int(value)

    pedestrian = count("collisions_pedestrian")
    vehicle = count("collisions_vehicle")
    layout = count("collisions_layout")
    scores = record.get("scores", {})
    return {
        "eligible": bool(payload.get("eligible")),
        "route_completion_percent": float(scores.get("score_route")),
        "pedestrian_collisions": pedestrian,
        "vehicle_collisions": vehicle,
        "layout_collisions": layout,
        "collisions": pedestrian + vehicle + layout,
        "score_penalty": float(scores.get("score_penalty")),
        "status": record.get("status"),
    }


def _matches(expected, observed):
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, (int, float)):
        return math.isclose(float(expected), float(observed), abs_tol=1e-6)
    return expected == observed


def verify(register_path: Path, results_root: Path) -> dict[str, Any]:
    register = json.loads(register_path.read_text(encoding="utf-8"))
    checks = []
    for case in register.get("existing_evidence_register", []):
        for fact in case.get("facts", []):
            job_id = int(fact["job_id"])
            eval_path = _find_eval(results_root, job_id)
            observed = _observed(eval_path)
            mismatches = []
            for key in EVALUATOR_FIELDS.intersection(fact):
                if not _matches(fact[key], observed[key]):
                    mismatches.append({
                        "field": key,
                        "expected": fact[key],
                        "observed": observed[key],
                    })
            checks.append({
                "job_id": job_id,
                "route_index": case["route_index"],
                "condition": fact["condition"],
                "eval_path": str(eval_path.resolve()),
                "observed": observed,
                "mismatches": mismatches,
                "passed": not mismatches,
            })
    return {
        "schema": "orion.closedloop_safety_outcomes_verification.v1",
        "register": str(register_path.resolve()),
        "checks": checks,
        "passed": bool(checks) and all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.register, args.results_root)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit("refusing to overwrite verification report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
