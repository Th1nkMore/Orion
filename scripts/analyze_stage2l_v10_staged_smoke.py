#!/usr/bin/env python3
"""Reduce a Stage2-L v10 staged report to its gate decision and trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.scenario_factory_lib import sha256_file


REPORT_SCHEMA = "orion.stage2l_v10_staged_smoke.v1"
SUMMARY_SCHEMA = "orion.stage2l_v10_staged_analysis.v1"


def _trend(phase: Mapping[str, Any], key: str) -> Dict[str, Any]:
    history = list(phase.get("history", []))
    values = [float(item[key]) for item in history if item.get(key) is not None]
    if not values:
        return {"available": False}
    return {
        "available": True,
        "first": values[0],
        "last": values[-1],
        "absolute_change": values[-1] - values[0],
        "relative_change": (
            (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else None
        ),
        "minimum": min(values),
        "maximum": max(values),
        "all_finite_flags_true": all(
            bool(item.get("finite")) for item in history
        ),
    }


def analyze(report: Mapping[str, Any], *, report_sha256: str) -> Dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported Stage2-L v10 report schema")
    if (
        report.get("engineering_preexperiment_only") is not True
        or report.get("formal_stage2l_ready") is not False
        or report.get("stage2p_ready") is not False
        or report.get("locks", {}).get("density_uq_or_governor_used") is not False
        or report.get("locks", {}).get("trajectory_or_control_loss_used") is not False
    ):
        raise ValueError("Stage2-L v10 claim or responsibility locks differ")
    phases = report.get("phases", {})
    phase_summary = {}
    for name, phase in phases.items():
        checks = dict(phase.get("checks", {}))
        trends = {}
        if name in {"A_map_pretrain", "B_risk_alignment"}:
            trends["loss"] = _trend(phase, "loss")
            trends["map_loss"] = _trend(phase, "map_loss")
            trends["ranking_loss"] = _trend(phase, "ranking_loss")
        elif name == "C_language_grounding":
            trends["mean_target_nll"] = _trend(phase, "mean_target_nll")
        phase_summary[name] = {
            "status": phase.get("status"),
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "optimizer_steps_completed": len(phase.get("history", [])),
            "trends": trends,
        }
    status = str(report.get("status"))
    decisions = {
        "stopped_after_phase_a_failed_gate": "stop_and_revise_spatial_R_objective_or_support_labels",
        "stopped_after_phase_b_failed_gate": "retain_R_result_and_revise_matched_risk_alignment",
        "stopped_after_phase_c_failed_gate": "retain_map_ranking_result_and_revise_language_bridge",
        "all_bounded_v10_phases_pass": "eligible_for_one_clean_corrupt_engineering_interface_smoke_only",
    }
    if status not in decisions:
        raise ValueError("unrecognized Stage2-L v10 terminal status")
    return {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "decision": decisions[status],
        "completed_phases": list(report.get("completed_phases", [])),
        "phases": phase_summary,
        "provenance": {
            "report_sha256": report_sha256,
            "trainer_sha256": report["provenance"]["trainer"]["sha256"],
            "protocol_sha256": report["provenance"]["protocol"]["sha256"],
            "frozen_u_tokenizer_sha256": report["provenance"][
                "frozen_u_tokenizer"
            ]["sha256"],
        },
        "unlocks": {
            "one_clean_corrupt_engineering_interface_smoke": status
            == "all_bounded_v10_phases_pass",
            "formal_stage2l": False,
            "stage2p": False,
            "closed_loop_matrix": False,
            "safety_claim": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v10 analysis")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = analyze(report, report_sha256=sha256_file(args.report.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
