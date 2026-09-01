#!/usr/bin/env python3
"""Build a diversity-aware, hash-bound human review queue for scenario events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


QUEUE_SCHEMA = "orion.scenario_event_review_queue.v1"
DECISIONS_SCHEMA = "orion.scenario_event_review_decisions.v1"
REPORT_SCHEMA = "orion.scenario_factory.batch_screen_report.v1"
PACKAGE_SCHEMA = "orion.scenario_event_package.v1"

OUTCOME_PRIORITY = {
    "VALID_COLLISION": 0,
    "VALID_SERIOUS_INFRACTION": 1,
    "VALID_MODEL_INCOMPLETE": 2,
    "VALID_SEVERE_TTC": 3,
    "VALID_NEAR_MISS_OR_CONFLICT": 4,
    "VALID_SAFE_NO_ACTOR_GROUNDED_EVENT": 5,
    "INVALID_RUNTIME": 99,
}

HUMAN_CHECKS = (
    "visual_stream_integrity",
    "actor_event_semantics",
    "front_bev_temporal_alignment",
    "no_actor_disappearance_or_spawn_artifact",
)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object: %s" % path)
    return payload


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _finite_metric(package: Mapping[str, Any], key: str) -> Any:
    value = package.get("continuous_safety", {}).get("safety", {}).get(key)
    return value if isinstance(value, (int, float)) else None


def _review_warnings(package: Mapping[str, Any]) -> List[str]:
    """Surface suspicious-but-reviewable events without changing eligibility."""
    warnings: List[str] = []
    event = package.get("critical_event")
    if isinstance(event, Mapping):
        progress = event.get("route_progress")
        if isinstance(progress, (int, float)):
            if float(progress) <= 0.02:
                warnings.append("critical_event_near_route_start")
            if float(progress) >= 0.98:
                warnings.append("critical_event_near_route_end")
    continuous = package.get("continuous_safety", {})
    duration = continuous.get("duration_seconds")
    stopped = continuous.get("efficiency", {}).get(
        "stopped_below_0_25_mps_seconds"
    )
    if (
        isinstance(duration, (int, float))
        and isinstance(stopped, (int, float))
        and float(duration) > 0.0
        and float(stopped) / float(duration) >= 0.5
    ):
        warnings.append("majority_of_route_nearly_stopped")
    return warnings


def _visual_paths(package: Mapping[str, Any], package_path: Path) -> List[str]:
    reference = package.get("visualization")
    if not isinstance(reference, dict):
        return []
    manifest_path = _resolve(str(reference.get("path", "")), package_path.parent)
    if not manifest_path.is_file() or sha256_file(manifest_path) != reference.get("sha256"):
        return []
    manifest = _load_json(manifest_path)
    paths: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.lower().endswith((".gif", ".png", ".mp4")):
            candidate = _resolve(value, manifest_path.parent)
            if candidate.is_file():
                paths.append(str(candidate.resolve()))

    visit(manifest)
    return sorted(set(paths))


def _rows_from_report(report_path: Path) -> Iterable[Dict[str, Any]]:
    report = _load_json(report_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported batch report schema: %s" % report_path)
    route_meta = {int(row["route_index"]): row for row in report.get("routes", [])}
    packaged_routes = set()
    for reference in report.get("event_packages", []):
        package_path = _resolve(str(reference["path"]), report_path.parent)
        if not package_path.is_file():
            raise FileNotFoundError(package_path)
        package_sha = sha256_file(package_path)
        if package_sha != reference.get("sha256"):
            raise ValueError("event-package SHA-256 mismatch: %s" % package_path)
        package = _load_json(package_path)
        if package.get("schema") != PACKAGE_SCHEMA:
            raise ValueError("unsupported event-package schema: %s" % package_path)
        route_index = int(package["route"]["route_index"])
        packaged_routes.add(route_index)
        meta = route_meta.get(route_index)
        if meta is None:
            raise ValueError("batch report lacks route metadata for %d" % route_index)
        event = package.get("critical_event")
        runtime_valid = package.get("runtime", {}).get("valid") is True
        qa_ready = package.get("qa_input_ready") is True
        eligible = runtime_valid and qa_ready and isinstance(event, dict)
        event_id = (
            "route%d_step%d" % (route_index, int(event["step"]))
            if isinstance(event, dict)
            else "route%d_no_actor_event" % route_index
        )
        yield {
            "event_id": event_id,
            "route_index": route_index,
            "split_origin": package.get("split"),
            "town": meta.get("town"),
            "scenario_family": meta.get("scenario_type"),
            "screen_role": meta.get("screen_role"),
            "outcome_class": package.get("outcome_class"),
            "runtime_valid": runtime_valid,
            "qa_input_ready": qa_ready,
            "actor_grounded_event": isinstance(event, dict),
            "critical_event": event,
            "official_endpoint": package.get("official_endpoint", {}),
            "continuous_safety_summary": {
                "min_obb_ttc_seconds": _finite_metric(package, "min_obb_ttc_seconds"),
                "min_obb_separating_axis_gap_m": _finite_metric(
                    package, "min_obb_separating_axis_gap_m"
                ),
                "min_predicted_disc_clearance_m": _finite_metric(
                    package, "min_predicted_disc_clearance_m"
                ),
            },
            "event_package": {
                "path": str(package_path.resolve()),
                "sha256": package_sha,
            },
            "visual_artifacts": _visual_paths(package, package_path),
            "automatic_review_warnings": _review_warnings(package),
            "automatic_status": (
                "eligible_for_human_event_review"
                if eligible
                else "excluded_before_human_review"
            ),
            "automatic_exclusion_reasons": [
                name
                for name, passed in (
                    ("runtime_invalid", runtime_valid),
                    ("qa_inputs_incomplete", qa_ready),
                    ("no_actor_grounded_event", isinstance(event, dict)),
                )
                if not passed
            ],
        }
    for route_index, meta in sorted(route_meta.items()):
        if route_index in packaged_routes:
            continue
        yield {
            "event_id": "route%d_unpackageable_runtime" % route_index,
            "route_index": route_index,
            "split_origin": report.get("split"),
            "town": meta.get("town"),
            "scenario_family": meta.get("scenario_type"),
            "screen_role": meta.get("screen_role"),
            "outcome_class": meta.get(
                "outcome_class", "INVALID_RUNTIME_UNPACKAGEABLE"
            ),
            "runtime_valid": False,
            "qa_input_ready": False,
            "actor_grounded_event": False,
            "critical_event": None,
            "official_endpoint": {},
            "continuous_safety_summary": {},
            "event_package": None,
            "visual_artifacts": [],
            "automatic_status": "excluded_before_human_review",
            "automatic_exclusion_reasons": [
                "runtime_invalid",
                "event_package_unbuildable",
            ],
            "runtime_attempts": meta.get("attempts", []),
        }


def _diversity_order(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remaining = list(rows)
    ordered: List[Dict[str, Any]] = []
    seen_towns = set()
    seen_families = set()
    seen_roles = set()
    while remaining:
        def key(row: Mapping[str, Any]) -> Any:
            return (
                -int(row.get("scenario_family") not in seen_families),
                -int(row.get("town") not in seen_towns),
                -int(row.get("screen_role") not in seen_roles),
                OUTCOME_PRIORITY.get(str(row.get("outcome_class")), 50),
                int(row["route_index"]),
                str(row["event_id"]),
            )

        selected = min(remaining, key=key)
        remaining.remove(selected)
        selected = dict(selected)
        selected["review_rank"] = len(ordered) + 1
        ordered.append(selected)
        seen_towns.add(selected.get("town"))
        seen_families.add(selected.get("scenario_family"))
        seen_roles.add(selected.get("screen_role"))
    return ordered


def build_review_queue(report_paths: Sequence[Path]) -> Dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one batch report is required")
    rows: List[Dict[str, Any]] = []
    seen_events = set()
    for report_path in report_paths:
        for row in _rows_from_report(report_path.resolve()):
            package = row.get("event_package")
            identity = (
                row["event_id"],
                package.get("sha256") if isinstance(package, dict) else None,
            )
            if identity in seen_events:
                continue
            seen_events.add(identity)
            rows.append(row)
    eligible = _diversity_order(
        [row for row in rows if row["automatic_status"] == "eligible_for_human_event_review"]
    )
    excluded = sorted(
        [row for row in rows if row["automatic_status"] != "eligible_for_human_event_review"],
        key=lambda row: (int(row["route_index"]), str(row["event_id"])),
    )
    sources = [
        {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for path in report_paths
    ]
    return {
        "schema": QUEUE_SCHEMA,
        "status": "pending_human_event_review",
        "source_batch_reports": sources,
        "candidate_count": len(rows),
        "human_review_count": len(eligible),
        "automatically_excluded_count": len(excluded),
        "review_order": eligible,
        "automatically_excluded": excluded,
        "human_review_contract": {
            "required_checks": list(HUMAN_CHECKS),
            "decision_values": ["accept", "reject", "pending"],
            "locked_test_rejection_must_be_technical_only": True,
            "prohibited_rejection_basis": [
                "learned UQ behavior",
                "Stage2 behavior",
                "whether ORION succeeded or failed",
            ],
        },
        "claim_boundary": (
            "Queue priority is for review efficiency only. Human acceptance establishes "
            "event integrity, not UQ validity or closed-loop benefit."
        ),
    }


def decisions_template(queue: Mapping[str, Any], queue_path: Path) -> Dict[str, Any]:
    return {
        "schema": DECISIONS_SCHEMA,
        "status": "unreviewed_template",
        "reviewer": None,
        "reviewed_at": None,
        "review_queue": {
            "path": str(queue_path.resolve()),
            "sha256": sha256_file(queue_path),
        },
        "decisions": [
            {
                "event_id": row["event_id"],
                "event_package_sha256": row["event_package"]["sha256"],
                "decision": "pending",
                "checks": {name: "pending" for name in HUMAN_CHECKS},
                "rejection_basis": None,
                "notes": "",
            }
            for row in queue["review_order"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty review output")
    queue = build_review_queue(args.batch_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "scenario_event_review_queue.json"
    queue_path.write_text(
        json.dumps(queue, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    template = decisions_template(queue, queue_path)
    template_path = args.output_dir / "scenario_event_review_decisions.template.json"
    template_path.write_text(
        json.dumps(template, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "queue": str(queue_path.resolve()),
                "decisions_template": str(template_path.resolve()),
                "human_review_count": queue["human_review_count"],
                "automatically_excluded_count": queue["automatically_excluded_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
