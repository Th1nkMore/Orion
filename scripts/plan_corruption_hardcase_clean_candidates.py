#!/usr/bin/env python3
"""Plan the next clean-qualification wave from reviewed current-runtime events.

This is a metadata-only planner.  It never launches CARLA, loads ORION, reads
corruption-conditioned results, or authorizes a scheduler submission.  The
planner deliberately combines the reviewed event bank with each event's clean
runtime package so an old published endpoint cannot hide a current collision or
an obviously non-live baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA = "orion.corruption_hardcase_clean_candidate_plan.v1"
ALLOWED_ACTOR_CATEGORIES = {"vehicle", "walker"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_event_package(path: str, roots: Sequence[Path]) -> Path:
    direct = Path(path)
    if direct.is_file():
        return direct
    marker = "/event_packages/"
    if marker not in path:
        raise FileNotFoundError("event package has no portable suffix: %s" % path)
    suffix = path.split(marker, 1)[1]
    matches = [root / suffix for root in roots if (root / suffix).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected one event-package mirror for %s, found %d" % (path, len(matches))
        )
    return matches[0]


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_or_reasons(
    event: Mapping[str, Any],
    package: Mapping[str, Any],
    *,
    excluded_routes: set[int],
    allowed_splits: set[str],
    maximum_total_stopped_seconds: float,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    route_index = int(event["route_index"])
    reasons: list[str] = []
    endpoint = package.get("official_endpoint", {})
    scores = endpoint.get("scores", {}) if isinstance(endpoint, Mapping) else {}
    efficiency = package.get("continuous_safety", {}).get("efficiency", {})
    critical = package.get("continuous_safety", {}).get("safety", {}).get(
        "critical_frame", {}
    )
    actor = critical.get("actor", {}) if isinstance(critical, Mapping) else {}

    if route_index in excluded_routes:
        reasons.append("explicitly_excluded_by_prior_current_runtime_evidence")
    if str(event.get("formal_split")) not in allowed_splits:
        reasons.append("protected_or_non_development_split")
    if event.get("runtime_valid") is not True or package.get("runtime", {}).get(
        "valid"
    ) is not True:
        reasons.append("runtime_invalid")
    if event.get("human_review", {}).get("decision") != "accept":
        reasons.append("human_event_review_not_accepted")
    if endpoint.get("status") != "Completed" or scores.get("score_route") != 100:
        reasons.append("clean_route_not_completed")
    if int(endpoint.get("collision_count", 0)) != 0:
        reasons.append("clean_collision")
    if int(endpoint.get("serious_infraction_count", 0)) != 0:
        reasons.append("clean_serious_infraction")

    total_stopped = _finite_number(efficiency.get("stopped_below_0_25_mps_seconds"))
    if total_stopped is None:
        reasons.append("missing_clean_stopped_exposure")
    elif total_stopped > maximum_total_stopped_seconds:
        # Total stopped exposure upper-bounds the longest contiguous interval.
        reasons.append("clean_total_stopped_exposure_exceeds_liveness_bound")

    actor_category = str(actor.get("category", ""))
    closing_speed = _finite_number(actor.get("closing_speed_mps"))
    minimum_ttc = _finite_number(actor.get("obb_collision_ttc_seconds"))
    if actor_category not in ALLOWED_ACTOR_CATEGORIES:
        reasons.append("no_supported_dynamic_actor_at_clean_critical_frame")
    if closing_speed is None or closing_speed <= 0:
        reasons.append("critical_actor_not_closing")
    if minimum_ttc is None or minimum_ttc <= 0:
        reasons.append("clean_critical_ttc_not_positive_finite")

    if reasons:
        return None, reasons

    event_package_ref = event["event_package"]
    return {
        "route_index": route_index,
        "event_id": event["event_id"],
        "town": event["town"],
        "scenario_family": event["scenario_family"],
        "screen_role": event["screen_role"],
        "formal_split": event["formal_split"],
        "clean_job_id": int(package["route"]["slurm_job_id"]),
        "clean_outcome_class": package["outcome_class"],
        "clean_total_stopped_below_0_25_mps_seconds": total_stopped,
        "clean_critical_actor_category": actor_category,
        "clean_critical_actor_type": actor.get("type_id"),
        "clean_critical_closing_speed_mps": closing_speed,
        "clean_critical_obb_ttc_seconds": minimum_ttc,
        "clean_critical_obb_gap_m": _finite_number(
            actor.get("obb_separating_axis_gap_m")
        ),
        "event_package": {
            "path": event_package_ref["path"],
            "sha256": event_package_ref["sha256"],
        },
        "route_xml": package["source_files"]["route_xml"],
        "selection_uses_clean_evidence_only": True,
    }, []


def build_plan(
    event_bank: Mapping[str, Any],
    *,
    event_bank_path: Path,
    event_package_roots: Sequence[Path],
    excluded_routes: Iterable[int],
    allowed_splits: Iterable[str],
    maximum_total_stopped_seconds: float,
    limit: int,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("candidate limit must be positive")
    if maximum_total_stopped_seconds <= 0:
        raise ValueError("liveness bound must be positive")

    excluded = {int(value) for value in excluded_routes}
    splits = {str(value) for value in allowed_splits}
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in event_bank.get("events", []):
        package_path = _resolve_event_package(
            str(event["event_package"]["path"]), event_package_roots
        )
        observed_hash = sha256_file(package_path)
        expected_hash = str(event["event_package"]["sha256"])
        if observed_hash != expected_hash:
            raise ValueError("event package hash differs: %s" % package_path)
        package = json.loads(package_path.read_text(encoding="utf-8"))
        candidate, reasons = _candidate_or_reasons(
            event,
            package,
            excluded_routes=excluded,
            allowed_splits=splits,
            maximum_total_stopped_seconds=maximum_total_stopped_seconds,
        )
        if candidate is not None:
            candidates.append(candidate)
        else:
            exclusions.append(
                {
                    "route_index": int(event["route_index"]),
                    "event_id": event["event_id"],
                    "reasons": sorted(set(reasons)),
                }
            )

    # Smaller clean TTC is a stronger already-present interaction.  Closing
    # speed breaks ties without consulting any corruption-conditioned output.
    candidates.sort(
        key=lambda row: (
            float(row["clean_critical_obb_ttc_seconds"]),
            -float(row["clean_critical_closing_speed_mps"]),
            int(row["route_index"]),
        )
    )
    selected = candidates[:limit]
    for rank, row in enumerate(selected, 1):
        row["selection_rank"] = rank

    return {
        "schema": SCHEMA,
        "status": "planned_not_activated_no_jobs_authorized",
        "source": {
            "event_bank_path": str(event_bank_path),
            "event_bank_sha256": sha256_file(event_bank_path),
            "reviewed_event_count": len(event_bank.get("events", [])),
        },
        "policy": {
            "allowed_formal_splits": sorted(splits),
            "explicitly_excluded_routes": sorted(excluded),
            "maximum_total_stopped_below_0_25_mps_seconds": maximum_total_stopped_seconds,
            "required_clean_endpoint": "Completed, 100% route, zero collision, zero serious infraction",
            "required_event": "human-accepted current-runtime event with a closing vehicle or walker and positive finite OBB TTC",
            "ranking": "ascending clean critical OBB TTC, descending closing speed, ascending route index",
            "corruption_conditioned_outputs_used": False,
            "learned_uq_or_stage2_outputs_used": False,
        },
        "counts": {
            "eligible_before_limit": len(candidates),
            "selected": len(selected),
            "excluded": len(exclusions),
        },
        "selected_candidates": selected,
        "remaining_eligible_candidates": candidates[limit:],
        "excluded_reviewed_events": sorted(
            exclusions, key=lambda row: int(row["route_index"])
        ),
        "execution_locks": {
            "clean_submission": False,
            "corruption_submission": False,
            "heldout_read": False,
            "severity_change": False,
            "learned_uq_or_control": False,
            "stage2p": False,
            "formal_200_route_evaluation": False,
        },
        "next_gate": "Freeze a separate prospective clean Q1/Q2 activation after user resumes experiments.",
        "claim_boundary": "Clean-candidate planning only; no new runtime, corruption, UQ, control, or safety result.",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-bank", type=Path, required=True)
    parser.add_argument(
        "--event-package-root", type=Path, action="append", required=True
    )
    parser.add_argument("--exclude-route", type=int, nargs="*", default=[])
    parser.add_argument("--allowed-split", action="append", default=["train"])
    parser.add_argument("--maximum-total-stopped-seconds", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite %s" % args.output)
    event_bank = json.loads(args.event_bank.read_text(encoding="utf-8"))
    plan = build_plan(
        event_bank,
        event_bank_path=args.event_bank,
        event_package_roots=args.event_package_root,
        excluded_routes=args.exclude_route,
        allowed_splits=args.allowed_split,
        maximum_total_stopped_seconds=args.maximum_total_stopped_seconds,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "selected_routes": [
            row["route_index"] for row in plan["selected_candidates"]
        ],
        "eligible_before_limit": plan["counts"]["eligible_before_limit"],
        "status": plan["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
