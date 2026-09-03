#!/usr/bin/env python3
"""Deterministically screen Bench2Drive routes for spatial-UQ pilots.

The script is intentionally read-only: it parses per-route XML files and, when
provided, an existing leaderboard result JSON.  It prints a JSON manifest to
stdout and never launches CARLA or submits a scheduler job.
"""

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ALLOWED_TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD")
DEFAULT_EXCLUDED_INDICES = (146, 148, 203)

# Larger is better.  The ordering favors a localized, dynamic conflict with the
# ego path over static obstacles or failures that mainly test route following.
SCENARIO_PRIORITY: Mapping[str, int] = {
    "DynamicObjectCrossing": 100,
    "ParkingCrossingPedestrian": 98,
    "PedestrianCrossing": 97,
    "OppositeVehicleRunningRedLight": 96,
    "ParkingCutIn": 94,
    "HazardAtSideLane": 92,
    "HazardAtSideLaneTwoWays": 90,
    "NonSignalizedJunctionLeftTurnEnterFlow": 88,
    "SignalizedJunctionLeftTurnEnterFlow": 86,
    "EnterActorFlow": 84,
    "SignalizedJunctionRightTurn": 82,
    # Secondary development-screen roles.  These do not replace the dynamic
    # conflict pool above; they add static hazards and hard negatives needed to
    # teach the VLM that observation uncertainty is not automatically relevant.
    "StaticCutIn": 80,
    "HardBreakRoute": 78,
    "ConstructionObstacle": 76,
    "Accident": 75,
    "ParkedObstacle": 74,
    "ParkedObstacleTwoWays": 73,
    "BlockedIntersection": 72,
    "T_Junction": 70,
    "ControlLoss": 68,
}

SCENARIO_SCREEN_ROLE: Mapping[str, str] = {
    "DynamicObjectCrossing": "dynamic_path_conflict",
    "ParkingCrossingPedestrian": "dynamic_path_conflict",
    "PedestrianCrossing": "dynamic_path_conflict",
    "OppositeVehicleRunningRedLight": "dynamic_path_conflict",
    "ParkingCutIn": "dynamic_path_conflict",
    "HazardAtSideLane": "dynamic_path_conflict",
    "HazardAtSideLaneTwoWays": "dynamic_path_conflict_with_background_flow",
    "NonSignalizedJunctionLeftTurnEnterFlow": "dynamic_junction_conflict",
    "SignalizedJunctionLeftTurnEnterFlow": "dynamic_junction_conflict",
    "EnterActorFlow": "dynamic_junction_conflict",
    "SignalizedJunctionRightTurn": "dynamic_junction_conflict",
    "StaticCutIn": "static_visual_hazard",
    "HardBreakRoute": "longitudinal_response_hazard",
    "ConstructionObstacle": "static_visual_hazard",
    "Accident": "static_visual_hazard",
    "ParkedObstacle": "static_visual_hazard",
    "ParkedObstacleTwoWays": "static_visual_hazard_with_opposing_flow",
    "BlockedIntersection": "liveness_and_rule_hard_negative",
    "T_Junction": "route_geometry_hard_negative",
    "ControlLoss": "non_perceptual_control_hard_negative",
}

ROUTE_FILE_RE = re.compile(r"bench2drive220_(\d+)_orion_traj\.xml$")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes-dir",
        type=Path,
        required=True,
        help="Directory containing bench2drive220_<index>_orion_traj.xml files.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Optional leaderboard JSON used to require a valid clean baseline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of ranked candidates to emit (default: 8).",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=list(DEFAULT_EXCLUDED_INDICES),
        metavar="INDEX",
        help="Route indices to exclude (default: 146 148 203).",
    )
    parser.add_argument(
        "--allow-invalid-clean",
        action="store_true",
        help="Keep routes with absent or invalid clean results for diagnosis.",
    )
    return parser.parse_args(argv)


def _route_element(root: ET.Element) -> Optional[ET.Element]:
    if root.tag == "route":
        return root
    return root.find("route")


def _polyline_projection(
    points: Sequence[Tuple[float, float]], target: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Return route length, along-route distance, and lateral projection error."""
    total = 0.0
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float, float]] = []
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        segments.append((start, end, length, total))
        total += length

    best: Optional[Tuple[float, float]] = None
    for start, end, length, prefix in segments:
        if length <= 1e-9:
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        u = ((target[0] - start[0]) * dx + (target[1] - start[1]) * dy) / (
            length * length
        )
        u = min(1.0, max(0.0, u))
        projected = (start[0] + u * dx, start[1] + u * dy)
        lateral = math.hypot(target[0] - projected[0], target[1] - projected[1])
        candidate = (lateral, prefix + u * length)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return total, 0.0, math.inf
    return total, best[1], best[0]


def _result_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(payload, dict):
        return ()
    checkpoint = payload.get("_checkpoint", {})
    if not isinstance(checkpoint, dict):
        return ()
    records = checkpoint.get("records", ())
    if not isinstance(records, list):
        return ()
    return (record for record in records if isinstance(record, dict))


def load_baseline_results(path: Path) -> Dict[str, Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    by_xml_route_id: Dict[str, Mapping[str, Any]] = {}
    for record in _result_records(payload):
        match = re.search(r"RouteScenario_(\d+)_rep\d+$", str(record.get("route_id", "")))
        if match:
            by_xml_route_id[match.group(1)] = record
    return by_xml_route_id


def _count_infractions(record: Mapping[str, Any], names: Sequence[str]) -> int:
    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return 0
    count = 0
    for name in names:
        entries = infractions.get(name, ())
        if isinstance(entries, list):
            count += len(entries)
    return count


def summarize_clean(record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if record is None:
        return {"available": False, "valid": False}

    scores = record.get("scores", {})
    scores = scores if isinstance(scores, dict) else {}
    collisions = _count_infractions(
        record,
        ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"),
    )
    completion_failures = _count_infractions(
        record,
        ("vehicle_blocked", "route_dev", "route_timeout", "scenario_timeouts"),
    )
    status = record.get("status")
    route_score = scores.get("score_route")
    penalty_score = scores.get("score_penalty")
    valid = (
        status == "Completed"
        and route_score == 100
        and penalty_score == 1
        and collisions == 0
        and completion_failures == 0
    )
    return {
        "available": True,
        "valid": valid,
        "status": status,
        "score_route": route_score,
        "score_penalty": penalty_score,
        "score_composed": scores.get("score_composed"),
        "collisions": collisions,
        "completion_failures": completion_failures,
    }


def _scenario_parameters(scenario: ET.Element) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}
    for child in scenario:
        if child.tag == "trigger_point":
            continue
        parameters[child.tag] = dict(sorted(child.attrib.items()))
    return parameters


def parse_route(path: Path, baseline: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    match = ROUTE_FILE_RE.search(path.name)
    if match is None:
        return []
    route_index = int(match.group(1))
    route = _route_element(ET.parse(path).getroot())
    if route is None or route.get("town") not in ALLOWED_TOWNS:
        return []

    points = [
        (float(position.get("x", "nan")), float(position.get("y", "nan")))
        for position in route.findall("./waypoints/position")
    ]
    if len(points) < 2:
        return []

    xml_route_id = str(route.get("id", ""))
    clean = summarize_clean(baseline.get(xml_route_id))
    rows: List[Dict[str, Any]] = []
    for scenario in route.findall("./scenarios/scenario"):
        scenario_type = str(scenario.get("type", ""))
        priority = SCENARIO_PRIORITY.get(scenario_type)
        trigger = scenario.find("trigger_point")
        if priority is None or trigger is None:
            continue

        trigger_xy = (float(trigger.get("x", "nan")), float(trigger.get("y", "nan")))
        route_length, trigger_along, trigger_lateral = _polyline_projection(points, trigger_xy)
        rows.append(
            {
                "route_index": route_index,
                "xml_route_id": xml_route_id,
                "source_xml": path.name,
                "town": route.get("town"),
                "scenario_name": scenario.get("name"),
                "scenario_type": scenario_type,
                "screen_role": SCENARIO_SCREEN_ROLE[scenario_type],
                "priority": priority,
                "trigger_point": {
                    key: float(trigger.get(key, "0")) for key in ("x", "y", "z", "yaw")
                },
                "route_length_m": round(route_length, 3),
                "trigger_along_route_m": round(trigger_along, 3),
                "trigger_progress": round(trigger_along / route_length, 6),
                "trigger_lateral_error_m": round(trigger_lateral, 3),
                "scenario_parameters": _scenario_parameters(scenario),
                "clean_baseline": clean,
            }
        )
    return rows


def select_candidates(
    routes_dir: Path,
    baseline: Mapping[str, Mapping[str, Any]],
    excluded: Sequence[int],
    limit: int,
    allow_invalid_clean: bool,
) -> List[Dict[str, Any]]:
    excluded_set = set(excluded)
    rows: List[Dict[str, Any]] = []
    for path in sorted(routes_dir.glob("bench2drive220_*_orion_traj.xml")):
        for row in parse_route(path, baseline):
            if row["route_index"] in excluded_set:
                continue
            clean = row["clean_baseline"]
            if baseline and not allow_invalid_clean and not clean["valid"]:
                continue
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["clean_baseline"].get("valid", False)),
            -int(row["priority"]),
            str(row["town"]),
            int(row["route_index"]),
        )
    )
    return rows[:limit]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    baseline = load_baseline_results(args.baseline_results) if args.baseline_results else {}
    candidates = select_candidates(
        routes_dir=args.routes_dir,
        baseline=baseline,
        excluded=args.exclude,
        limit=args.limit,
        allow_invalid_clean=args.allow_invalid_clean,
    )
    payload = {
        "schema": "orion.closedloop_heldout_route_screen.v1",
        "allowed_towns": list(ALLOWED_TOWNS),
        "excluded_route_indices": sorted(set(args.exclude)),
        "clean_baseline_required": bool(baseline) and not args.allow_invalid_clean,
        "selection_inputs": {
            "static_route_geometry_used": True,
            "scenario_family_used": True,
            "published_orion_outcomes_used": bool(baseline),
            "learned_uq_outcomes_used": False,
            "stage2_outcomes_used": False,
        },
        "locked_test_selection_eligible": not bool(baseline),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
