"""CPU-only event packages for automatic closed-loop scenario screening."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    # Package import used by tests and reusable callers.
    from scripts.summarize_closedloop_safety import summarize_records
except ModuleNotFoundError:
    # Direct ``python scripts/build_scenario_event_package.py`` execution puts
    # the scripts directory, not the project root, on sys.path.
    from summarize_closedloop_safety import summarize_records


EVENT_PACKAGE_SCHEMA = "orion.scenario_event_package.v1"
RENDER_CONDITION_SCHEMA = "orion.closedloop_render_condition.v1"
# Historical clean-off traces created before the ORION closed-loop agent had
# any native-render override interface remain admissible only when their
# recorded agent hash is in this explicit allow-list.  Once that agent changes,
# a missing render_condition fails closed instead of silently becoming clean.
PRE_NATIVE_GLARE_AGENT_SHA256 = frozenset({
    "185ba32012718d13322efdf19f1b23d71ec64001bcd1ee57625b1904d0eeb8a6",
})
CAMERA_DIRECTORIES = (
    "rgb_front",
    "rgb_front_model_input",
    "rgb_front_model_tensor",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
    "bev",
)
COLLISION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
)
SERIOUS_INFRACTION_KEYS = (
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "scenario_timeouts",
    "route_timeout",
)
ALLOWED_SPLITS = (
    "development_screen",
    "qa_train_candidate",
    "qa_dev_candidate",
    "locked_test",
)
CRITICAL_EVENT_MAX_ROUTE_PROGRESS = 0.98
CRITICAL_EVENT_PRE_SECONDS = 3.0
CRITICAL_EVENT_POST_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one %r below %s, found %d"
            % (pattern, root, len(matches))
        )
    return matches[0]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("empty JSONL file: %s" % path)
    return rows


def _count_entries(infractions: Mapping[str, Any], keys: Sequence[str]) -> int:
    return sum(
        len(infractions.get(key, []))
        for key in keys
        if isinstance(infractions.get(key, []), list)
    )


def _terminal_record(eval_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    records = eval_payload.get("_checkpoint", {}).get("records", [])
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError("expected exactly one terminal evaluator record")
    if not isinstance(records[0], dict):
        raise RuntimeError("terminal evaluator record is not an object")
    return records[0]


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def select_critical_event(
    records: Sequence[Mapping[str, Any]],
    preferred_actor_categories: Sequence[str] = (),
    require_complete_review_window: bool = False,
) -> Optional[Dict[str, Any]]:
    """Select one actor-grounded review event without learned-UQ outcomes.

    Actor-category preference may only choose the human-review window for an
    explicitly named scenario.  If that category has no candidate, selection
    falls back to all actors.  This preference is never a model input or a
    Stage-1/Stage2-L supervision source.
    """

    first_time = float(records[0]["sim_time_seconds"])
    last_time = float(records[-1]["sim_time_seconds"])
    ttc_candidates: List[Tuple[float, int, Mapping[str, Any], Mapping[str, Any]]] = []
    gap_candidates: List[Tuple[float, int, Mapping[str, Any], Mapping[str, Any]]] = []
    for row in records:
        # Bench2Drive actors remain live while route completion is being
        # registered.  Relative-motion TTCs in this terminal tail can therefore
        # select a stopped-at-finish artifact rather than a driving event.
        # Keep the boundary fixed and independent of model outcome.
        if float(row.get("route_progress", 0.0)) >= CRITICAL_EVENT_MAX_ROUTE_PROGRESS:
            continue
        center = float(row["sim_time_seconds"])
        if require_complete_review_window and (
            center - first_time < CRITICAL_EVENT_PRE_SECONDS
            or last_time - center < CRITICAL_EVENT_POST_SECONDS
        ):
            continue
        safety = row.get("closedloop_safety") or {}
        for actor in safety.get("actors", []):
            if not isinstance(actor, dict):
                continue
            ttc = _finite(actor.get("obb_collision_ttc_seconds"))
            gap = _finite(actor.get("obb_separating_axis_gap_m"))
            step = int(row["step"])
            if ttc is not None:
                ttc_candidates.append((ttc, step, row, actor))
            if gap is not None:
                gap_candidates.append((gap, step, row, actor))

    preferred = {str(value) for value in preferred_actor_categories}
    preferred_ttc = [
        item for item in ttc_candidates
        if str(item[3].get("category")) in preferred
    ]
    preferred_gap = [
        item for item in gap_candidates
        if str(item[3].get("category")) in preferred
    ]
    basis = None
    selected = None
    if preferred_ttc:
        selected = min(preferred_ttc, key=lambda item: (item[0], item[1]))
        basis = "minimum_finite_preferred_actor_obb_ttc"
    elif preferred_gap:
        selected = min(preferred_gap, key=lambda item: (item[0], item[1]))
        basis = "minimum_finite_preferred_actor_obb_gap"
    elif ttc_candidates:
        selected = min(ttc_candidates, key=lambda item: (item[0], item[1]))
        basis = "minimum_finite_dynamic_actor_obb_ttc"
    elif gap_candidates:
        selected = min(gap_candidates, key=lambda item: (item[0], item[1]))
        basis = "minimum_finite_dynamic_actor_obb_gap"
    if selected is None:
        return None

    _, _, row, actor = selected
    center = float(row["sim_time_seconds"])
    return {
        "selection_basis": basis,
        "selection_policy": {
            "maximum_route_progress_exclusive": CRITICAL_EVENT_MAX_ROUTE_PROGRESS,
            "terminal_route_tail_excluded": True,
            "complete_review_window_required": require_complete_review_window,
            "minimum_pre_event_seconds": (
                CRITICAL_EVENT_PRE_SECONDS if require_complete_review_window else 0.0
            ),
            "minimum_post_event_seconds": (
                CRITICAL_EVENT_POST_SECONDS if require_complete_review_window else 0.0
            ),
        },
        "preferred_actor_categories": sorted(preferred),
        "step": int(row["step"]),
        "sim_time_seconds": center,
        "route_progress": float(row["route_progress"]),
        "speed_mps": float(row["speed"]),
        "window_seconds": [max(first_time, center - 3.0), min(last_time, center + 2.0)],
        "actor": {
            "actor_id": actor.get("actor_id"),
            "category": actor.get("category"),
            "type_id": actor.get("type_id"),
            "obb_ttc_seconds": _finite(actor.get("obb_collision_ttc_seconds")),
            "obb_separating_axis_gap_m": _finite(
                actor.get("obb_separating_axis_gap_m")
            ),
        },
    }


def validate_trace(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = [int(row["step"]) for row in records]
    times = [float(row["sim_time_seconds"]) for row in records]
    safety_rows = [row.get("closedloop_safety") for row in records]
    contiguous = steps == list(range(steps[0], steps[0] + len(steps)))
    monotonic = all(right > left for left, right in zip(times, times[1:]))
    coverage = sum(
        isinstance(row, dict) and row.get("available") is True
        for row in safety_rows
    ) / len(safety_rows)
    return {
        "steps_contiguous": contiguous,
        "times_strictly_monotonic": monotonic,
        "safety_telemetry_coverage": coverage,
        "valid": contiguous and monotonic and coverage == 1.0,
    }


def validate_clean_runtime_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    render_condition = manifest.get("render_condition")
    render_attestation = "explicit"
    if isinstance(render_condition, Mapping):
        render_condition_clean = (
            render_condition.get("schema") == RENDER_CONDITION_SCHEMA
            and render_condition.get("kind") == "standard_carla_rgb"
            and render_condition.get("native_glare_profile") == "none"
            and render_condition.get("camera_postprocess_override") is False
        )
    else:
        agent_hash = manifest.get("source_sha256", {}).get(
            "team_code/orion_b2d_agent.py"
        )
        render_condition_clean = agent_hash in PRE_NATIVE_GLARE_AGENT_SHA256
        render_attestation = "legacy_pre_native_glare_agent_hash_allowlist"
    checks = {
        "condition_clean_off": manifest.get("pilot_condition") == "clean_off",
        "uq_mode_none": manifest.get("orion_closedloop_uq_mode") == "none",
        "risk_mode_off": manifest.get("orion_closedloop_risk_mode") == "off",
        "planning_response_off": manifest.get("orion_planning_response_mode") in (
            None,
            "",
            "off",
        ),
        "legacy_density_disabled": str(
            manifest.get("orion_enable_legacy_density_uq", "0")
        ).lower()
        in ("0", "false", "no", "off"),
        "corruption_absent": manifest.get("orion_closedloop_corruption") in (
            None,
            "",
        ),
        "render_condition_clean": render_condition_clean,
    }
    return {
        "checks": checks,
        "render_condition_attestation": render_attestation,
        "valid": all(checks.values()),
    }


def classify_outcome(
    terminal: Mapping[str, Any],
    safety_summary: Mapping[str, Any],
    *,
    runtime_valid: bool,
    critical_event: Optional[Mapping[str, Any]],
) -> str:
    if not runtime_valid:
        return "INVALID_RUNTIME"
    infractions = terminal.get("infractions", {})
    if not isinstance(infractions, dict):
        infractions = {}
    if _count_entries(infractions, COLLISION_KEYS):
        return "VALID_COLLISION"
    if _count_entries(infractions, SERIOUS_INFRACTION_KEYS):
        return "VALID_SERIOUS_INFRACTION"
    scores = terminal.get("scores", {})
    complete = (
        terminal.get("status") == "Completed"
        and float(scores.get("score_route", -1.0)) == 100.0
    )
    if not complete:
        return "VALID_MODEL_INCOMPLETE"
    safety = safety_summary["safety"]
    min_ttc = _finite(safety.get("min_obb_ttc_seconds"))
    min_gap = _finite(safety.get("min_obb_separating_axis_gap_m"))
    exposure = float(safety.get("low_ttc_exposure_seconds", {}).get("2.0", 0.0))
    if (
        min_ttc is not None
        and min_ttc <= 1.0
        and exposure >= 0.5
        and min_gap is not None
        and min_gap <= 0.5
    ):
        return "VALID_SEVERE_TTC"
    if critical_event is not None:
        return "VALID_NEAR_MISS_OR_CONFLICT"
    return "VALID_SAFE_NO_ACTOR_GROUNDED_EVENT"


def _camera_inventory(scenario_dir: Path) -> Dict[str, Any]:
    inventory: Dict[str, Any] = {}
    for directory in CAMERA_DIRECTORIES:
        root = scenario_dir / directory
        frames = sorted(root.glob("*.png")) if root.is_dir() else []
        inventory[directory] = {
            "path": str(root.resolve()),
            "frame_count": len(frames),
            "first_frame": frames[0].name if frames else None,
            "last_frame": frames[-1].name if frames else None,
        }
    return inventory


def build_event_package(
    run_dir: Path,
    *,
    split: str,
    batch_manifest_path: Optional[Path] = None,
    visualization_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("unsupported scenario-factory split: %s" % split)
    manifest_path = run_dir / "manifest.json"
    eval_path = find_one(run_dir, "eval_*.json")
    trace_path = find_one(run_dir, "records_*/**/control_trace.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    terminal = _terminal_record(eval_payload)
    records = load_jsonl(trace_path)
    trace_validation = validate_trace(records)
    manifest_validation = validate_clean_runtime_manifest(manifest)
    runtime_valid = (
        trace_validation["valid"]
        and manifest_validation["valid"]
        and bool(eval_payload.get("eligible"))
        and eval_payload.get("entry_status") == "Finished"
    )
    safety_summary = summarize_records(records)
    preferred_actor_categories: Tuple[str, ...] = ()
    if batch_manifest_path is not None:
        batch = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        route_index = int(manifest["pilot_route_index"])
        matches = [
            row for row in batch.get("routes", [])
            if int(row.get("route_index", -1)) == route_index
        ]
        if len(matches) != 1:
            raise ValueError("batch manifest does not uniquely identify route")
        scenario_type = str(matches[0].get("scenario_type", ""))
        if "pedestrian" in scenario_type.lower():
            preferred_actor_categories = ("walker",)
    critical_event = select_critical_event(
        records,
        preferred_actor_categories=preferred_actor_categories,
        require_complete_review_window=True,
    )
    outcome_class = classify_outcome(
        terminal,
        safety_summary,
        runtime_valid=runtime_valid,
        critical_event=critical_event,
    )
    scenario_dir = trace_path.parent
    camera_inventory = _camera_inventory(scenario_dir)
    all_required_streams = all(
        item["frame_count"] > 0 for item in camera_inventory.values()
    )
    route_path_value = manifest.get("pilot_route_file")
    route_path = Path(route_path_value) if route_path_value else None
    source_files: Dict[str, Any] = {
        "run_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "evaluator": {
            "path": str(eval_path.resolve()),
            "sha256": sha256_file(eval_path),
        },
        "control_trace": {
            "path": str(trace_path.resolve()),
            "sha256": sha256_file(trace_path),
        },
    }
    if route_path is not None and route_path.is_file():
        source_files["route_xml"] = {
            "path": str(route_path.resolve()),
            "sha256": sha256_file(route_path),
        }
    if batch_manifest_path is not None:
        source_files["batch_manifest"] = {
            "path": str(batch_manifest_path.resolve()),
            "sha256": sha256_file(batch_manifest_path),
        }
    visualization = None
    if visualization_manifest_path is not None:
        visualization = {
            "path": str(visualization_manifest_path.resolve()),
            "sha256": sha256_file(visualization_manifest_path),
        }

    return {
        "schema": EVENT_PACKAGE_SCHEMA,
        "status": "pending_human_review",
        "split": split,
        "route": {
            "route_index": int(manifest["pilot_route_index"]),
            "variant": manifest.get("pilot_variant"),
            "run_id": manifest.get("pilot_run_id"),
            "slurm_job_id": manifest.get("slurm_job_id"),
        },
        "runtime": {
            "valid": runtime_valid,
            "manifest": manifest_validation,
            "trace": trace_validation,
            "evaluator_eligible": bool(eval_payload.get("eligible")),
            "evaluator_entry_status": eval_payload.get("entry_status"),
        },
        "outcome_class": outcome_class,
        "official_endpoint": {
            "status": terminal.get("status"),
            "scores": terminal.get("scores", {}),
            "collision_count": _count_entries(
                terminal.get("infractions", {}), COLLISION_KEYS
            ),
            "serious_infraction_count": _count_entries(
                terminal.get("infractions", {}), SERIOUS_INFRACTION_KEYS
            ),
        },
        "continuous_safety": safety_summary,
        "critical_event": critical_event,
        "camera_inventory": camera_inventory,
        "qa_input_ready": runtime_valid and all_required_streams,
        "stage1_observation_uq": {
            "status": "pending_offline_precomputation",
            "control_influence": False,
        },
        "task_relevance_target": {
            "status": "pending_geometry_and_counterfactual_construction",
            "must_not_use_corruption_family_as_ground_truth": True,
        },
        "visualization": visualization,
        "source_files": source_files,
        "claim_boundary": (
            "Development scenario-screening artifact only. A useful event does not "
            "validate Stage-1 UQ, VLM task relevance or closed-loop benefit."
        ),
    }


__all__ = [
    "ALLOWED_SPLITS",
    "CAMERA_DIRECTORIES",
    "CRITICAL_EVENT_MAX_ROUTE_PROGRESS",
    "CRITICAL_EVENT_POST_SECONDS",
    "CRITICAL_EVENT_PRE_SECONDS",
    "EVENT_PACKAGE_SCHEMA",
    "PRE_NATIVE_GLARE_AGENT_SHA256",
    "RENDER_CONDITION_SCHEMA",
    "build_event_package",
    "classify_outcome",
    "select_critical_event",
    "sha256_file",
    "validate_clean_runtime_manifest",
    "validate_trace",
]
