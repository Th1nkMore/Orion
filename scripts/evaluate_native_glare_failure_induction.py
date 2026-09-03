#!/usr/bin/env python3
"""Fail-closed evaluation for one CARLA-native glare closed-loop pair.

Unlike the synthetic-corruption gate, this evaluator does not look for a
``corruption_active`` flag.  Native glare is a renderer condition that is
active for the whole route.  The surrogate comparison therefore uses a
pre-registered route-progress window, while hard endpoint degradation remains
valid even when the degraded run never reaches that window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from scripts.evaluate_failure_induction_gate import (
    hard_endpoint_comparison,
    surrogate_comparison,
)
from scripts.summarize_closedloop_safety import (
    compare_summaries,
    find_control_trace,
    load_records,
    summarize_records,
)


SCHEMA = "orion.native_glare_failure_induction_decision.v1"
SAFETY_SCHEMA = "orion.closedloop_dynamic_actor_safety.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_one(root: Path, pattern: str) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one {pattern!r} below {root}, found {len(paths)}"
        )
    return paths[0], _load_json(paths[0])


def _close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return str(left).strip().lower() == str(right).strip().lower()


def _camera_contract(
    readback: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    cameras = readback.get("cameras") or {}
    details: dict[str, Any] = {}
    passed = True
    for sensor_id in ("CAM_FRONT", "bev"):
        actual = (cameras.get(sensor_id) or {}).get("attributes") or {}
        sensor_expected = expected.get(sensor_id) or {}
        checks = {
            key: {
                "expected": value,
                "actual": actual.get(key),
                "passed": _close(actual.get(key), value),
            }
            for key, value in sensor_expected.items()
        }
        sensor_passed = bool(actual) and all(
            check["passed"] for check in checks.values()
        )
        details[sensor_id] = {
            "present": bool(actual),
            "checks": checks,
            "passed": sensor_passed,
        }
        passed = passed and sensor_passed
    return {"passed": passed, "sensors": details}


def _render_evidence(
    run: Path,
    *,
    expected_profile: str,
    expected_camera: dict[str, Any],
    expected_weather: dict[str, Any],
) -> dict[str, Any]:
    manifest_path, manifest = _load_one(run, "manifest.json")
    readback_path, readback = _load_one(run, "render_condition_readback.json")
    render = manifest.get("render_condition") or {}
    readback_link = render.get("actual_readback") or {}
    camera = _camera_contract(readback, expected_camera)
    weather_actual = readback.get("weather") or {}
    weather_checks = {
        key: {
            "expected": value,
            "actual": weather_actual.get(key),
            "passed": _close(weather_actual.get(key), value, tolerance=1e-4),
        }
        for key, value in expected_weather.items()
    }
    checks = {
        "manifest_profile": render.get("native_glare_profile")
        == expected_profile,
        "manifest_kind": render.get("kind")
        == "carla_native_low_sun_glare",
        "manifest_readback_verified": readback_link.get("status") == "verified",
        "manifest_readback_schema": readback_link.get("schema")
        == "orion.closedloop_render_condition_readback.v1",
        "readback_schema": readback.get("schema")
        == "orion.closedloop_render_condition_readback.v1",
        "readback_status": readback.get("status") == "verified",
        "readback_profile": readback.get("native_glare_profile")
        == expected_profile,
        "camera_contract": camera["passed"],
        "weather_contract": all(item["passed"] for item in weather_checks.values()),
    }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "readback_path": str(readback_path.resolve()),
        "readback_sha256": _sha256(readback_path),
        "manifest": manifest,
        "readback": readback,
        "camera": camera,
        "weather": weather_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _nearest_record(records: list[dict[str, Any]], step: int) -> dict[str, Any]:
    return min(records, key=lambda row: abs(int(row["step"]) - step))


def _model_tensor_dir(run: Path) -> Path:
    trace = find_control_trace(run)
    path = trace.parent / "rgb_front_model_tensor"
    if not path.is_dir():
        raise RuntimeError(f"missing exact ORION front tensor directory: {path}")
    return path


def compare_exact_model_tensors(
    clean_run: Path,
    degraded_run: Path,
    *,
    progress_window: tuple[float, float],
    minimum_pairs: int,
    minimum_median_mad_8bit: float,
) -> dict[str, Any]:
    clean_trace = find_control_trace(clean_run)
    degraded_trace = find_control_trace(degraded_run)
    clean_records = load_records(clean_trace)
    degraded_records = load_records(degraded_trace)
    clean_dir = _model_tensor_dir(clean_run)
    degraded_dir = _model_tensor_dir(degraded_run)
    clean_paths = {path.stem: path for path in clean_dir.glob("*.png")}
    degraded_paths = {path.stem: path for path in degraded_dir.glob("*.png")}
    start, end = progress_window
    candidates = []
    for stem in sorted(set(clean_paths) & set(degraded_paths), key=int):
        step = int(stem) * 10
        clean_row = _nearest_record(clean_records, step)
        degraded_row = _nearest_record(degraded_records, step)
        clean_progress = float(clean_row["route_progress"])
        degraded_progress = float(degraded_row["route_progress"])
        if not (
            start <= clean_progress <= end
            and start <= degraded_progress <= end
        ):
            continue
        clean_image = Image.open(clean_paths[stem]).convert("RGB")
        degraded_image = Image.open(degraded_paths[stem]).convert("RGB")
        if clean_image.size != (640, 640) or degraded_image.size != (640, 640):
            raise RuntimeError("exact ORION tensor image is not 640x640")
        difference = ImageChops.difference(clean_image, degraded_image)
        channel_means = ImageStat.Stat(difference).mean
        candidates.append({
            "saved_frame": int(stem),
            "control_step": step,
            "clean_route_progress": clean_progress,
            "degraded_route_progress": degraded_progress,
            "clean_sha256": _sha256(clean_paths[stem]),
            "degraded_sha256": _sha256(degraded_paths[stem]),
            "mean_absolute_delta_8bit": float(sum(channel_means) / 3.0),
        })
    deltas = [row["mean_absolute_delta_8bit"] for row in candidates]
    median_delta = float(statistics.median(deltas)) if deltas else None
    changed_pairs = sum(
        row["clean_sha256"] != row["degraded_sha256"] for row in candidates
    )
    checks = {
        "minimum_event_pairs": len(candidates) >= int(minimum_pairs),
        "all_event_pairs_hash_differ": bool(candidates)
        and changed_pairs == len(candidates),
        "median_mad_large_enough": median_delta is not None
        and median_delta >= float(minimum_median_mad_8bit),
    }
    return {
        "clean_tensor_dir": str(clean_dir.resolve()),
        "degraded_tensor_dir": str(degraded_dir.resolve()),
        "progress_window": [start, end],
        "pairs": candidates,
        "pair_count": len(candidates),
        "changed_pair_count": changed_pairs,
        "median_mean_absolute_delta_8bit": median_delta,
        "thresholds": {
            "minimum_pairs": int(minimum_pairs),
            "minimum_median_mad_8bit": float(minimum_median_mad_8bit),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _select_progress_interval(
    records: list[dict[str, Any]],
    start_progress: float,
    end_progress: float,
    recovery_seconds: float,
) -> list[dict[str, Any]]:
    start_index = next(
        (index for index, row in enumerate(records)
         if float(row["route_progress"]) >= start_progress),
        None,
    )
    if start_index is None:
        raise RuntimeError("run never reached the native-glare event start")
    end_index = next(
        (index for index in range(start_index, len(records))
         if float(records[index]["route_progress"]) >= end_progress),
        None,
    )
    if end_index is None:
        raise RuntimeError("run never reached the native-glare event end")
    end_time = float(records[end_index]["sim_time_seconds"]) + recovery_seconds
    selected = [
        row for row in records[start_index:]
        if float(row["sim_time_seconds"]) <= end_time
    ]
    if not selected:
        raise RuntimeError("native-glare event interval is empty")
    return selected


def build_progress_event_report(
    clean_records: list[dict[str, Any]],
    degraded_records: list[dict[str, Any]],
    *,
    start_progress: float,
    end_progress: float,
    recovery_seconds: float,
) -> dict[str, Any]:
    clean_interval = _select_progress_interval(
        clean_records, start_progress, end_progress, recovery_seconds
    )
    degraded_interval = _select_progress_interval(
        degraded_records, start_progress, end_progress, recovery_seconds
    )
    clean_summary = summarize_records(clean_interval)
    degraded_summary = summarize_records(degraded_interval)
    comparison = compare_summaries(
        {"summary": clean_summary}, {"summary": degraded_summary}
    )
    return {
        "alignment": "each_run_pre_registered_route_progress_window",
        "event_progress_window": [start_progress, end_progress],
        "recovery_seconds_requested": recovery_seconds,
        "event_plus_recovery": {
            "clean": clean_summary,
            "degraded": degraded_summary,
            "comparison": comparison,
        },
    }


def _safety_valid(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(
        row.get("closedloop_safety", {}).get("available") is True
        and row.get("closedloop_safety", {}).get("schema") == SAFETY_SCHEMA
        for row in records
    )


def evaluate(
    *,
    protocol_path: Path,
    render_protocol_path: Path,
    gate_path: Path,
    clean_run: Path,
    degraded_run: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    render_protocol = _load_json(render_protocol_path)
    gate = _load_json(gate_path)
    pair = protocol["first_pair"]
    route_index = str(pair["route"])
    event = pair["event_window"]
    tensor_gate = pair["exact_model_tensor_gate"]
    camera_profiles = render_protocol["methods"]["carla_native_low_sun"][
        "camera_profiles"
    ]
    expected_weather = render_protocol["methods"]["carla_native_low_sun"][
        "weather_shared_by_all_profiles"
    ]
    expected_camera = {
        profile: {
            "CAM_FRONT": {
                "enable_postprocess_effects": "true",
                "exposure_mode": "histogram",
                **camera_profiles[profile],
            },
            "bev": {
                "enable_postprocess_effects": "false",
                "lens_flare_intensity": 0.0,
                "bloom_intensity": 0.0,
            },
        }
        for profile in ("clean", "medium")
    }
    clean_render = _render_evidence(
        clean_run,
        expected_profile="clean",
        expected_camera=expected_camera["clean"],
        expected_weather=expected_weather,
    )
    degraded_render = _render_evidence(
        degraded_run,
        expected_profile="medium",
        expected_camera=expected_camera["medium"],
        expected_weather=expected_weather,
    )
    clean_manifest = clean_render["manifest"]
    degraded_manifest = degraded_render["manifest"]
    clean_eval_path, clean_eval = _load_one(clean_run, "eval*.json")
    degraded_eval_path, degraded_eval = _load_one(degraded_run, "eval*.json")
    clean_gate_path, clean_gate = _load_one(
        clean_run, "clean_safety_gate.json"
    )
    clean_trace = find_control_trace(clean_run)
    degraded_trace = find_control_trace(degraded_run)
    clean_records = load_records(clean_trace)
    degraded_records = load_records(degraded_trace)
    tensor = compare_exact_model_tensors(
        clean_run,
        degraded_run,
        progress_window=(
            float(event["start_route_progress"]),
            float(event["end_route_progress"]),
        ),
        minimum_pairs=int(tensor_gate["minimum_event_pairs"]),
        minimum_median_mad_8bit=float(
            tensor_gate["minimum_median_mean_absolute_delta_8bit"]
        ),
    )
    run_contract = {
        "route_indices_match": clean_manifest.get("pilot_route_index")
        == route_index
        and degraded_manifest.get("pilot_route_index") == route_index,
        "route_hashes_match": bool(clean_manifest.get("route_sha256"))
        and clean_manifest.get("route_sha256")
        == degraded_manifest.get("route_sha256"),
        "epic_quality": clean_manifest.get("carla_quality_level") == "Epic"
        and degraded_manifest.get("carla_quality_level") == "Epic",
        "original_orion_checkpoint": clean_manifest.get("base_checkpoint_path")
        == degraded_manifest.get("base_checkpoint_path")
        and str(clean_manifest.get("base_checkpoint_path") or "").endswith(
            "/checkpoints/Orion.pth"
        ),
        "risk_off": clean_manifest.get("orion_closedloop_risk_mode") == "off"
        and degraded_manifest.get("orion_closedloop_risk_mode") == "off",
        "uq_none": clean_manifest.get("orion_closedloop_uq_mode") == "none"
        and degraded_manifest.get("orion_closedloop_uq_mode") == "none",
        "planning_response_off": clean_manifest.get(
            "orion_planning_response_mode"
        ) == "off"
        and degraded_manifest.get("orion_planning_response_mode") == "off",
        "no_synthetic_corruption": not clean_manifest.get(
            "orion_closedloop_corruption"
        )
        and not degraded_manifest.get("orion_closedloop_corruption"),
        "effective_conditioning_none": clean_manifest.get(
            "orion_effective_conditioning"
        ) == "none"
        and degraded_manifest.get("orion_effective_conditioning") == "none",
    }
    validity_checks = {
        "clean_safety_gate": clean_gate.get("gate_passed") is True,
        **run_contract,
        "clean_render_readback": clean_render["passed"],
        "degraded_render_readback": degraded_render["passed"],
        "exact_model_tensor_changed": tensor["passed"],
        "clean_safety_telemetry": _safety_valid(clean_records),
        "degraded_safety_telemetry": _safety_valid(degraded_records),
    }
    valid = all(validity_checks.values())
    hard = hard_endpoint_comparison(clean_eval, degraded_eval)
    paired_event = None
    paired_error = None
    surrogate = None
    try:
        paired_event = build_progress_event_report(
            clean_records,
            degraded_records,
            start_progress=float(event["start_route_progress"]),
            end_progress=float(event["end_route_progress"]),
            recovery_seconds=float(event["recovery_seconds"]),
        )
        actor_category = gate["route_actor_category"][route_index]
        surrogate = surrogate_comparison(
            paired_event,
            actor_category=actor_category,
            thresholds=gate["surrogate_safety_margin_degradation"][
                "thresholds"
            ],
        )
    except (KeyError, RuntimeError, ValueError) as error:
        paired_error = str(error)
    hard_pass = bool(hard["degraded"])
    surrogate_pass = surrogate is not None and bool(surrogate["degraded"])
    failure_pass = valid and (hard_pass or surrogate_pass)
    if not valid:
        tier = "invalid"
    elif hard_pass:
        tier = "hard_failure_induction"
    elif surrogate_pass:
        tier = "near_miss_surrogate_failure_induction"
    else:
        tier = "failure_induction_not_demonstrated"
    return {
        "schema": SCHEMA,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path),
        },
        "render_protocol": {
            "path": str(render_protocol_path.resolve()),
            "sha256": _sha256(render_protocol_path),
        },
        "safety_gate": {
            "path": str(gate_path.resolve()),
            "sha256": _sha256(gate_path),
        },
        "source_files": {
            "clean_eval": str(clean_eval_path.resolve()),
            "degraded_eval": str(degraded_eval_path.resolve()),
            "clean_gate": str(clean_gate_path.resolve()),
            "clean_trace": str(clean_trace.resolve()),
            "degraded_trace": str(degraded_trace.resolve()),
        },
        "render_evidence": {
            "clean": {key: value for key, value in clean_render.items()
                      if key not in {"manifest", "readback"}},
            "degraded": {key: value for key, value in degraded_render.items()
                         if key not in {"manifest", "readback"}},
        },
        "exact_model_tensor": tensor,
        "validity": {"valid": valid, "checks": validity_checks},
        "hard_endpoint": hard,
        "surrogate_safety_margin": surrogate,
        "paired_event": paired_event,
        "paired_event_error": paired_error,
        "decision": {
            "failure_induction_pass": failure_pass,
            "evidence_tier": tier,
            "counts_toward_final_hard_case_target": valid and hard_pass,
            "next_action": (
                "expand_at_most_three_to_four_outcome_blind_routes"
                if failure_pass
                else "stop_or_repair_according_to_validity"
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--render-protocol", type=Path, required=True)
    parser.add_argument("--safety-gate", type=Path, required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--degraded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite native-glare decision")
    report = evaluate(
        protocol_path=args.protocol.resolve(),
        render_protocol_path=args.render_protocol.resolve(),
        gate_path=args.safety_gate.resolve(),
        clean_run=args.clean_run.resolve(),
        degraded_run=args.degraded_run.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "valid": report["validity"]["valid"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))
    return 0 if report["validity"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
