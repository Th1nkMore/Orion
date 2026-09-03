#!/usr/bin/env python3
"""Diagnose geometry coverage in a terminal, incomplete train-only collection."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TaskRelevanceGeometryError,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_v12_partial_coverage_collection_diagnostic.v1"
CAMERA_DIRECTORY_BY_VIEW = {
    "CAM_FRONT": "rgb_front_model_input",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected JSONL objects: %s" % path)
            rows.append(value)
    if not rows:
        raise ValueError("control trace is empty: %s" % path)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _single_scenario_root(run_dir: Path) -> Path:
    roots = sorted(
        path
        for path in (run_dir / "records_orion_traj_0").glob("RouteScenario_*")
        if path.is_dir()
    )
    if len(roots) != 1:
        raise ValueError("expected exactly one partial scenario root, found %d" % len(roots))
    return roots[0]


def _aligned_frames(scenario_root: Path) -> tuple[Path, list[int]]:
    meta_root = scenario_root / "meta"
    streams = [{int(path.stem) for path in meta_root.glob("*.json")}]
    for view in CAMERA_ORDER:
        directory = CAMERA_DIRECTORY_BY_VIEW[view]
        streams.append(
            {int(path.stem) for path in (scenario_root / directory).glob("*.png")}
        )
    aligned = sorted(set.intersection(*streams))
    if not aligned:
        raise ValueError("partial collection has no aligned six-view/meta frames")
    return meta_root, aligned


def diagnose(
    *,
    protocol_path: Path,
    batch_manifest_path: Path,
    run_dir: Path,
    slurm_job_id: int,
    slurm_state: str,
    slurm_exit_code: str,
) -> Dict[str, Any]:
    protocol = _read_json(protocol_path)
    batch = _read_json(batch_manifest_path)
    if protocol.get("schema") != "orion.stage2l_v12_train_coverage_repair_protocol.v1":
        raise ValueError("coverage repair protocol schema differs")
    if batch.get("schema") != "orion.scenario_factory.batch.v1":
        raise ValueError("batch schema differs")
    if batch.get("split") != "train_coverage_repair":
        raise ValueError("partial diagnostic accepts train coverage repair only")
    if len(batch.get("routes", [])) != 1:
        raise ValueError("partial diagnostic requires one route")
    if slurm_state == "COMPLETED" or slurm_exit_code == "0:0":
        raise ValueError("partial diagnostic requires a non-completed terminal job")

    scenario_root = _single_scenario_root(run_dir)
    meta_root, frames = _aligned_frames(scenario_root)
    trace_paths = sorted(scenario_root.glob("control_trace.jsonl"))
    if len(trace_paths) != 1:
        raise ValueError("expected one control trace")
    trace_path = trace_paths[0]
    trace_rows = _read_jsonl(trace_path)

    per_view_frames = {view: [] for view in CAMERA_ORDER}
    per_view_positive_cells = {view: 0 for view in CAMERA_ORDER}
    per_view_target_mass = {view: 0.0 for view in CAMERA_ORDER}
    modes = Counter()
    geometry_errors = Counter()
    for frame in frames:
        meta = _read_json(meta_root / ("%04d.json" % frame))
        try:
            geometry = build_task_relevance_map(
                meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
            )
        except TaskRelevanceGeometryError as error:
            geometry_errors[str(error)] += 1
            continue
        modes[geometry.provenance["support_mode"]] += 1
        for index, view in enumerate(CAMERA_ORDER):
            view_target = geometry.relevance[index]
            positive_cells = int((view_target > 0.0).sum().item())
            if positive_cells:
                per_view_frames[view].append(frame)
                per_view_positive_cells[view] += positive_cells
                per_view_target_mass[view] += float(view_target.sum().item())

    minimum = int(
        protocol["candidate_gate"][
            "minimum_positive_saved_frames_in_one_independent_event"
        ]
    )
    back_right_count = len(per_view_frames["CAM_BACK_RIGHT"])
    coverage_gate_passed = back_right_count >= minimum
    last_trace = trace_rows[-1]
    route = batch["routes"][0]
    return {
        "schema": SCHEMA,
        "status": (
            "partial_collection_back_right_geometry_present"
            if coverage_gate_passed
            else "partial_collection_back_right_geometry_absent"
        ),
        "diagnostic_completed": True,
        "runtime_complete": False,
        "candidate_accepted": False,
        "coverage_gate_passed_on_partial_frames": coverage_gate_passed,
        "formal_bank_modified": False,
        "gpu_used_by_diagnostic": False,
        "orion_forward_run_by_diagnostic": False,
        "training_started": False,
        "inputs": {
            "protocol": _reference(protocol_path),
            "batch_manifest": _reference(batch_manifest_path),
            "run_manifest": _reference(run_dir / "manifest.json"),
            "control_trace": _reference(trace_path),
        },
        "terminal_job": {
            "job_id": int(slurm_job_id),
            "state": slurm_state,
            "exit_code": slurm_exit_code,
            "run_dir": str(run_dir.resolve()),
            "scenario_root": str(scenario_root.resolve()),
        },
        "route": {
            "route_index": int(route["route_index"]),
            "town": route["town"],
            "scenario_type": route["scenario_type"],
            "formal_plan_member": False,
            "held_out_evidence_eligible": False,
        },
        "partial_runtime": {
            "control_trace_rows": len(trace_rows),
            "last_step": int(last_trace["step"]),
            "last_sim_time_seconds": float(last_trace["sim_time_seconds"]),
            "last_route_progress": float(last_trace["route_progress"]),
            "aligned_six_view_meta_frames": frames,
            "aligned_frame_count": len(frames),
            "event_package_built": False,
        },
        "geometry": {
            "grid": [40, 40],
            "minimum_back_right_positive_frames": minimum,
            "geometry_valid_frame_count": int(sum(modes.values())),
            "per_view_positive_frame_count": {
                view: len(per_view_frames[view]) for view in CAMERA_ORDER
            },
            "per_view_positive_frames": per_view_frames,
            "per_view_positive_cell_total": per_view_positive_cells,
            "per_view_target_mass": per_view_target_mass,
            "support_mode_counts": dict(sorted(modes.items())),
            "geometry_error_counts": dict(sorted(geometry_errors.items())),
        },
        "inspection_boundary": {
            "partial_train_only_frames_read": True,
            "dev_or_locked_test_artifacts_read": False,
            "observation_uq_or_qa_answers_read": False,
            "model_predictions_or_losses_used_for_selection": False,
            "collision_or_infraction_outcomes_used_for_selection": False,
        },
        "disposition": (
            "do not accept a partial runtime; inspect geometry and human visuals before any recollection decision"
            if coverage_gate_passed
            else "retire Route167 as the CAM_BACK_RIGHT coverage fallback; do not spend another GPU retry on this candidate"
        ),
        "launch_locks": {
            "additional_route167_retry": False,
            "gpu_r_only_smoke": False,
            "language_bridge": False,
            "stage2p": False,
        },
        "claim_boundary": "CPU geometry diagnosis of aligned frames salvaged from an incomplete train-only collection. It cannot be accepted as a complete event package or used as held-out, model, U, language, planning, closed-loop, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--slurm-state", required=True)
    parser.add_argument("--slurm-exit-code", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite partial collection diagnostic")
    value = diagnose(
        protocol_path=args.protocol,
        batch_manifest_path=args.batch_manifest,
        run_dir=args.run_dir,
        slurm_job_id=args.slurm_job_id,
        slurm_state=args.slurm_state,
        slurm_exit_code=args.slurm_exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "coverage_gate_passed_on_partial_frames": value[
                    "coverage_gate_passed_on_partial_frames"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
