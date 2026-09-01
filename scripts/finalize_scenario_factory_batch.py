#!/usr/bin/env python3
"""Finalize one scenario-screen batch into reviewable event packages and GIFs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scripts.scenario_factory_lib import build_event_package, sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import build_event_package, sha256_file


SCHEMA = "orion.scenario_factory.batch_screen_report.v1"


def _job_id(run_dir: Path) -> int:
    try:
        return int(run_dir.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _candidate_package(
    run_dir: Path, batch_manifest: Path, split: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return (
            build_event_package(
                run_dir,
                split=split,
                batch_manifest_path=batch_manifest,
            ),
            None,
        )
    except Exception as error:
        return None, "%s: %s" % (type(error).__name__, error)


def select_run(
    results_root: Path, route_index: int, batch_manifest: Path, split: str
) -> Tuple[Optional[Path], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    attempts = []
    usable = []
    for run_dir in sorted(
        results_root.glob("route%d_hazard_clean_off-*" % route_index),
        key=_job_id,
    ):
        package, error = _candidate_package(run_dir, batch_manifest, split)
        attempts.append(
            {
                "run_dir": str(run_dir.resolve()),
                "job_id": _job_id(run_dir),
                "package_buildable": package is not None,
                "runtime_valid": (
                    package["runtime"]["valid"] if package is not None else False
                ),
                "error": error,
            }
        )
        if package is not None:
            usable.append((bool(package["runtime"]["valid"]), _job_id(run_dir), run_dir, package))
    if not usable:
        return None, None, attempts
    _, _, selected_dir, selected_package = max(
        usable, key=lambda item: (item[0], item[1])
    )
    return selected_dir, selected_package, attempts


def _render_visuals(
    project_root: Path,
    run_dir: Path,
    output_dir: Path,
    package: Dict[str, Any],
) -> Optional[Path]:
    event = package.get("critical_event")
    if event is None:
        return None
    command = [
        sys.executable,
        str(project_root / "scripts" / "render_closedloop_front_bev_gifs.py"),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--center-time-seconds",
        str(float(event["sim_time_seconds"])),
        "--pre-seconds",
        "3",
        "--post-seconds",
        "2",
        "--fps",
        "2",
    ]
    subprocess.run(command, check=True)
    manifest = output_dir / "visualization_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("visual renderer did not create its manifest")
    return manifest


def finalize_batch(
    *,
    project_root: Path,
    batch_manifest_path: Path,
    results_root: Path,
    output_root: Path,
    render_visuals: bool,
) -> Dict[str, Any]:
    batch = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    if batch.get("schema") != "orion.scenario_factory.batch.v1":
        raise ValueError("invalid scenario-factory batch schema")
    split = batch.get("split")
    if split not in ("development_screen", "locked_test"):
        raise ValueError("unsupported scenario-factory batch split")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty finalization root")

    selected_rows = []
    prepared = []
    for route in batch["routes"]:
        route_index = int(route["route_index"])
        run_dir, initial_package, attempts = select_run(
            results_root, route_index, batch_manifest_path, split
        )
        if run_dir is None or initial_package is None:
            selected_rows.append(
                {
                    "route_index": route_index,
                    "town": route["town"],
                    "scenario_type": route["scenario_type"],
                    "screen_role": route["screen_role"],
                    "selected_run_dir": None,
                    "selected_job_id": None,
                    "attempts": attempts,
                    "runtime_valid": False,
                    "outcome_class": "INVALID_RUNTIME_UNPACKAGEABLE",
                    "qa_input_ready": False,
                    "has_actor_grounded_event": False,
                }
            )
            continue
        route_root = output_root / ("route%d" % route_index)
        visualization_manifest = None
        if render_visuals and initial_package.get("critical_event") is not None:
            visualization_manifest = _render_visuals(
                project_root,
                run_dir,
                route_root / "visuals",
                initial_package,
            )
        package = build_event_package(
            run_dir,
            split=split,
            batch_manifest_path=batch_manifest_path,
            visualization_manifest_path=visualization_manifest,
        )
        prepared.append((route_root, package))
        selected_rows.append(
            {
                "route_index": route_index,
                "town": route["town"],
                "scenario_type": route["scenario_type"],
                "screen_role": route["screen_role"],
                "selected_run_dir": str(run_dir.resolve()),
                "selected_job_id": _job_id(run_dir),
                "attempts": attempts,
                "runtime_valid": package["runtime"]["valid"],
                "outcome_class": package["outcome_class"],
                "qa_input_ready": package["qa_input_ready"],
                "has_actor_grounded_event": package["critical_event"] is not None,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    package_index = []
    for route_root, package in prepared:
        route_root.mkdir(parents=True, exist_ok=True)
        package_path = route_root / "event_package.json"
        package_path.write_text(
            json.dumps(package, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        package_index.append(
            {
                "route_index": package["route"]["route_index"],
                "path": str(package_path.resolve()),
                "sha256": sha256_file(package_path),
            }
        )

    outcome_counts = Counter(row["outcome_class"] for row in selected_rows)
    report = {
        "schema": SCHEMA,
        "status": "pending_human_review",
        "run_id": batch["run_id"],
        "split": split,
        "batch_manifest": {
            "path": str(batch_manifest_path.resolve()),
            "sha256": sha256_file(batch_manifest_path),
        },
        "results_root": str(results_root.resolve()),
        "output_root": str(output_root.resolve()),
        "route_count": len(selected_rows),
        "runtime_valid_count": sum(row["runtime_valid"] for row in selected_rows),
        "qa_input_ready_count": sum(row["qa_input_ready"] for row in selected_rows),
        "actor_grounded_event_count": sum(
            row["has_actor_grounded_event"] for row in selected_rows
        ),
        "unpackageable_route_count": sum(
            row["selected_run_dir"] is None for row in selected_rows
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "routes": selected_rows,
        "event_packages": package_index,
        "claim_boundary": (
            (
                "Automated locked-test runtime report awaiting integrity review. Routes "
                "must not be accepted or replaced based on model outcome."
            )
            if split == "locked_test"
            else (
                "Automated development-screen report awaiting human scene review; "
                "not a training-set acceptance decision or held-out result."
            )
        ),
    }
    report_path = output_root / "batch_screen_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-visuals", action="store_true")
    args = parser.parse_args()
    report = finalize_batch(
        project_root=args.project_root,
        batch_manifest_path=args.batch_manifest,
        results_root=args.results_root,
        output_root=args.output_root,
        render_visuals=not args.skip_visuals,
    )
    print(
        json.dumps(
            {
                "output": str((args.output_root / "batch_screen_report.json").resolve()),
                "route_count": report["route_count"],
                "runtime_valid_count": report["runtime_valid_count"],
                "qa_input_ready_count": report["qa_input_ready_count"],
                "outcome_counts": report["outcome_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
