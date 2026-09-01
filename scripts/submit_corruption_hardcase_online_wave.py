#!/usr/bin/env python3
"""Plan or submit the explicitly activated corruption hard-case online wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCHEMA = "orion.corruption_hardcase_online_wave_submission.v1"
CORRUPTION_ENV_KEYS = {
    "ORION_CLOSEDLOOP_CORRUPTION",
    "ORION_CLOSEDLOOP_CORRUPTION_VIEWS",
    "ORION_CLOSEDLOOP_CORRUPTION_SEED",
    "ORION_CLOSEDLOOP_CORRUPTION_REGION",
    "ORION_CLOSEDLOOP_CORRUPTION_START_SECONDS",
    "ORION_CLOSEDLOOP_CORRUPTION_END_SECONDS",
    "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS",
    "ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS",
    "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS",
    "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY",
    "ORION_PAIRED_WATERDROP_PROFILE",
    "ORION_PAIRED_WATERDROP_BANK",
    "ORION_NATIVE_MOTION_BLUR_PROFILE",
    "ORION_NATIVE_GLARE_PROFILE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_jobs(
    *,
    config: dict[str, Any],
    activation: dict[str, Any],
    project_root: Path,
    asset_root: Path,
) -> list[dict[str, Any]]:
    exact = activation["primary_exact_conditions"]
    if exact != {
        "front_stale": "delay_ms:200",
        "lens_waterdrop_paired_template": "profile:medium",
        "native_motion_blur": "profile:medium",
    }:
        raise ValueError("Wave0 exact-condition set differs from authorization")
    runtime = activation["runtime"]
    jobs = []
    for route in config["routes"]:
        route_index = int(route["route_index"])
        route_file = Path(route["route_xml"]["path"])
        if not route_file.is_file():
            raise FileNotFoundError(str(route_file))
        route_sha = sha256(route_file)
        if route_sha != route["route_xml"]["sha256"]:
            raise ValueError("route XML hash differs for route%d" % route_index)
        common = {
            "PROJECT_ROOT": str(project_root),
            "ASSET_ROOT": str(asset_root),
            "PILOT_RUN_ID": runtime["pilot_run_id"],
            "PILOT_ROUTE_FILE": str(route_file),
            "PILOT_ROUTE_FILE_SHA256": route_sha,
            "ORION_CORRUPTION_VISUAL_APPROVAL_GATE": str(
                project_root / activation["visual_gate"]["path"]
            ),
            "SLURM_PARTITION": runtime["slurm_partition"],
            "SLURM_CPUS_PER_TASK": str(runtime["cpus_per_job"]),
            "SLURM_MEM": runtime["memory_per_job"],
            "SLURM_TIME": runtime["time_limit"],
            "ORION_ENABLE_LEGACY_DENSITY_UQ": "0",
            "ORION_CLOSEDLOOP_UQ_MODE": "none",
            "ORION_CLOSEDLOOP_CONDITIONING": "none",
            "ORION_CLOSEDLOOP_RISK_MODE": "off",
            "ORION_PLANNING_RESPONSE_MODE": "off",
            "ORION_STAGE2_SPATIAL_UQ_SOURCE": "disabled",
        }
        start_progress = str(route["corruption_window"]["start_progress"])
        duration = str(route["corruption_window"]["duration_seconds"])
        conditions = [
            ("clean_off", {}),
            (
                "front_stale_transient_off",
                {
                    "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS": start_progress,
                    "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS": duration,
                    "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY": "2",
                },
            ),
            (
                "lens_waterdrop_paired_template_transient_off",
                {
                    "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS": start_progress,
                    "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS": duration,
                    "ORION_PAIRED_WATERDROP_PROFILE": "medium",
                    "ORION_PAIRED_WATERDROP_BANK": str(
                        project_root
                        / "assets/waterdrop_patterns/icra2023_paired_template_v1"
                    ),
                },
            ),
            (
                "native_motion_blur_off",
                {"ORION_NATIVE_MOTION_BLUR_PROFILE": "medium"},
            ),
        ]
        for condition, condition_env in conditions:
            environment = dict(common)
            environment.update(condition_env)
            jobs.append({
                "job_key": "route%d_%s" % (route_index, condition),
                "route_index": route_index,
                "variant": "hazard",
                "condition": condition,
                "route_xml": str(route_file),
                "route_xml_sha256": route_sha,
                "environment": environment,
            })
    if len(jobs) != activation["matrix"]["total_jobs"]:
        raise ValueError("planned job count differs from activation")
    return jobs


def verify_environment_isolation(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        condition = job["condition"]
        environment = job["environment"]
        present = CORRUPTION_ENV_KEYS.intersection(environment)
        if condition == "clean_off" and present:
            raise ValueError("clean job inherits corruption variables")
        if condition == "front_stale_transient_off":
            expected = {
                "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS",
                "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS",
                "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY",
            }
            if present != expected:
                raise ValueError("front-stale environment differs")
        if condition == "lens_waterdrop_paired_template_transient_off":
            expected = {
                "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS",
                "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS",
                "ORION_PAIRED_WATERDROP_PROFILE",
                "ORION_PAIRED_WATERDROP_BANK",
            }
            if present != expected:
                raise ValueError("waterdrop environment differs")
        if condition == "native_motion_blur_off":
            if present != {"ORION_NATIVE_MOTION_BLUR_PROFILE"}:
                raise ValueError("native-motion-blur environment differs")


def strict_preflight(
    *, project_root: Path, config_path: Path, activation_path: Path
) -> None:
    command = [
        sys.executable,
        str(project_root / "scripts/verify_corruption_hardcase_online_screen_wave.py"),
        "--config", str(config_path),
        "--activation", str(activation_path),
        "--repository-root", str(project_root),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def sanitized_environment(updates: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    for key in list(environment):
        if (
            key in CORRUPTION_ENV_KEYS
            or key.startswith("ORION_OBSERVATION_UQ_")
            or key.startswith("ORION_STAGE2_")
            or key.startswith("ORION_CLOSEDLOOP_RISK_")
            or key.startswith("ORION_PLANNING_")
        ):
            environment.pop(key, None)
    environment.update(updates)
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-key", action="append", default=[])
    parser.add_argument("--exclude-node", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite wave submission record")
    project_root = args.repository_root.resolve()
    config = read_json(args.config)
    activation = read_json(args.activation)
    if config.get("schema") != "orion.corruption_hardcase_online_screen_wave.v1":
        raise ValueError("unexpected wave schema")
    if activation.get("schema") != (
        "orion.corruption_hardcase_online_screen_wave_activation.v1"
    ):
        raise ValueError("unexpected activation schema")
    if sha256(args.config) != activation["base_wave"]["sha256"]:
        raise ValueError("activation base-wave hash differs")
    if sha256(args.activation) == "0" * 64:
        raise ValueError("invalid activation hash")
    strict_preflight(
        project_root=project_root,
        config_path=args.config,
        activation_path=args.activation,
    )
    jobs = build_jobs(
        config=config,
        activation=activation,
        project_root=project_root,
        asset_root=args.asset_root.resolve(),
    )
    if args.job_key:
        requested = set(args.job_key)
        known = {job["job_key"] for job in jobs}
        missing = sorted(requested - known)
        if missing:
            raise ValueError("unknown requested job keys: %s" % missing)
        jobs = [job for job in jobs if job["job_key"] in requested]
        if len(jobs) != len(requested):
            raise ValueError("duplicate requested job keys")
    if args.exclude_node:
        exclude_nodes = ",".join(sorted(set(args.exclude_node)))
        for job in jobs:
            job["environment"]["SLURM_EXCLUDE"] = exclude_nodes
    verify_environment_isolation(jobs)
    payload = {
        "schema": SCHEMA,
        "status": "validated_plan_no_jobs_submitted",
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "activation": {
            "path": str(args.activation), "sha256": sha256(args.activation)
        },
        "submitter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "jobs": jobs,
        "requested_job_keys": list(args.job_key),
        "excluded_nodes": sorted(set(args.exclude_node)),
        "job_ids": [],
        "jobs_submitted": 0,
    }
    exit_code = 0
    if args.execute:
        payload["status"] = "submission_in_progress"
        submitter = project_root / "scripts/submit_closedloop_uq_pilot.sh"
        for job in jobs:
            command = [
                "bash", str(submitter), str(job["route_index"]),
                job["condition"], job["variant"],
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=sanitized_environment(job["environment"]),
                )
                lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                job_id = lines[-1] if lines else ""
                if not re.fullmatch(r"[0-9]+", job_id):
                    raise RuntimeError("could not parse sbatch JobID from submitter output")
                payload["job_ids"].append({
                    "job_key": job["job_key"],
                    "job_id": int(job_id),
                })
                payload["jobs_submitted"] = len(payload["job_ids"])
            except Exception as error:
                payload["status"] = "partial_submission_failed"
                payload["error_type"] = type(error).__name__
                payload["error"] = str(error)
                exit_code = 2
                break
        if exit_code == 0:
            payload["status"] = "submitted"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "jobs_planned": len(jobs),
        "jobs_submitted": payload["jobs_submitted"],
        "job_ids": payload["job_ids"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
