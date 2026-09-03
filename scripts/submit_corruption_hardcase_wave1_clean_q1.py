#!/usr/bin/env python3
"""Plan or submit the preregistered Wave1 Q1 clean qualification jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


SCHEMA = "orion.corruption_hardcase_wave1_clean_q1_submission.v1"
CONFIG_SCHEMA = "orion.corruption_hardcase_wave1_clean_qualification.v1"
CORRUPTION_KEYS = {
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


def verify_lineage(config: dict[str, Any], repository_root: Path) -> None:
    for label, reference in config["lineage"].items():
        path = repository_root / reference["path"]
        if not path.is_file():
            raise FileNotFoundError("missing %s: %s" % (label, path))
        if sha256(path) != reference["sha256"]:
            raise ValueError("%s lineage hash differs" % label)


def build_jobs(
    config: dict[str, Any], repository_root: Path, asset_root: Path
) -> list[dict[str, Any]]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected Wave1 clean qualification schema")
    if config["qualification_protocol"]["q1"].get("authorized") is not True:
        raise ValueError("Wave1 Q1 is not authorized")
    if config["execution_locks"].get("q1_clean_submission") is not True:
        raise ValueError("Wave1 Q1 submission lock is closed")
    verify_lineage(config, repository_root)
    resource = config["resources"]
    excluded = ",".join(sorted(set(resource["excluded_nodes"])))
    run_id = config["qualification_protocol"]["q1"]["run_id"]
    jobs: list[dict[str, Any]] = []
    selected = list(config["selection"]["routes"])
    for route in config["routes"]:
        route_index = int(route["route_index"])
        route_path = Path(route["route_xml"]["path"])
        if not route_path.is_file():
            raise FileNotFoundError(str(route_path))
        route_hash = sha256(route_path)
        if route_hash != route["route_xml"]["sha256"]:
            raise ValueError("route XML hash differs for route%d" % route_index)
        environment = {
            "PROJECT_ROOT": str(repository_root),
            "ASSET_ROOT": str(asset_root),
            "PILOT_RUN_ID": run_id,
            "PILOT_ROUTE_FILE": str(route_path),
            "PILOT_ROUTE_FILE_SHA256": route_hash,
            "SLURM_PARTITION": resource["partition"],
            "SLURM_CPUS_PER_TASK": str(resource["cpus_per_job"]),
            "SLURM_MEM": resource["memory_per_job"],
            "SLURM_TIME": resource["time_limit"],
            "SLURM_EXCLUDE": excluded,
            "ORION_ENABLE_LEGACY_DENSITY_UQ": "0",
            "ORION_CLOSEDLOOP_UQ_MODE": "none",
            "ORION_CLOSEDLOOP_CONDITIONING": "none",
            "ORION_CLOSEDLOOP_RISK_MODE": "off",
            "ORION_PLANNING_RESPONSE_MODE": "off",
            "ORION_STAGE2_SPATIAL_UQ_SOURCE": "disabled",
        }
        if CORRUPTION_KEYS.intersection(environment):
            raise ValueError("clean qualification inherited corruption variables")
        jobs.append({
            "job_key": "route%d_clean_q1" % route_index,
            "route_index": route_index,
            "variant": "hazard",
            "condition": "clean_off",
            "route_xml": str(route_path),
            "route_xml_sha256": route_hash,
            "environment": environment,
        })
    if [job["route_index"] for job in jobs] != selected:
        raise ValueError("route order/identity differs from frozen selection")
    return jobs


def sanitized_environment(updates: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    for key in list(environment):
        if (
            key in CORRUPTION_KEYS
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
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-index", type=int, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Wave1 Q1 submission record")
    repository_root = args.repository_root.resolve()
    config = read_json(args.config)
    jobs = build_jobs(config, repository_root, args.asset_root.resolve())
    if args.route_index:
        requested = set(args.route_index)
        known = {job["route_index"] for job in jobs}
        if requested - known:
            raise ValueError("unknown route indices: %s" % sorted(requested - known))
        jobs = [job for job in jobs if job["route_index"] in requested]
        if len(jobs) != len(requested):
            raise ValueError("duplicate route selection")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "validated_plan_no_jobs_submitted",
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "submitter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "jobs": jobs,
        "job_ids": [],
        "jobs_submitted": 0,
    }
    exit_code = 0
    if args.execute:
        payload["status"] = "submission_in_progress"
        submitter = repository_root / "scripts/submit_closedloop_uq_pilot.sh"
        for job in jobs:
            try:
                completed = subprocess.run(
                    [
                        "bash",
                        str(submitter),
                        str(job["route_index"]),
                        "clean_off",
                        "hazard",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=sanitized_environment(job["environment"]),
                )
                lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                job_id = lines[-1] if lines else ""
                if not re.fullmatch(r"[0-9]+", job_id):
                    raise RuntimeError("could not parse sbatch JobID")
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
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
