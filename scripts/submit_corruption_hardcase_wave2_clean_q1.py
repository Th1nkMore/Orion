#!/usr/bin/env python3
"""Validate or submit Wave2 clean Q1 only with a separate resume activation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Optional


SUBMISSION_SCHEMA = "orion.corruption_hardcase_wave2_clean_q1_submission.v1"
PREREG_SCHEMA = (
    "orion.corruption_hardcase_wave2_clean_qualification_preregistration.v1"
)
ACTIVATION_SCHEMA = "orion.corruption_hardcase_wave2_clean_q1_activation.v1"
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_submission_record(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist the current submission journal.

    The record is rewritten before and after every sbatch attempt.  This keeps
    already returned JobIDs recoverable if the submitter is interrupted while
    submitting a later route.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _resolve(repository_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def verify_reference(
    repository_root: Path,
    reference: Mapping[str, Any],
    *,
    expected_path: Optional[Path] = None,
) -> Path:
    path = _resolve(repository_root, str(reference["path"]))
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ValueError("activation references a different preregistration")
    if sha256(path) != reference.get("sha256"):
        raise ValueError("lineage hash differs for %s" % path)
    return path


def validate_prereg(prereg: Mapping[str, Any], repository_root: Path) -> None:
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("unexpected Wave2 clean preregistration schema")
    if prereg.get("status") != "prepared_paused_no_submission_authority":
        raise ValueError("unexpected Wave2 clean preregistration status")
    if prereg["qualification_protocol"]["q1"].get("authorized") is not False:
        raise ValueError("base preregistration must remain non-authorizing")
    if prereg["execution_locks"].get("q1_clean_submission") is not False:
        raise ValueError("base preregistration Q1 lock must remain closed")
    for reference in prereg["lineage"].values():
        verify_reference(repository_root, reference)


def validate_activation(
    activation: Mapping[str, Any],
    *,
    prereg: Mapping[str, Any],
    prereg_path: Path,
    repository_root: Path,
) -> None:
    if activation.get("schema") != ACTIVATION_SCHEMA:
        raise ValueError("Wave2 Q1 activation is absent or is only a template")
    if activation.get("status") != "authorized_after_user_resume":
        raise ValueError("Wave2 Q1 activation is not authorized")
    verify_reference(
        repository_root,
        activation["base_prereg"],
        expected_path=prereg_path,
    )
    scope = activation["scope"]
    if list(scope.get("routes", [])) != list(prereg["selection"]["routes"]):
        raise ValueError("activation route order differs from the frozen selection")
    if scope.get("condition") != "clean_off" or int(
        scope.get("runs_per_route", 0)
    ) != 1:
        raise ValueError("activation scope is not one clean Q1 run per route")
    if scope.get("run_id") != prereg["qualification_protocol"]["q1"]["run_id"]:
        raise ValueError("activation run ID differs from the preregistration")
    authority = activation.get("authorization", {})
    if (
        authority.get("q1_clean_submission") is not True
        or authority.get("user_resume_recorded") is not True
    ):
        raise ValueError("activation lacks explicit post-resume authority")
    forbidden = (
        "q2_clean_submission",
        "corruption_submission",
        "heldout_confirmation",
        "severity_change",
        "learned_uq_or_governor",
        "stage2p",
        "formal_200_route_evaluation",
    )
    if any(authority.get(key) is not False for key in forbidden):
        raise ValueError("activation opens scope beyond clean Q1")


def build_jobs(
    *,
    prereg: dict[str, Any],
    prereg_path: Path,
    activation: dict[str, Any],
    repository_root: Path,
    asset_root: Path,
) -> list[dict[str, Any]]:
    validate_prereg(prereg, repository_root)
    validate_activation(
        activation,
        prereg=prereg,
        prereg_path=prereg_path,
        repository_root=repository_root,
    )
    resource = prereg["resources"]
    excluded = ",".join(sorted(set(resource["excluded_nodes"])))
    run_id = activation["scope"]["run_id"]
    jobs: list[dict[str, Any]] = []
    for route in prereg["routes"]:
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
            "PILOT_RUN_ID": str(run_id),
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
            "ORION_EXACT_FRAME_SPEEDOMETER": "1",
            "ORION_SENSOR_QUEUE_DIAGNOSTICS": "1",
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
    if [job["route_index"] for job in jobs] != list(
        activation["scope"]["routes"]
    ):
        raise ValueError("built job order differs from activated scope")
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
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Wave2 Q1 submission record")
    repository_root = args.repository_root.resolve()
    prereg_path = args.prereg.resolve()
    prereg = read_json(prereg_path)
    activation = read_json(args.activation)
    jobs = build_jobs(
        prereg=prereg,
        prereg_path=prereg_path,
        activation=activation,
        repository_root=repository_root,
        asset_root=args.asset_root.resolve(),
    )
    payload: dict[str, Any] = {
        "schema": SUBMISSION_SCHEMA,
        "status": "validated_authorized_plan_no_jobs_submitted",
        "created_at": utc_now(),
        "prereg": {"path": str(prereg_path), "sha256": sha256(prereg_path)},
        "activation": {
            "path": str(args.activation.resolve()),
            "sha256": sha256(args.activation),
        },
        "submitter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "jobs": jobs,
        "job_ids": [],
        "jobs_submitted": 0,
        "submission_attempts": [],
    }
    exit_code = 0
    if args.execute:
        payload["status"] = "submission_in_progress"
        write_submission_record(args.output, payload)
        submitter = repository_root / "scripts/submit_closedloop_uq_pilot.sh"
        for job in jobs:
            attempt: dict[str, Any] = {
                "job_key": job["job_key"],
                "route_index": job["route_index"],
                "state": "submitting",
                "attempted_at": utc_now(),
            }
            payload["submission_attempts"].append(attempt)
            write_submission_record(args.output, payload)
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
                attempt["submission_stdout"] = completed.stdout
                attempt["sbatch_returned_at"] = utc_now()
                write_submission_record(args.output, payload)
                lines = [
                    line.strip()
                    for line in completed.stdout.splitlines()
                    if line.strip()
                ]
                job_id = lines[-1] if lines else ""
                if not re.fullmatch(r"[0-9]+", job_id):
                    raise RuntimeError("could not parse sbatch JobID")
                attempt["state"] = "submitted"
                attempt["job_id"] = int(job_id)
                attempt["recorded_at"] = utc_now()
                payload["job_ids"].append({
                    "job_key": job["job_key"],
                    "job_id": int(job_id),
                })
                payload["jobs_submitted"] = len(payload["job_ids"])
                write_submission_record(args.output, payload)
            except Exception as error:
                attempt["state"] = "failed"
                attempt["failed_at"] = utc_now()
                attempt["error_type"] = type(error).__name__
                attempt["error"] = str(error)
                payload["status"] = "partial_submission_failed"
                payload["error_type"] = type(error).__name__
                payload["error"] = str(error)
                exit_code = 2
                payload["completed_at"] = utc_now()
                write_submission_record(args.output, payload)
                break
        if exit_code == 0:
            payload["status"] = "submitted"
            payload["completed_at"] = utc_now()
    write_submission_record(args.output, payload)
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
