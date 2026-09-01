#!/usr/bin/env python3
"""Validate, plan, or submit the frozen Route158 Wave1 corruption screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APPROVAL_PATH = ROOT / "uq_estimator/corruption_visual_approval.py"
APPROVAL_SPEC = importlib.util.spec_from_file_location(
    "orion_route158_corruption_visual_approval", APPROVAL_PATH
)
APPROVAL_MODULE = importlib.util.module_from_spec(APPROVAL_SPEC)
sys.modules[APPROVAL_SPEC.name] = APPROVAL_MODULE
APPROVAL_SPEC.loader.exec_module(APPROVAL_MODULE)
verify_visual_approval = APPROVAL_MODULE.verify_visual_approval


ACTIVATION_SCHEMA = "orion.corruption_hardcase_wave1_route158_corruption_activation.v1"
SUBMISSION_SCHEMA = "orion.corruption_hardcase_wave1_route158_corruption_submission.v1"
Q2_RESULT_SCHEMA = "orion.corruption_hardcase_wave1_clean_q2_result_amendment.v1"
Q2_REPORT_SCHEMA = "orion.corruption_hardcase_clean_qualification.v1"
EVENT_WINDOW_SCHEMA = "orion.corruption_hardcase_event_windows.v1"
EXACT_CONDITIONS = {
    "front_stale": "delay_ms:200",
    "lens_waterdrop_paired_template": "profile:medium",
    "native_motion_blur": "profile:medium",
}
CONDITION_NAMES = (
    "front_stale_transient_off",
    "lens_waterdrop_paired_template_transient_off",
    "native_motion_blur_off",
)
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


def repository_path(repository_root: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return candidate.resolve()


def verify_reference(
    repository_root: Path, reference: dict[str, Any], *, schema: str | None = None
) -> tuple[Path, dict[str, Any] | None]:
    path = repository_path(repository_root, str(reference["path"]))
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if sha256(path) != reference["sha256"]:
        raise ValueError("hash differs: %s" % path)
    if schema is None:
        return path, None
    payload = read_json(path)
    if payload.get("schema") != schema:
        raise ValueError("unexpected schema: %s" % path)
    return path, payload


def validate_activation(
    *, activation: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    if activation.get("schema") != ACTIVATION_SCHEMA:
        raise ValueError("unexpected Route158 activation schema")
    if activation["execution_authority"].get(
        "route158_three_condition_development_screen"
    ) is not True:
        raise ValueError("Route158 corruption screen is not authorized")
    for locked in (
        "additional_routes",
        "additional_severities",
        "heldout_confirmation",
        "route203",
        "stage2p",
        "formal_200_route_evaluation",
    ):
        if activation["execution_authority"].get(locked) is not False:
            raise ValueError("activation broadens locked scope: %s" % locked)
    if activation["visual_approval"]["exact_conditions"] != EXACT_CONDITIONS:
        raise ValueError("exact corruption conditions differ from approval")
    if activation["visual_approval"].get("severity_escalation_allowed") is not False:
        raise ValueError("severity escalation is not locked")
    if activation["matrix"]["corruption_conditions"] != list(CONDITION_NAMES):
        raise ValueError("condition matrix differs")
    if int(activation["matrix"]["total_jobs"]) != 3:
        raise ValueError("Route158 activation must contain exactly three jobs")
    if int(activation["route"]["route_index"]) != 158:
        raise ValueError("activation route is not Route158")

    _, q2_result = verify_reference(
        repository_root, activation["q2_result"], schema=Q2_RESULT_SCHEMA
    )
    assert q2_result is not None
    if q2_result["decision"]["stable_clean_routes"] != [158]:
        raise ValueError("Q2 result does not freeze Route158 as sole stable-clean route")
    q2_report_path, q2_report = verify_reference(
        repository_root,
        activation["clean_comparator"]["q2_report"],
        schema=Q2_REPORT_SCHEMA,
    )
    assert q2_report is not None
    if not (
        q2_report.get("phase") == "q2"
        and q2_report.get("route_index") == 158
        and q2_report.get("status") == "clean_qualified"
        and q2_report.get("qualified_for_corruption_screen") is True
    ):
        raise ValueError("Q2 clean comparator is not a valid Route158 pass")

    _, event_windows = verify_reference(
        repository_root,
        activation["event_window"]["source"],
        schema=EVENT_WINDOW_SCHEMA,
    )
    assert event_windows is not None
    expected_event_id = activation["event_window"]["source"]["event_id"]
    events = [row for row in event_windows["events"] if row["event_id"] == expected_event_id]
    if len(events) != 1 or int(events[0]["route_index"]) != 158:
        raise ValueError("frozen Route158 event is absent or ambiguous")
    event = events[0]
    if activation["event_window"]["start_progress"] != event["route_progress_window"][0]:
        raise ValueError("Route158 event start progress differs")
    if activation["event_window"]["end_progress_provenance"] != event["route_progress_window"][1]:
        raise ValueError("Route158 event end provenance differs")
    if activation["event_window"]["anchor_progress"] != event["route_progress_anchor"]:
        raise ValueError("Route158 event anchor differs")
    if float(activation["event_window"]["duration_seconds"]) != 5.0:
        raise ValueError("Route158 corruption duration differs")

    gate_path, _ = verify_reference(
        repository_root, activation["visual_approval"]["gate"]
    )
    verify_reference(
        repository_root, activation["visual_approval"]["authorization_provenance"]
    )
    approvals = {}
    for family, condition in EXACT_CONDITIONS.items():
        approvals[family] = verify_visual_approval(
            gate_path=gate_path,
            repository_root=repository_root,
            family=family,
            condition=condition,
            require_approved=True,
        ).to_dict()

    bank = repository_path(repository_root, activation["waterdrop_bank"]["path"])
    for name, expected in activation["waterdrop_bank"]["files"].items():
        path = bank / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("waterdrop bank differs: %s" % path)
    for relative, expected in activation["runtime_source_contract"].items():
        path = repository_path(repository_root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("runtime source differs: %s" % relative)

    route_path = repository_path(repository_root, activation["route"]["route_xml"]["path"])
    if not route_path.is_file() or sha256(route_path) != activation["route"]["route_xml"]["sha256"]:
        raise ValueError("Route158 XML differs")
    return {
        "route_path": route_path,
        "q2_report_path": q2_report_path,
        "gate_path": gate_path,
        "bank_path": bank,
        "approvals": approvals,
    }


def build_jobs(
    *, activation: dict[str, Any], repository_root: Path, asset_root: Path,
    validated: dict[str, Any]
) -> list[dict[str, Any]]:
    resource = activation["resources"]
    route_path = Path(validated["route_path"])
    common = {
        "PROJECT_ROOT": str(repository_root),
        "ASSET_ROOT": str(asset_root),
        "PILOT_RUN_ID": activation["matrix"]["run_id"],
        "PILOT_ROUTE_FILE": str(route_path),
        "PILOT_ROUTE_FILE_SHA256": activation["route"]["route_xml"]["sha256"],
        "ORION_CORRUPTION_VISUAL_APPROVAL_GATE": str(validated["gate_path"]),
        "SLURM_PARTITION": resource["partition"],
        "SLURM_CPUS_PER_TASK": str(resource["cpus_per_job"]),
        "SLURM_MEM": resource["memory_per_job"],
        "SLURM_TIME": resource["time_limit"],
        "SLURM_EXCLUDE": ",".join(sorted(set(resource["excluded_nodes"]))),
        "ORION_ENABLE_LEGACY_DENSITY_UQ": "0",
        "ORION_CLOSEDLOOP_UQ_MODE": "none",
        "ORION_CLOSEDLOOP_CONDITIONING": "none",
        "ORION_CLOSEDLOOP_RISK_MODE": "off",
        "ORION_PLANNING_RESPONSE_MODE": "off",
        "ORION_STAGE2_SPATIAL_UQ_SOURCE": "disabled",
    }
    start = str(activation["event_window"]["start_progress"])
    duration = str(activation["event_window"]["duration_seconds"])
    specifications = (
        (
            "front_stale_transient_off",
            {
                "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS": start,
                "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS": duration,
                "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY": "2",
            },
        ),
        (
            "lens_waterdrop_paired_template_transient_off",
            {
                "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS": start,
                "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS": duration,
                "ORION_PAIRED_WATERDROP_PROFILE": "medium",
                "ORION_PAIRED_WATERDROP_BANK": str(validated["bank_path"]),
            },
        ),
        (
            "native_motion_blur_off",
            {"ORION_NATIVE_MOTION_BLUR_PROFILE": "medium"},
        ),
    )
    jobs = []
    for condition, condition_environment in specifications:
        environment = dict(common)
        environment.update(condition_environment)
        jobs.append({
            "job_key": "route158_%s" % condition,
            "route_index": 158,
            "variant": "hazard",
            "condition": condition,
            "route_xml": str(route_path),
            "route_xml_sha256": activation["route"]["route_xml"]["sha256"],
            "environment": environment,
        })
    verify_environment_isolation(jobs)
    return jobs


def verify_environment_isolation(jobs: list[dict[str, Any]]) -> None:
    expected_by_condition = {
        "front_stale_transient_off": {
            "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS",
            "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS",
            "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY",
        },
        "lens_waterdrop_paired_template_transient_off": {
            "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS",
            "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS",
            "ORION_PAIRED_WATERDROP_PROFILE",
            "ORION_PAIRED_WATERDROP_BANK",
        },
        "native_motion_blur_off": {"ORION_NATIVE_MOTION_BLUR_PROFILE"},
    }
    if [job["condition"] for job in jobs] != list(CONDITION_NAMES):
        raise ValueError("planned condition order differs")
    for job in jobs:
        present = CORRUPTION_ENV_KEYS.intersection(job["environment"])
        if present != expected_by_condition[job["condition"]]:
            raise ValueError("corruption environment differs for %s" % job["condition"])
        if job["environment"].get("ORION_CLOSEDLOOP_RISK_MODE") != "off":
            raise ValueError("risk mode is not off")


def select_jobs(
    jobs: list[dict[str, Any]], job_keys: list[str]
) -> list[dict[str, Any]]:
    if not job_keys:
        return jobs
    requested = set(job_keys)
    known = {str(job["job_key"]) for job in jobs}
    if requested - known:
        raise ValueError("unknown Route158 job keys: %s" % sorted(requested - known))
    selected = [job for job in jobs if job["job_key"] in requested]
    if len(selected) != len(requested):
        raise ValueError("duplicate Route158 job selection")
    return selected


def extend_excluded_nodes(
    jobs: list[dict[str, Any]], extra_nodes: list[str]
) -> list[dict[str, Any]]:
    if not extra_nodes:
        return jobs
    for job in jobs:
        current = set(filter(None, job["environment"]["SLURM_EXCLUDE"].split(",")))
        current.update(extra_nodes)
        job["environment"]["SLURM_EXCLUDE"] = ",".join(sorted(current))
    return jobs


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
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-key", action="append", default=[])
    parser.add_argument("--extra-exclude-node", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Route158 submission record")
    repository_root = args.repository_root.resolve()
    activation = read_json(args.activation)
    validated = validate_activation(
        activation=activation, repository_root=repository_root
    )
    jobs = build_jobs(
        activation=activation,
        repository_root=repository_root,
        asset_root=args.asset_root.resolve(),
        validated=validated,
    )
    jobs = select_jobs(jobs, args.job_key)
    jobs = extend_excluded_nodes(jobs, args.extra_exclude_node)
    payload: dict[str, Any] = {
        "schema": SUBMISSION_SCHEMA,
        "status": "validated_plan_no_jobs_submitted",
        "activation": {"path": str(args.activation), "sha256": sha256(args.activation)},
        "submitter": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "q2_clean_comparator": {
            "job_id": activation["clean_comparator"]["q2_job_id"],
            "report": str(validated["q2_report_path"]),
            "sha256": activation["clean_comparator"]["q2_report"]["sha256"],
        },
        "approvals": validated["approvals"],
        "jobs": jobs,
        "requested_job_keys": list(args.job_key),
        "extra_excluded_nodes": sorted(set(args.extra_exclude_node)),
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
                    ["bash", str(submitter), "158", job["condition"], "hazard"],
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
                payload["job_ids"].append({"job_key": job["job_key"], "job_id": int(job_id)})
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
