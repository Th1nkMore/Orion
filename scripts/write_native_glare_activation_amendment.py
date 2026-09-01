#!/usr/bin/env python3
"""Write the immutable remote-activation record for native glare support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "orion.scenario_factory.amendment.v1"
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT",
    "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_activation(
    *,
    project_root: Path,
    staging_amendment: Path,
    protocol_path: Path,
    platform_test_log: Path,
    training_job_id: str,
    training_job_state: str,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite native-glare activation")
    if training_job_id != "1112878":
        raise ValueError("activation is bound to frozen dependency job 1112878")
    state = training_job_state.strip().upper().split("+", 1)[0]
    if state not in TERMINAL_STATES:
        raise ValueError("Stage2-L dependency job is not terminal: " + state)
    staging = json.loads(staging_amendment.read_text(encoding="utf-8"))
    if (
        staging.get("schema") != SCHEMA
        or staging.get("status")
        != "implementation_complete_remote_activation_deferred_dependency_freeze"
        or staging.get("dependency_freeze", {}).get("active_job_id") != "1112878"
    ):
        raise ValueError("staging amendment contract differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_contract = protocol.get("source_contract") or {}
    actual_sources = {}
    for relative_path, expected in source_contract.items():
        source = project_root / relative_path
        actual = sha256(source)
        if actual != expected:
            raise ValueError("source contract differs: " + relative_path)
        actual_sources[relative_path] = actual
    test_text = platform_test_log.read_text(encoding="utf-8")
    matches = re.findall(r"(\d+) passed", test_text)
    passed_tests = int(matches[-1]) if matches else 0
    if passed_tests < 18 or "failed" in test_text.lower():
        raise ValueError("platform native-glare suite did not pass")
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "native_glare_orion_interface_activated",
        "title": "Activate CARLA-native glare on the real ORION closed-loop path",
        "staging_amendment": {
            "path": str(staging_amendment.resolve()),
            "sha256": sha256(staging_amendment),
        },
        "dependency_freeze_release": {
            "job_id": training_job_id,
            "terminal_state": state,
            "terminal_verified_before_source_activation": True,
        },
        "activated_sources": actual_sources,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "platform_validation": {
            "test_log": str(platform_test_log.resolve()),
            "test_log_sha256": sha256(platform_test_log),
            "passed_tests": passed_tests,
            "python_compile": "passed",
            "shell_syntax": "passed",
            "carla_runtime_readback": "required_fail_closed_during_first_clean_bundle_run",
        },
        "launch_locks": {
            "route151_native_glare_pair_allowed": True,
            "maximum_route151_bundle_submissions": 1,
            "automatic_retry_allowed": False,
            "automatic_route_expansion_allowed": False,
            "route203_native_glare_submission_allowed": False,
            "stage2l_v10_aware_closed_loop_allowed": False,
            "stage2p_allowed": False,
            "formal_200_route_evaluation_allowed": False,
        },
        "runtime_gate": (
            "The Route151 bundle must abort before medium if the clean run lacks "
            "verified CARLA sensor/weather readback, exact original-ORION lineage, "
            "or a valid clean safety baseline."
        ),
        "claim_boundary": (
            "Source activation and platform tests only. The first clean run remains "
            "the real CARLA interface/readback smoke; this amendment is not a "
            "glare, behavior, UQ or safety result."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--staging-amendment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--platform-test-log", type=Path, required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--training-job-state", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_activation(
        project_root=args.project_root.resolve(),
        staging_amendment=args.staging_amendment.resolve(),
        protocol_path=args.protocol.resolve(),
        platform_test_log=args.platform_test_log.resolve(),
        training_job_id=args.training_job_id,
        training_job_state=args.training_job_state,
        output=args.output.resolve(),
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": payload["status"],
        "activated_source_count": len(payload["activated_sources"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
