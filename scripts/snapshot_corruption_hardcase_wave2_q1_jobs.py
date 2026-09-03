#!/usr/bin/env python3
"""Write one immutable Slurm status snapshot for a Wave2 Q1 submission journal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


SUBMISSION_SCHEMA = "orion.corruption_hardcase_wave2_clean_q1_submission.v1"
SNAPSHOT_SCHEMA = "orion.corruption_hardcase_wave2_clean_q1_job_snapshot.v1"
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_state(state: str) -> str:
    return state.strip().split()[0].rstrip("+") if state.strip() else "UNKNOWN"


def parse_squeue(text: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split("|", 3)
        if len(fields) != 4 or not fields[0].strip().isdigit():
            raise ValueError("unexpected squeue row: %s" % raw)
        job_id = int(fields[0].strip())
        rows[job_id] = {
            "state": normalized_state(fields[1]),
            "elapsed": fields[2].strip(),
            "reason_or_nodes": fields[3].strip(),
        }
    return rows


def parse_sacct(text: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split("|", 4)
        if len(fields) != 5:
            raise ValueError("unexpected sacct row: %s" % raw)
        job_id_raw = fields[0].strip()
        # Ignore job steps such as 123.batch and 123.extern.  The allocation
        # row is the authoritative terminal state for this snapshot.
        if not job_id_raw.isdigit():
            continue
        job_id = int(job_id_raw)
        rows[job_id] = {
            "state": normalized_state(fields[1]),
            "elapsed": fields[2].strip(),
            "exit_code": fields[3].strip(),
            "node_list": fields[4].strip(),
        }
    return rows


def submission_job_ids(submission: Mapping[str, Any]) -> list[dict[str, Any]]:
    if submission.get("schema") != SUBMISSION_SCHEMA:
        raise ValueError("unexpected Wave2 Q1 submission schema")
    rows = list(submission.get("job_ids", []))
    if not rows:
        raise ValueError("submission journal contains no returned JobIDs")
    ids = [int(row["job_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("submission journal contains duplicate JobIDs")
    if int(submission.get("jobs_submitted", -1)) != len(rows):
        raise ValueError("jobs_submitted differs from returned JobID count")
    return rows


def build_snapshot(
    *,
    submission: Mapping[str, Any],
    submission_path: Path,
    squeue_text: str,
    sacct_text: str,
    captured_at: str,
) -> dict[str, Any]:
    job_refs = submission_job_ids(submission)
    active = parse_squeue(squeue_text)
    accounting = parse_sacct(sacct_text)
    jobs = []
    for reference in job_refs:
        job_id = int(reference["job_id"])
        if job_id in active:
            source = "squeue"
            row = active[job_id]
        elif job_id in accounting:
            source = "sacct"
            row = accounting[job_id]
        else:
            source = "unobserved"
            row = {"state": "UNKNOWN"}
        state = normalized_state(row["state"])
        jobs.append({
            "job_key": reference["job_key"],
            "job_id": job_id,
            "source": source,
            "state": state,
            "terminal": state in TERMINAL_STATES,
            "details": row,
        })
    return {
        "schema": SNAPSHOT_SCHEMA,
        "captured_at": captured_at,
        "submission": {
            "path": str(submission_path.resolve()),
            "sha256": sha256(submission_path),
            "status": submission.get("status"),
        },
        "jobs": jobs,
        "job_count": len(jobs),
        "terminal_count": sum(bool(job["terminal"]) for job in jobs),
        "all_terminal": all(bool(job["terminal"]) for job in jobs),
        "unknown_job_ids": [
            job["job_id"] for job in jobs if job["source"] == "unobserved"
        ],
        "raw": {
            "squeue": squeue_text,
            "sacct": sacct_text,
        },
        "claim_boundary": (
            "Read-only Slurm status snapshot. This record neither submits nor "
            "cancels a job and is not a scientific result."
        ),
    }


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Slurm status snapshot")
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    job_refs = submission_job_ids(submission)
    ids = ",".join(str(int(row["job_id"])) for row in job_refs)
    squeue_text = run_command(
        ["squeue", "--noheader", "--jobs", ids, "--format", "%i|%T|%M|%R"]
    )
    sacct_text = run_command(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            ids,
            "--format",
            "JobIDRaw,State,Elapsed,ExitCode,NodeList",
        ]
    )
    snapshot = build_snapshot(
        submission=submission,
        submission_path=args.submission,
        squeue_text=squeue_text,
        sacct_text=sacct_text,
        captured_at=utc_now(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "job_count": snapshot["job_count"],
        "terminal_count": snapshot["terminal_count"],
        "all_terminal": snapshot["all_terminal"],
        "unknown_job_ids": snapshot["unknown_job_ids"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
