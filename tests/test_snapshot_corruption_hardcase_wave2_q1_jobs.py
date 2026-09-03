import hashlib
import json
from pathlib import Path

import pytest

from scripts.snapshot_corruption_hardcase_wave2_q1_jobs import (
    SUBMISSION_SCHEMA,
    build_snapshot,
    parse_sacct,
    parse_squeue,
    submission_job_ids,
)


def _submission(path: Path):
    payload = {
        "schema": SUBMISSION_SCHEMA,
        "status": "submitted",
        "jobs_submitted": 3,
        "job_ids": [
            {"job_key": "route160_clean_q1", "job_id": 1001},
            {"job_key": "route165_clean_q1", "job_id": 1002},
            {"job_key": "route161_clean_q1", "job_id": 1003},
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def test_parses_queue_and_allocation_rows_only():
    queue = parse_squeue("1001|PENDING|0:00|Priority\n1002|RUNNING|1:02|gpu4\n")
    assert queue[1001]["state"] == "PENDING"
    assert queue[1002]["reason_or_nodes"] == "gpu4"

    accounting = parse_sacct(
        "1003|COMPLETED|00:03:00|0:0|gpu3\n"
        "1003.batch|COMPLETED|00:03:00|0:0|gpu3\n"
    )
    assert list(accounting) == [1003]
    assert accounting[1003]["state"] == "COMPLETED"


def test_builds_mixed_pending_running_terminal_snapshot(tmp_path):
    path = tmp_path / "submission.json"
    submission = _submission(path)
    snapshot = build_snapshot(
        submission=submission,
        submission_path=path,
        squeue_text="1001|PENDING|0:00|Priority\n1002|RUNNING|1:02|gpu4\n",
        sacct_text="1003|COMPLETED|00:03:00|0:0|gpu3\n",
        captured_at="2026-09-01T00:00:00+00:00",
    )
    assert [row["state"] for row in snapshot["jobs"]] == [
        "PENDING",
        "RUNNING",
        "COMPLETED",
    ]
    assert snapshot["terminal_count"] == 1
    assert snapshot["all_terminal"] is False
    assert snapshot["unknown_job_ids"] == []
    assert snapshot["submission"]["sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_rejects_empty_or_inconsistent_job_journal():
    with pytest.raises(ValueError, match="no returned JobIDs"):
        submission_job_ids({
            "schema": SUBMISSION_SCHEMA,
            "jobs_submitted": 0,
            "job_ids": [],
        })
    with pytest.raises(ValueError, match="differs"):
        submission_job_ids({
            "schema": SUBMISSION_SCHEMA,
            "jobs_submitted": 2,
            "job_ids": [{"job_key": "one", "job_id": 1001}],
        })


def test_unobserved_job_is_not_terminal(tmp_path):
    path = tmp_path / "submission.json"
    submission = _submission(path)
    snapshot = build_snapshot(
        submission=submission,
        submission_path=path,
        squeue_text="",
        sacct_text="",
        captured_at="2026-09-01T00:00:00+00:00",
    )
    assert snapshot["all_terminal"] is False
    assert snapshot["terminal_count"] == 0
    assert snapshot["unknown_job_ids"] == [1001, 1002, 1003]
