import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "configs/scenario_factory/corruption_hardcase_event_windows_dev8_v1.json"
FUNNEL = ROOT / "configs/scenario_factory/corruption_hardcase_funnel12_v1.json"


def test_event_windows_match_development_roles_and_keep_heldout_closed():
    windows = json.loads(WINDOWS.read_text(encoding="utf-8"))
    funnel = json.loads(FUNNEL.read_text(encoding="utf-8"))
    expected = {row["event_id"] for row in funnel["route_roles"]["development_screen"]}
    rows = windows["events"]
    assert {row["event_id"] for row in rows} == expected
    assert len(rows) == 8
    assert sum(bool(row["positive_case_eligible"]) for row in rows) == 7
    route168 = next(row for row in rows if row["route_index"] == 168)
    assert route168["positive_case_eligible"] is False
    assert route168["clean_outcome"] == "VALID_COLLISION"
    assert windows["selection_rule"]["heldout_routes_read"] is False
    assert windows["execution_locks"]["read_heldout_confirmation"] is False


def test_windows_are_ordered_progress_schedules_with_source_provenance():
    windows = json.loads(WINDOWS.read_text(encoding="utf-8"))
    assert windows["selection_rule"]["runtime_schedule"] == "route_progress"
    assert windows["selection_rule"]["corrupt_run_outcomes_read"] is False
    for row in windows["events"]:
        start, end = row["route_progress_window"]
        anchor = row["route_progress_anchor"]
        assert 0.0 <= start <= anchor <= end <= 1.0
        assert len(row["source_steps"]) == 3
        assert row["source_steps"] == sorted(row["source_steps"])
        assert len(row["event_package_sha256"]) == 64
        assert len(row["control_trace_sha256"]) == 64


def test_visual_freeze_and_no_route_submission_locks_remain_closed():
    windows = json.loads(WINDOWS.read_text(encoding="utf-8"))
    locks = windows["execution_locks"]
    assert locks["use_before_visual_severity_freeze"] is False
    assert locks["route_jobs_submitted_by_this_freeze"] is False
    assert locks["change_windows_after_corrupt_orion_output"] is False
    assert locks["accept_route168_as_positive"] is False
