import pytest

from scripts.summarize_closedloop_safety import (
    build_paired_event_report,
    summarize_records,
)


def _row(step, ttc, speed):
    actors = [] if ttc is None else [{
        "actor_id": 7,
        "category": "walker",
        "obb_collision_ttc_seconds": ttc,
        "obb_separating_axis_gap_m": 2.0 + step,
    }]
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "route_progress": step * 0.01,
        "speed": speed,
        "closedloop_safety": {
            "available": True,
            "min_obb_collision_ttc_seconds": ttc,
            "min_obb_separating_axis_gap_m": 2.0 + step,
            "min_disc_clearance_m": 1.0 + step,
            "critical_actor": {"actor_id": 7},
            "actors": actors,
        },
    }


def test_safety_summary_integrates_low_ttc_exposure_and_efficiency():
    records = [
        _row(0, None, 2.0),
        _row(1, 2.5, 0.1),
        _row(2, 1.5, 0.0),
        _row(3, 0.5, 2.0),
    ]
    result = summarize_records(records)
    assert result["safety"]["min_obb_ttc_seconds"] == 0.5
    assert result["safety"]["low_ttc_exposure_seconds"]["1.0"] == pytest.approx(0.05)
    assert result["safety"]["low_ttc_exposure_seconds"]["2.0"] == pytest.approx(0.1)
    assert result["efficiency"]["stopped_below_0_25_mps_seconds"] == pytest.approx(0.1)
    assert result["safety"]["critical_frame"]["step"] == 3
    assert result["safety"]["by_category"]["walker"]["min_obb_ttc_seconds"] == 0.5
    assert result["safety"]["by_category"]["vehicle"]["min_obb_ttc_seconds"] is None


def test_paired_event_report_uses_degraded_event_steps_for_both_runs():
    clean = [_row(step, 3.0, 2.0) for step in range(8)]
    degraded = [_row(step, 1.5 if 2 <= step <= 4 else 3.0, 1.5) for step in range(8)]
    for row in clean:
        row["corruption_active"] = False
    for row in degraded:
        row["corruption_active"] = 2 <= row["step"] <= 4
    report = build_paired_event_report(clean, degraded, recovery_seconds=0.1)
    active = report["active_event"]
    assert (active["start_step"], active["end_step"]) == (2, 4)
    assert active["clean"]["frames"] == 3
    assert active["degraded"]["safety"]["min_obb_ttc_seconds"] == 1.5
    assert active["comparison"]["min_obb_ttc_seconds"] == -1.5
    assert active["comparison"]["by_category"]["walker"]["min_obb_ttc_seconds"] == -1.5
    assert active["comparison"]["by_category"]["walker"][
        "low_ttc_exposure_seconds"
    ]["2.0"] == pytest.approx(0.15)
    prefix = report["pre_event_alignment"]
    assert prefix["frames"] == 2
    assert prefix["control_step_sequences_equal"] is True
    assert prefix["degraded_corruption_absent"] is True
    assert prefix["absolute_delta"]["route_progress"]["max"] == 0.0
    assert prefix["absolute_delta"]["speed"]["mean"] == 0.5
