import json

from scripts.evaluate_clean_liveness_screen import (
    evaluate,
    longest_low_speed_interval,
)


def _row(step, speed, progress=None):
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "route_progress": step / 1000 if progress is None else progress,
        "speed": speed,
    }


def test_long_stop_interval_ignores_initialization_and_triggers():
    rows = []
    for step in range(260):
        if step < 40:
            speed = 0.0
        elif step < 80:
            speed = 4.0
        else:
            speed = 0.1
        rows.append(_row(step, speed))
    interval = longest_low_speed_interval(rows)
    assert interval["start_step"] == 80
    assert abs(interval["duration_seconds"] - 9.0) < 1e-9


def test_evaluate_marks_stage_b_unqualified(tmp_path):
    run = tmp_path / "run"
    trace_dir = run / "records_x" / "route"
    trace_dir.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "pilot_condition": "clean_off",
        "orion_closedloop_risk_mode": "off",
    }))
    rows = [_row(step, 4.0 if step < 50 else 0.0) for step in range(230)]
    (trace_dir / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = evaluate(run)
    assert report["fast_screen_triggered"]
    assert not report["stage_b_qualified"]
    assert report["last_observed"]["step"] == 229
