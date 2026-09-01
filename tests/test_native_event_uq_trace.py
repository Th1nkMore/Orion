from scripts.analyze_native_event_uq_trace import analyze


def _row(step, score):
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "route_progress": step / 400.0,
        "density_uq_score": score,
    }


def test_global_uq_report_separates_startup_raw_trigger_from_valid_lead():
    rows = []
    for step in range(500):
        time = step * 0.05
        score = 0.2
        if step == 0:
            score = 0.5
        if 14.0 <= time < 16.0:
            score = 0.8
        rows.append(_row(step, score))
    report = analyze(rows, event_time_seconds=16.0)
    assert report["first_threshold_trigger_any_time"]["sim_time_seconds"] == 0.0
    assert report["first_threshold_trigger_post_baseline"]["sim_time_seconds"] == 14.0
    assert report["post_baseline_threshold_trigger_lead_seconds"] == 2.0
    assert report["pre_lead_false_trigger_window"][
        "frames_at_or_above_threshold"
    ] == 0
