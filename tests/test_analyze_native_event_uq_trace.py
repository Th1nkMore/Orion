import pytest

from scripts.analyze_native_event_uq_trace import analyze


def test_native_event_uq_analysis_reports_causal_trigger_lead():
    rows = []
    for step in range(241):
        current_time = step * 0.05
        score = 0.1 if current_time < 6.0 else 0.8
        rows.append(
            {
                "step": step,
                "sim_time_seconds": current_time,
                "route_progress": current_time / 12.0,
                "density_uq_score": score,
            }
        )
    report = analyze(rows, event_time_seconds=10.0, threshold=0.4)
    assert report["first_threshold_trigger"]["sim_time_seconds"] == 6.0
    assert report["trigger_lead_seconds"] == 4.0
    assert report["approach"]["fraction_at_or_above_threshold"] == 1.0
    assert report["baseline"]["robust_median"] == pytest.approx(0.1)


def test_native_event_uq_analysis_rejects_inverted_windows():
    with pytest.raises(ValueError, match="lead_seconds"):
        analyze([], event_time_seconds=10.0, lead_seconds=1.0, approach_seconds=2.0)
