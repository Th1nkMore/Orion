import pytest

from scripts.analyze_clean_global_uq_activation import analyze


def _row(step, score, *, speed=2.0, ttc=None, corruption=False, mode="off"):
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "route_progress": step / 400.0,
        "speed": speed,
        "density_uq_score": score,
        "corruption_active": corruption,
        "risk": {"mode": mode},
        "closedloop_safety": {
            "available": True,
            "min_obb_collision_ttc_seconds": ttc,
        },
    }


def test_clean_activation_separates_conflict_and_no_conflict_exposure():
    rows = []
    for step in range(140):
        time = step * 0.05
        score, speed, ttc = 0.2, 2.0, None
        if 4.0 <= time < 5.0:
            score = 0.5
        if 5.0 <= time < 6.0:
            score, speed, ttc = 0.8, 0.0, 2.0
        rows.append(_row(step, score, speed=speed, ttc=ttc))
    report = analyze(rows)
    post = report["post_baseline"]
    assert post["threshold_exposure_seconds"] == pytest.approx(2.0)
    assert post["threshold_exposure_while_moving_seconds"] == pytest.approx(1.0)
    assert post["threshold_exposure_while_stopped_seconds"] == pytest.approx(1.0)
    assert post[
        "threshold_exposure_without_near_term_obb_conflict_seconds"
    ] == pytest.approx(1.0)
    assert report["longest_continuous_threshold_exposure"][
        "duration_seconds"
    ] == pytest.approx(2.0)


def test_clean_activation_rejects_corruption_or_active_governor():
    rows = [_row(step, 0.2) for step in range(100)]
    rows[90]["corruption_active"] = True
    with pytest.raises(ValueError, match="active corruption"):
        analyze(rows)
    rows[90]["corruption_active"] = False
    rows[90]["risk"]["mode"] = "learned"
    with pytest.raises(ValueError, match="risk mode off"):
        analyze(rows)
