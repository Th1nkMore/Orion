import pytest

from uq_estimator.corruption_schedule import (
    RouteTriggeredTimedWindow,
    project_route_progress,
)


def test_project_route_progress_on_straight_route():
    route = [(0.0, 0.0), (10.0, 0.0)]
    assert project_route_progress((2.5, 3.0), route) == pytest.approx(0.25)


def test_project_route_progress_uses_polyline_arc_length():
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert project_route_progress((10.5, 5.0), route) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("position", "expected"),
    [((-5.0, 0.0), 0.0), ((15.0, 0.0), 1.0)],
)
def test_project_route_progress_clamps_route_ends(position, expected):
    assert project_route_progress(position, [(0.0, 0.0), (10.0, 0.0)]) == expected


def test_project_route_progress_rejects_degenerate_route():
    with pytest.raises(ValueError, match="positive length"):
        project_route_progress((0.0, 0.0), [(1.0, 1.0), (1.0, 1.0)])


def test_route_triggered_timed_window_ends_without_progress():
    window = RouteTriggeredTimedWindow(start_progress=0.3, duration_seconds=5.0)
    assert not window.is_active(route_progress=0.29, sim_time_seconds=10.0)
    assert window.is_active(route_progress=0.30, sim_time_seconds=10.5)
    assert window.is_active(route_progress=0.30, sim_time_seconds=15.49)
    assert not window.is_active(route_progress=0.30, sim_time_seconds=15.5)


def test_route_triggered_timed_window_does_not_retrigger():
    window = RouteTriggeredTimedWindow(start_progress=0.3, duration_seconds=1.0)
    assert window.is_active(route_progress=0.31, sim_time_seconds=2.0)
    assert not window.is_active(route_progress=0.31, sim_time_seconds=3.0)
    assert not window.is_active(route_progress=0.8, sim_time_seconds=4.0)
    assert window.trigger_time_seconds == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("start_progress", "duration_seconds"),
    [(-0.1, 1.0), (1.1, 1.0), (0.3, 0.0), (0.3, float("inf"))],
)
def test_route_triggered_timed_window_rejects_invalid_settings(
    start_progress, duration_seconds
):
    with pytest.raises(ValueError):
        RouteTriggeredTimedWindow(start_progress, duration_seconds)
