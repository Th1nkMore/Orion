"""Tests for the non-terminal Route197 structural-deadlock report."""

from scripts.report_route197_v4_structural_deadlock import (
    longest_true_interval,
    parse_actor_flow_defaults,
)


def test_actor_flow_defaults_imply_short_headway():
    source = """
self._flow_speed = get_value_parameter(config, 'flow_speed', float, 20)
self._source_dist_interval = get_interval_parameter(config, 'source_dist_interval', float, [25, 50])
self._scenario_timeout = 240
root.add_child(ActorFlow(a, b, c, 2, d, initial_actors=True))
"""
    parsed = parse_actor_flow_defaults(source)
    assert parsed["flow_speed_mps"] == 20.0
    assert parsed["source_headway_interval_seconds"] == [1.25, 2.5]
    assert parsed["scenario_timeout_seconds"] == 240.0
    assert parsed["continuous_actor_flow_present"] is True
    assert parsed["initial_actors_enabled"] is True


def test_longest_stationary_interval_preserves_progress_evidence():
    rows = [
        {
            "step": step,
            "sim_time_seconds": step * 0.05,
            "route_progress": progress,
            "speed": speed,
        }
        for step, progress, speed in (
            (0, 0.1, 1.0),
            (1, 0.2, 0.1),
            (2, 0.2001, 0.0),
            (3, 0.2001, 0.0),
            (4, 0.3, 1.0),
        )
    ]
    result = longest_true_interval(rows, lambda row: row["speed"] < 0.25)
    assert result["observed"] is True
    assert result["start_step"] == 1
    assert result["end_step"] == 3
    assert result["frame_count"] == 3
    assert abs(result["duration_seconds"] - 0.1) < 1e-12
    assert abs(result["route_progress_delta"] - 0.0001) < 1e-12
