import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_closedloop_spatial_event.py"
)
SPEC = importlib.util.spec_from_file_location(
    "freeze_closedloop_spatial_event", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(time_seconds: float, progress: float, speed: float) -> dict:
    return {
        "sim_time_seconds": time_seconds,
        "route_progress": progress,
        "speed": speed,
    }


def test_pre_event_liveness_ignores_legitimate_post_event_stop():
    rows = [
        row(0.0, 0.00, 0.0),
        row(0.5, 0.05, 2.0),
        row(1.0, 0.10, 3.0),
        row(1.5, 0.20, 4.0),
        row(2.0, 0.40, 2.0),
    ]
    rows.extend(row(2.5 + index * 0.5, 0.41, 0.0) for index in range(30))

    prefix = MODULE.rows_before_progress(rows, 0.40)

    assert [item["route_progress"] for item in prefix] == [0.00, 0.05, 0.10, 0.20]
    assert MODULE.longest_low_speed(prefix) == 0.0
    assert MODULE.longest_low_speed(rows) >= 14.5


def test_pre_event_liveness_still_rejects_a_stuck_clean_prefix():
    rows = [row(index * 0.5, min(index * 0.001, 0.10), 0.0) for index in range(25)]
    rows.append(row(12.5, 0.40, 2.0))

    prefix = MODULE.rows_before_progress(rows, 0.40)

    assert MODULE.longest_low_speed(prefix) >= 10.0
