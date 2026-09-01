import json
from pathlib import Path

from PIL import Image

from scripts.render_closedloop_front_bev_gifs import (
    choose_time_window,
    critical_row,
    render_channel,
)


def row(step, *, active=False, ttc=None):
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "route_progress": step / 100.0,
        "speed": 2.0,
        "corruption_active": active,
        "closedloop_safety": {
            "available": True,
            "min_obb_collision_ttc_seconds": ttc,
            "min_obb_separating_axis_gap_m": 1.0,
            "critical_actor": {"category": "walker", "actor_id": 7},
        },
    }


def test_event_window_takes_priority_over_critical_row():
    rows = [row(step, active=20 <= step <= 30, ttc=0.1 if step == 5 else None)
            for step in range(41)]
    start, end, basis = choose_time_window(
        rows,
        pre_seconds=0.25,
        post_seconds=0.5,
        full_route=False,
        auto_critical_window=True,
    )
    assert (start, end, basis) == (0.75, 2.0, "corruption_event_window")


def test_critical_row_uses_minimum_finite_ttc_then_earliest_step():
    rows = [row(0), row(1, ttc=0.5), row(2, ttc=0.2), row(3, ttc=0.2)]
    assert critical_row(rows)["step"] == 2


def test_explicit_center_time_overrides_automatic_critical_window():
    rows = [row(step, ttc=0.1 if step == 5 else None) for step in range(101)]
    start, end, basis = choose_time_window(
        rows,
        pre_seconds=1.0,
        post_seconds=2.0,
        full_route=False,
        auto_critical_window=True,
        center_time_seconds=3.0,
    )
    assert (start, end, basis) == (2.0, 5.0, "explicit_center_time")


def test_render_channel_writes_animated_gif(tmp_path: Path):
    frame_dir = tmp_path / "rgb_front"
    frame_dir.mkdir()
    rows = [row(step, active=10 <= step <= 20, ttc=1.0) for step in range(31)]
    rows_by_step = {item["step"]: item for item in rows}
    for frame in range(4):
        Image.new("RGB", (80, 48), (frame * 40, 20, 30)).save(
            frame_dir / f"{frame:04d}.png"
        )
    output = tmp_path / "front.gif"
    report = render_channel(
        frame_dir,
        output,
        label="front",
        rows_by_step=rows_by_step,
        start_time=0.0,
        end_time=1.5,
        fps=2.0,
        max_size=(80, 48),
    )
    assert output.is_file()
    assert report["frame_count"] == 4
    with Image.open(output) as image:
        assert image.n_frames == 4
