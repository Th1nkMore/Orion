from scripts.preflight_stage2l_v12_route196_coverage_reuse import select_fixed_frames


def test_select_fixed_frames_reuses_frozen_offsets_without_duplicates():
    trace = [
        {"step": step, "sim_time_seconds": step * 0.05}
        for step in range(250, 371)
    ]
    result = select_fixed_frames(
        trace=trace,
        available_frames=list(range(25, 38)),
        critical_step=313,
        offsets_seconds=[-2.0, -1.0, 0.0, 1.0, 2.0],
    )
    frames = [row["selected_saved_frame_index"] for row in result]
    assert frames == [27, 29, 31, 33, 35]
    assert len(set(frames)) == 5


def test_select_fixed_frames_rejects_offsets_without_center():
    trace = [{"step": step, "sim_time_seconds": step * 0.05} for step in range(100)]
    try:
        select_fixed_frames(
            trace=trace,
            available_frames=list(range(10)),
            critical_step=50,
            offsets_seconds=[-1.0, 1.0],
        )
    except ValueError as error:
        assert "include zero" in str(error)
    else:
        raise AssertionError("missing center offset should fail")
