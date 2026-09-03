from scripts.scan_stage2l_v12_train_back_right_support import rank_candidates


def _row(route, frames, locked=False):
    return {
        "route_index": route,
        "per_view_positive_frame_count": {"CAM_BACK_RIGHT": frames},
        "frozen_geometry_reselection_locked": locked,
    }


def test_rank_candidates_requires_three_frames_and_excludes_locked_event():
    rows = [_row(10, 2), _row(11, 3), _row(12, 5, True), _row(13, 4)]
    assert [row["route_index"] for row in rank_candidates(rows, minimum_positive_frames=3)] == [13, 11]
