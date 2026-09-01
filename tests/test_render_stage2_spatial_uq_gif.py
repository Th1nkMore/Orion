from scripts.render_stage2_spatial_uq_gif import _critical_time


def _row(step, *, state="go", obb_ttc=None, disc_ttc=None):
    return {
        "step": step,
        "sim_time_seconds": step * 0.05,
        "planning_response": {"yield_label": {"state": state}},
        "closedloop_safety": {
            "min_obb_collision_ttc_seconds": obb_ttc,
            "min_disc_collision_ttc_seconds": disc_ttc,
        },
    }


def test_privileged_yield_response_has_priority_over_unrelated_ttc():
    rows = [
        _row(0, obb_ttc=0.1),
        _row(10, state="prepare_yield"),
        _row(20, state="hold"),
    ]
    assert _critical_time(rows) == (0.5, "first_privileged_yield_response")


def test_critical_time_falls_back_from_obb_to_disc_then_midpoint():
    rows = [_row(0, disc_ttc=0.8), _row(10, disc_ttc=0.4), _row(20)]
    assert _critical_time(rows) == (0.5, "minimum_finite_disc_ttc")
    assert _critical_time([_row(0), _row(10), _row(20)]) == (
        0.5,
        "route_midpoint_fallback",
    )
