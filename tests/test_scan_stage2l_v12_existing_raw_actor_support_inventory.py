from types import SimpleNamespace

import numpy as np

from scripts.scan_stage2l_v12_existing_raw_actor_support_inventory import (
    CAMERA_ORDER,
    choose_per_route,
    component_flags,
    summarize_routes,
)


def fake_geometry(*, route=False, actor=False):
    route_map = np.zeros((len(CAMERA_ORDER), 2, 2), dtype=np.float32)
    actor_map = np.zeros_like(route_map)
    if route:
        route_map[1, 0, 0] = 0.75
    if actor:
        actor_map[1, 1, 1] = 1.0
    return SimpleNamespace(
        relevance=np.maximum(route_map, actor_map),
        route_corridor=route_map,
        relevant_actor_support=actor_map,
    )


def component_counts(*, actor_front=0, actor_right=0, union_right=0):
    value = {
        component: {view: 0 for view in CAMERA_ORDER}
        for component in (
            "union_relevance",
            "route_support",
            "actor_support",
            "route_only",
            "actor_only",
            "route_and_actor",
        )
    }
    value["actor_support"]["CAM_FRONT"] = actor_front
    value["actor_support"]["CAM_FRONT_RIGHT"] = actor_right
    value["union_relevance"]["CAM_FRONT_RIGHT"] = union_right
    return value


def test_component_flags_distinguish_route_actor_and_union():
    assert component_flags(fake_geometry(route=True), 1) == {
        "union_relevance": True,
        "route_support": True,
        "actor_support": False,
        "route_only": True,
        "actor_only": False,
        "route_and_actor": False,
    }
    assert component_flags(fake_geometry(actor=True), 1)["actor_only"] is True
    both = component_flags(fake_geometry(route=True, actor=True), 1)
    assert both["route_and_actor"] is True
    assert both["route_only"] is False
    assert both["actor_only"] is False


def test_choose_per_route_prefers_actor_view_diversity_before_frame_volume():
    front_only = {
        "route_index": 1,
        "job_id": 10,
        "geometry_valid_frame_count": 20,
        "aligned_frame_count": 20,
        "component_positive_frame_count": component_counts(actor_front=10),
    }
    two_views = {
        "route_index": 1,
        "job_id": 11,
        "geometry_valid_frame_count": 5,
        "aligned_frame_count": 5,
        "component_positive_frame_count": component_counts(
            actor_front=1, actor_right=1
        ),
    }
    assert choose_per_route([front_only, two_views]) == [two_views]


def test_summary_does_not_promote_union_only_to_actor_support():
    row = {
        "route_index": 2,
        "formal_split": None,
        "component_positive_frame_count": component_counts(union_right=4),
    }
    summary = summarize_routes([row])
    assert summary["independent_route_count_by_component_and_view"][
        "union_relevance"
    ]["CAM_FRONT_RIGHT"] == 1
    assert summary["independent_route_count_by_component_and_view"][
        "actor_support"
    ]["CAM_FRONT_RIGHT"] == 0
    assert "CAM_FRONT_RIGHT" in summary["zero_actor_support_views"]
