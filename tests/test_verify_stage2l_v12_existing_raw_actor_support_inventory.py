import copy

import pytest

from scripts.scan_stage2l_v12_existing_raw_actor_support_inventory import (
    CAMERA_ORDER,
    COMPONENTS,
    SCHEMA,
    summarize_routes,
)
from scripts.verify_stage2l_v12_existing_raw_actor_support_inventory import (
    verify_inventory,
)


def empty_component_values():
    counts = {
        component: {view: 0 for view in CAMERA_ORDER} for component in COMPONENTS
    }
    frames = {
        component: {view: [] for view in CAMERA_ORDER} for component in COMPONENTS
    }
    return counts, frames


def valid_inventory():
    counts, frames = empty_component_values()
    for component in ("union_relevance", "route_support", "route_only"):
        counts[component]["CAM_FRONT"] = 1
        frames[component]["CAM_FRONT"] = [3]
    row = {
        "route_index": 1,
        "formal_split": "train",
        "component_positive_frame_count": counts,
        "component_positive_frames": frames,
    }
    return {
        "schema": SCHEMA,
        "inventory": {"deduplicated_route_count": 1},
        "all_geometry_runs": [row],
        "deduplicated_routes": [row],
        "summary": summarize_routes([row]),
        "actor_support_route_indices_by_view": {view: [] for view in CAMERA_ORDER},
    }


def test_verify_inventory_accepts_exact_partition():
    result = verify_inventory(valid_inventory())
    assert result["deduplicated_route_count"] == 1
    assert result["zero_actor_support_views"] == list(CAMERA_ORDER)


def test_verify_inventory_rejects_union_that_does_not_match_components():
    value = copy.deepcopy(valid_inventory())
    value["all_geometry_runs"][0]["component_positive_frames"][
        "union_relevance"
    ]["CAM_FRONT"] = []
    value["all_geometry_runs"][0]["component_positive_frame_count"][
        "union_relevance"
    ]["CAM_FRONT"] = 0
    with pytest.raises(ValueError, match="union frames differ"):
        verify_inventory(value)
