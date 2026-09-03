from scripts.scan_stage2l_v12_existing_raw_coverage_inventory import (
    choose_per_route,
    strict_clean_off_reason,
)


def clean_manifest():
    return {
        "pilot_condition": "clean_off",
        "pilot_variant": "hazard",
        "pilot_route_index": "167",
        "orion_closedloop_conditioning": "none",
        "orion_closedloop_uq_mode": "none",
        "orion_enable_legacy_density_uq": "0",
        "orion_observation_uq_checkpoint": "",
        "orion_stage1_spatial_uq_checkpoint": "",
        "orion_stage2_spatial_uq_source": "disabled",
        "orion_stage2_task_checkpoint": "",
        "orion_closedloop_risk_mode": "off",
        "orion_planning_response_mode": "off",
        "orion_closedloop_corruption": "",
        "orion_native_glare_profile": "none",
        "orion_native_motion_blur_profile": "none",
        "orion_closedloop_safety_telemetry": "1",
    }


def test_strict_filter_rejects_corruption_and_accepts_clean_off():
    manifest = clean_manifest()
    assert strict_clean_off_reason(manifest) is None
    manifest["orion_closedloop_corruption"] = "front_stale"
    assert strict_clean_off_reason(manifest) == "corruption"


def test_choose_per_route_prefers_more_back_right_then_more_frames():
    base = {
        "route_index": 1,
        "job_id": 10,
        "aligned_frame_count": 5,
        "per_view_positive_frame_count": {"CAM_BACK_RIGHT": 2},
    }
    better = {
        **base,
        "job_id": 11,
        "aligned_frame_count": 3,
        "per_view_positive_frame_count": {"CAM_BACK_RIGHT": 3},
    }
    chosen = choose_per_route([base, better])
    assert chosen == [better]
