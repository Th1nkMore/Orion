import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/scenario_factory/corruption_hardcase_funnel12_v1.json"
EXPECTED = {
    "route147_step223": (147, "Town02", "DynamicObjectCrossing", "dev"),
    "route151_step218": (151, "Town02", "ParkingCrossingPedestrian", "train"),
    "route152_step125": (152, "Town10HD", "HardBreakRoute", "dev"),
    "route158_step154": (158, "Town01", "HardBreakRoute", "train"),
    "route162_step277": (162, "Town03", "Accident", "dev"),
    "route164_step522": (164, "Town04", "HazardAtSideLane", "train"),
    "route168_step482": (168, "Town05", "HazardAtSideLane", "train"),
    "route180_step696": (180, "Town05", "HazardAtSideLaneTwoWays", "train"),
    "route185_step304": (185, "Town05", "StaticCutIn", "train"),
    "route194_step684": (194, "Town04", "OppositeVehicleRunningRedLight", "train"),
    "route195_step230": (195, "Town03", "OppositeVehicleRunningRedLight", "dev"),
    "route207_step924": (207, "Town02", "T_Junction", "train"),
}


def test_funnel_roles_match_existing_whole_event_splits_and_stay_locked():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    development = protocol["route_roles"]["development_screen"]
    heldout = protocol["route_roles"]["heldout_confirmation"]
    assert len(development) == 8
    assert len(heldout) == 4
    assert len({row["event_id"] for row in development + heldout}) == 12
    for row in development:
        route, town, family, split = EXPECTED[row["event_id"]]
        assert split == "train"
        assert (route, town, family) == (
            row["route_index"], row["town"], row["scenario_family"]
        )
    for row in heldout:
        route, town, family, split = EXPECTED[row["event_id"]]
        assert split == "dev"
        assert (route, town, family) == (
            row["route_index"], row["town"], row["scenario_family"]
        )
    assert protocol["event_source"]["sha256"] == (
        "c386c1ab6eeb292103d8cf4089836d74b7fc4ffe011b53e4d87deec7cb8eb5f3"
    )
    assert protocol["route_roles"]["counts"] == {
        "development": 8,
        "heldout_confirmation": 4,
        "total": 12,
        "towns_total": 6,
        "scenario_families_total": 9,
    }
    assert protocol["funnel"][1]["maximum_orion_loads"] == 1
    assert protocol["funnel"][2]["maximum_route_condition_pairs"] == 6
    assert protocol["execution_locks"]["full_12_route_closed_loop_matrix"] is False
    assert protocol["execution_locks"]["heldout_confirmation_before_development_gate"] is False
    assert protocol["execution_locks"]["formal_200_route_evaluation"] is False
    assert protocol["execution_locks"]["route203_native_glare_submission"] is False
    route168 = next(row for row in development if row["event_id"] == "route168_step482")
    assert "negative" in route168["role"]
    assert "never positive-case eligible" in route168["role"]
    assert protocol["amendments"] == [
        "configs/scenario_factory/amendments/20260831_corruption_hardcase_funnel_clean_validity_audit_v1.json",
        "configs/scenario_factory/amendments/20260831_corruption_hardcase_visual_rejection_v1.json",
        "configs/scenario_factory/amendments/20260831_corruption_hardcase_clean_render_diagnostic_prereg_v1.json",
        "configs/scenario_factory/amendments/20260831_corruption_hardcase_clean_render_diagnostic_submission_v1.json",
        "configs/scenario_factory/amendments/20260831_corruption_hardcase_clean_render_diagnostic_postprocess_recovery_v1.json",
        "configs/scenario_factory/amendments/20260831_corruption_hardcase_clean_render_diagnostic_result_v1.json",
        "configs/scenario_factory/amendments/20260831_lens_waterdrop_v2_real_mask_source_freeze_v1.json",
    ]


def test_failure_gate_does_not_accept_image_metrics_or_ade_only():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rejected = protocol["prospective_failure_induction_gate"]["not_a_positive_case"]
    assert "only ADE/FDE changes" in rejected
    assert "only image-quality metrics change" in rejected
    assert protocol["architecture_boundary"]["candidate_selection_uses_stage2_outputs"] is False
    assert protocol["architecture_boundary"]["candidate_selection_uses_learned_uq"] is False
