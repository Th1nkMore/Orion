import json

from scripts.collect_corruption_hardcase_wave_decisions import (
    CONDITIONS,
    ROUTES,
    collect,
    decision_filename,
)


def test_collect_requires_exact_nine_and_preserves_scope_locks(tmp_path):
    for route in ROUTES:
        for condition in CONDITIONS:
            positive = route == "180" and condition == "waterdrop_medium"
            payload = {
                "schema": "orion.corruption_hardcase_wave_pair_decision.v1",
                "route_index": route,
                "condition": condition,
                "validity": {"valid": True},
                "decision": {
                    "positive_case": positive,
                    "evidence_tier": (
                        "hard_failure_induction" if positive else "valid_negative"
                    ),
                },
                "hard_endpoint": {"degraded": positive},
                "continuous_safety_margin": {"degraded": False},
            }
            (tmp_path / decision_filename(route, condition)).write_text(
                json.dumps(payload), encoding="utf-8"
            )
    report = collect(tmp_path)
    assert report["all_pairs_present"]
    assert report["all_pairs_valid"]
    assert report["positive_pair_count"] == 1
    assert report["by_condition"]["waterdrop_medium"]["positive_routes"] == [
        "180"
    ]
    assert not report["decision_boundary"][
        "heldout_confirmation_automatically_authorized"
    ]
