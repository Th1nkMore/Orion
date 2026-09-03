import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/scenario_factory/corruption_hardcase_severity_freeze_v1.json"


def test_historical_choices_are_retained_only_as_rejected_provenance():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "revoked_visual_failure_do_not_use"
    assert freeze["revocation"][
        "all_retained_conditions_below_are_historical_rejected_choices"
    ] is True
    retained = freeze["retained_conditions"]
    assert set(retained) == {"front_stale", "lens_waterdrop", "native_motion_blur"}
    assert retained["front_stale"]["delays_ms"] == [200, 400]
    assert retained["lens_waterdrop"]["severity"] == 2
    assert retained["native_motion_blur"]["profile"] == "medium"
    assert "native_glare" in freeze["non_screen_conditions"]


def test_visual_rejection_keeps_every_experimental_stage_locked():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    locks = freeze["execution_locks"]
    assert locks["offline_development_screen_unlocked"] is False
    assert locks["closed_loop_route_jobs_unlocked"] is False
    assert locks["heldout_confirmation_unlocked"] is False
    assert locks["stage2p_unlocked"] is False
    assert locks["formal_200_route_evaluation_unlocked"] is False
    assert locks["replacement_visual_preview_required"] is True
    assert freeze["source_result"]["orion_loaded"] is False
