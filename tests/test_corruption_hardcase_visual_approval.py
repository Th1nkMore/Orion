import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "configs/scenario_factory/corruption_hardcase_visual_approval_gate_v2.json"
CONTRACT = ROOT / "uq_estimator/corruption_visual_approval.py"
SPEC = importlib.util.spec_from_file_location(
    "orion_corruption_visual_approval_contract_test", CONTRACT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
VisualApprovalError = MODULE.VisualApprovalError
verify_visual_approval = MODULE.verify_visual_approval


@pytest.mark.parametrize(
    "family,approved,unapproved",
    [
        ("front_stale", "delay_ms:200", "delay_ms:400"),
        ("lens_waterdrop_paired_template", "profile:medium", "profile:light"),
        ("native_motion_blur", "profile:medium", "profile:light"),
    ],
)
def test_explicitly_approved_exact_condition_passes_and_others_fail(
    family, approved, unapproved
):
    record = verify_visual_approval(
        gate_path=GATE,
        repository_root=ROOT,
        family=family,
        condition=approved,
        require_approved=True,
    )
    assert record.decision_status == "approved"
    assert record.approved_conditions == (approved,)
    assert record.human_authorization["source"] == "explicit_user_message"
    with pytest.raises(VisualApprovalError, match="is not approved"):
        verify_visual_approval(
            gate_path=GATE,
            repository_root=ROOT,
            family=family,
            condition=unapproved,
            require_approved=True,
        )


def test_gate_unlock_is_exact_and_failed_waterdrops_remain_retired():
    value = json.loads(GATE.read_text())
    assert value["authority"]["orion_screen_unlocked_families"] == [
        "front_stale",
        "lens_waterdrop_paired_template",
        "native_motion_blur",
    ]
    assert value["execution_locks"]["orion_closed_loop_screen"] is True
    assert value["execution_locks"]["severity_freeze"] is True
    for locked in (
        "orion_offline_screen", "heldout_confirmation", "stage2p",
        "formal_200_route_evaluation",
    ):
        assert value["execution_locks"][locked] is False
    retired = {row["path"] for row in value["retired_implementations"]}
    assert "uq_estimator/lens_waterdrop.py" in retired
    assert "uq_estimator/lens_waterdrop_v2.py" in retired
    assert value["families"]["lens_waterdrop_paired_template"]["implementation"][
        "path"
    ] not in retired


def test_hash_change_fails_closed(tmp_path):
    gate = json.loads(GATE.read_text())
    gate["families"]["front_stale"]["implementation"]["sha256"] = "0" * 64
    changed = tmp_path / "gate.json"
    changed.write_text(json.dumps(gate))
    with pytest.raises(VisualApprovalError, match="hash differs"):
        verify_visual_approval(
            gate_path=changed,
            repository_root=ROOT,
            family="front_stale",
            require_approved=False,
        )
