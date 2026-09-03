from __future__ import annotations

import pytest
import torch

from scripts.train_stage2l_v122_vertical_slice_semantic_smoke import (
    _hard_factorization_checks,
    derived_union_logit,
    validate_v121_terminal_validation,
)


def test_derived_union_logit_matches_probability_max() -> None:
    logits = torch.zeros((1, 2, 1, 1, 2), dtype=torch.float32)
    logits[0, 0, 0, 0] = torch.tensor([-2.0, 0.5])
    logits[0, 1, 0, 0] = torch.tensor([1.0, -0.25])
    union = derived_union_logit(logits)
    expected = logits.sigmoid().amax(dim=1)
    assert torch.equal(union, logits.amax(dim=1))
    assert torch.allclose(union.sigmoid(), expected, atol=0.0, rtol=0.0)


def test_derived_union_logit_rejects_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        derived_union_logit(torch.zeros(1, 6, 10, 10))
    value = torch.zeros(1, 2, 6, 10, 10)
    value[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        derived_union_logit(value)


def test_hard_factorization_checks_ignore_soft_quality_failures() -> None:
    invariant = {
        "shared_r_bitwise_exact": True,
        "zero_u_and_k_exact": True,
        "on_off_magnitude_matched": True,
        "on_off_support_spatially_distinct": True,
        "on_over_off_fraction": False,
        "off_path_low_risk_fraction": False,
    }
    factorization = {
        "train": {"release_checks": dict(invariant)},
        "dev": {"release_checks": dict(invariant)},
    }
    checks = _hard_factorization_checks(factorization)
    assert all(checks.values())
    factorization["dev"]["release_checks"]["zero_u_and_k_exact"] = False
    checks = _hard_factorization_checks(factorization)
    assert checks["dev_zero_u_and_k_exact"] is False
    assert not all(checks.values())


def test_terminal_validation_uses_authoritative_decision_field() -> None:
    value = {
        "schema": "orion.stage2l_v12_1_factorized_r_validation.v1",
        "status": "validated_failed_gate",
        "integrity_valid": True,
        "decision": "held_out_factorized_r_transfer_failed",
        "optimizer_steps": 40,
    }
    assert (
        validate_v121_terminal_validation(value)
        == "held_out_factorized_r_transfer_failed"
    )
    broken = dict(value)
    broken.pop("decision")
    broken["terminal_decision"] = "held_out_factorized_r_transfer_failed"
    with pytest.raises(ValueError, match="terminal validation differs"):
        validate_v121_terminal_validation(broken)
