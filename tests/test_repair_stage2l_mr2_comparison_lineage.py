from scripts.repair_stage2l_mr2_comparison_lineage import (
    NEW_CONSTANT,
    NEW_GUARD,
    OLD_GUARD,
    _decision,
    _shared_history_exact,
    reconstruct_reference_trainer,
)


def test_reconstruct_reference_trainer_only_reverses_admission_guard():
    current = "prefix\n" + NEW_CONSTANT + "middle\n" + NEW_GUARD + "\nsuffix\n"
    assert reconstruct_reference_trainer(current) == (
        "prefix\nmiddle\n" + OLD_GUARD + "\nsuffix\n"
    )


def test_shared_history_requires_exact_first_40_shared_fields():
    old = [{"optimizer_step": step, "loss": float(step)} for step in range(1, 41)]
    new = [dict(row, extra=True) for row in old] + [{"optimizer_step": 41}]
    assert _shared_history_exact(old, new)
    new[12]["loss"] += 1e-8
    assert not _shared_history_exact(old, new)


def test_frozen_otherwise_rule_stops_coverage_scaling():
    result = _decision(
        {"train": 0, "dev": -2},
        {"train_a": False, "dev_a": False, "unrelated": True},
    )
    assert result["decision"] == "expanded_coverage_not_sufficient_revise_objective"
    assert "Stop route or step scaling" in result["next_action"]
