from pathlib import Path

import pytest


pytest.importorskip("torch")

from scripts.train_stage2l_mr1_smoke import (
    ALLOWED_BOUNDED_OPTIMIZER_STEPS,
    EventBalancedSampler,
)


ROOT = Path(__file__).resolve().parents[1]


def _event_groups():
    return {
        "event_%02d" % event: tuple(
            "event_%02d_group_%02d" % (event, group)
            for group in range(3 + (event % 3))
        )
        for event in range(6)
    }


def test_event_balanced_sampler_uses_one_group_per_event_every_step():
    event_groups = _event_groups()
    sampler = EventBalancedSampler(event_groups, seed=17)
    group_to_event = {
        group: event for event, groups in event_groups.items() for group in groups
    }
    for _ in range(80):
        selected = sampler.next()
        assert len(selected) == 6
        assert len(set(selected)) == 6
        assert {group_to_event[group] for group in selected} == set(event_groups)


def test_event_balanced_sampler_is_deterministic():
    left = EventBalancedSampler(_event_groups(), seed=20260831)
    right = EventBalancedSampler(_event_groups(), seed=20260831)
    assert [left.next() for _ in range(20)] == [right.next() for _ in range(20)]


def test_event_balanced_sampler_fails_closed_on_wrong_event_count():
    with pytest.raises(ValueError, match="exactly six"):
        EventBalancedSampler({"only": ("g0",)}, seed=1)


def test_mr1_only_allows_the_two_preregistered_bounded_durations():
    assert ALLOWED_BOUNDED_OPTIMIZER_STEPS == (40, 80)


def test_mr1_single_answer_nll_does_not_inherit_v7_negative_requirement():
    source = (ROOT / "scripts/train_stage2l_mr1_smoke.py").read_text(
        encoding="utf-8"
    )
    function = source[
        source.index("def _answer_nlls_mr1(") : source.index(
            "\n\nclass EventBalancedSampler"
        )
    ]
    assert "if not answers:" in function
    assert "len(answers) < 2" not in function
    assert "_candidate_answer_nlls_v7" not in source


def test_dev_and_language_diagnostics_are_not_smuggled_into_release_gates():
    source = (ROOT / "scripts/train_stage2l_mr1_smoke.py").read_text(
        encoding="utf-8"
    )
    checks_start = source.index("    checks = {")
    diagnostics_start = source.index("    diagnostics = {", checks_start)
    checks = source[checks_start:diagnostics_start]
    diagnostics = source[
        diagnostics_start : source.index(
            "    passed = all(checks.values())", diagnostics_start
        )
    ]
    assert "auxiliary_language_nll_decreases" not in checks
    assert "auxiliary_language_nll_decreases" in diagnostics
    assert '"free_generation_is_release_evidence": False' in diagnostics
    assert '"dev_labels_never_enter_optimizer"' in checks
