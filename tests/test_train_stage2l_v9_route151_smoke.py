from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from scripts.train_stage2l_v9_route151_smoke import (
    _field_loss,
    _field_metrics,
)
from uq_estimator.stage2l_structured_field_head import TASK_FIELD_VOCABULARIES


ROOT = Path(__file__).resolve().parents[1]


def _probabilities(targets):
    result = {}
    for field, vocabulary in TASK_FIELD_VOCABULARIES.items():
        values = torch.full((1, len(vocabulary)), 0.01)
        values[0, vocabulary.index(targets[field])] = 0.96
        result[field] = values / values.sum(dim=-1, keepdim=True)
    return result


def _targets():
    return {
        "zero_uq": {
            "relevance_level": "not_applicable",
            "risk_level": "none",
            "risk_view": "none",
            "risk_region": "none",
            "stance": "maintain",
        },
        "off_path_uq": {
            "relevance_level": "low",
            "risk_level": "none",
            "risk_view": "none",
            "risk_region": "none",
            "stance": "maintain",
        },
        "on_path_uq": {
            "relevance_level": "high",
            "risk_level": "medium",
            "risk_view": "CAM_FRONT",
            "risk_region": "lower_center",
            "stance": "prepare_to_yield",
        },
    }


def test_field_metrics_make_zero_and_off_path_separately_visible():
    targets = _targets()
    entries = [
        ("g0", variant, _probabilities(fields), fields)
        for variant, fields in targets.items()
    ]
    metrics = _field_metrics(entries)
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["zero_uq_complete_field_accuracy"] == 1.0
    assert metrics["supported_class_recall"]["relevance_level"] == {
        "not_applicable": 1.0,
        "low": 1.0,
        "high": 1.0,
    }


def test_partial_field_loss_uses_dataset_counts_and_backpropagates():
    targets = _targets()
    conditioned = {}
    for variant in targets:
        conditioned[variant] = {
            "field_logits": {
                field: torch.randn(
                    1, len(vocabulary), requires_grad=True
                )
                for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
            }
        }
    counts = {
        field: {
            value: sum(row[field] == value for row in targets.values())
            for value in vocabulary
        }
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
    }
    loss, per_field = _field_loss(conditioned, targets, counts)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(per_field) == set(TASK_FIELD_VOCABULARIES)
    assert all(
        conditioned[variant]["field_logits"][field].grad is not None
        for variant in targets
        for field in TASK_FIELD_VOCABULARIES
    )


def test_free_language_nll_is_diagnostic_not_release_evidence():
    source = (
        ROOT / "scripts/train_stage2l_v9_route151_smoke.py"
    ).read_text(encoding="utf-8")
    checks_start = source.index("    checks = {")
    diagnostics_start = source.index("    diagnostics = {", checks_start)
    checks_block = source[checks_start:diagnostics_start]
    diagnostics_block = source[diagnostics_start:source.index(
        "    passed = all(checks.values())", diagnostics_start
    )]
    assert "auxiliary_language_nll_decreases" not in checks_block
    assert "auxiliary_language_nll_decreases" in diagnostics_block
    assert '"free_generation_is_release_evidence": False' in diagnostics_block
