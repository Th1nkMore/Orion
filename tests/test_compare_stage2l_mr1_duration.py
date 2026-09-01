import copy
import json

from scripts.compare_stage2l_mr1_duration import compare_duration


def _split(value=0.9):
    return {
        "relevance_support": {
            "foreground_recall": value,
            "background_false_positive_rate": 0.05,
        },
        "ranking": {
            "positive_order_fraction": value,
            "minimum_attained_fraction": value,
        },
        "task_fields": {
            "overall_accuracy": value,
            "supported_class_macro_recall": value,
            "per_field_accuracy": {"stance": value},
            "zero_uq_complete_field_accuracy": value,
        },
        "deterministic_render": {
            "semantic_answer_exact_match": value,
            "semantic_field_accuracy": value,
        },
        "per_event": {
            "event1": {
                "ranking": {
                    "positive_order_fraction": value,
                    "minimum_attained_fraction": value,
                },
                "relevance_support": {
                    "foreground_recall": value,
                    "background_false_positive_rate": 0.05,
                },
            }
        },
    }


def _history(steps):
    return [
        {
            "optimizer_step": step + 1,
            "loss": 1.0 / (step + 1),
            "language_nll": 2.0 / (step + 1),
            "support_aligned_relevance": 0.8 / (step + 1),
            "background_support_hinge": 0.2 / (step + 1),
            "ranking_loss": 0.6 / (step + 1),
            "task_field_loss": 0.9 / (step + 1),
        }
        for step in range(steps)
    ]


def _checks(train_passed, dev_passed):
    return {
        "train_ranking": train_passed,
        "train_relevance": train_passed,
        "dev_ranking": dev_passed,
        "dev_relevance": dev_passed,
        "runtime_first_two_steps_finite": True,
    }


def _report(steps, train_passed, dev_passed):
    stable = {
        "aggregate_audit_sha256": "a",
        "base_orion_checkpoint_sha256": "b",
        "dataset_manifest_sha256": "c",
        "dev_audit_sha256": "d",
        "orion_config_sha256": "e",
        "records_sha256": "f",
        "reference_audit_sha256": "g",
        "train_audit_sha256": "h",
        "visual_cache_sha256_by_event": {"event1": "i"},
    }
    return {
        "schema": "orion.stage2l_mr1_report.v1",
        "optimizer_steps": steps,
        "history": _history(steps),
        "architecture": {"kind": "same"},
        "train_events": ["event1"],
        "dev_events": ["event1"],
        "before": {"train": _split(0.1), "dev": _split(0.1)},
        "after": {"train": _split(0.9), "dev": _split(0.9)},
        "checks": _checks(train_passed, dev_passed),
        "provenance": {"validated_inputs": stable},
    }


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_classifies_train_and_dev_pass(tmp_path):
    forty = _report(40, False, False)
    eighty = _report(80, True, True)
    result = compare_duration(
        _write(tmp_path / "40.json", forty),
        _write(tmp_path / "80.json", eighty),
    )
    assert result["controlled_comparison"]["valid"] is True
    assert result["decision"] == "engineering_multievent_paradigm_passes"
    assert result["locks"]["formal_stage2l_allowed"] is False


def test_classifies_train_pass_dev_stall(tmp_path):
    result = compare_duration(
        _write(tmp_path / "40.json", _report(40, False, False)),
        _write(tmp_path / "80.json", _report(80, True, False)),
    )
    assert result["decision"] == "train_passes_dev_stalls_data_coverage_bottleneck"


def test_classifies_gate_count_overfit_before_train_failure(tmp_path):
    forty = _report(40, False, True)
    eighty = _report(80, True, False)
    result = compare_duration(
        _write(tmp_path / "40.json", forty),
        _write(tmp_path / "80.json", eighty),
    )
    assert result["overfit_diagnostic"]["gate_count_overfit_established"] is True
    assert result["decision"] == "duration_overfit_stop"


def test_invalidates_nonidentical_first_40_step_replay(tmp_path):
    forty = _report(40, False, False)
    eighty = _report(80, True, True)
    eighty["history"][3]["loss"] += 0.01
    result = compare_duration(
        _write(tmp_path / "40.json", forty),
        _write(tmp_path / "80.json", eighty),
    )
    assert result["controlled_comparison"]["valid"] is False
    assert result["decision"] == "invalid_duration_only_comparison"


def test_invalidates_stable_input_drift(tmp_path):
    forty = _report(40, False, False)
    eighty = copy.deepcopy(_report(80, True, True))
    eighty["provenance"]["validated_inputs"]["records_sha256"] = "changed"
    result = compare_duration(
        _write(tmp_path / "40.json", forty),
        _write(tmp_path / "80.json", eighty),
    )
    assert result["controlled_comparison"]["valid"] is False
    assert result["controlled_comparison"]["stable_input_matches"][
        "records_sha256"
    ] is False
