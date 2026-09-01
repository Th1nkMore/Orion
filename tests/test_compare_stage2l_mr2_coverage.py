import json
from pathlib import Path

from scripts.compare_stage2l_mr2_coverage import compare_coverage


def _metric_block():
    return {
        "relevance_support": {
            "foreground_recall": 0.9,
            "background_false_positive_rate": 0.08,
        },
        "ranking": {
            "positive_order_fraction": 1.0,
            "minimum_attained_fraction": 0.85,
        },
        "task_fields": {
            "overall_accuracy": 0.9,
            "supported_class_macro_recall": 0.8,
            "zero_uq_complete_field_accuracy": 1.0,
            "per_field_accuracy": {"stance": 0.9},
        },
        "deterministic_render": {
            "semantic_field_accuracy": 0.9,
            "semantic_answer_exact_match": 0.9,
        },
    }


def _report(schema, train_events, dev_events, passed, mr2=False):
    block = _metric_block()
    train = json.loads(json.dumps(block))
    dev = json.loads(json.dumps(block))
    train["per_event"] = {
        event: json.loads(json.dumps(block)) for event in train_events
    }
    dev["per_event"] = {
        event: json.loads(json.dumps(block)) for event in dev_events
    }
    checks = {
        "train_a": passed,
        "train_b": passed,
        "dev_a": passed,
        "dev_b": passed,
        "trajectory_control_density_and_governor_disabled": True,
    }
    value = {
        "schema": schema,
        "optimizer_steps": 40,
        "history": [
            {
                "finite_loss": True,
                "finite_gradient_norm": True,
                "finite_gradients": True,
                "primary_event_ids": list(train_events),
            }
            for _ in range(40)
        ],
        "engineering_preexperiment_only": True,
        "formal_training_ready": False,
        "stage2p_ready": False,
        "train_events": train_events,
        "dev_events": dev_events,
        "checks": checks,
        "after": {"train": train, "dev": dev},
        "provenance": {
            "validated_inputs": {
                "trainer_sha256": "base",
                "base_mr1_trainer_sha256": "base" if mr2 else None,
                "base_orion_checkpoint_sha256": "checkpoint",
                "orion_config_sha256": "config",
            }
        },
    }
    if mr2:
        value["diagnostic_identity"] = {"not_formal_training": True}
    return value


def _protocol(schema):
    return {
        "schema": schema,
        "bounded_preexperiment": {
            "optimizer_steps": 40,
            "fresh_initialization_from_original_orion_checkpoint": True,
        },
        "losses": {"trajectory": 0.0},
        "release_gates": {"interpretation": "different prose", "x": 1},
        "launch_locks": {
            "formal_stage2l_training_allowed": False,
            "stage2p_allowed": False,
        },
    }


def _write(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_expanded_all_gate_passes_is_engineering_pass(tmp_path):
    old_train = ["t%d" % i for i in range(6)]
    old_dev = ["d%d" % i for i in range(2)]
    new_train = old_train + ["nt%d" % i for i in range(7)]
    new_dev = old_dev + ["nd%d" % i for i in range(2)]
    reference = _report(
        "orion.stage2l_mr1_multiroute_smoke.v1",
        old_train,
        old_dev,
        False,
    )
    expanded = _report(
        "orion.stage2l_mr2_expanded_coverage_smoke.v1",
        new_train,
        new_dev,
        True,
        mr2=True,
    )
    result = compare_coverage(
        reference_report_path=_write(tmp_path / "old.json", reference),
        expanded_report_path=_write(tmp_path / "new.json", expanded),
        reference_protocol_path=_write(
            tmp_path / "old_protocol.json",
            _protocol("orion.stage2l_mr1_training_protocol.v1"),
        ),
        expanded_protocol_path=_write(
            tmp_path / "new_protocol.json",
            _protocol("orion.stage2l_mr2_training_protocol.v1"),
        ),
    )
    assert result["controlled_comparison"]["valid"] is True
    assert result["decision"] == "expanded_coverage_engineering_paradigm_passes"
    assert result["locks"]["formal_stage2l_training_allowed"] is False
    assert result["locks"]["stage2p_allowed"] is False


def test_train_gain_dev_regression_stops_expansion(tmp_path):
    old_train = ["t%d" % i for i in range(6)]
    old_dev = ["d%d" % i for i in range(2)]
    new_train = old_train + ["nt%d" % i for i in range(7)]
    new_dev = old_dev + ["nd%d" % i for i in range(2)]
    reference = _report(
        "orion.stage2l_mr1_multiroute_smoke.v1",
        old_train,
        old_dev,
        False,
    )
    reference["checks"].update({"train_a": False, "train_b": False, "dev_a": True, "dev_b": True})
    expanded = _report(
        "orion.stage2l_mr2_expanded_coverage_smoke.v1",
        new_train,
        new_dev,
        False,
        mr2=True,
    )
    expanded["checks"].update({"train_a": True, "train_b": False, "dev_a": True, "dev_b": False})
    result = compare_coverage(
        reference_report_path=_write(tmp_path / "old.json", reference),
        expanded_report_path=_write(tmp_path / "new.json", expanded),
        reference_protocol_path=_write(
            tmp_path / "old_protocol.json",
            _protocol("orion.stage2l_mr1_training_protocol.v1"),
        ),
        expanded_protocol_path=_write(
            tmp_path / "new_protocol.json",
            _protocol("orion.stage2l_mr2_training_protocol.v1"),
        ),
    )
    assert result["decision"] == "expanded_coverage_overfit_or_objective_mismatch_stop"
    assert result["next_action"].startswith("Stop repeated MR scaling")


def test_gate_key_or_balanced_presentation_drift_invalidates_comparison(tmp_path):
    old_train = ["t%d" % i for i in range(6)]
    old_dev = ["d%d" % i for i in range(2)]
    new_train = old_train + ["nt%d" % i for i in range(7)]
    new_dev = old_dev + ["nd%d" % i for i in range(2)]
    reference = _report(
        "orion.stage2l_mr1_multiroute_smoke.v1", old_train, old_dev, False
    )
    expanded = _report(
        "orion.stage2l_mr2_expanded_coverage_smoke.v1",
        new_train,
        new_dev,
        False,
        mr2=True,
    )
    expanded["checks"].pop("dev_b")
    expanded["history"][0]["primary_event_ids"] = new_train[:-1]

    result = compare_coverage(
        reference_report_path=_write(tmp_path / "old.json", reference),
        expanded_report_path=_write(tmp_path / "new.json", expanded),
        reference_protocol_path=_write(
            tmp_path / "old_protocol.json",
            _protocol("orion.stage2l_mr1_training_protocol.v1"),
        ),
        expanded_protocol_path=_write(
            tmp_path / "new_protocol.json",
            _protocol("orion.stage2l_mr2_training_protocol.v1"),
        ),
    )

    checks = result["controlled_comparison"]["integrity_checks"]
    assert checks["train_dev_gate_keys_unchanged"] is False
    assert checks["every_step_covers_entire_train_split"] is False
    assert checks["same_primary_presentations_per_train_event"] is False
    assert result["decision"] == "invalid_coverage_comparison"
