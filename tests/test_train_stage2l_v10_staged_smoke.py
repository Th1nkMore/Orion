import pytest


pytest.importorskip("torch")

from scripts.train_stage2l_v10_staged_smoke import (
    _clear_gradients,
    _phase_a_gate,
    _phase_b_gate,
    _phase_c_gate,
    _protocol_checks,
)


GATES = {
    "train_min_foreground_recall": 0.9,
    "train_max_background_fpr": 0.1,
    "dev_min_foreground_recall": 0.8,
    "dev_max_background_fpr": 0.1,
    "dev_min_positive_order_fraction": 0.8,
    "dev_min_mean_attained_fraction": 0.5,
    "dev_min_target_preference_fraction": 0.6,
    "dev_min_zero_uq_target_preference_fraction": 0.5,
    "dev_min_on_path_target_preference_fraction": 0.5,
    "map_metric_retention_tolerance": 1e-8,
}


def _map_metrics(train=(0.95, 0.05), dev=(0.85, 0.05), ranking=(0.9, 0.6)):
    def split(values):
        return {
            "relevance_support": {
                "foreground_recall": values[0],
                "background_false_positive_rate": values[1],
                "foreground_mean_probability": 0.8,
                "background_mean_probability": 0.1,
                "foreground_background_probability_gap": 0.7,
            },
            "ranking": {
                "positive_order_fraction": ranking[0],
                "mean_attained_fraction": ranking[1],
            },
        }
    return {"train": split(train), "dev": split(dev)}


def test_phase_a_and_b_gates_are_strictly_sequential():
    healthy = _map_metrics()
    assert all(_phase_a_gate(healthy, GATES).values())
    assert all(_phase_b_gate(healthy, GATES).values())
    bad_map = _map_metrics(dev=(0.85, 0.2))
    assert not all(_phase_a_gate(bad_map, GATES).values())
    assert _phase_b_gate(bad_map, GATES)["phase_a_map_gates_retained"] is False


def test_phase_c_requires_language_improvement_preferences_and_map_retention():
    before = {
        "train": {"mean_target_nll": 4.0},
        "dev": {"mean_target_nll": 4.0},
    }
    after = {
        "train": {"mean_target_nll": 3.0},
        "dev": {
            "mean_target_nll": 3.5,
            "target_preference_fraction": 0.75,
            "zero_uq_target_preference_fraction": 0.5,
            "on_path_target_preference_fraction": 1.0,
        },
    }
    maps = _map_metrics()
    assert all(
        _phase_c_gate(
            before=before,
            after=after,
            map_before=maps,
            map_after=maps,
            gates=GATES,
        ).values()
    )
    changed = _map_metrics(dev=(0.84, 0.05))
    assert _phase_c_gate(
        before=before,
        after=after,
        map_before=maps,
        map_after=changed,
        gates=GATES,
    )["frozen_map_metrics_retained"] is False


def test_protocol_checks_forbid_task_leakage_and_control_paths():
    protocol = {
        "schema": "orion.stage2l_v10_accelerated_protocol.v1",
        "architecture": {
            "stage1_adapter_trainable": False,
            "uq_tokenizer_trainable": False,
            "learned_structured_field_head_used": False,
            "trajectory_training_enabled": False,
            "direct_control_training_enabled": False,
            "density_uq_used": False,
            "governor_used": False,
            "task_relevance_owner": "ORION/VLM",
        },
        "launch_locks": {"real_training_allowed": False},
    }
    _protocol_checks(protocol)
    protocol["architecture"]["stage1_adapter_trainable"] = True
    with pytest.raises(ValueError, match="contract"):
        _protocol_checks(protocol)


def test_phase_boundary_gradient_clear_discards_stale_grads():
    import torch

    module = torch.nn.Linear(2, 1)
    module(torch.ones(1, 2)).sum().backward()
    assert any(parameter.grad is not None for parameter in module.parameters())
    _clear_gradients((module,))
    assert all(parameter.grad is None for parameter in module.parameters())
