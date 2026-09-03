import pytest


torch = pytest.importorskip("torch")

from uq_estimator.stage2l_gradient_routed_objective import (
    dataset_frequency_balanced_stance_loss,
)


def test_dataset_frequency_weights_equalize_epoch_class_mass():
    counts = {"maintain": 10, "caution": 2, "prepare_to_yield": 3}
    total = sum(counts.values())
    weights = {name: total / (3 * count) for name, count in counts.items()}
    masses = {name: counts[name] * weights[name] for name in counts}
    assert len({round(value, 6) for value in masses.values()}) == 1


def test_dataset_frequency_balanced_loss_has_gradients():
    logits = {
        "zero_uq": torch.randn(2, 3, requires_grad=True),
        "off_path_uq": torch.randn(2, 3, requires_grad=True),
        "on_path_uq": torch.randn(2, 3, requires_grad=True),
    }
    loss = dataset_frequency_balanced_stance_loss(
        logits,
        {
            "zero_uq": ["maintain", "maintain"],
            "off_path_uq": ["maintain", "maintain"],
            "on_path_uq": ["caution", "prepare_to_yield"],
        },
        {"maintain": 10, "caution": 2, "prepare_to_yield": 3},
        supervised_variants=("zero_uq", "off_path_uq", "on_path_uq"),
    )
    loss.backward()
    assert all(value.grad is not None for value in logits.values())


def test_loss_uses_fixed_sample_mean_not_current_weight_sum():
    logits = {
        "zero_uq": torch.tensor([[1.0, 0.0, 0.0]]),
        "off_path_uq": torch.tensor([[0.0, 1.0, 0.0]]),
        "on_path_uq": torch.tensor([[0.0, 0.0, 1.0]]),
    }
    targets = {
        "zero_uq": "maintain",
        "off_path_uq": "maintain",
        "on_path_uq": "caution",
    }
    counts = {"maintain": 10, "caution": 2, "prepare_to_yield": 3}
    actual = dataset_frequency_balanced_stance_loss(
        logits,
        targets,
        counts,
        supervised_variants=("zero_uq", "off_path_uq", "on_path_uq"),
    )
    joined = torch.cat(tuple(logits.values()))
    encoded = torch.tensor([0, 0, 1])
    weights = torch.tensor([0.5, 2.5, 5.0 / 3.0])
    expected = (
        torch.nn.functional.cross_entropy(joined, encoded, reduction="none")
        * weights[encoded]
    ).mean()
    assert torch.allclose(actual, expected)


def test_dataset_frequency_balanced_loss_requires_all_classes():
    logits = {name: torch.randn(1, 3) for name in ("z", "o", "n")}
    with pytest.raises(ValueError, match="every stance"):
        dataset_frequency_balanced_stance_loss(
            logits,
            {"z": "maintain", "o": "maintain", "n": "caution"},
            {"maintain": 2, "caution": 1, "prepare_to_yield": 0},
            supervised_variants=("z", "o", "n"),
        )
