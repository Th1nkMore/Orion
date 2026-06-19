import torch
import torch.nn as nn

from uq_estimator.training import (
    count_parameter_groups,
    freeze_for_uq_token_training,
    load_uq_token_weights,
    low_uq_consistency_loss,
)
from scripts.train_uq_token import trajectory_metrics


class DummyAdaptationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.uq_token_projector = nn.Linear(4, 4)
        self.uq_grounding_head = nn.Linear(4, 1)
        self.llm = nn.Module()
        self.llm.lora_A = nn.Linear(4, 2, bias=False)
        self.llm.lora_B = nn.Linear(2, 4, bias=False)


def test_freeze_for_uq_token_training():
    model = DummyAdaptationModel()
    groups = freeze_for_uq_token_training(model)
    counts = count_parameter_groups(groups)
    assert counts["projector"] == 20
    assert counts["grounding"] == 5
    assert counts["lora"] == 16
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.uq_token_projector.parameters()
    )


def test_load_uq_token_weights():
    source = DummyAdaptationModel()
    target = DummyAdaptationModel()
    with torch.no_grad():
        source.uq_token_projector.weight.fill_(2.0)
        source.llm.lora_A.weight.fill_(3.0)
    payload = {
        "model_state": {
            name: tensor.clone()
            for name, tensor in source.state_dict().items()
            if name.startswith("uq_token_projector.")
            or name.startswith("uq_grounding_head.")
            or "lora_" in name
        }
    }
    loaded = load_uq_token_weights(target, payload)
    assert loaded == 6
    torch.testing.assert_close(
        target.uq_token_projector.weight,
        source.uq_token_projector.weight,
    )
    torch.testing.assert_close(target.llm.lora_A.weight, source.llm.lora_A.weight)


def test_low_uq_consistency_loss_weights_by_score():
    baseline = torch.zeros(2, 4)
    conditioned = torch.ones(2, 4)
    score = torch.tensor([[0.0], [1.0]])
    loss = low_uq_consistency_loss(conditioned, baseline, score)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_low_uq_consistency_loss_has_conditioned_gradient_only():
    baseline = torch.randn(2, 4, requires_grad=True)
    conditioned = torch.randn(2, 4, requires_grad=True)
    score = torch.full((2, 1), 0.25)
    loss = low_uq_consistency_loss(conditioned, baseline, score)
    loss.backward()
    assert conditioned.grad is not None
    assert baseline.grad is None


def test_trajectory_metrics_use_cumulative_positions_and_masks():
    prediction = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [5.0, 0.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    metrics = trajectory_metrics([prediction], [target], [mask])
    assert metrics["count"] == 1
    assert metrics["ade"] == 1.5
    assert metrics["fde"] == 2.0
