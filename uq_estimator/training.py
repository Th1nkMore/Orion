"""Training utilities for UQ-token adaptation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_for_uq_token_training(
    model: nn.Module,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Freeze the model and enable only UQ projector and LLM LoRA parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    groups: dict[str, list[tuple[str, nn.Parameter]]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if name.startswith("uq_token_projector."):
            group = "projector"
        elif "lora_" in name:
            group = "lora"
        else:
            continue
        parameter.requires_grad = True
        groups[group].append((name, parameter))

    if not groups["projector"]:
        raise RuntimeError("No UQ token projector parameters were found")
    if not groups["lora"]:
        raise RuntimeError("No LLM LoRA parameters were found")

    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("uq_token_projector.")
        and "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:10]}")
    return dict(groups)


def count_parameter_groups(
    groups: dict[str, list[tuple[str, nn.Parameter]]],
) -> dict[str, int]:
    return {
        group: sum(parameter.numel() for _, parameter in parameters)
        for group, parameters in groups.items()
    }


def load_uq_token_weights(
    model: nn.Module,
    checkpoint: str | Path | dict,
) -> int:
    """Load projector and LoRA weights from a UQ-token adaptation checkpoint."""
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        payload = checkpoint
    state = payload.get("model_state", payload)
    adaptation_state = {
        name: tensor
        for name, tensor in state.items()
        if name.startswith("uq_token_projector.") or "lora_" in name
    }
    if not adaptation_state:
        raise KeyError("Checkpoint contains no UQ projector or LoRA weights")
    _, unexpected = model.load_state_dict(adaptation_state, strict=False)
    unexpected_adaptation = [
        name for name in unexpected
        if name.startswith("uq_token_projector.") or "lora_" in name
    ]
    if unexpected_adaptation:
        raise RuntimeError(
            f"Unexpected UQ-token checkpoint keys: {unexpected_adaptation[:10]}"
        )
    return len(adaptation_state)


def low_uq_consistency_loss(
    conditioned_feature: torch.Tensor,
    baseline_feature: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """Keep low-UQ waypoint representations close to the no-token baseline."""
    if conditioned_feature.shape != baseline_feature.shape:
        raise ValueError(
            "conditioned_feature and baseline_feature must have identical shapes"
        )
    if score.ndim == 1:
        score = score.unsqueeze(-1)
    if score.ndim != 2 or score.shape[-1] != 1:
        raise ValueError("score must have shape [B, 1]")
    if score.shape[0] != conditioned_feature.shape[0]:
        raise ValueError("score and feature batch sizes must match")

    per_sample = F.mse_loss(
        conditioned_feature.float(),
        baseline_feature.detach().float(),
        reduction="none",
    ).flatten(start_dim=1).mean(dim=1)
    weight = (1.0 - score.detach().float().squeeze(-1)).clamp(0.0, 1.0)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)
