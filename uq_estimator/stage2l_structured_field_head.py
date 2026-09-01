"""VLM-owned categorical task fields for interpretable Stage2-L output.

The head consumes only the task-risk bridge produced after the VLM predicts
dense relevance R and combines it with frozen Stage1 U.  It does not change or
replace the task-agnostic Stage1 adapter.  Field predictions are supervised
directly; optional language conditioning receives detached probabilities so
free-text loss cannot redefine task relevance, risk, or stance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uq_estimator.stage2l_qa_contract_v5 import (
    FIELD_VOCABULARIES,
    TASK_FIELD_KEYS,
)


SCHEMA = "orion.stage2l_vlm_task_semantic_fields.v1"
RAW_TASK_RISK_FEATURE_COUNT = 6
TASK_FIELD_VOCABULARIES = {
    field: (
        FIELD_VOCABULARIES["uq_view"]
        if field == "risk_view"
        else FIELD_VOCABULARIES["uq_region"]
        if field == "risk_region"
        else FIELD_VOCABULARIES[field]
    )
    for field in TASK_FIELD_KEYS
}


@dataclass(frozen=True)
class StructuredTaskFieldOutput:
    token: torch.Tensor
    logits: Mapping[str, torch.Tensor]
    probabilities: Mapping[str, torch.Tensor]
    predicted_indices: Mapping[str, torch.Tensor]
    raw_observation_global_features: torch.Tensor
    raw_task_risk_global_features: torch.Tensor
    probabilities_detached_for_language: bool
    schema: str = SCHEMA


class VLMTaskSemanticFieldHead(nn.Module):
    """Predict task fields from VLM-conditioned K context and magnitude."""

    def __init__(
        self,
        model_dim: int = 4096,
        hidden_dim: int = 256,
        magnitude_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if min(model_dim, hidden_dim) <= 0 or magnitude_scale <= 0.0:
            raise ValueError("field-head dimensions and scales must be positive")
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.magnitude_scale = float(magnitude_scale)
        self.observation_magnitude_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.observation_location_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.task_risk_magnitude_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.task_risk_location_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        fused_dim = hidden_dim * 5
        self.classifiers = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(vocabulary)),
            )
            for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
        })
        self.field_embeddings = nn.ModuleDict({
            field: nn.Embedding(len(vocabulary), model_dim)
            for field, vocabulary in TASK_FIELD_VOCABULARIES.items()
        })
        self.token_type_embedding = nn.Parameter(torch.zeros(model_dim))
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        task_risk_bridge_tokens: torch.Tensor,
        raw_observation_global_features: torch.Tensor,
        raw_task_risk_global_features: torch.Tensor,
        *,
        detach_probabilities_for_language: bool = True,
    ) -> StructuredTaskFieldOutput:
        if task_risk_bridge_tokens.ndim != 3:
            raise ValueError("task-risk bridge tokens must have shape [B,N,D]")
        batch, token_count, model_dim = task_risk_bridge_tokens.shape
        if token_count < 1 or model_dim != self.model_dim:
            raise ValueError("task-risk bridge token shape is incompatible")
        if raw_observation_global_features.shape != (
            batch, RAW_TASK_RISK_FEATURE_COUNT
        ):
            raise ValueError("raw observation summary must have shape [B,6]")
        if raw_task_risk_global_features.shape != (
            batch, RAW_TASK_RISK_FEATURE_COUNT
        ):
            raise ValueError("raw task-risk summary must have shape [B,6]")
        if not (
            bool(torch.isfinite(task_risk_bridge_tokens).all())
            and bool(torch.isfinite(raw_observation_global_features).all())
            and bool(torch.isfinite(raw_task_risk_global_features).all())
        ):
            raise ValueError("structured field inputs must be finite")
        if bool((raw_observation_global_features[:, :3] < 0.0).any()):
            raise ValueError("observation magnitude summaries must be non-negative")
        if bool((raw_task_risk_global_features[:, :3] < 0.0).any()):
            raise ValueError("task-risk magnitude summaries must be non-negative")

        observation_magnitude = torch.log1p(
            self.magnitude_scale * raw_observation_global_features[:, :3]
        )
        task_risk_magnitude = torch.log1p(
            self.magnitude_scale * raw_task_risk_global_features[:, :3]
        )
        fused = torch.cat(
            (
                self.observation_magnitude_projection(observation_magnitude),
                self.observation_location_projection(
                    raw_observation_global_features[:, 3:]
                ),
                self.task_risk_magnitude_projection(task_risk_magnitude),
                self.task_risk_location_projection(
                    raw_task_risk_global_features[:, 3:]
                ),
                self.context_projection(task_risk_bridge_tokens[:, -1]),
            ),
            dim=-1,
        )
        logits = {
            field: classifier(fused)
            for field, classifier in self.classifiers.items()
        }
        probabilities = {
            field: F.softmax(value, dim=-1)
            for field, value in logits.items()
        }
        predicted = {
            field: value.argmax(dim=-1)
            for field, value in logits.items()
        }
        language_probabilities = {
            field: (
                value.detach()
                if detach_probabilities_for_language else value
            )
            for field, value in probabilities.items()
        }
        field_tokens = [
            language_probabilities[field]
            @ self.field_embeddings[field].weight
            for field in TASK_FIELD_VOCABULARIES
        ]
        token = self.output_norm(
            torch.stack(field_tokens, dim=1).mean(dim=1)
            + self.token_type_embedding[None]
        ).unsqueeze(1)
        return StructuredTaskFieldOutput(
            token=token,
            logits=logits,
            probabilities=probabilities,
            predicted_indices=predicted,
            raw_observation_global_features=raw_observation_global_features,
            raw_task_risk_global_features=raw_task_risk_global_features,
            probabilities_detached_for_language=(
                detach_probabilities_for_language
            ),
        )


def encode_task_field_targets(
    targets: Sequence[Mapping[str, str]], *, device: torch.device = None
) -> Dict[str, torch.Tensor]:
    if not targets:
        raise ValueError("task-field targets cannot be empty")
    encoded: Dict[str, torch.Tensor] = {}
    for field, vocabulary in TASK_FIELD_VOCABULARIES.items():
        index = {value: position for position, value in enumerate(vocabulary)}
        values = []
        for target in targets:
            if set(target) != set(TASK_FIELD_VOCABULARIES):
                raise ValueError("task-field target keys are incomplete")
            value = str(target[field])
            if value not in index:
                raise ValueError("unsupported task-field target: %s=%s" % (field, value))
            values.append(index[value])
        encoded[field] = torch.tensor(values, dtype=torch.long, device=device)
    return encoded


def dataset_frequency_balanced_field_loss(
    logits: Mapping[str, torch.Tensor],
    target_indices: Mapping[str, torch.Tensor],
    class_counts: Mapping[str, Mapping[str, int]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if set(logits) != set(TASK_FIELD_VOCABULARIES):
        raise ValueError("task-field logits are incomplete")
    if set(target_indices) != set(TASK_FIELD_VOCABULARIES):
        raise ValueError("task-field target indices are incomplete")
    if set(class_counts) != set(TASK_FIELD_VOCABULARIES):
        raise ValueError("task-field class counts are incomplete")
    return dataset_frequency_balanced_partial_field_loss(
        logits, target_indices, class_counts
    )


def dataset_frequency_balanced_partial_field_loss(
    logits: Mapping[str, torch.Tensor],
    target_indices: Mapping[str, torch.Tensor],
    class_counts: Mapping[str, Mapping[str, int]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    active_fields = tuple(target_indices)
    if not active_fields:
        raise ValueError("at least one task field must be active")
    if any(field not in TASK_FIELD_VOCABULARIES for field in active_fields):
        raise ValueError("unsupported active task field")
    if not set(active_fields).issubset(logits):
        raise ValueError("active task-field logits are incomplete")
    if set(class_counts) != set(active_fields):
        raise ValueError("active task-field class counts differ")
    losses: Dict[str, torch.Tensor] = {}
    for field in active_fields:
        vocabulary = TASK_FIELD_VOCABULARIES[field]
        field_logits = logits[field]
        targets = target_indices[field]
        if field_logits.ndim != 2 or field_logits.shape[1] != len(vocabulary):
            raise ValueError("task-field logit shape is invalid: %s" % field)
        if targets.shape != (field_logits.shape[0],):
            raise ValueError("task-field target shape is invalid: %s" % field)
        counts = class_counts[field]
        if set(counts) != set(vocabulary):
            raise ValueError("task-field class-count vocabulary differs: %s" % field)
        total = sum(int(value) for value in counts.values())
        active = sum(int(value) > 0 for value in counts.values())
        if total <= 0 or active <= 0:
            raise ValueError("task-field class counts are empty: %s" % field)
        weights = torch.zeros(
            len(vocabulary), dtype=field_logits.dtype, device=field_logits.device
        )
        for index, value in enumerate(vocabulary):
            count = int(counts[value])
            if count > 0:
                weights[index] = float(total) / float(active * count)
        sample_losses = F.cross_entropy(
            field_logits, targets, reduction="none"
        )
        losses[field] = (sample_losses * weights[targets]).mean()
    return torch.stack(tuple(losses.values())).mean(), losses


def decode_task_field_predictions(
    predicted_indices: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, str], ...]:
    if set(predicted_indices) != set(TASK_FIELD_VOCABULARIES):
        raise ValueError("task-field predictions are incomplete")
    batch_sizes = {
        int(value.shape[0])
        for value in predicted_indices.values()
        if value.ndim == 1
    }
    if len(batch_sizes) != 1 or any(
        value.ndim != 1 for value in predicted_indices.values()
    ):
        raise ValueError("task-field predictions must be one-dimensional batches")
    batch = batch_sizes.pop()
    output = []
    for row in range(batch):
        fields = {}
        for field, vocabulary in TASK_FIELD_VOCABULARIES.items():
            index = int(predicted_indices[field][row].item())
            if index < 0 or index >= len(vocabulary):
                raise ValueError("task-field prediction index is out of range")
            fields[field] = vocabulary[index]
        output.append(fields)
    return tuple(output)


__all__ = [
    "RAW_TASK_RISK_FEATURE_COUNT",
    "SCHEMA",
    "StructuredTaskFieldOutput",
    "TASK_FIELD_VOCABULARIES",
    "VLMTaskSemanticFieldHead",
    "dataset_frequency_balanced_field_loss",
    "dataset_frequency_balanced_partial_field_loss",
    "decode_task_field_predictions",
    "encode_task_field_targets",
]
