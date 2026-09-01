"""Gradient-routed Stage2-L planning-semantics bottleneck.

The structured stance classifier is trained only by its explicit stance loss.
Language generation receives the predicted (never target) stance distribution,
but that probability path is detached before constructing the language token.
This prevents QA loss from changing the classifier while still allowing QA
loss to train the stance embeddings and their language-side representation.
"""

from __future__ import annotations

import torch

from uq_estimator.stage2l_semantic_bottleneck_v2 import (
    MagnitudePreservingPlanningStanceBottleneck,
    MagnitudeSemanticTokens,
)


SCHEMA = "orion.stage2l_gradient_routed_semantic_bottleneck.v3"


class GradientRoutedPlanningStanceBottleneck(
    MagnitudePreservingPlanningStanceBottleneck
):
    """Keep stance prediction teacher-free while isolating QA gradients."""

    def forward(
        self,
        task_risk_bridge_tokens: torch.Tensor,
        raw_global_features: torch.Tensor,
        *,
        detach_probabilities_for_language: bool = True,
    ) -> MagnitudeSemanticTokens:
        structural = super().forward(task_risk_bridge_tokens, raw_global_features)
        if not detach_probabilities_for_language:
            return structural
        language_probabilities = structural.probabilities.detach()
        soft_stance = language_probabilities @ self.stance_embedding.weight
        token = self.output_norm(
            soft_stance + self.token_type_embedding[None]
        ).unsqueeze(1)
        return MagnitudeSemanticTokens(
            token=token,
            logits=structural.logits,
            probabilities=structural.probabilities,
            predicted_indices=structural.predicted_indices,
            raw_global_features=structural.raw_global_features,
            magnitude_features=structural.magnitude_features,
            schema=SCHEMA,
        )


__all__ = ["GradientRoutedPlanningStanceBottleneck", "SCHEMA"]
