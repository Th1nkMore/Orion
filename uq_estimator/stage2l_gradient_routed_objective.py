"""Additional Stage2-L v8 objectives with frozen loss ownership."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.stage2l_semantic_bottleneck import (
    PLANNING_STANCES,
    encode_planning_stances,
)


SCHEMA = "orion.stage2l_gradient_routed_objective.v1"


def dataset_frequency_balanced_stance_loss(
    logits_by_variant: Mapping[str, torch.Tensor],
    target_stance_by_variant: Mapping[str, Any],
    dataset_class_counts: Mapping[str, int],
    *,
    supervised_variants: Sequence[str],
) -> torch.Tensor:
    """Use frozen dataset-level class counts for unbiased per-group updates.

    The v7 loss balanced only classes represented inside each three-variant
    group.  With five groups this still visited ``prepare_to_yield`` more often
    than ``caution``.  Fixed inverse-frequency weights make the accumulated
    contribution of every stance class equal over a complete epoch.
    """

    variants = tuple(str(value) for value in supervised_variants)
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("supervised variants must be unique and non-empty")
    if set(logits_by_variant) != set(variants):
        raise ValueError("stance logits do not match supervised variants")
    if set(target_stance_by_variant) != set(variants):
        raise ValueError("stance targets do not match supervised variants")
    if set(dataset_class_counts) != set(PLANNING_STANCES):
        raise ValueError("dataset class counts must cover every stance")
    counts = [int(dataset_class_counts[name]) for name in PLANNING_STANCES]
    if min(counts) <= 0:
        raise ValueError("every stance needs positive dataset support")

    logits = []
    targets = []
    for variant in variants:
        value = logits_by_variant[variant]
        if value.ndim != 2 or value.shape[-1] != len(PLANNING_STANCES):
            raise ValueError("stance logits must have shape [B,3]")
        raw_target = target_stance_by_variant[variant]
        if isinstance(raw_target, str):
            names = [raw_target] * value.shape[0]
        else:
            names = [str(item) for item in raw_target]
        if len(names) != value.shape[0]:
            raise ValueError("stance target batch size differs from logits")
        logits.append(value)
        targets.append(encode_planning_stances(names, device=value.device))
    joined_logits = torch.cat(logits, dim=0)
    joined_targets = torch.cat(targets, dim=0)
    total = float(sum(counts))
    class_count = float(len(counts))
    weights = torch.tensor(
        [total / (class_count * count) for count in counts],
        dtype=joined_logits.dtype,
        device=joined_logits.device,
    )
    # Do not use CrossEntropy's default weighted-mean reduction: its
    # denominator is the sum of weights present in the *current group*, which
    # changes between caution and prepare_to_yield groups.  A fixed sample
    # mean makes the accumulated mass over one frozen epoch exactly
    # ``class_count * inverse_frequency_weight`` for every class.
    per_sample = F.cross_entropy(
        joined_logits, joined_targets, reduction="none"
    )
    return (per_sample * weights[joined_targets]).mean()


__all__ = ["SCHEMA", "dataset_frequency_balanced_stance_loss"]
