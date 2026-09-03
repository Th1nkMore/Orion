"""Pairwise Stage-1 supervision without blanket reference-zero labels.

The adapter is still deployable from an observation sequence alone.  A paired
reference is privileged training data used only to supervise the difference
between two adapter outputs at a matched world pose.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.counterfactual_evidence import (
    EVIDENCE_COMPONENTS,
    CounterfactualEvidenceError,
    CounterfactualEvidenceTarget,
    ObservationEvidenceHurdleAdapter,
    scale_counterfactual_target,
)
from uq_estimator.counterfactual_evidence_training import (
    CounterfactualEvidenceRecord,
    _collate_record_features,
    _group_batches,
    _project_model_input,
    _targets_from_records_or_features,
    measured_target_ranking_loss,
)
from uq_estimator.native_weather_audit import (
    CONDITION_SEVERITY,
    validate_native_weather_payload,
)


PAIRWISE_TRAINING_SCHEMA_VERSION = "orion.counterfactual-evidence-pairwise/v1"


def records_from_native_weather_payload(
    payload: Mapping[str, Any], split: str = "native_train"
) -> List[CounterfactualEvidenceRecord]:
    """Convert paired clear/fog features to train records.

    ``clear`` is a matched reference, not an absolute zero target.  The two fog
    conditions are native renderer interventions and no pixel corruption mask
    or condition metadata is exposed to the adapter.
    """

    validate_native_weather_payload(payload)
    if not str(split).strip():
        raise CounterfactualEvidenceError("native pairwise split is empty")
    items = payload["items"]
    features = payload["features_by_condition"]
    lookup = {
        (str(item["route_id"]), int(item["sequence_index"])): index
        for index, item in enumerate(items)
    }
    records: List[CounterfactualEvidenceRecord] = []
    for index, item in enumerate(items):
        route_id = str(item["route_id"])
        sequence_index = int(item["sequence_index"])
        prior = lookup.get((route_id, sequence_index - 1))
        reference_current = features["clear"][index]
        reference_previous = (
            features["clear"][prior]
            if prior is not None
            else torch.zeros_like(reference_current)
        )
        pair_id = str(item["sample_id"])
        for condition in ("fog_light", "fog_heavy"):
            observed_current = features[condition][index]
            observed_previous = (
                features[condition][prior]
                if prior is not None
                else torch.zeros_like(observed_current)
            )
            records.append(
                CounterfactualEvidenceRecord(
                    sample_id="%s/%s" % (pair_id, condition),
                    pair_id=pair_id,
                    route_id=route_id,
                    frame_idx=sequence_index,
                    split=str(split),
                    family="native_fog",
                    severity=float(CONDITION_SEVERITY[condition]),
                    reference_current=reference_current,
                    observed_current=observed_current,
                    reference_previous=reference_previous,
                    observed_previous=observed_previous,
                    previous_valid=prior is not None,
                    corruption_mask=None,
                )
            )
    return records


def pairwise_evidence_delta_loss(
    observed_score: torch.Tensor,
    reference_score: torch.Tensor,
    target: CounterfactualEvidenceTarget,
    responsive_weight: float = 0.75,
    response_floor: float = 1e-6,
) -> torch.Tensor:
    """Regress signed score increments against non-negative paired targets.

    ``asinh`` keeps the loss defined when the model initially predicts a
    negative increment, unlike log1p.  Responsive and unchanged cells are
    normalized separately per component so local interventions cannot produce
    an all-zero shortcut.
    """

    if (
        observed_score.shape != target.values.shape
        or reference_score.shape != target.values.shape
    ):
        raise CounterfactualEvidenceError("pairwise prediction/target shapes differ")
    if not 0.0 < responsive_weight < 1.0 or response_floor < 0:
        raise CounterfactualEvidenceError("invalid pairwise regression settings")
    predicted_delta = observed_score - reference_score
    per_cell = F.smooth_l1_loss(
        torch.asinh(predicted_delta), torch.asinh(target.values), reduction="none"
    )
    component_losses = []
    for component in range(len(EVIDENCE_COMPONENTS)):
        valid = target.component_valid[..., component]
        responsive = valid & (target.values[..., component] > response_floor)
        background = valid & ~responsive
        terms = []
        weights = []
        if bool(responsive.any()):
            amplitude = target.values[..., component]
            amplitude_weight = 1.0 + 3.0 * amplitude / (1.0 + amplitude)
            selected_weight = amplitude_weight[responsive]
            terms.append(
                (per_cell[..., component][responsive] * selected_weight).sum()
                / selected_weight.sum().clamp_min(1.0)
            )
            weights.append(responsive_weight)
        if bool(background.any()):
            terms.append(per_cell[..., component][background].mean())
            weights.append(1.0 - responsive_weight)
        if terms:
            component_losses.append(
                sum(weight * term for weight, term in zip(weights, terms))
                / sum(weights)
            )
    if not component_losses:
        raise CounterfactualEvidenceError("pairwise regression has no valid cells")
    return torch.stack(component_losses).mean()


def run_pairwise_hurdle_epoch(
    model: ObservationEvidenceHurdleAdapter,
    records: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    pair_batch_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    seed: int = 0,
    responsive_weight: float = 0.75,
    ranking_weight: float = 0.25,
    response_floor: float = 1e-6,
    input_projection: Optional[torch.Tensor] = None,
    input_quantization: Optional[str] = None,
) -> Dict[str, float]:
    """Train/evaluate score differences; never force a reference toward zero."""

    if not records:
        raise CounterfactualEvidenceError("pairwise epoch records are empty")
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "delta": 0.0, "ranking": 0.0}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in _group_batches(
            records, pair_batch_size, shuffle=training, seed=seed
        ):
            reference, observed, reference_previous, observed_previous, valid = (
                _collate_record_features(batch, device)
            )
            target = scale_counterfactual_target(
                _targets_from_records_or_features(
                    batch,
                    reference,
                    observed,
                    reference_previous,
                    observed_previous,
                    valid,
                ),
                scales,
            )
            observed_prediction = model.predict_parts(
                _project_model_input(observed, input_projection, input_quantization),
                _project_model_input(
                    observed_previous, input_projection, input_quantization
                ),
                valid,
            )
            reference_prediction = model.predict_parts(
                _project_model_input(reference, input_projection, input_quantization),
                _project_model_input(
                    reference_previous, input_projection, input_quantization
                ),
                valid,
            )
            delta_loss = pairwise_evidence_delta_loss(
                observed_prediction.score,
                reference_prediction.score,
                target,
                responsive_weight=responsive_weight,
                response_floor=response_floor,
            )
            predicted_delta = observed_prediction.score - reference_prediction.score
            ranking_loss = measured_target_ranking_loss(
                predicted_delta, target, batch
            )
            loss = delta_loss + ranking_weight * ranking_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            weight = len(batch)
            totals["total"] += float(loss.detach()) * weight
            totals["delta"] += float(delta_loss.detach()) * weight
            totals["ranking"] += float(ranking_loss.detach()) * weight
            count += weight
    return {name: value / max(count, 1) for name, value in totals.items()}


__all__ = [
    "PAIRWISE_TRAINING_SCHEMA_VERSION",
    "pairwise_evidence_delta_loss",
    "records_from_native_weather_payload",
    "run_pairwise_hurdle_epoch",
]
