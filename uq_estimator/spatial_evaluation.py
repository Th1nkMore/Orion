"""Independent evaluation and calibration for Stage-1 spatial UQ.

The evaluator is intentionally separate from the training loop.  It loads only
the distilled student, treats the route-disjoint manifest embedded in the
checkpoint as immutable, fits scalar calibration parameters on the calibration
split, and never pools actual perception failures with representation-error
proxies.

Optional metadata conventions are deliberately small and fail closed:

* ``path_mask`` / ``path_corridor_mask`` / ``relevance_mask``: a spatial mask
  with the same shape as one record's patch target;
* ``relevance``: ``"on_path"`` or ``"off_path"`` (``on_path: bool`` is also
  accepted) for matched counterfactual records;
* ``event_id``, ``timestamp_seconds``, and ``event_active``: a temporal event
  sequence.  ``timestamp`` is accepted as an alias for ``timestamp_seconds``.

No metric emitted by this module supports a closed-loop safety or semantic-UQ
claim on its own.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from uq_estimator.spatial_metrics import (
    binary_spatial_metrics,
    spearman_correlation,
    temporal_event_metrics,
)
from uq_estimator.spatial_training import (
    DEFAULT_CLAIM_BOUNDARY,
    PAIRED_RECORD_SCHEMA_VERSION,
    SPATIAL_CHECKPOINT_SCHEMA_VERSION,
    TARGET_CONTRACT_SCHEMA_VERSION,
    TARGET_ACTUAL_FAILURE,
    TARGET_REPRESENTATION_PROXY,
    PairedSpatialFeatureRecord,
    RouteDisjointManifest,
    SpatialTrainingDataError,
    collate_paired_spatial_records,
    summarize_target_provenance,
    validate_manifest_coverage,
)
from uq_estimator.spatial_uq import SpatialPatchUQHead, SpatialUQOutput


SPATIAL_EVALUATION_SCHEMA_VERSION = "spatial-uq-stage1-evaluation/v2"
SPATIAL_CALIBRATION_SCHEMA_VERSION = "spatial-uq-stage1-calibration/v2"
EVALUATED_SPLITS = ("validation", "calibration", "held_out")
TARGET_PROVENANCES = (TARGET_ACTUAL_FAILURE, TARGET_REPRESENTATION_PROXY)


class SpatialEvaluationError(ValueError):
    """Raised when evaluation inputs violate the preregistered contract."""


@dataclass(frozen=True)
class MonotonicCalibration:
    """Scalar logit-temperature plus an operational decision threshold."""

    temperature: float
    threshold: float
    failure_event_threshold: float
    fit_split: str = "calibration"
    objective: str = "binary_nll_grid_then_f1_threshold"
    schema_version: str = SPATIAL_CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPATIAL_CALIBRATION_SCHEMA_VERSION:
            raise SpatialEvaluationError(
                f"unsupported calibration schema {self.schema_version!r}"
            )
        if self.fit_split != "calibration":
            raise SpatialEvaluationError("calibration fit_split must be 'calibration'")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise SpatialEvaluationError("temperature must be finite and positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise SpatialEvaluationError("threshold must lie in [0, 1]")
        if not 0.0 < self.failure_event_threshold < 1.0:
            raise SpatialEvaluationError(
                "failure_event_threshold must lie in (0, 1)"
            )

    def apply(self, probability: torch.Tensor) -> torch.Tensor:
        return temperature_scale_probability(probability, self.temperature)


@dataclass(frozen=True)
class SpatialPredictionRecord:
    record: PairedSpatialFeatureRecord
    failure_probability: torch.Tensor
    expected_error: torch.Tensor
    error_severity_target: torch.Tensor
    error_severity_valid_mask: torch.Tensor
    failure_event_target: torch.Tensor
    failure_event_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        expected_shape = self.record.observed_patch_features.shape[:-1]
        for name in (
            "failure_probability",
            "expected_error",
            "error_severity_target",
            "error_severity_valid_mask",
            "failure_event_target",
            "failure_event_valid_mask",
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise SpatialEvaluationError(
                    f"{name} shape {tuple(value.shape)} does not match "
                    f"record spatial shape {tuple(expected_shape)}"
                )
            if not torch.isfinite(value).all():
                raise SpatialEvaluationError(f"{name} contains non-finite values")
        if self.error_severity_valid_mask.dtype != torch.bool:
            raise SpatialEvaluationError("error_severity_valid_mask must be boolean")
        if self.failure_event_valid_mask.dtype != torch.bool:
            raise SpatialEvaluationError("failure_event_valid_mask must be boolean")


def _safe_torch_load(path: Path | str) -> Mapping[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise SpatialEvaluationError("checkpoint must contain a mapping")
    return payload


def load_stage1_student(
    checkpoint_path: Path | str,
) -> Tuple[SpatialPatchUQHead, Mapping[str, Any], RouteDisjointManifest]:
    """Load a Stage-1 student with strict schema and state-dict checks."""
    payload = _safe_torch_load(checkpoint_path)
    if payload.get("schema_version") != SPATIAL_CHECKPOINT_SCHEMA_VERSION:
        raise SpatialEvaluationError(
            f"unsupported checkpoint schema {payload.get('schema_version')!r}"
        )
    if payload.get("target_contract_schema_version") != TARGET_CONTRACT_SCHEMA_VERSION:
        raise SpatialEvaluationError(
            "checkpoint target-contract schema is missing or incompatible"
        )
    if payload.get("spatial_output_schema_version") != SpatialUQOutput.CURRENT_SCHEMA_VERSION:
        raise SpatialEvaluationError("checkpoint spatial-output schema is incompatible")
    if payload.get("paired_record_schema_version") != PAIRED_RECORD_SCHEMA_VERSION:
        raise SpatialEvaluationError("checkpoint paired-record schema is incompatible")

    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise SpatialEvaluationError("checkpoint model_config must be a mapping")
    try:
        feature_dim = int(model_config["feature_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        min_log_variance = float(model_config["min_log_variance"])
        max_log_variance = float(model_config["max_log_variance"])
        predict_epistemic = model_config["predict_epistemic"]
    except (KeyError, TypeError, ValueError) as error:
        raise SpatialEvaluationError("checkpoint model_config is incomplete") from error
    if feature_dim <= 0 or hidden_dim <= 0:
        raise SpatialEvaluationError("checkpoint model dimensions must be positive")
    if not math.isfinite(min_log_variance) or not math.isfinite(max_log_variance):
        raise SpatialEvaluationError("checkpoint log-variance bounds must be finite")
    if min_log_variance >= max_log_variance:
        raise SpatialEvaluationError(
            "checkpoint min_log_variance must be smaller than max_log_variance"
        )
    if predict_epistemic is not True:
        raise SpatialEvaluationError("Stage-1 student must declare predict_epistemic=true")

    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        raise SpatialEvaluationError("checkpoint claim_boundary must be a mapping")
    for key, required_value in DEFAULT_CLAIM_BOUNDARY.items():
        if claim_boundary.get(key) != required_value:
            raise SpatialEvaluationError(
                f"checkpoint claim boundary changed required field {key!r}"
            )

    manifest_payload = payload.get("route_disjoint_manifest")
    if not isinstance(manifest_payload, Mapping):
        raise SpatialEvaluationError("checkpoint has no route-disjoint manifest")
    try:
        manifest = RouteDisjointManifest.from_dict(manifest_payload)
    except SpatialTrainingDataError as error:
        raise SpatialEvaluationError(str(error)) from error

    state = payload.get("student_state")
    if not isinstance(state, Mapping):
        raise SpatialEvaluationError("checkpoint student_state must be a state dict")
    student = SpatialPatchUQHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        min_log_variance=min_log_variance,
        max_log_variance=max_log_variance,
        predict_epistemic=True,
    )
    try:
        student.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise SpatialEvaluationError(
            f"checkpoint student_state is incompatible: {error}"
        ) from error
    student.eval()
    return student, payload, manifest


def validate_evaluation_inputs(
    records: Sequence[PairedSpatialFeatureRecord],
    manifest: RouteDisjointManifest,
    checkpoint: Mapping[str, Any],
    supplied_manifest: Optional[RouteDisjointManifest] = None,
) -> None:
    """Fail closed on split, feature, provenance, or manifest drift."""
    try:
        validate_manifest_coverage(records, manifest)
    except SpatialTrainingDataError as error:
        raise SpatialEvaluationError(str(error)) from error
    if supplied_manifest is not None and supplied_manifest.to_dict() != manifest.to_dict():
        raise SpatialEvaluationError(
            "supplied manifest differs from the immutable checkpoint manifest"
        )
    config = checkpoint["model_config"]
    feature_dim = int(config["feature_dim"])
    if any(record.observed_patch_features.shape[-1] != feature_dim for record in records):
        raise SpatialEvaluationError("record feature dimension differs from checkpoint")
    checkpoint_provenance = checkpoint.get("target_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise SpatialEvaluationError("checkpoint target_provenance is missing")
    current = summarize_target_provenance(records)
    if checkpoint_provenance != current:
        raise SpatialEvaluationError(
            "evaluation records do not match checkpoint target-provenance inventory"
        )


def temperature_scale_probability(
    probability: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Apply a strictly monotone scalar temperature in logit space."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise SpatialEvaluationError("temperature must be finite and positive")
    if not torch.isfinite(probability).all():
        raise SpatialEvaluationError("probability contains non-finite values")
    if torch.any((probability < 0) | (probability > 1)):
        raise SpatialEvaluationError("probability must lie in [0, 1]")
    epsilon = torch.finfo(probability.dtype).eps
    bounded = probability.clamp(epsilon, 1.0 - epsilon)
    return torch.sigmoid(torch.logit(bounded) / float(temperature))


def fit_monotonic_calibration(
    probability: torch.Tensor,
    binary_target: torch.Tensor,
    failure_event_threshold: float = 0.5,
) -> MonotonicCalibration:
    """Fit temperature and threshold using calibration data only.

    A deterministic log-spaced grid avoids optimizer/version sensitivity.  The
    temperature minimizes binary NLL and therefore cannot change ranking.  The
    threshold maximizes patch-level F1 after temperature scaling.
    """
    probability = probability.float().reshape(-1)
    binary_target = binary_target.float().reshape(-1)
    if probability.numel() == 0 or probability.shape != binary_target.shape:
        raise SpatialEvaluationError("calibration tensors must be non-empty and aligned")
    if not torch.isfinite(probability).all() or not torch.isfinite(binary_target).all():
        raise SpatialEvaluationError("calibration tensors must be finite")
    if torch.any((probability < 0) | (probability > 1)):
        raise SpatialEvaluationError("calibration probability must lie in [0, 1]")
    if torch.any((binary_target != 0) & (binary_target != 1)):
        raise SpatialEvaluationError("calibration target must be binary")
    if binary_target.min() == binary_target.max():
        raise SpatialEvaluationError(
            "calibration requires both positive and negative target cells"
        )

    temperatures = torch.logspace(-1.30103, 1.30103, 401)  # [0.05, 20]
    epsilon = torch.finfo(probability.dtype).eps
    logits = torch.logit(probability.clamp(epsilon, 1.0 - epsilon))
    scaled = torch.sigmoid(logits.unsqueeze(0) / temperatures.unsqueeze(1))
    nll = -(
        binary_target.unsqueeze(0) * torch.log(scaled.clamp_min(epsilon))
        + (1.0 - binary_target.unsqueeze(0))
        * torch.log((1.0 - scaled).clamp_min(epsilon))
    ).mean(dim=1)
    best_loss = nll.min()
    tied = torch.nonzero(torch.isclose(nll, best_loss, rtol=1e-7, atol=1e-9)).flatten()
    best_index = min(tied.tolist(), key=lambda index: abs(float(temperatures[index]) - 1.0))
    temperature = float(temperatures[best_index])
    calibrated = scaled[best_index]

    unique = torch.unique(calibrated).sort().values
    candidates = torch.cat(
        (calibrated.new_tensor([0.0]), unique, calibrated.new_tensor([1.0]))
    ).unique(sorted=True)
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -float("inf"))
    for threshold_tensor in candidates:
        prediction = calibrated >= threshold_tensor
        target_bool = binary_target.bool()
        true_positive = float((prediction & target_bool).sum())
        false_positive = float((prediction & ~target_bool).sum())
        false_negative = float((~prediction & target_bool).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        threshold = float(threshold_tensor)
        # Deterministic ties: greater recall, then threshold closest to 0.5.
        key = (f1, recall, -abs(threshold - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return MonotonicCalibration(
        temperature=temperature,
        threshold=best_threshold,
        failure_event_threshold=failure_event_threshold,
    )


def predict_records(
    student: SpatialPatchUQHead,
    records: Sequence[PairedSpatialFeatureRecord],
    failure_event_threshold: float,
    device: torch.device | str = "cpu",
) -> List[SpatialPredictionRecord]:
    if not 0.0 < failure_event_threshold < 1.0:
        raise SpatialEvaluationError("failure_event_threshold must lie in (0, 1)")
    device = torch.device(device)
    student = student.to(device)
    student.eval()
    predictions: List[SpatialPredictionRecord] = []
    with torch.no_grad():
        for record in records:
            batch = collate_paired_spatial_records([record]).to(device)
            output = student(batch.observed_features)
            probability = output.failure_probability[0].detach().cpu()
            expected_error = output.expected_error[0].detach().cpu()
            predictions.append(
                SpatialPredictionRecord(
                    record=record,
                    failure_probability=probability,
                    expected_error=expected_error,
                    error_severity_target=(
                        batch.error_severity_target[0].detach().cpu()
                    ),
                    error_severity_valid_mask=(
                        batch.error_severity_valid_mask[0].detach().cpu()
                    ),
                    failure_event_target=(
                        batch.failure_event_target[0].detach().cpu()
                        >= failure_event_threshold
                    ).float(),
                    failure_event_valid_mask=(
                        batch.failure_event_valid_mask[0].detach().cpu()
                    ),
                )
            )
    return predictions


def _flatten_failure_predictions(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
) -> Tuple[torch.Tensor, torch.Tensor]:
    probability_parts = []
    target_parts = []
    for prediction in predictions:
        valid = prediction.failure_event_valid_mask
        probability_parts.append(prediction.failure_probability[valid])
        target_parts.append(prediction.failure_event_target[valid])
    probability = torch.cat(probability_parts)
    if calibration is not None:
        probability = calibration.apply(probability)
    return probability, torch.cat(target_parts)


def _flatten_severity_predictions(
    predictions: Sequence[SpatialPredictionRecord],
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected_parts = []
    target_parts = []
    for prediction in predictions:
        valid = prediction.error_severity_valid_mask
        expected_parts.append(prediction.expected_error[valid])
        target_parts.append(prediction.error_severity_target[valid])
    return torch.cat(expected_parts), torch.cat(target_parts)


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _core_metrics(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
) -> Dict[str, Any]:
    expected_error, target_error = _flatten_severity_predictions(predictions)
    result: Dict[str, Any] = {
        "spearman_expected_vs_target_error": (
            _finite_or_none(spearman_correlation(expected_error, target_error))
            if target_error.numel() > 0
            else None
        ),
        "severity_valid_patch_cells": int(target_error.numel()),
    }
    probability, binary_target = _flatten_failure_predictions(
        predictions, calibration
    )
    if probability.numel() == 0:
        result["failure_event_metrics"] = {
            "status": "unavailable",
            "reason": "no valid failure_event_target cells for this provenance",
            "valid_patch_cells": 0,
        }
    else:
        binary = binary_spatial_metrics(probability, binary_target)
        result["failure_event_metrics"] = {
            "status": "ok",
            "average_precision": _finite_or_none(binary.average_precision),
            "auroc": _finite_or_none(binary.auroc),
            "fpr_at_95_tpr": _finite_or_none(binary.fpr_at_95_tpr),
            "brier": _finite_or_none(binary.brier),
            "ece": _finite_or_none(binary.ece),
            "aurc": _finite_or_none(binary.aurc),
            "positives": binary.positives,
            "negatives": binary.negatives,
            "valid_patch_cells": int(binary_target.numel()),
        }
    component_values: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for prediction in predictions:
        components = prediction.record.component_errors
        if components is None:
            continue
        axis = prediction.record.component_error_axis
        normalized_axis = axis if axis >= 0 else components.ndim + axis
        components_last = components.movedim(normalized_axis, -1)
        valid = prediction.error_severity_valid_mask
        for index, name in enumerate(prediction.record.component_error_names):
            component_values.setdefault(name, []).append(
                (prediction.expected_error[valid], components_last[..., index][valid])
            )
    result["component_error_metrics"] = {
        name: {
            "status": "ok",
            "spearman_expected_vs_component_error": _finite_or_none(
                spearman_correlation(
                    torch.cat([value[0] for value in values]),
                    torch.cat([value[1] for value in values]),
                )
            ),
            "valid_patch_cells": sum(value[1].numel() for value in values),
        }
        for name, values in sorted(component_values.items())
    }
    if not component_values:
        result["component_error_metrics"] = {
            "status": "unavailable",
            "reason": "records contain no component_errors",
        }
    return result


def _metadata_spatial_mask(prediction: SpatialPredictionRecord) -> Optional[torch.Tensor]:
    metadata = prediction.record.metadata
    keys = ("path_mask", "path_corridor_mask", "relevance_mask")
    present = [key for key in keys if key in metadata]
    if len(present) > 1:
        raise SpatialEvaluationError(
            f"record {prediction.record.record_id} provides multiple path-mask aliases"
        )
    if not present:
        return None
    value = metadata[present[0]]
    mask = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if mask.shape != prediction.failure_probability.shape:
        raise SpatialEvaluationError(
            f"record {prediction.record.record_id} metadata {present[0]} shape mismatch"
        )
    if not torch.isfinite(mask.float()).all() or torch.any(mask.float() < 0):
        raise SpatialEvaluationError("path metadata mask must be finite and non-negative")
    return mask.float() > 0


def _record_relevance(prediction: SpatialPredictionRecord) -> Optional[str]:
    metadata = prediction.record.metadata
    if "relevance" in metadata and "on_path" in metadata:
        raise SpatialEvaluationError(
            f"record {prediction.record.record_id} provides two relevance labels"
        )
    if "relevance" in metadata:
        value = str(metadata["relevance"]).strip().lower().replace("-", "_")
        aliases = {"onpath": "on_path", "offpath": "off_path"}
        value = aliases.get(value, value)
        if value not in {"on_path", "off_path"}:
            raise SpatialEvaluationError(
                f"record {prediction.record.record_id} has invalid relevance {value!r}"
            )
        return value
    if "on_path" in metadata:
        if not isinstance(metadata["on_path"], bool):
            raise SpatialEvaluationError("metadata on_path must be boolean")
        return "on_path" if metadata["on_path"] else "off_path"
    return None


def _top_q_mean(value: torch.Tensor, mask: Optional[torch.Tensor], top_q: float = 0.1) -> float:
    selected = value[mask] if mask is not None else value.reshape(-1)
    if selected.numel() == 0:
        raise SpatialEvaluationError("record risk mask selects no patch cells")
    count = max(1, int(math.ceil(selected.numel() * top_q)))
    return float(torch.topk(selected, count).values.mean())


def _record_score(
    prediction: SpatialPredictionRecord,
    calibration: Optional[MonotonicCalibration],
) -> float:
    probability = prediction.failure_probability
    if calibration is not None:
        probability = calibration.apply(probability)
    return _top_q_mean(probability, _metadata_spatial_mask(prediction))


def _clean_false_positive(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
) -> Dict[str, Any]:
    clean = [prediction for prediction in predictions if prediction.record.severity == 0]
    if not clean:
        return {"status": "unavailable", "reason": "no severity==0 records"}
    probability, binary_target = _flatten_failure_predictions(clean, calibration)
    if probability.numel() == 0:
        return {
            "status": "unavailable",
            "reason": "clean records have no valid failure_event_target cells",
            "record_count": len(clean),
        }
    negative = binary_target == 0
    if not negative.any():
        return {
            "status": "unavailable",
            "reason": "severity==0 records contain no negative target cells",
            "record_count": len(clean),
        }
    threshold = calibration.threshold if calibration is not None else 0.5
    return {
        "status": "ok",
        "record_count": len(clean),
        "patch_cells": int(probability.numel()),
        "negative_patch_cells": int(negative.sum()),
        "mean_probability": float(probability.mean()),
        "false_positive_rate": float(
            (probability[negative] >= threshold).float().mean()
        ),
        "threshold": float(threshold),
    }


def _on_off_path_contrast(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
) -> Dict[str, Any]:
    cell_on: List[torch.Tensor] = []
    cell_off: List[torch.Tensor] = []
    for prediction in predictions:
        mask = _metadata_spatial_mask(prediction)
        if mask is None or not mask.any() or mask.all():
            continue
        probability = prediction.failure_probability
        if calibration is not None:
            probability = calibration.apply(probability)
        cell_on.append(probability[mask])
        cell_off.append(probability[~mask])

    labelled: Dict[str, List[float]] = {"on_path": [], "off_path": []}
    for prediction in predictions:
        relevance = _record_relevance(prediction)
        if relevance is not None:
            labelled[relevance].append(_record_score(prediction, calibration))

    result: Dict[str, Any] = {}
    if cell_on and cell_off:
        on_mean = float(torch.cat(cell_on).mean())
        off_mean = float(torch.cat(cell_off).mean())
        result["path_mask_cell_contrast"] = {
            "status": "ok",
            "on_path_mean_probability": on_mean,
            "off_path_mean_probability": off_mean,
            "difference_on_minus_off": on_mean - off_mean,
            "on_path_cells": sum(value.numel() for value in cell_on),
            "off_path_cells": sum(value.numel() for value in cell_off),
        }
    else:
        result["path_mask_cell_contrast"] = {
            "status": "unavailable",
            "reason": "no non-degenerate path mask metadata",
        }

    if labelled["on_path"] and labelled["off_path"]:
        on_mean = sum(labelled["on_path"]) / len(labelled["on_path"])
        off_mean = sum(labelled["off_path"]) / len(labelled["off_path"])
        result["matched_relevance_record_contrast"] = {
            "status": "ok",
            "on_path_mean_topq_probability": on_mean,
            "off_path_mean_topq_probability": off_mean,
            "difference_on_minus_off": on_mean - off_mean,
            "on_path_records": len(labelled["on_path"]),
            "off_path_records": len(labelled["off_path"]),
        }
    else:
        result["matched_relevance_record_contrast"] = {
            "status": "unavailable",
            "reason": "metadata lacks both on_path and off_path records",
        }
    return result


def _temporal_metrics(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
) -> Dict[str, Any]:
    groups: Dict[str, List[Tuple[float, bool, SpatialPredictionRecord]]] = {}
    partial_metadata: List[str] = []
    for prediction in predictions:
        metadata = prediction.record.metadata
        present = {
            "event_id": "event_id" in metadata,
            "timestamp": "timestamp_seconds" in metadata or "timestamp" in metadata,
            "event_active": "event_active" in metadata,
        }
        if any(present.values()) and not all(present.values()):
            partial_metadata.append(prediction.record.record_id)
            continue
        if not any(present.values()):
            continue
        event_id = str(metadata["event_id"]).strip()
        if not event_id:
            raise SpatialEvaluationError("metadata event_id must be non-empty")
        timestamp = float(metadata.get("timestamp_seconds", metadata.get("timestamp")))
        active = metadata["event_active"]
        if not math.isfinite(timestamp) or not isinstance(active, bool):
            raise SpatialEvaluationError(
                "temporal metadata requires finite timestamp and boolean event_active"
            )
        groups.setdefault(event_id, []).append((timestamp, active, prediction))
    if partial_metadata:
        raise SpatialEvaluationError(
            "partial temporal metadata on records: " + ", ".join(partial_metadata[:5])
        )
    if not groups:
        return {"status": "unavailable", "reason": "no temporal event metadata"}

    threshold = calibration.threshold if calibration is not None else 0.5
    events = []
    for event_id, items in sorted(groups.items()):
        items.sort(key=lambda item: item[0])
        timestamps = [item[0] for item in items]
        if len(set(timestamps)) != len(timestamps):
            raise SpatialEvaluationError(f"event {event_id!r} has duplicate timestamps")
        if len(items) < 2:
            events.append({"event_id": event_id, "status": "insufficient", "frames": 1})
            continue
        steps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if min(steps) <= 0 or max(steps) - min(steps) > max(1e-6, 0.01 * min(steps)):
            raise SpatialEvaluationError(
                f"event {event_id!r} timestamps are not uniformly spaced"
            )
        score = torch.tensor(
            [_record_score(item[2], calibration) for item in items], dtype=torch.float32
        )
        active = torch.tensor([item[1] for item in items], dtype=torch.bool)
        if not active.any():
            events.append(
                {
                    "event_id": event_id,
                    "status": "no_active_window",
                    "frames": len(items),
                    "false_trigger_seconds": float((score >= threshold).sum()) * steps[0],
                }
            )
            continue
        metrics = temporal_event_metrics(score, active, threshold, steps[0])
        item = {"event_id": event_id, "status": "ok", "frames": len(items)}
        item.update({key: _finite_or_none(value) for key, value in asdict(metrics).items()})
        events.append(item)
    return {
        "status": "ok",
        "threshold": float(threshold),
        "record_score": "top_10_percent_mean_within_path_mask_when_available",
        "event_count": len(events),
        "events": events,
    }


def _bootstrap_ci(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
    cluster_attribute: str,
    replicates: int,
    seed: int,
    minimum_clusters: int = 5,
) -> Dict[str, Any]:
    groups: Dict[str, List[SpatialPredictionRecord]] = {}
    for prediction in predictions:
        cluster = str(getattr(prediction.record, cluster_attribute))
        groups.setdefault(cluster, []).append(prediction)
    cluster_ids = sorted(groups)
    if len(cluster_ids) < minimum_clusters:
        return {
            "status": "insufficient",
            "unique_clusters": len(cluster_ids),
            "minimum_clusters": minimum_clusters,
        }
    rng = random.Random(seed)
    values: Dict[str, List[float]] = {}
    for _ in range(replicates):
        sample: List[SpatialPredictionRecord] = []
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            sample.extend(groups[cluster_id])
        metrics = _core_metrics(sample, calibration)
        for key, value in metrics.items():
            if isinstance(value, float) and math.isfinite(value):
                values.setdefault(key, []).append(value)
    required_valid = max(20, replicates // 4)
    intervals: Dict[str, Any] = {}
    for key, samples in sorted(values.items()):
        if len(samples) < required_valid:
            intervals[key] = {
                "status": "insufficient",
                "valid_replicates": len(samples),
                "required_valid_replicates": required_valid,
            }
            continue
        ordered = torch.tensor(samples).sort().values
        lower = float(torch.quantile(ordered, 0.025))
        upper = float(torch.quantile(ordered, 0.975))
        intervals[key] = {
            "status": "ok",
            "lower_95": lower,
            "upper_95": upper,
            "valid_replicates": len(samples),
        }
    return {
        "status": "ok",
        "unique_clusters": len(cluster_ids),
        "replicates": replicates,
        "intervals": intervals,
    }


def _evaluate_group(
    predictions: Sequence[SpatialPredictionRecord],
    calibration: Optional[MonotonicCalibration],
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "record_count": len(predictions),
        "route_count": len({prediction.record.route_id for prediction in predictions}),
        "pair_count": len({prediction.record.pair_id for prediction in predictions}),
        "uncalibrated_metrics": _core_metrics(predictions, None),
        "clean_false_positive_uncalibrated": _clean_false_positive(predictions, None),
        "on_off_path_uncalibrated": _on_off_path_contrast(predictions, None),
        "temporal_uncalibrated": _temporal_metrics(predictions, None),
    }
    if calibration is None:
        result["calibrated_metrics"] = {
            "status": "unavailable",
            "reason": "no same-provenance calibrator was fit on calibration split",
        }
        result["bootstrap_ci"] = {
            "route": _bootstrap_ci(
                predictions, None, "route_id", bootstrap_replicates, seed
            ),
            "pair": _bootstrap_ci(
                predictions, None, "pair_id", bootstrap_replicates, seed + 1
            ),
            "calibration": "uncalibrated_only",
        }
    else:
        result["calibration"] = asdict(calibration)
        result["calibrated_metrics"] = _core_metrics(predictions, calibration)
        result["clean_false_positive_calibrated"] = _clean_false_positive(
            predictions, calibration
        )
        result["on_off_path_calibrated"] = _on_off_path_contrast(
            predictions, calibration
        )
        result["temporal_calibrated"] = _temporal_metrics(predictions, calibration)
        result["bootstrap_ci"] = {
            "route": _bootstrap_ci(
                predictions, calibration, "route_id", bootstrap_replicates, seed
            ),
            "pair": _bootstrap_ci(
                predictions, calibration, "pair_id", bootstrap_replicates, seed + 1
            ),
            "calibration": "temperature_fit_on_calibration_split_only",
        }
    return result


def evaluate_stage1_checkpoint(
    checkpoint_path: Path | str,
    records: Sequence[PairedSpatialFeatureRecord],
    supplied_manifest: Optional[RouteDisjointManifest] = None,
    failure_event_threshold: float = 0.5,
    bootstrap_replicates: int = 200,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Evaluate validation/calibration/held-out without touching train routes."""
    if bootstrap_replicates < 20:
        raise SpatialEvaluationError("bootstrap_replicates must be at least 20")
    student, checkpoint, manifest = load_stage1_student(checkpoint_path)
    validate_evaluation_inputs(records, manifest, checkpoint, supplied_manifest)
    evaluation_routes = {
        route_id
        for split in EVALUATED_SPLITS
        for route_id in manifest.splits[split]
    }
    evaluation_records = [
        record for record in records if record.route_id in evaluation_routes
    ]
    predictions = predict_records(
        student,
        evaluation_records,
        failure_event_threshold=failure_event_threshold,
        device=device,
    )

    by_split: Dict[str, List[SpatialPredictionRecord]] = {}
    for split in EVALUATED_SPLITS:
        allowed = set(manifest.splits[split])
        by_split[split] = [
            prediction
            for prediction in predictions
            if prediction.record.route_id in allowed
        ]
        if not by_split[split]:
            raise SpatialEvaluationError(f"split {split!r} has no predictions")

    calibrators: Dict[str, MonotonicCalibration] = {}
    calibration_status: Dict[str, Any] = {}
    for provenance in TARGET_PROVENANCES:
        calibration_records = [
            prediction
            for prediction in by_split["calibration"]
            if prediction.record.target_provenance == provenance
        ]
        if not calibration_records:
            calibration_status[provenance] = {
                "status": "unavailable",
                "reason": "calibration split contains no records of this provenance",
            }
            continue
        probability, binary_target = _flatten_failure_predictions(
            calibration_records, calibration=None
        )
        if probability.numel() == 0:
            calibration_status[provenance] = {
                "status": "unavailable",
                "reason": "no valid failure_event_target cells for this provenance",
                "record_count": len(calibration_records),
            }
            continue
        try:
            calibrator = fit_monotonic_calibration(
                probability,
                binary_target,
                failure_event_threshold=failure_event_threshold,
            )
        except SpatialEvaluationError as error:
            calibration_status[provenance] = {
                "status": "insufficient",
                "reason": str(error),
                "record_count": len(calibration_records),
            }
            continue
        calibrators[provenance] = calibrator
        calibration_status[provenance] = {
            "status": "ok",
            "record_count": len(calibration_records),
            **asdict(calibrator),
        }

    split_reports: Dict[str, Any] = {}
    for split_index, split in enumerate(EVALUATED_SPLITS):
        provenance_reports: Dict[str, Any] = {}
        for provenance in TARGET_PROVENANCES:
            subset = [
                prediction
                for prediction in by_split[split]
                if prediction.record.target_provenance == provenance
            ]
            if not subset:
                provenance_reports[provenance] = {
                    "status": "unavailable",
                    "reason": "split contains no records of this provenance",
                }
                continue
            provenance_reports[provenance] = {
                "status": "ok",
                "target_interpretation": (
                    "actual perception failure"
                    if provenance == TARGET_ACTUAL_FAILURE
                    else "paired frozen-representation error proxy; not semantic UQ"
                ),
                **_evaluate_group(
                    subset,
                    calibrators.get(provenance),
                    bootstrap_replicates,
                    seed + 1000 * split_index,
                ),
            }
        split_reports[split] = {
            "route_ids": list(manifest.splits[split]),
            "record_count": len(by_split[split]),
            "by_target_provenance": provenance_reports,
            "pooled_cross_provenance_metrics": {
                "status": "prohibited",
                "reason": "actual failures and representation proxies have different meanings",
            },
        }

    return {
        "schema_version": SPATIAL_EVALUATION_SCHEMA_VERSION,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_schema_version": checkpoint["schema_version"],
        "manifest": manifest.to_dict(),
        "evaluated_splits": list(EVALUATED_SPLITS),
        "train_split_evaluated": False,
        "failure_event_binarization_threshold": failure_event_threshold,
        "calibration": {
            "fit_split": "calibration",
            "held_out_used_for_fitting": False,
            "validation_used_for_fitting": False,
            "rank_preserving_temperature": True,
            "by_target_provenance": calibration_status,
        },
        "claim_boundary": {
            **dict(checkpoint["claim_boundary"]),
            "actual_and_proxy_reported_separately": True,
            "pooled_cross_provenance_metrics_prohibited": True,
            "held_out_metrics_support_semantic_uq_claim": False,
        },
        "splits": split_reports,
    }


def save_evaluation_report(report: Mapping[str, Any], path: Path | str) -> None:
    if report.get("schema_version") != SPATIAL_EVALUATION_SCHEMA_VERSION:
        raise SpatialEvaluationError("refusing to save report with incompatible schema")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


__all__ = [
    "SPATIAL_EVALUATION_SCHEMA_VERSION",
    "SPATIAL_CALIBRATION_SCHEMA_VERSION",
    "EVALUATED_SPLITS",
    "SpatialEvaluationError",
    "MonotonicCalibration",
    "SpatialPredictionRecord",
    "load_stage1_student",
    "validate_evaluation_inputs",
    "temperature_scale_probability",
    "fit_monotonic_calibration",
    "predict_records",
    "evaluate_stage1_checkpoint",
    "save_evaluation_report",
]
