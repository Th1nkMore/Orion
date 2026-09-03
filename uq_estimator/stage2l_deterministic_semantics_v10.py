"""Deterministic Stage2-L semantics derived from U, VLM-owned R, and K=U*R.

The V5 QA factory already defines task-risk view, region, magnitude and stance
as deterministic functions of the spatial maps.  Re-learning those fields in
an auxiliary classifier adds approximation error without adding information.
This module provides the inference-side deterministic interface while leaving
task relevance R entirely owned by the VLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from uq_estimator.uq_relevance_tokenizer import fixed_task_risk


SCHEMA = "orion.stage2l_deterministic_semantics.v10"
DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
REAR_VIEWS = frozenset(("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"))


@dataclass(frozen=True)
class DeterministicTaskSemantics:
    task_risk: torch.Tensor
    structured_fields: Tuple[Mapping[str, str], ...]
    numeric_summaries: Tuple[Mapping[str, Any], ...]
    learned_structured_field_classifier_used: bool = False
    schema: str = SCHEMA


def _region(row: int, column: int, height: int, width: int) -> str:
    vertical = ("upper", "middle", "lower")[min(2, (3 * row) // height)]
    horizontal = ("left", "center", "right")[
        min(2, (3 * column) // width)
    ]
    return vertical + "_" + horizontal


def _risk_level(value: float, *, medium: float, high: float) -> str:
    return "high" if value >= high else "medium" if value >= medium else "low"


@torch.no_grad()
def deterministic_task_semantics(
    latest_scalar_uq: torch.Tensor,
    relevance_logits: torch.Tensor,
    *,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    relevance_high_threshold: float = 0.66,
    risk_medium_threshold: float = 0.33,
    risk_high_threshold: float = 0.66,
    caution_threshold: float = 0.25,
    prepare_to_yield_threshold: float = 0.55,
    observation_absence_threshold: float = 1e-8,
    task_risk_absence_threshold: float = 1e-4,
    rearward_high_risk_stance_cap: str = "caution",
) -> DeterministicTaskSemantics:
    """Decode the structured interface without a learned field classifier.

    ``relevance_level`` intentionally follows the current V5 vocabulary:
    ``not_applicable`` for absent U, otherwise ``high`` or ``low``.  Task risk
    retains the four-way ``none/low/medium/high`` vocabulary.
    """

    if latest_scalar_uq.shape != relevance_logits.shape:
        raise ValueError("U and R shapes differ")
    if latest_scalar_uq.ndim != 4:
        raise ValueError("U and R must have shape [B,V,H,W]")
    if latest_scalar_uq.shape[1] != len(camera_order):
        raise ValueError("camera order does not match U/R views")
    thresholds = (
        relevance_high_threshold,
        risk_medium_threshold,
        risk_high_threshold,
        caution_threshold,
        prepare_to_yield_threshold,
        observation_absence_threshold,
        task_risk_absence_threshold,
    )
    if not all(float(value) >= 0.0 for value in thresholds):
        raise ValueError("semantic thresholds must be non-negative")
    if not (
        risk_medium_threshold < risk_high_threshold
        and caution_threshold < prepare_to_yield_threshold
        and relevance_high_threshold <= 1.0
        and risk_high_threshold <= 1.0
        and prepare_to_yield_threshold <= 1.0
    ):
        raise ValueError("semantic threshold ordering is invalid")
    if rearward_high_risk_stance_cap not in ("maintain", "caution"):
        raise ValueError("rearward stance cap must be maintain or caution")
    if not (
        latest_scalar_uq.is_floating_point()
        and relevance_logits.is_floating_point()
        and bool(torch.isfinite(latest_scalar_uq).all())
        and bool(torch.isfinite(relevance_logits).all())
        and not bool((latest_scalar_uq < 0.0).any())
        and not bool((latest_scalar_uq > 1.0).any())
    ):
        raise ValueError("U and R logits must be finite; U must lie in [0,1]")

    task_risk = fixed_task_risk(latest_scalar_uq, relevance_logits)
    relevance = relevance_logits.sigmoid()
    batch, views, height, width = latest_scalar_uq.shape
    structured = []
    numeric = []
    for index in range(batch):
        u_flat = int(latest_scalar_uq[index].flatten().argmax().item())
        u_view_index = u_flat // (height * width)
        u_remainder = u_flat % (height * width)
        u_row, u_column = divmod(u_remainder, width)
        u_peak = float(latest_scalar_uq[index].flatten()[u_flat].item())

        k_flat = int(task_risk[index].flatten().argmax().item())
        k_view_index = k_flat // (height * width)
        k_remainder = k_flat % (height * width)
        k_row, k_column = divmod(k_remainder, width)
        k_peak = float(task_risk[index].flatten()[k_flat].item())

        if u_peak <= observation_absence_threshold:
            relevance_at_u = 0.0
            relevance_level = "not_applicable"
            observation_view = "none"
            observation_region = "none"
        else:
            relevance_at_u = float(
                relevance[index, u_view_index, u_row, u_column].item()
            )
            relevance_level = (
                "high"
                if relevance_at_u >= relevance_high_threshold
                else "low"
            )
            observation_view = str(camera_order[u_view_index])
            observation_region = _region(u_row, u_column, height, width)

        if k_peak <= task_risk_absence_threshold:
            risk_level = "none"
            risk_view = "none"
            risk_region = "none"
            risk_bearing = "none"
            stance = "maintain"
        else:
            risk_level = _risk_level(
                k_peak, medium=risk_medium_threshold, high=risk_high_threshold
            )
            risk_view = str(camera_order[k_view_index])
            risk_region = _region(k_row, k_column, height, width)
            risk_bearing = (
                "rearward" if risk_view in REAR_VIEWS else "forward_or_crossing"
            )
            if k_peak >= prepare_to_yield_threshold:
                stance = (
                    rearward_high_risk_stance_cap
                    if risk_view in REAR_VIEWS
                    else "prepare_to_yield"
                )
            elif k_peak >= caution_threshold:
                stance = "caution"
            else:
                stance = "maintain"

        structured.append(
            {
                "relevance_level": relevance_level,
                "risk_level": risk_level,
                "risk_view": risk_view,
                "risk_region": risk_region,
                "stance": stance,
                "direct_control": "no",
                "response_basis": "observation_uncertainty",
            }
        )
        numeric.append(
            {
                "observation_peak": u_peak,
                "observation_view": observation_view,
                "observation_region": observation_region,
                "relevance_at_observation_peak": relevance_at_u,
                "task_risk_peak": k_peak,
                "task_risk_view": risk_view,
                "task_risk_region": risk_region,
                "risk_bearing": risk_bearing,
            }
        )
    return DeterministicTaskSemantics(
        task_risk=task_risk,
        structured_fields=tuple(structured),
        numeric_summaries=tuple(numeric),
    )


__all__ = [
    "DEFAULT_CAMERA_ORDER",
    "DeterministicTaskSemantics",
    "SCHEMA",
    "deterministic_task_semantics",
]
