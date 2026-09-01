"""Frozen extraction plan helpers for counterfactual evidence supervision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from uq_estimator.counterfactual_evidence import CounterfactualEvidenceError


COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION = (
    "orion.counterfactual-evidence-extraction/v1"
)
COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V2 = (
    "orion.counterfactual-evidence-extraction/v2"
)
COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V3 = (
    "orion.counterfactual-evidence-extraction/v3"
)
EXPECTED_PROTOCOL_SCHEMA_VERSIONS = {
    "orion.observation-uq-counterfactual-evidence/v1",
    "orion.observation-uq-counterfactual-evidence/v2",
    "orion.observation-uq-counterfactual-evidence/v3",
}


def load_counterfactual_protocol(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") not in EXPECTED_PROTOCOL_SCHEMA_VERSIONS:
        raise CounterfactualEvidenceError("unexpected counterfactual protocol schema")
    split = payload.get("intervention_split", {})
    families = tuple(split.get("optimizer_families", ()))
    severities = tuple(int(value) for value in split.get("optimizer_severities", ()))
    if families != ("local_blur", "local_dark") or severities != (1, 3):
        raise CounterfactualEvidenceError("optimizer intervention set changed")
    expected_heldout = (
        "local_glare on validation and held-out B2D routes; read only after train diagnostics"
        if payload["schema_version"].endswith("/v3")
        else "local_glare on validation and held-out B2D routes"
    )
    if split.get("heldout_family_development") != expected_heldout:
        raise CounterfactualEvidenceError("held-out intervention family changed")
    if payload["schema_version"].endswith(("/v2", "/v3")):
        schedule = split.get("view_schedule", {})
        if schedule.get("version") != "route_condition_window_cycle/v2":
            raise CounterfactualEvidenceError("v2 view schedule changed")
        if int(schedule.get("window_frames", 0)) != 4:
            raise CounterfactualEvidenceError("v2 view window changed")
    return payload


def split_interventions(
    split: str, protocol: Mapping[str, Any]
) -> Tuple[Tuple[str, int], ...]:
    optimizer_families = tuple(protocol["intervention_split"]["optimizer_families"])
    severities = tuple(
        int(value) for value in protocol["intervention_split"]["optimizer_severities"]
    )
    optimizer = tuple(
        (family, severity)
        for family in optimizer_families
        for severity in severities
    )
    glare = tuple(("local_glare", severity) for severity in severities)
    if split == "train":
        return optimizer
    if split == "validation":
        return optimizer + glare
    if split == "held_out":
        return glare
    raise CounterfactualEvidenceError("unsupported counterfactual extraction split")


def deterministic_condition_view(
    route_id: str,
    family: str,
    severity: int,
    view_count: int,
    seed: int,
) -> int:
    if not str(route_id).strip() or not str(family).strip():
        raise CounterfactualEvidenceError("route/family must be non-empty")
    if severity <= 0 or view_count <= 0:
        raise CounterfactualEvidenceError("severity/view count must be positive")
    raw = "%d|%s|%s|%d" % (seed, route_id, family, severity)
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big") % view_count


def deterministic_window_balanced_view(
    route_id: str,
    frame_idx: int,
    family: str,
    severity: int,
    view_count: int,
    seed: int,
    window_frames: int = 4,
) -> int:
    """Cycle cameras across short windows instead of fixing one per route.

    The route/condition hash supplies only the cycle offset.  Consecutive
    frames remain on one sensor for a bounded window, while every 16-frame
    route/condition sequence covers at least four cameras.  This removes the
    deterministic route-condition-to-positive-camera label used by v1.
    """

    if frame_idx < 0 or window_frames <= 0:
        raise CounterfactualEvidenceError("frame/window values must be valid")
    offset = deterministic_condition_view(
        route_id, family, severity, view_count, seed
    )
    return (offset + frame_idx // window_frames) % view_count


def projected_feature_counts(
    routes_by_split: Mapping[str, int], frames_per_route: int
) -> Dict[str, int]:
    if frames_per_route <= 0:
        raise CounterfactualEvidenceError("frames per route must be positive")
    expected_splits = {"train", "validation", "held_out"}
    if set(routes_by_split) != expected_splits or any(
        int(value) <= 0 for value in routes_by_split.values()
    ):
        raise CounterfactualEvidenceError("route quotas must cover three positive splits")
    reference = sum(int(value) for value in routes_by_split.values()) * frames_per_route
    observed_by_split = {
        split: int(route_count)
        * frames_per_route
        * len(split_interventions(split, _minimal_protocol()))
        for split, route_count in routes_by_split.items()
    }
    return {
        "reference": reference,
        "observed": sum(observed_by_split.values()),
        "total": reference + sum(observed_by_split.values()),
        **{"observed_%s" % key: value for key, value in observed_by_split.items()},
    }


def _minimal_protocol() -> Dict[str, Any]:
    return {
        "intervention_split": {
            "optimizer_families": ["local_blur", "local_dark"],
            "optimizer_severities": [1, 3],
        }
    }


__all__ = [
    "COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION",
    "COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V2",
    "COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V3",
    "EXPECTED_PROTOCOL_SCHEMA_VERSIONS",
    "deterministic_condition_view",
    "deterministic_window_balanced_view",
    "load_counterfactual_protocol",
    "projected_feature_counts",
    "split_interventions",
]
