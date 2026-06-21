"""Structured targets and parsing utilities for Risk QA."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import numpy as np


RISK_QA_QUESTION = (
    "Assess the reliability of the current visual observations. "
    "Identify the critical objects in the scene, then explain how visual "
    "reliability should affect how cautiously those observations are used."
)
RELIABILITY_QA_QUESTION = (
    "How reliable are the current visual observations? "
    "Answer with exactly one sentence in the form: "
    "Visual reliability is LEVEL."
)

ROAD_USER_NAMES = {
    "car",
    "van",
    "truck",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
    "traffic_cone",
}


@dataclass(frozen=True)
class CriticalObject:
    category: str
    position: str
    distance_m: float


@dataclass(frozen=True)
class RiskQAAnswer:
    reliability_percentile: int
    reliability_level: str
    critical_objects: tuple[CriticalObject, ...]
    risk_interpretation: str


def reliability_percentile(uq_score: float) -> int:
    score = min(max(float(uq_score), 0.0), 1.0)
    return int(round(100.0 * (1.0 - score)))


def reliability_level(percentile: int) -> str:
    value = min(max(int(percentile), 0), 100)
    if value < 25:
        return "very low"
    if value < 50:
        return "low"
    if value < 75:
        return "moderate"
    if value < 90:
        return "high"
    return "very high"


def relative_position(x: float, y: float) -> str:
    longitudinal = "front" if x >= 0 else "back"
    if y > 2.5:
        return f"{longitudinal}-left"
    if y < -2.5:
        return f"{longitudinal}-right"
    return longitudinal


def normalize_b2d_category(raw_name: object) -> str:
    name = str(raw_name).lower()
    if name.startswith("walker.pedestrian."):
        return "pedestrian"
    if "parkedvehicles" in name:
        return "car"
    if name.startswith("vehicle."):
        if any(item in name for item in ("crossbike", "century", "omafiets")):
            return "bicycle"
        if "firetruck" in name:
            return "truck"
        if "ambulance" in name:
            return "van"
        return "car"
    if name == "traffic.traffic_light":
        return "traffic_light"
    if name.startswith("traffic."):
        return "traffic_sign"
    if name.startswith("static.prop."):
        return "traffic_cone"
    return name


def select_critical_objects(
    boxes: np.ndarray,
    names: Sequence[str],
    max_distance_m: float = 30.0,
    max_objects: int = 3,
) -> tuple[CriticalObject, ...]:
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[-1] < 2:
        raise ValueError("boxes must have shape [N, >=2]")
    if len(boxes) != len(names):
        raise ValueError("boxes and names must have the same length")

    candidates = []
    for box, raw_name in zip(boxes, names):
        category = normalize_b2d_category(raw_name)
        if category not in ROAD_USER_NAMES:
            continue
        x, y = float(box[0]), float(box[1])
        distance = math.hypot(x, y)
        if distance > max_distance_m:
            continue
        candidates.append(
            CriticalObject(
                category=category,
                position=relative_position(x, y),
                distance_m=round(distance, 1),
            )
        )
    candidates.sort(key=lambda item: item.distance_m)
    return tuple(candidates[:max_objects])


def build_risk_qa_answer(
    uq_score: float,
    critical_objects: Iterable[CriticalObject],
) -> RiskQAAnswer:
    percentile = reliability_percentile(uq_score)
    level = reliability_level(percentile)
    interpretation = (
        f"Visual evidence has {level} reliability. Treat the listed "
        "critical-object observations according to this reliability and "
        "verify uncertain evidence before committing to the maneuver."
    )
    return RiskQAAnswer(
        reliability_percentile=percentile,
        reliability_level=level,
        critical_objects=tuple(critical_objects),
        risk_interpretation=interpretation,
    )


def render_risk_qa_answer(answer: RiskQAAnswer) -> str:
    if answer.critical_objects:
        objects = "; ".join(
            f"{item.category}|{item.position}|{item.distance_m:.1f}m"
            for item in answer.critical_objects
        )
    else:
        objects = "none"
    return "\n".join(
        (
            f"VISUAL_RELIABILITY_PERCENTILE: {answer.reliability_percentile}",
            f"RELIABILITY_LEVEL: {answer.reliability_level}",
            f"CRITICAL_OBJECTS: {objects}",
            f"RISK_INTERPRETATION: {answer.risk_interpretation}",
        )
    )


def render_natural_risk_qa_answer(answer: RiskQAAnswer) -> str:
    if answer.critical_objects:
        objects = "; ".join(
            f"{item.category.replace('_', ' ')} in the {item.position}"
            for item in answer.critical_objects
        )
    else:
        objects = "none"
    return (
        f"Visual reliability is {answer.reliability_level}. "
        f"Critical objects: {objects}. "
        f"Risk assessment: {answer.risk_interpretation}"
    )


def render_reliability_answer(answer: RiskQAAnswer) -> str:
    return f"Visual reliability is {answer.reliability_level}."


_PERCENTILE_RE = re.compile(r"VISUAL_RELIABILITY_PERCENTILE:\s*(\d{1,3})")
_LEVEL_RE = re.compile(r"RELIABILITY_LEVEL:\s*([^\n]+)")
_OBJECTS_RE = re.compile(r"CRITICAL_OBJECTS:\s*([^\n]+)")
_INTERPRETATION_RE = re.compile(r"RISK_INTERPRETATION:\s*([^\n]+)")


def parse_risk_qa_answer(text: str) -> RiskQAAnswer:
    percentile_match = _PERCENTILE_RE.search(text)
    level_match = _LEVEL_RE.search(text)
    objects_match = _OBJECTS_RE.search(text)
    interpretation_match = _INTERPRETATION_RE.search(text)
    if not all(
        (percentile_match, level_match, objects_match, interpretation_match)
    ):
        raise ValueError("Risk QA answer is missing one or more required fields")

    percentile = int(percentile_match.group(1))
    if not 0 <= percentile <= 100:
        raise ValueError("Reliability percentile must be in [0, 100]")

    objects_text = objects_match.group(1).strip()
    objects = []
    if objects_text.lower() != "none":
        for item in objects_text.split(";"):
            parts = [part.strip() for part in item.split("|")]
            if len(parts) != 3 or not parts[2].endswith("m"):
                raise ValueError(f"Malformed critical object: {item!r}")
            objects.append(
                CriticalObject(
                    category=parts[0],
                    position=parts[1],
                    distance_m=float(parts[2][:-1]),
                )
            )
    return RiskQAAnswer(
        reliability_percentile=percentile,
        reliability_level=level_match.group(1).strip(),
        critical_objects=tuple(objects),
        risk_interpretation=interpretation_match.group(1).strip(),
    )


_NATURAL_LEVEL_RE = re.compile(
    r"Visual reliability is "
    r"(very low|low|moderate|high|very high)",
    re.IGNORECASE,
)
_NATURAL_OBJECTS_RE = re.compile(
    r"Critical objects:\s*(.*?)\.\s*Risk assessment:",
    re.IGNORECASE | re.DOTALL,
)
_NATURAL_RISK_RE = re.compile(
    r"Risk assessment:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_natural_risk_qa_answer(text: str) -> tuple[str, tuple[str, ...]]:
    level_match = _NATURAL_LEVEL_RE.search(text)
    objects_match = _NATURAL_OBJECTS_RE.search(text)
    risk_match = _NATURAL_RISK_RE.search(text)
    if not all((level_match, objects_match, risk_match)):
        raise ValueError("Natural Risk QA answer is missing required sections")
    object_text = objects_match.group(1).strip()
    objects = ()
    if object_text.lower() != "none":
        objects = tuple(
            item.strip() for item in object_text.split(";") if item.strip()
        )
    return level_match.group(1).lower(), objects


def parse_reliability_answer(text: str) -> str:
    level_match = _NATURAL_LEVEL_RE.search(text)
    if level_match is None:
        raise ValueError("Reliability answer does not contain a valid level")
    return level_match.group(1).lower()


def select_balanced_sample_ids(
    score_by_sample_id: dict[str, float],
    per_level: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    if per_level <= 0:
        raise ValueError("per_level must be positive")
    buckets: dict[str, list[str]] = {
        level: []
        for level in ("very low", "low", "moderate", "high", "very high")
    }
    for sample_id, score in score_by_sample_id.items():
        level = reliability_level(reliability_percentile(score))
        buckets[level].append(sample_id)

    generator = np.random.default_rng(seed)
    selected = []
    counts = {}
    for level, sample_ids in buckets.items():
        if len(sample_ids) < per_level:
            raise ValueError(
                f"Reliability level {level!r} has only {len(sample_ids)} "
                f"samples, fewer than requested {per_level}"
            )
        indices = generator.choice(
            len(sample_ids), size=per_level, replace=False
        )
        chosen = [sample_ids[int(index)] for index in indices]
        selected.extend(chosen)
        counts[level] = len(chosen)
    generator.shuffle(selected)
    return selected, counts
