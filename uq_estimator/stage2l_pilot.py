"""Pure Stage2-L pilot data and metric contracts.

This module deliberately avoids importing ORION, CARLA, or CUDA libraries so
the frozen 6/2 pilot can be audited before any expensive model is loaded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PILOT_SCHEMA = "orion.stage2_l.pilot_dataset.v1"
CACHE_MANIFEST_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)
QUESTION_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def resolve_reference(
    reference: Mapping[str, Any], base: Path, name: str
) -> Path:
    path = Path(str(reference.get("path", reference.get("output", ""))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


@dataclass(frozen=True)
class PilotInputs:
    manifest: Dict[str, Any]
    records: List[Dict[str, Any]]
    records_path: Path
    event_cache_paths: Dict[str, Path]
    event_cache_manifests: Dict[str, Dict[str, Any]]


def load_pilot_inputs(manifest_path: Path) -> PilotInputs:
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != PILOT_SCHEMA
        or manifest.get("status") != "assembled_ready_for_stage2l_pilot_training"
        or manifest.get("formal_training_ready") is not False
    ):
        raise ValueError("pilot dataset is not at the expected engineering stage")
    records_path = resolve_reference(
        manifest.get("records", {}), manifest_path.parent, "pilot QA records"
    )
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != int(manifest.get("qa_record_count", -1)):
        raise ValueError("pilot QA record count mismatch")

    events = manifest.get("events", [])
    if not isinstance(events, list) or len(events) != 8:
        raise ValueError("Stage2-L pilot requires exactly eight events")
    event_ids = [str(row.get("event_id", "")) for row in events]
    routes = [int(row.get("route_index", -1)) for row in events]
    if len(set(event_ids)) != 8 or len(set(routes)) != 8:
        raise ValueError("pilot events and routes must be unique")
    expected_split = {str(row["event_id"]): str(row["split"]) for row in events}
    if Counter(expected_split.values()) != {"train": 6, "dev": 2}:
        raise ValueError("pilot event split must be exactly 6 train / 2 dev")

    event_cache_paths: Dict[str, Path] = {}
    event_cache_manifests: Dict[str, Dict[str, Any]] = {}
    cache_groups: Dict[str, set] = {}
    for row in events:
        event_id = str(row["event_id"])
        cache_path = resolve_reference(
            row.get("visual_cache", {}), manifest_path.parent,
            "visual cache for %s" % event_id,
        )
        cache_manifest_path = resolve_reference(
            row.get("visual_cache_manifest", {}), manifest_path.parent,
            "visual-cache manifest for %s" % event_id,
        )
        cache_manifest = _load_json(cache_manifest_path)
        if cache_manifest.get("schema") != CACHE_MANIFEST_SCHEMA:
            raise ValueError("unsupported visual-cache manifest for %s" % event_id)
        if resolve_reference(
            cache_manifest, cache_manifest_path.parent,
            "visual-cache payload for %s" % event_id,
        ) != cache_path:
            raise ValueError("pilot and per-event visual-cache paths differ")
        forbidden = (
            "privileged_safety_inputs_used",
            "stage1_uq_inputs_used",
            "task_relevance_targets_used",
            "qa_answers_used",
        )
        if any(cache_manifest.get(key) is not False for key in forbidden):
            raise ValueError("visual cache contains prohibited pilot inputs")
        groups = {str(value) for value in cache_manifest.get("group_ids", [])}
        if not groups:
            raise ValueError("visual cache has no counterfactual groups")
        event_cache_paths[event_id] = cache_path
        event_cache_manifests[event_id] = cache_manifest
        cache_groups[event_id] = groups

    group_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    event_groups: Dict[str, set] = defaultdict(set)
    seen_sample_ids = set()
    for row in records:
        event_id = str(row.get("event_id", ""))
        if event_id not in expected_split or row.get("split") != expected_split[event_id]:
            raise ValueError("record event/split disagrees with the frozen pilot")
        if row.get("split") not in ("train", "dev"):
            raise ValueError("pilot records may not contain a test split")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError("pilot sample ids must be non-empty and unique")
        seen_sample_ids.add(sample_id)
        group_id = str(row.get("counterfactual", {}).get("group_id", ""))
        if not group_id:
            raise ValueError("pilot record lacks a counterfactual group")
        group_rows[group_id].append(row)
        event_groups[event_id].add(group_id)
    expected_pairs = {(variant, family) for variant in VARIANTS for family in QUESTION_FAMILIES}
    for group_id, rows in group_rows.items():
        pairs = {
            (str(row["counterfactual"]["variant"]), str(row["question_family"]))
            for row in rows
        }
        if len(rows) != 20 or pairs != expected_pairs:
            raise ValueError("counterfactual group is not five variants x four QA families: %s" % group_id)
        if len({str(row["event_id"]) for row in rows}) != 1:
            raise ValueError("counterfactual group crosses event boundaries")
    if set(event_groups) != set(expected_split):
        raise ValueError("one or more frozen pilot events have no QA records")
    for event_id, groups in event_groups.items():
        if groups != cache_groups[event_id]:
            raise ValueError("QA and visual-cache groups differ for %s" % event_id)

    if len(records) < 480 or len(records) > 800:
        raise ValueError("pilot QA count is outside the frozen 480-800 range")
    return PilotInputs(
        manifest=manifest,
        records=records,
        records_path=records_path,
        event_cache_paths=event_cache_paths,
        event_cache_manifests=event_cache_manifests,
    )


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if values.ndim != 1 or truth.ndim != 1 or values.shape != truth.shape:
        raise ValueError("AUROC scores and labels must be equal-length vectors")
    if not np.isfinite(values).all():
        raise ValueError("AUROC scores must be finite")
    positive = int(truth.sum())
    negative = int((~truth).sum())
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[truth].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def planning_stance(peak_task_risk: float, peak_view: str = "CAM_FRONT") -> str:
    value = float(peak_task_risk)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("peak task risk must lie in [0,1]")
    if value >= 0.55:
        if str(peak_view) in {"CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}:
            return "caution"
        return "prepare_to_yield"
    if value >= 0.25:
        return "caution"
    return "maintain"


def parse_planning_stance(text: str) -> Optional[str]:
    normalized = str(text).lower().replace("-", "_").replace(" ", "_")
    for stance in ("prepare_to_yield", "caution", "maintain"):
        if stance in normalized:
            return stance
    return None


def _target_driving_stance(record: Mapping[str, Any]) -> Optional[str]:
    if record.get("question_family") != "driving_implication":
        return None
    try:
        stance = str(
            record["target"]["structured_summary"]["planning_implication"]["stance"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("driving-implication record lacks a target stance") from exc
    if stance not in {"maintain", "caution", "prepare_to_yield"}:
        raise ValueError("unsupported driving target stance: %s" % stance)
    return stance


def driving_stance_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Count only language-driving targets; other QA families are excluded."""
    counts = Counter(
        stance for stance in (_target_driving_stance(row) for row in records)
        if stance is not None
    )
    return {
        stance: int(counts.get(stance, 0))
        for stance in ("maintain", "caution", "prepare_to_yield")
    }


def balanced_driving_epoch(
    records: Sequence[Mapping[str, Any]], rng: random.Random
) -> List[Mapping[str, Any]]:
    """Return one epoch with balanced driving stances and untouched other QA.

    Minority driving-implication records are sampled with replacement up to the
    majority stance count.  Every non-driving record appears exactly once.
    This changes sampling frequency only; labels and loss weights are unchanged.
    """
    non_driving: List[Mapping[str, Any]] = []
    by_stance: Dict[str, List[Mapping[str, Any]]] = {
        "maintain": [],
        "caution": [],
        "prepare_to_yield": [],
    }
    for row in records:
        stance = _target_driving_stance(row)
        if stance is None:
            non_driving.append(row)
        else:
            by_stance[stance].append(row)
    missing = [stance for stance, rows in by_stance.items() if not rows]
    if missing:
        raise ValueError(
            "balanced driving sampling requires all three stances; missing: %s"
            % ", ".join(missing)
        )
    target_count = max(len(rows) for rows in by_stance.values())
    epoch: List[Mapping[str, Any]] = list(non_driving)
    for stance in ("maintain", "caution", "prepare_to_yield"):
        rows = by_stance[stance]
        pool: List[Mapping[str, Any]] = []
        while len(pool) < target_count:
            cycle = list(rows)
            rng.shuffle(cycle)
            pool.extend(cycle)
        epoch.extend(pool[:target_count])
    rng.shuffle(epoch)
    return epoch


def matched_answer_preference_loss(
    expected_answer_nll: torch.Tensor,
    alternative_answer_nll: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Require the matched target answer to outrank the opposite stance."""
    if expected_answer_nll.shape != alternative_answer_nll.shape:
        raise ValueError("expected and alternative answer losses must match")
    if margin < 0.0:
        raise ValueError("answer preference margin must be non-negative")
    if not (
        bool(torch.isfinite(expected_answer_nll).all())
        and bool(torch.isfinite(alternative_answer_nll).all())
    ):
        raise ValueError("answer preference losses must be finite")
    return F.relu(
        float(margin) + expected_answer_nll - alternative_answer_nll
    ).mean()


__all__ = [
    "CACHE_MANIFEST_SCHEMA",
    "PILOT_SCHEMA",
    "PilotInputs",
    "QUESTION_FAMILIES",
    "VARIANTS",
    "balanced_driving_epoch",
    "binary_auroc",
    "driving_stance_counts",
    "load_pilot_inputs",
    "matched_answer_preference_loss",
    "parse_planning_stance",
    "planning_stance",
    "resolve_reference",
]
