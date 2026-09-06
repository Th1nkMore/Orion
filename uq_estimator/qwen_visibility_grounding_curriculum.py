"""Deterministic row-addressed curriculum for Qwen visibility grounding.

The curriculum uses only real frontier records from the immutable Route-151
oracle-token manifest. Complete-row permutations place different physical rows
at the same queried sequence slot, so image identity and slot identity cannot
determine the answer. Zero-U and spatial-shuffle artifacts are used only to
predeclare evaluation targets and never enter the optimizer schedule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .qwen_visibility_belief import VISIBILITY_TOKEN_FEATURE_NAMES
from .qwen_visibility_grounding import (
    FACTORIZED_GROUNDING_QUESTIONS,
    GroundingThresholds,
    VISIBILITY_GROUNDING_MANIFEST_SCHEMA,
    derive_visibility_grounding_row_target,
    derive_visibility_grounding_target,
    permute_frontier_rows,
)


ROW_ADDRESSED_CURRICULUM_SCHEMA = (
    "orion.qwen-visibility-row-grounding-curriculum/v1"
)
ROW_ADDRESSED_FIELDS = ("frontier", "route", "margin", "action")
FRONTIER_TARGET_SLOTS = (3, 13, 23)
ROW_QUERY_SLOTS = {"route": 0, "margin": 1, "action": 2}
FIELD_LABEL_ORDER = {
    "frontier": tuple("F%02d" % index for index in FRONTIER_TARGET_SLOTS),
    "route": ("ON_ROUTE", "OFF_ROUTE"),
    "margin": ("INSIDE", "NEAR", "CLEAR"),
    "action": ("KEEP", "SLOW", "STOP"),
}
CONTROL_PREFIXES = {
    "true_u": "visibility_tokens",
    "zero_u": "visibility_tokens_zero_u",
    "spatial_shuffle": "visibility_tokens_spatial_shuffle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_permutation_moving_row(
    count: int, source_index: int, target_index: int
) -> np.ndarray:
    """Return ``new -> old`` cyclic order moving one complete row to a slot."""

    values = (count, source_index, target_index)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("row permutation arguments must be integers")
    count, source_index, target_index = (int(value) for value in values)
    if count <= 0 or not 0 <= source_index < count or not 0 <= target_index < count:
        raise ValueError("row permutation indices must address valid rows")
    order = np.roll(np.arange(count, dtype=np.int64), target_index - source_index)
    if int(order[target_index]) != source_index:
        raise RuntimeError("failed to move the requested row")
    return order


def row_grounding_question(field: str, frontier_label: str) -> str:
    """Create a self-contained exact-answer question for one physical row."""

    if field not in ROW_QUERY_SLOTS:
        raise ValueError("row-addressed field must be route, margin, or action")
    if (
        not isinstance(frontier_label, str)
        or len(frontier_label) != 3
        or not frontier_label.startswith("F")
        or not frontier_label[1:].isdigit()
        or not 0 <= int(frontier_label[1:]) < 32
    ):
        raise ValueError("frontier_label must be F00 through F31")
    prefix = (
        "Read the continuous visibility-belief tokens inserted after the camera "
        "block. Use the complete physical record at frontier token %s. "
        % frontier_label
    )
    if field == "route":
        return (
            prefix
            + "Is route_weight_mean at least 0.2? Reply with exactly ON_ROUTE "
            "or OFF_ROUTE."
        )
    if field == "margin":
        return (
            prefix
            + "Classify frontier_stopping_margin_normalized: INSIDE for at most "
            "0, NEAR for above 0 through 5/60, otherwise CLEAR. Reply with "
            "exactly INSIDE, NEAR, or CLEAR."
        )
    return (
        prefix
        + "Reply STOP when route_weight_mean is at least 0.2 and "
        "frontier_stopping_margin_normalized is at most 0. Reply SLOW when "
        "route_weight_mean is at least 0.2 and either the normalized margin is "
        "above 0 through 5/60 or urgency_max is at least 0.1. Otherwise reply "
        "KEEP. Reply with exactly KEEP, SLOW, or STOP."
    )


def _load_control_frontiers(record: Mapping[str, object]):
    token_path = Path(str(record["token_artifact"]))
    if not token_path.is_file() or _sha256(token_path) != record.get("token_sha256"):
        raise ValueError("base grounding token hash changed")
    base_permutation = np.asarray(
        record["frontier_permutation_new_to_old"], dtype=np.int64
    )
    controls = {}
    feature_names = None
    reference_mask = None
    with np.load(token_path, allow_pickle=False) as artifact:
        for control, prefix in CONTROL_PREFIXES.items():
            tokens = np.asarray(artifact[prefix + "_frontier"], dtype=np.float32)
            mask = np.asarray(artifact[prefix + "_frontier_mask"], dtype=bool)
            names = tuple(
                str(value)
                for value in artifact[prefix + "_feature_names"].tolist()
            )
            if names != tuple(VISIBILITY_TOKEN_FEATURE_NAMES):
                raise ValueError("control feature order changed")
            permuted, permuted_mask = permute_frontier_rows(
                tokens, mask, base_permutation
            )
            if feature_names is None:
                feature_names = names
                reference_mask = permuted_mask
            elif not np.array_equal(permuted_mask, reference_mask):
                raise ValueError("paired control masks changed")
            controls[control] = permuted
    if reference_mask is None or int(reference_mask.sum()) != 32:
        raise ValueError("V1e requires all 32 real frontier rows")
    return controls, reference_mask, feature_names


def _target_for_example(
    controls: Mapping[str, np.ndarray],
    mask: np.ndarray,
    feature_names: Sequence[str],
    sequence_permutation: Sequence[int],
    field: str,
    query_index: Optional[int],
    thresholds: GroundingThresholds,
) -> Tuple[Dict[str, str], object]:
    targets = {}
    true_target = None
    for control, tokens in controls.items():
        if field == "frontier":
            target = derive_visibility_grounding_target(
                tokens,
                mask,
                feature_names,
                sequence_permutation,
                thresholds=thresholds,
            )
            answer = target.frontier
        else:
            target = derive_visibility_grounding_row_target(
                tokens,
                mask,
                feature_names,
                sequence_permutation,
                frontier_index=int(query_index),
                thresholds=thresholds,
            )
            answer = target.answer_dict()[field]
        targets[control] = answer
        if control == "true_u":
            true_target = target
    if true_target is None:
        raise RuntimeError("true-U target was not derived")
    return targets, true_target


def _make_example(
    record: Mapping[str, object],
    controls: Mapping[str, np.ndarray],
    mask: np.ndarray,
    feature_names: Sequence[str],
    field: str,
    source_index: int,
    query_index: int,
    expected_answer: str,
    example_suffix: str,
    thresholds: GroundingThresholds,
) -> dict:
    permutation = complete_permutation_moving_row(
        int(mask.sum()), source_index, query_index
    )
    targets, true_target = _target_for_example(
        controls,
        mask,
        feature_names,
        permutation,
        field,
        query_index if field != "frontier" else None,
        thresholds,
    )
    if targets["true_u"] != expected_answer:
        raise RuntimeError("derived true-U curriculum target changed")
    if targets["spatial_shuffle"] == expected_answer:
        raise ValueError("V1e example lacks a spatial-shuffle target change")
    frontier_label = "F%02d" % query_index
    question = (
        FACTORIZED_GROUNDING_QUESTIONS["frontier"]
        if field == "frontier"
        else row_grounding_question(field, frontier_label)
    )
    example_id = "%s-%s-%s" % (
        field,
        str(record["sample_id"]),
        example_suffix,
    )
    return {
        "example_id": example_id,
        "sample_id": record["sample_id"],
        "task_field": field,
        "question": question,
        "query_frontier": None if field == "frontier" else frontier_label,
        "source_manifest_frontier": "F%02d" % source_index,
        "sequence_permutation_new_to_manifest": permutation.tolist(),
        "expected_answer": expected_answer,
        "control_expected_answers": targets,
        "spatial_shuffle_changes_target": True,
        "target_evidence": true_target.evidence_dict(),
    }


def _balanced_schedule(examples: Sequence[Mapping[str, object]], steps_per_field: int):
    if isinstance(steps_per_field, bool) or int(steps_per_field) != steps_per_field:
        raise ValueError("steps_per_field must be an integer")
    steps_per_field = int(steps_per_field)
    if steps_per_field <= 0:
        raise ValueError("steps_per_field must be positive")
    pools = defaultdict(list)
    for example in examples:
        pools[(example["task_field"], example["expected_answer"])].append(
            example["example_id"]
        )
    for pool in pools.values():
        pool.sort()
    schedule = []
    for round_index in range(steps_per_field):
        for field in ROW_ADDRESSED_FIELDS:
            labels = FIELD_LABEL_ORDER[field]
            if steps_per_field % len(labels):
                raise ValueError("steps_per_field must balance every field label")
            label_index = round_index % len(labels)
            label = labels[label_index]
            pool = pools.get((field, label), [])
            if not pool:
                raise ValueError("curriculum lacks %s/%s examples" % (field, label))
            occurrence = round_index // len(labels)
            schedule.append(pool[occurrence % len(pool)])
    return schedule


def build_route151_row_grounding_curriculum(
    base_manifest_path: Path,
    output_path: Path,
    steps_per_field: int = 90,
    thresholds: GroundingThresholds = GroundingThresholds(),
) -> dict:
    """Build the fail-closed V1e curriculum from real immutable frontier rows."""

    base_manifest_path = Path(base_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite row curriculum: %s" % output_path)
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base.get("schema") != VISIBILITY_GROUNDING_MANIFEST_SCHEMA:
        raise ValueError("unexpected base grounding manifest schema")
    if (
        base.get("reportable_generalization") is not False
        or base.get("controls_used_for_optimizer") is not False
        or base.get("hidden_actor_labels_used") is not False
        or base.get("planning_expert_used_for_optimizer") is not False
    ):
        raise ValueError("base manifest violates the V1 boundary")

    examples = []
    row_candidates = defaultdict(lambda: defaultdict(list))
    for record in base["records"]:
        for image, digest in zip(record["camera_images"], record["camera_sha256"]):
            if _sha256(Path(image)) != digest:
                raise ValueError("base grounding image hash changed")
        controls, mask, feature_names = _load_control_frontiers(record)
        identity = np.arange(int(mask.sum()), dtype=np.int64)
        true_max = derive_visibility_grounding_target(
            controls["true_u"],
            mask,
            feature_names,
            identity,
            thresholds=thresholds,
        )
        for target_slot in FRONTIER_TARGET_SLOTS:
            examples.append(
                _make_example(
                    record,
                    controls,
                    mask,
                    feature_names,
                    field="frontier",
                    source_index=true_max.frontier_index,
                    query_index=target_slot,
                    expected_answer="F%02d" % target_slot,
                    example_suffix="max-to-F%02d" % target_slot,
                    thresholds=thresholds,
                )
            )

        for row_index in range(int(mask.sum())):
            true_target = derive_visibility_grounding_row_target(
                controls["true_u"],
                mask,
                feature_names,
                identity,
                frontier_index=row_index,
                thresholds=thresholds,
            )
            shuffle_target = derive_visibility_grounding_row_target(
                controls["spatial_shuffle"],
                mask,
                feature_names,
                identity,
                frontier_index=row_index,
                thresholds=thresholds,
            )
            for field in ("route", "margin", "action"):
                label = true_target.answer_dict()[field]
                if shuffle_target.answer_dict()[field] == label:
                    continue
                row_candidates[field][label].append(
                    {
                        "record": record,
                        "controls": controls,
                        "mask": mask,
                        "feature_names": feature_names,
                        "source_index": row_index,
                        "selection_score": true_target.frontier_selection_score,
                    }
                )

    selected_counts = {"route": 5, "margin": 4, "action": 2}
    for field in ("route", "margin", "action"):
        for label in FIELD_LABEL_ORDER[field]:
            all_candidates = sorted(
                row_candidates[field][label],
                key=lambda value: (
                    -float(value["selection_score"]),
                    str(value["record"]["sample_id"]),
                    int(value["source_index"]),
                ),
            )
            best_by_sample = {}
            for candidate in all_candidates:
                sample_id = str(candidate["record"]["sample_id"])
                best_by_sample.setdefault(sample_id, candidate)
            candidates = sorted(
                best_by_sample.values(),
                key=lambda value: (
                    -float(value["selection_score"]),
                    str(value["record"]["sample_id"]),
                    int(value["source_index"]),
                ),
            )
            needed = selected_counts[field]
            if len(candidates) < needed:
                raise ValueError(
                    "V1e lacks %d shuffle-sensitive %s/%s rows"
                    % (needed, field, label)
                )
            for candidate in candidates[:needed]:
                source_index = int(candidate["source_index"])
                query_index = ROW_QUERY_SLOTS[field]
                examples.append(
                    _make_example(
                        candidate["record"],
                        candidate["controls"],
                        candidate["mask"],
                        candidate["feature_names"],
                        field=field,
                        source_index=source_index,
                        query_index=query_index,
                        expected_answer=label,
                        example_suffix=(
                            "%s-src-F%02d-to-F%02d"
                            % (label.lower(), source_index, query_index)
                        ),
                        thresholds=thresholds,
                    )
                )

    examples.sort(key=lambda value: value["example_id"])
    if len({example["example_id"] for example in examples}) != len(examples):
        raise RuntimeError("row curriculum contains duplicate example ids")
    field_counts = Counter(example["task_field"] for example in examples)
    if field_counts != Counter({"frontier": 15, "route": 10, "margin": 12, "action": 6}):
        raise RuntimeError("unexpected V1e field balance")
    label_counts = {
        field: dict(
            sorted(
                Counter(
                    example["expected_answer"]
                    for example in examples
                    if example["task_field"] == field
                ).items()
            )
        )
        for field in ROW_ADDRESSED_FIELDS
    }
    if any(len(set(counts.values())) != 1 for counts in label_counts.values()):
        raise RuntimeError("V1e example labels are not balanced within field")
    schedule = _balanced_schedule(examples, steps_per_field)
    schedule_counts = Counter(
        next(
            example["task_field"]
            for example in examples
            if example["example_id"] == example_id
        )
        for example_id in schedule
    )
    if schedule_counts != Counter(
        {field: int(steps_per_field) for field in ROW_ADDRESSED_FIELDS}
    ):
        raise RuntimeError("V1e optimizer fields are not balanced")

    curriculum = {
        "schema": ROW_ADDRESSED_CURRICULUM_SCHEMA,
        "purpose": "Route 151 row-addressed grounding plumbing overfit only",
        "base_manifest_path": str(base_manifest_path),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "reportable_generalization": False,
        "oracle_depth": True,
        "hidden_actor_labels_used": False,
        "controls_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "complete_row_permutations_only": True,
        "spatial_shuffle_target_changed_for_every_example": True,
        "frontier_target_slots": list(FRONTIER_TARGET_SLOTS),
        "row_query_slots": dict(ROW_QUERY_SLOTS),
        "thresholds": thresholds.as_dict(),
        "example_count": len(examples),
        "field_counts": dict(sorted(field_counts.items())),
        "label_counts": label_counts,
        "steps_per_field": int(steps_per_field),
        "optimizer_steps": len(schedule),
        "training_schedule": schedule,
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return curriculum
