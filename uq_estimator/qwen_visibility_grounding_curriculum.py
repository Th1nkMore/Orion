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
ROUTE_READOUT_CURRICULUM_SCHEMA = (
    "orion.qwen-visibility-route-readout-curriculum/v1"
)
TARGET_ROW_ROUTE_READOUT_CURRICULUM_SCHEMA = (
    "orion.qwen-visibility-target-row-route-readout-curriculum/v1"
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

V1J_TARGET_ROW_PAIRS = {
    "route151-step-000000": {
        "train": ((6, 24), (14, 28)),
        "held_out_target_rows": ((1, 20),),
    },
    "route151-step-000200": {
        "train": (),
        "held_out_target_rows": ((11, 31),),
    },
    "route151-step-000260": {
        "train": ((4, 15), (10, 28), (17, 31)),
        "held_out_target_rows": ((0, 8),),
    },
    "route151-step-000280": {
        "train": ((9, 3), (11, 5), (15, 20), (29, 23), (31, 25)),
        "held_out_target_rows": ((0, 2),),
    },
    "route151-step-000300": {
        "train": ((6, 24), (9, 26), (11, 30)),
        "held_out_target_rows": ((5, 16),),
    },
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


def paired_route_readout_permutations(
    count: int,
    on_route_source_index: int,
    off_route_source_index: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a matched pair differing only by a complete-row swap at F00."""

    values = (count, on_route_source_index, off_route_source_index, seed)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("paired route-readout arguments must be integers")
    count, on_route_source_index, off_route_source_index, seed = (
        int(value) for value in values
    )
    if count <= 1:
        raise ValueError("paired route readout requires at least two rows")
    if not 0 <= on_route_source_index < count:
        raise ValueError("ON_ROUTE source index is invalid")
    if not 0 <= off_route_source_index < count:
        raise ValueError("OFF_ROUTE source index is invalid")
    if on_route_source_index == off_route_source_index:
        raise ValueError("paired route rows must be distinct")
    generator = np.random.default_rng(seed)
    on_route = generator.permutation(count).astype(np.int64, copy=False)
    on_position = int(np.flatnonzero(on_route == on_route_source_index)[0])
    on_route[0], on_route[on_position] = (
        on_route[on_position],
        on_route[0],
    )
    off_position = int(np.flatnonzero(on_route == off_route_source_index)[0])
    off_route = on_route.copy()
    off_route[0], off_route[off_position] = (
        off_route[off_position],
        off_route[0],
    )
    if int(on_route[0]) != on_route_source_index:
        raise RuntimeError("failed to place the ON_ROUTE row at F00")
    if int(off_route[0]) != off_route_source_index:
        raise RuntimeError("failed to place the OFF_ROUTE row at F00")
    changed = np.flatnonzero(on_route != off_route).tolist()
    if changed != [0, off_position]:
        raise RuntimeError("matched route pair changed more than two row slots")
    return on_route, off_route


def _route_readout_example(
    record: Mapping[str, object],
    controls: Mapping[str, np.ndarray],
    mask: np.ndarray,
    feature_names: Sequence[str],
    permutation: Sequence[int],
    expected_answer: str,
    source_index: int,
    paired_source_index: int,
    split: str,
    pair_id: str,
    variant_index: int,
    seed: int,
    thresholds: GroundingThresholds,
) -> dict:
    targets, true_target = _target_for_example(
        controls,
        mask,
        feature_names,
        permutation,
        "route",
        0,
        thresholds,
    )
    if targets["true_u"] != expected_answer:
        raise RuntimeError("paired route-readout true target changed")
    if targets["spatial_shuffle"] == expected_answer:
        raise ValueError("paired route-readout shuffle target did not change")
    example_id = "%s-%s-%s-v%02d" % (
        pair_id,
        expected_answer.lower(),
        split,
        variant_index,
    )
    return {
        "example_id": example_id,
        "sample_id": record["sample_id"],
        "task_field": "route",
        "split": split,
        "pair_id": pair_id + "-%s-v%02d" % (split, variant_index),
        "question": row_grounding_question("route", "F00"),
        "query_frontier": "F00",
        "source_manifest_frontier": "F%02d" % int(source_index),
        "paired_source_manifest_frontier": "F%02d" % int(paired_source_index),
        "decoy_permutation_seed": int(seed),
        "sequence_permutation_new_to_manifest": [
            int(value) for value in permutation
        ],
        "expected_answer": expected_answer,
        "control_expected_answers": targets,
        "spatial_shuffle_changes_target": True,
        "target_evidence": true_target.evidence_dict(),
    }


def build_route151_route_readout_curriculum(
    base_manifest_path: Path,
    output_path: Path,
    train_pair_variants: int = 3,
    evaluation_pair_variants: int = 2,
    optimizer_steps: int = 240,
    seed: int = 1701,
    thresholds: GroundingThresholds = GroundingThresholds(),
) -> dict:
    """Build the V1f matched-pair route-scalar readout diagnostic."""

    integer_values = (
        train_pair_variants,
        evaluation_pair_variants,
        optimizer_steps,
        seed,
    )
    if any(
        isinstance(value, bool) or int(value) != value
        for value in integer_values
    ):
        raise ValueError("route-readout curriculum counts and seed must be integers")
    train_pair_variants = int(train_pair_variants)
    evaluation_pair_variants = int(evaluation_pair_variants)
    optimizer_steps = int(optimizer_steps)
    seed = int(seed)
    if train_pair_variants <= 0 or evaluation_pair_variants <= 0:
        raise ValueError("route-readout train/evaluation variants must be positive")
    if optimizer_steps <= 0 or optimizer_steps % 2:
        raise ValueError("route-readout optimizer steps must be positive and even")

    base_manifest_path = Path(base_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite route-readout curriculum: %s" % output_path
        )
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
    for sample_index, record in enumerate(base["records"]):
        for image, digest in zip(record["camera_images"], record["camera_sha256"]):
            if _sha256(Path(image)) != digest:
                raise ValueError("base grounding image hash changed")
        controls, mask, feature_names = _load_control_frontiers(record)
        identity = np.arange(int(mask.sum()), dtype=np.int64)
        candidates = defaultdict(list)
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
            if true_target.route == shuffle_target.route:
                continue
            candidates[true_target.route].append(
                (float(true_target.frontier_selection_score), row_index)
            )
        selected = {}
        for label in ("ON_ROUTE", "OFF_ROUTE"):
            ranked = sorted(candidates[label], key=lambda value: (-value[0], value[1]))
            if not ranked:
                raise ValueError(
                    "route-readout curriculum lacks shuffle-sensitive %s for %s"
                    % (label, record["sample_id"])
                )
            selected[label] = int(ranked[0][1])

        pair_prefix = "route-readout-%s" % record["sample_id"]
        split_variants = (
            ("train", train_pair_variants),
            ("held_out_order", evaluation_pair_variants),
        )
        variant_offset = 0
        for split, variant_count in split_variants:
            for variant_index in range(variant_count):
                variant_seed = seed + sample_index * 1000 + variant_offset
                variant_offset += 1
                on_permutation, off_permutation = paired_route_readout_permutations(
                    int(mask.sum()),
                    selected["ON_ROUTE"],
                    selected["OFF_ROUTE"],
                    variant_seed,
                )
                pair_id = pair_prefix
                examples.extend(
                    [
                        _route_readout_example(
                            record,
                            controls,
                            mask,
                            feature_names,
                            on_permutation,
                            "ON_ROUTE",
                            selected["ON_ROUTE"],
                            selected["OFF_ROUTE"],
                            split,
                            pair_id,
                            variant_index,
                            variant_seed,
                            thresholds,
                        ),
                        _route_readout_example(
                            record,
                            controls,
                            mask,
                            feature_names,
                            off_permutation,
                            "OFF_ROUTE",
                            selected["OFF_ROUTE"],
                            selected["ON_ROUTE"],
                            split,
                            pair_id,
                            variant_index,
                            variant_seed,
                            thresholds,
                        ),
                    ]
                )

    examples.sort(key=lambda value: value["example_id"])
    by_id = {example["example_id"]: example for example in examples}
    if len(by_id) != len(examples):
        raise RuntimeError("route-readout curriculum contains duplicate ids")
    training_ids = sorted(
        example["example_id"]
        for example in examples
        if example["split"] == "train"
    )
    evaluation_ids = sorted(
        example["example_id"]
        for example in examples
        if example["split"] == "held_out_order"
    )
    training_pools = {
        label: sorted(
            example_id
            for example_id in training_ids
            if by_id[example_id]["expected_answer"] == label
        )
        for label in ("ON_ROUTE", "OFF_ROUTE")
    }
    if any(not pool for pool in training_pools.values()):
        raise RuntimeError("route-readout training split is missing a label")
    schedule = []
    for round_index in range(optimizer_steps // 2):
        for label in ("ON_ROUTE", "OFF_ROUTE"):
            pool = training_pools[label]
            schedule.append(pool[round_index % len(pool)])
    train_label_counts = Counter(by_id[value]["expected_answer"] for value in training_ids)
    evaluation_label_counts = Counter(
        by_id[value]["expected_answer"] for value in evaluation_ids
    )
    expected_samples = {
        "route151-step-000000",
        "route151-step-000200",
        "route151-step-000260",
        "route151-step-000280",
        "route151-step-000300",
    }
    for sample_id in expected_samples:
        for label in ("ON_ROUTE", "OFF_ROUTE"):
            train_orders = {
                tuple(by_id[value]["sequence_permutation_new_to_manifest"])
                for value in training_ids
                if by_id[value]["sample_id"] == sample_id
                and by_id[value]["expected_answer"] == label
            }
            evaluation_orders = {
                tuple(by_id[value]["sequence_permutation_new_to_manifest"])
                for value in evaluation_ids
                if by_id[value]["sample_id"] == sample_id
                and by_id[value]["expected_answer"] == label
            }
            if len(train_orders) != train_pair_variants:
                raise RuntimeError("route-readout training row orders repeated")
            if len(evaluation_orders) != evaluation_pair_variants:
                raise RuntimeError("route-readout evaluation row orders repeated")
            if train_orders & evaluation_orders:
                raise RuntimeError("route-readout evaluation row order leaked")
    schedule_label_counts = Counter(
        by_id[value]["expected_answer"] for value in schedule
    )
    if {str(record["sample_id"]) for record in base["records"]} != expected_samples:
        raise ValueError("V1f requires the five immutable Route 151 records")
    expected_train_per_label = len(expected_samples) * train_pair_variants
    expected_eval_per_label = len(expected_samples) * evaluation_pair_variants
    if train_label_counts != Counter(
        {"ON_ROUTE": expected_train_per_label, "OFF_ROUTE": expected_train_per_label}
    ):
        raise RuntimeError("route-readout training labels are not balanced")
    if evaluation_label_counts != Counter(
        {"ON_ROUTE": expected_eval_per_label, "OFF_ROUTE": expected_eval_per_label}
    ):
        raise RuntimeError("route-readout evaluation labels are not balanced")
    if schedule_label_counts != Counter(
        {"ON_ROUTE": optimizer_steps // 2, "OFF_ROUTE": optimizer_steps // 2}
    ):
        raise RuntimeError("route-readout optimizer labels are not balanced")

    curriculum = {
        "schema": ROUTE_READOUT_CURRICULUM_SCHEMA,
        "purpose": "Route 151 matched-pair route readout plumbing diagnostic only",
        "base_manifest_path": str(base_manifest_path),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "reportable_generalization": False,
        "oracle_depth": True,
        "hidden_actor_labels_used": False,
        "controls_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "complete_row_permutations_only": True,
        "matched_pairs_differ_only_by_query_row_swap": True,
        "non_query_order_randomized": True,
        "held_out_order_evaluation": True,
        "held_out_order_disjoint_verified": True,
        "spatial_shuffle_target_changed_for_every_example": True,
        "query_frontier": "F00",
        "thresholds": thresholds.as_dict(),
        "seed": seed,
        "train_pair_variants_per_sample": train_pair_variants,
        "evaluation_pair_variants_per_sample": evaluation_pair_variants,
        "example_count": len(examples),
        "training_example_ids": training_ids,
        "evaluation_example_ids": evaluation_ids,
        "training_label_counts": dict(sorted(train_label_counts.items())),
        "evaluation_label_counts": dict(sorted(evaluation_label_counts.items())),
        "optimizer_steps": optimizer_steps,
        "optimizer_label_counts": dict(sorted(schedule_label_counts.items())),
        "training_schedule": schedule,
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return curriculum


def build_route151_target_row_route_readout_curriculum(
    base_manifest_path: Path,
    output_path: Path,
    train_pair_variants: int = 3,
    evaluation_pair_variants: int = 2,
    optimizer_steps: int = 240,
    seed: int = 1701,
    thresholds: GroundingThresholds = GroundingThresholds(),
) -> dict:
    """Build V1j with disjoint queried target rows and one held-out frame."""

    integer_values = (
        train_pair_variants,
        evaluation_pair_variants,
        optimizer_steps,
        seed,
    )
    if any(
        isinstance(value, bool) or int(value) != value
        for value in integer_values
    ):
        raise ValueError("target-row curriculum counts and seed must be integers")
    train_pair_variants = int(train_pair_variants)
    evaluation_pair_variants = int(evaluation_pair_variants)
    optimizer_steps = int(optimizer_steps)
    seed = int(seed)
    if train_pair_variants <= 0 or evaluation_pair_variants <= 0:
        raise ValueError("target-row train/evaluation variants must be positive")
    if optimizer_steps <= 0 or optimizer_steps % 2:
        raise ValueError("target-row optimizer steps must be positive and even")

    base_manifest_path = Path(base_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            "refusing to overwrite target-row curriculum: %s" % output_path
        )
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

    records = {str(record["sample_id"]): record for record in base["records"]}
    if set(records) != set(V1J_TARGET_ROW_PAIRS):
        raise ValueError("V1j requires the five immutable Route 151 records")
    examples = []
    pair_ordinal = 0
    for sample_id in sorted(V1J_TARGET_ROW_PAIRS):
        record = records[sample_id]
        for image, digest in zip(record["camera_images"], record["camera_sha256"]):
            if _sha256(Path(image)) != digest:
                raise ValueError("base grounding image hash changed")
        controls, mask, feature_names = _load_control_frontiers(record)
        split_pairs = V1J_TARGET_ROW_PAIRS[sample_id]
        for split in ("train", "held_out_target_rows"):
            variant_count = (
                train_pair_variants
                if split == "train"
                else evaluation_pair_variants
            )
            for pair_index, (on_index, off_index) in enumerate(split_pairs[split]):
                pair_prefix = (
                    "target-row-readout-%s-p%02d-on-F%02d-off-F%02d"
                    % (sample_id, pair_index, on_index, off_index)
                )
                seen_orders = {"ON_ROUTE": set(), "OFF_ROUTE": set()}
                for variant_index in range(variant_count):
                    variant_seed = seed + pair_ordinal * 100 + variant_index
                    on_permutation, off_permutation = paired_route_readout_permutations(
                        int(mask.sum()), on_index, off_index, variant_seed
                    )
                    pair_examples = (
                        _route_readout_example(
                            record,
                            controls,
                            mask,
                            feature_names,
                            on_permutation,
                            "ON_ROUTE",
                            on_index,
                            off_index,
                            split,
                            pair_prefix,
                            variant_index,
                            variant_seed,
                            thresholds,
                        ),
                        _route_readout_example(
                            record,
                            controls,
                            mask,
                            feature_names,
                            off_permutation,
                            "OFF_ROUTE",
                            off_index,
                            on_index,
                            split,
                            pair_prefix,
                            variant_index,
                            variant_seed,
                            thresholds,
                        ),
                    )
                    for example in pair_examples:
                        order = tuple(example["sequence_permutation_new_to_manifest"])
                        label = str(example["expected_answer"])
                        if order in seen_orders[label]:
                            raise RuntimeError("target-row decoy order repeated")
                        seen_orders[label].add(order)
                        examples.append(example)
                pair_ordinal += 1

    examples.sort(key=lambda value: value["example_id"])
    by_id = {example["example_id"]: example for example in examples}
    if len(by_id) != len(examples):
        raise RuntimeError("target-row curriculum contains duplicate ids")
    training_ids = sorted(
        example["example_id"]
        for example in examples
        if example["split"] == "train"
    )
    evaluation_ids = sorted(
        example["example_id"]
        for example in examples
        if example["split"] == "held_out_target_rows"
    )
    target_rows = {"train": defaultdict(set), "held_out_target_rows": defaultdict(set)}
    for example in examples:
        target_rows[example["split"]][example["sample_id"]].add(
            int(str(example["source_manifest_frontier"])[1:])
        )
    for sample_id in V1J_TARGET_ROW_PAIRS:
        if target_rows["train"][sample_id] & target_rows["held_out_target_rows"][sample_id]:
            raise RuntimeError("V1j queried target row leaked across splits")
    held_out_frame = "route151-step-000200"
    if target_rows["train"][held_out_frame]:
        raise RuntimeError("V1j held-out frame entered the training split")

    training_pools = {
        label: sorted(
            example_id
            for example_id in training_ids
            if by_id[example_id]["expected_answer"] == label
        )
        for label in ("ON_ROUTE", "OFF_ROUTE")
    }
    schedule = []
    for round_index in range(optimizer_steps // 2):
        for label in ("ON_ROUTE", "OFF_ROUTE"):
            pool = training_pools[label]
            if not pool:
                raise RuntimeError("target-row training split is missing a label")
            schedule.append(pool[round_index % len(pool)])

    training_label_counts = Counter(
        by_id[value]["expected_answer"] for value in training_ids
    )
    evaluation_label_counts = Counter(
        by_id[value]["expected_answer"] for value in evaluation_ids
    )
    optimizer_label_counts = Counter(
        by_id[value]["expected_answer"] for value in schedule
    )
    if training_label_counts != Counter({"ON_ROUTE": 39, "OFF_ROUTE": 39}):
        raise RuntimeError("V1j training labels are not balanced")
    if evaluation_label_counts != Counter({"ON_ROUTE": 10, "OFF_ROUTE": 10}):
        raise RuntimeError("V1j evaluation labels are not balanced")
    if optimizer_label_counts != Counter({"ON_ROUTE": 120, "OFF_ROUTE": 120}):
        raise RuntimeError("V1j optimizer labels are not balanced")

    serialized_split = {
        sample_id: {
            split: [
                {"on_route_row": int(on_index), "off_route_row": int(off_index)}
                for on_index, off_index in V1J_TARGET_ROW_PAIRS[sample_id][split]
            ]
            for split in ("train", "held_out_target_rows")
        }
        for sample_id in sorted(V1J_TARGET_ROW_PAIRS)
    }
    curriculum = {
        "schema": TARGET_ROW_ROUTE_READOUT_CURRICULUM_SCHEMA,
        "purpose": (
            "Route 151 queried-target-row-disjoint matched-pair route readout "
            "plumbing diagnostic only"
        ),
        "base_manifest_path": str(base_manifest_path),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "reportable_generalization": False,
        "oracle_depth": True,
        "hidden_actor_labels_used": False,
        "controls_used_for_optimizer": False,
        "spatial_shuffle_examples_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "complete_row_permutations_only": True,
        "matched_pairs_differ_only_by_query_row_swap": True,
        "non_query_order_randomized": True,
        "held_out_target_row_evaluation": True,
        "queried_target_rows_disjoint_verified": True,
        "fully_held_out_frame": held_out_frame,
        "spatial_shuffle_target_changed_for_every_example": True,
        "spatial_shuffle_evaluation_role": "reported_diagnostic_not_hard_gate",
        "query_frontier": "F00",
        "thresholds": thresholds.as_dict(),
        "seed": seed,
        "train_pair_variants": train_pair_variants,
        "evaluation_pair_variants": evaluation_pair_variants,
        "distinct_training_target_pairs": 13,
        "distinct_evaluation_target_pairs": 5,
        "target_row_split": serialized_split,
        "example_count": len(examples),
        "training_example_ids": training_ids,
        "evaluation_example_ids": evaluation_ids,
        "training_label_counts": dict(sorted(training_label_counts.items())),
        "evaluation_label_counts": dict(sorted(evaluation_label_counts.items())),
        "optimizer_steps": optimizer_steps,
        "optimizer_label_counts": dict(sorted(optimizer_label_counts.items())),
        "training_schedule": schedule,
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return curriculum
