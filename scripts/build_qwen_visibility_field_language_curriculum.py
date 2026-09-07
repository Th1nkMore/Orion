#!/usr/bin/env python3
"""Build the immutable A1b multi-field threshold curriculum."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np


SCHEMA = "orion.qwen-visibility-field-language-curriculum/v1"
FIELDS = {
    "center_x_normalized": (0.25, "normalized ego-forward position; larger means farther forward"),
    "center_y_normalized": (0.0, "normalized ego-left position; positive is left and negative is right"),
    "occluded_unknown_height_ratio": (0.5, "fraction of the local vertical evidence that is occluded unknown space"),
    "observation_age_normalized": (0.2, "normalized time since the region was last directly observed"),
    "route_weight_mean": (0.2, "overlap with the planned route; at least 0.2 is route-relevant"),
    "stopping_weight_mean": (0.1, "overlap with the current stopping envelope"),
    "urgency_max": (0.05, "maximum deterministic frontier-route-stopping exposure; not an actor or action prediction"),
    "frontier_stopping_margin_normalized": (0.0, "signed normalized distance relative to the stopping envelope"),
}
SYSTEM_PROMPT = (
    "You read authoritative continuous visibility-belief records inserted between the camera tokens and this question. "
    "G00-G15 are global cells and F00-F31 are frame-local observed/unknown frontiers. "
    "Each record keeps an explicit slot and named physical fields. Values are normalized physical measurements, not hidden-actor claims. "
    + " ".join("%s means %s." % (name, description) for name, (_, description) in FIELDS.items())
    + " Read only the requested slot and field."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frontier_values(record, control):
    prefix = {"true_u": "visibility_tokens", "zero_u": "visibility_tokens_zero_u", "spatial_shuffle": "visibility_tokens_spatial_shuffle"}[control]
    with np.load(record["token_artifact"], allow_pickle=False) as artifact:
        rows = np.asarray(artifact[prefix + "_frontier"], dtype=np.float32)
        mask = np.asarray(artifact[prefix + "_frontier_mask"], dtype=bool)
        names = [str(x) for x in artifact[prefix + "_feature_names"].tolist()]
    valid = int(mask.sum())
    order = np.asarray(record["frontier_permutation_new_to_old"], dtype=np.int64)
    rows = rows.copy()
    rows[:valid] = rows[:valid][order]
    return rows[:valid], names


def answer(value, threshold):
    return "AT_OR_ABOVE" if float(value) >= float(threshold) else "BELOW"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202609071)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite A1b curriculum")
    if args.seed != 202609071 or sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("A1b seed or manifest lineage changed")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = []
    for record in manifest["records"]:
        true_rows, names = frontier_values(record, "true_u")
        zero_rows, zero_names = frontier_values(record, "zero_u")
        shuffled_rows, shuffled_names = frontier_values(record, "spatial_shuffle")
        if names != zero_names or names != shuffled_names:
            raise ValueError("control field order changed")
        for field_offset, (field, (threshold, _)) in enumerate(FIELDS.items()):
            digest = hashlib.sha256((record["sample_id"] + "/" + field).encode()).digest()
            slot = int.from_bytes(digest[:4], "big") % len(true_rows)
            expected = answer(true_rows[slot, names.index(field)], threshold)
            controls = {
                "true_u": expected,
                "zero_u": answer(zero_rows[slot, names.index(field)], threshold),
                "spatial_shuffle": answer(shuffled_rows[slot, names.index(field)], threshold),
            }
            examples.append({
                "example_id": "%s-%s" % (record["sample_id"], field),
                "sample_id": record["sample_id"],
                "split": record["split"],
                "task_field": field,
                "query_frontier": "F%02d" % slot,
                "threshold": threshold,
                "question": (
                    "Consider the continuous visibility-belief records. Is F%02d.%s at least %.6f? "
                    "Reply with exactly AT_OR_ABOVE or BELOW, with no explanation."
                    % (slot, field, threshold)
                ),
                "expected_answer": expected,
                "control_expected_answers": controls,
                "sequence_permutation_new_to_manifest": list(range(len(true_rows))),
                "field_offset": field_offset,
            })
    by_id = {row["example_id"]: row for row in examples}
    training = [row for row in examples if row["split"] == "train"]
    schedule = []
    rng = random.Random(args.seed)
    for field in FIELDS:
        grouped = defaultdict(list)
        for row in training:
            if row["task_field"] == field:
                grouped[row["expected_answer"]].append(row["example_id"])
        if set(grouped) != {"AT_OR_ABOVE", "BELOW"}:
            raise ValueError("training field lacks both labels: %s" % field)
        for label in ("AT_OR_ABOVE", "BELOW"):
            pool = sorted(grouped[label])
            rng.shuffle(pool)
            schedule.extend(pool[index % len(pool)] for index in range(35))
    rng.shuffle(schedule)
    evaluation = [row["example_id"] for row in examples if row["split"] != "train"]
    result = {
        "schema": SCHEMA,
        "base_manifest_path": str(args.manifest.resolve()),
        "base_manifest_sha256": args.expected_manifest_sha256,
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "fields": {key: {"threshold": value[0], "meaning": value[1]} for key, value in FIELDS.items()},
        "examples": examples,
        "example_count": len(examples),
        "training_example_ids": [row["example_id"] for row in training],
        "evaluation_example_ids": evaluation,
        "training_schedule": schedule,
        "optimizer_steps": len(schedule),
        "schedule_field_counts": dict(sorted(Counter(by_id[x]["task_field"] for x in schedule).items())),
        "schedule_label_counts": dict(sorted(Counter(by_id[x]["expected_answer"] for x in schedule).items())),
        "evaluation_counts": {
            split: {
                "examples": sum(row["split"] == split for row in examples),
                "labels": dict(sorted(Counter(row["expected_answer"] for row in examples if row["split"] == split).items())),
            }
            for split in ("validation", "held_out")
        },
        "route_disjoint_splits": True,
        "controls_used_for_optimizer": False,
        "hidden_actor_labels_used": False,
        "planning_expert_used_for_optimizer": False,
        "spatial_shuffle_role": "reported_diagnostic_not_hard_gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("example_count", "optimizer_steps", "schedule_field_counts", "schedule_label_counts", "evaluation_counts")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
