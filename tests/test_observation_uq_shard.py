"""Tests for deduplicated clean-first feature shards and Teacher gate."""

from __future__ import annotations

import re

from uq_estimator.observation_uq_shard import (
    FEATURE_SHARD_SCHEMA_VERSION,
    examples_from_feature_shard,
    load_feature_shard,
    save_feature_shard,
    validate_feature_shard,
)
from uq_estimator.observation_uq_v3 import (
    make_mock_examples,
    run_clean_only_adapter_training,
    run_teacher_viability_training,
)


def _mock_shard():
    examples = make_mock_examples(
        feature_dim=12,
        routes=12,
        frames_per_route=4,
        views=2,
        height=6,
        width=6,
        seed=23,
    )
    clean = [item for item in examples if item.family == "clean"]
    clean_index = {}
    clean_features = []
    clean_items = []
    for index, item in enumerate(clean):
        frame = int(re.search(r"frame_(\d+)", item.sample_id).group(1))
        clean_index[(item.route_id, frame)] = index
        clean_features.append(item.current.half())
        clean_items.append(
            {
                "clean_index": index,
                "sample_id": item.sample_id.rsplit("/clean", 1)[0],
                "route_id": item.route_id,
                "town": "MockTown",
                "frame_idx": frame,
                "split": item.split,
            }
        )
    observed_features = []
    observed_items = []
    for item in examples:
        if item.family != "local_glare" or item.split == "train":
            continue
        frame = int(re.search(r"frame_(\d+)", item.sample_id).group(1))
        index = len(observed_features)
        observed_features.append(item.current.half())
        observed_items.append(
            {
                "observed_index": index,
                "clean_index": clean_index[(item.route_id, frame)],
                "sample_id": item.sample_id,
                "route_id": item.route_id,
                "town": "MockTown",
                "frame_idx": frame,
                "split": item.split,
                "family": item.family,
                "severity": item.severity,
                "corruption_mask": item.corruption_mask.half(),
            }
        )
    return {
        "schema_version": FEATURE_SHARD_SCHEMA_VERSION,
        "clean_features": clean_features,
        "clean_items": clean_items,
        "observed_features": observed_features,
        "observed_items": observed_items,
        "provenance": {"mock": True, "clean_token_deduplicated": True},
    }


def test_feature_shard_round_trip_deduplicates_clean_and_preserves_sequence(tmp_path):
    payload = _mock_shard()
    summary = validate_feature_shard(payload)
    assert summary["clean_count"] == 48
    assert summary["observed_count"] == 48
    assert summary["routes_by_split"] == {"held_out": 2, "train": 8, "validation": 2}
    path = tmp_path / "shard.pt"
    save_feature_shard(payload, path)
    examples = examples_from_feature_shard(load_feature_shard(path))
    assert len(examples) == 96
    train_clean = [item for item in examples if item.split == "train"]
    assert {item.family for item in train_clean} == {"clean"}
    assert sum(item.previous_valid for item in train_clean) == 8 * 3


def test_teacher_viability_gate_uses_many_clean_examples_and_no_adapter(tmp_path):
    examples = examples_from_feature_shard(_mock_shard())
    first_output = tmp_path / "teacher_first.pt"
    first = run_teacher_viability_training(
        examples=examples,
        heldout_families=["local_glare"],
        output_path=first_output,
        feature_dim=12,
        hidden_dim=32,
        teacher_members=1,
        teacher_epochs=4,
        batch_size=8,
        learning_rate=3e-3,
        validation_interval=2,
        seed=23,
        device="cpu",
    )
    assert len(first["history"]["teacher_train"]) == 4
    assert first_output.with_suffix(".progress.pt").exists()
    checkpoint = run_teacher_viability_training(
        examples=examples,
        heldout_families=["local_glare"],
        output_path=tmp_path / "teacher.pt",
        feature_dim=12,
        hidden_dim=32,
        teacher_members=1,
        teacher_epochs=8,
        batch_size=8,
        learning_rate=3e-3,
        validation_interval=2,
        resume_path=first_output.with_suffix(".progress.pt"),
        seed=23,
        device="cpu",
    )
    history = checkpoint["history"]["teacher_train"]
    assert len(history) == 8
    assert history[-1]["loss"] < history[0]["loss"]
    assert checkpoint["data_attestation"] == {
        "teacher_optimizer_example_count": 32,
        "teacher_optimizer_families": ["clean"],
        "teacher_optimizer_route_count": 8,
        "corruption_metadata_used_as_target": False,
        "actual_target_tensor_read": False,
        "adapter_trained": False,
    }
    assert checkpoint["gate_inputs"]["validation_heldout_family"]["families"][
        "local_glare"
    ]["positive_uplift"]
    assert checkpoint["gate_inputs"]["heldout_route_and_family"]["families"][
        "local_glare"
    ]["positive_uplift"]

    adapter_first_output = tmp_path / "adapter_first.pt"
    adapter_first = run_clean_only_adapter_training(
        examples=examples,
        teacher_checkpoint=checkpoint,
        output_path=adapter_first_output,
        adapter_epochs=2,
        batch_size=8,
        learning_rate=3e-3,
        seed=123,
        device="cpu",
    )
    assert len(adapter_first["history"]["adapter_train"]) == 2
    adapter = run_clean_only_adapter_training(
        examples=examples,
        teacher_checkpoint=checkpoint,
        output_path=tmp_path / "adapter.pt",
        adapter_epochs=4,
        batch_size=8,
        learning_rate=3e-3,
        resume_path=adapter_first_output.with_suffix(".progress.pt"),
        seed=123,
        device="cpu",
    )
    assert len(adapter["history"]["adapter_train"]) == 4
    assert adapter["data_attestation"]["adapter_optimizer_families"] == [
        "clean"
    ]
    assert adapter["data_attestation"]["corruption_observation_used_by_optimizer"] is False
