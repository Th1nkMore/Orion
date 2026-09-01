"""Tests for generator-independent observation-UQ v3."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from uq_estimator.observation_uq_v3 import (
    _binary_auc,
    _spearman,
    CleanConditionalTeacher,
    ObservationUQAdapter,
    ObservationUQError,
    conditional_surprise,
    examples_from_paired_records,
    make_mock_examples,
    mask_phase,
    run_observation_uq_training,
    split_examples_for_training,
    train_clean_teacher_epoch,
    validate_family_protocol,
)


def test_rank_auc_matches_pairwise_definition_with_ties_and_large_input():
    scores = torch.tensor([0.1, 0.4, 0.4, 0.8, 0.9])
    labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool)
    positive = scores[labels]
    negative = scores[~labels]
    expected = (
        (positive[:, None] > negative[None, :]).float()
        + 0.5 * (positive[:, None] == negative[None, :]).float()
    ).mean()
    assert _binary_auc(scores, labels) == pytest.approx(float(expected))

    large_scores = torch.linspace(0, 1, 1_000_000)
    large_labels = large_scores > 0.5
    assert _binary_auc(large_scores, large_labels) == pytest.approx(1.0)


def test_spearman_uses_average_ranks_for_repeated_severity_levels():
    severity = torch.tensor([1.0, 1.0, 3.0, 3.0])
    score = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # Average severity ranks are [0.5, 0.5, 2.5, 2.5], not ordinal
    # [0, 1, 2, 3].  The latter would incorrectly report perfect rho.
    assert _spearman(severity, score) == pytest.approx(0.8944271909999159)


def test_nine_mask_phases_partition_grid_exactly_once():
    masks = [mask_phase(7, 9, phase, torch.device("cpu")) for phase in range(9)]
    coverage = torch.stack(masks).sum(dim=0)
    assert torch.equal(coverage, torch.ones_like(coverage))
    assert all(bool(mask.any()) for mask in masks)


def test_teacher_masks_same_target_in_current_and_previous_context():
    torch.manual_seed(2)
    teacher = CleanConditionalTeacher(
        8, hidden_dim=8, max_views=1, mask_block_size=1, mask_halo=0
    ).eval()
    current = torch.randn(1, 1, 4, 4, 8)
    previous_a = torch.randn_like(current)
    previous_b = previous_a.clone()
    target = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    target[:, :, 1, 2] = True
    current_b = current.clone()
    current_b[:, :, 1, 2] += 100.0
    previous_b[:, :, 1, 2] -= 100.0
    valid = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        prediction_a = teacher(
            current, target, previous_a, valid, context_mask=target
        )
        prediction_b = teacher(
            current_b, target, previous_b, valid, context_mask=target
        )
    torch.testing.assert_close(
        prediction_a[:, :, 1, 2], prediction_b[:, :, 1, 2]
    )


def test_family_protocol_rejects_leakage_and_training_split_holds_out_family():
    with pytest.raises(ObservationUQError, match="leakage"):
        validate_family_protocol(["local_blur"], ["local_blur"])

    examples = make_mock_examples(
        feature_dim=8, routes=6, frames_per_route=2, height=4, width=4, seed=3
    )
    splits = split_examples_for_training(
        examples, ["local_blur", "local_dark"], ["local_glare"]
    )
    assert {item.family for item in splits["teacher_train"]} == {"clean"}
    assert {item.family for item in splits["student_train"]} == {
        "clean",
        "local_blur",
        "local_dark",
    }
    assert {item.family for item in splits["heldout_route_and_family"]} == {
        "clean",
        "local_glare",
    }


def test_teacher_training_fails_closed_if_corrupt_example_is_present():
    examples = make_mock_examples(
        feature_dim=8, routes=6, frames_per_route=2, height=4, width=4, seed=4
    )
    corrupt = next(item for item in examples if item.family != "clean")
    teacher = CleanConditionalTeacher(8, hidden_dim=8, max_views=2)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    with pytest.raises(ObservationUQError, match="clean examples"):
        train_clean_teacher_epoch(
            [teacher], [corrupt], [optimizer], 1, torch.device("cpu"), seed=0
        )


def test_model_outputs_are_independent_of_audit_only_corruption_metadata():
    examples = make_mock_examples(
        feature_dim=8, routes=6, frames_per_route=2, height=4, width=4, seed=5
    )
    source = next(item for item in examples if item.family == "local_glare")
    relabelled = replace(
        source,
        sample_id=source.sample_id + "/relabelled",
        family="invented_label",
        severity=999.0,
        corruption_mask=1.0 - source.corruption_mask,
    )
    current = torch.stack((source.current, relabelled.current))
    previous = torch.stack((source.previous, relabelled.previous))
    valid = torch.tensor((source.previous_valid, relabelled.previous_valid))
    teacher = CleanConditionalTeacher(8, hidden_dim=8, max_views=2).eval()
    adapter = ObservationUQAdapter(8, hidden_dim=8, max_views=2).eval()
    with torch.no_grad():
        target = conditional_surprise([teacher], current, previous, valid)
        prediction = adapter(current, previous, valid)
    torch.testing.assert_close(target[0], target[1])
    torch.testing.assert_close(prediction[0], prediction[1])


class _ForbiddenActualTargetRecord(SimpleNamespace):
    @property
    def error_severity_target(self):  # pragma: no cover - only called on regression
        raise AssertionError("v3 must not read actual targets")

    @property
    def failure_event_target(self):  # pragma: no cover - only called on regression
        raise AssertionError("v3 must not read actual targets")


def test_paired_conversion_ignores_old_actual_and_representation_targets():
    feature = torch.randn(2, 4, 6)
    record = _ForbiddenActualTargetRecord(
        route_id="route_train",
        severity=2.0,
        clean_patch_features=feature,
        observed_patch_features=feature + 0.2,
        corruption_mask=torch.ones(2, 4),
        metadata={
            "source_identity": {"sample_token": "sample_0", "frame_idx": 0},
            "corruption": {"corruption": "local_blur"},
        },
    )
    examples = examples_from_paired_records(
        [record], {"route_train": "train"}, patch_height=2, patch_width=2
    )
    assert len(examples) == 2
    assert {item.family for item in examples} == {"clean", "local_blur"}


def test_bounded_mock_training_converges_without_family_label_supervision(tmp_path):
    examples = make_mock_examples(
        feature_dim=10,
        routes=8,
        frames_per_route=3,
        views=2,
        height=5,
        width=5,
        seed=9,
    )
    checkpoint = run_observation_uq_training(
        examples=examples,
        train_families=["local_blur", "local_dark"],
        heldout_families=["local_glare"],
        output_path=tmp_path / "v3.pt",
        feature_dim=10,
        hidden_dim=24,
        teacher_members=1,
        teacher_epochs=5,
        adapter_epochs=8,
        batch_size=8,
        learning_rate=3e-3,
        seed=9,
        device="cpu",
    )
    teacher = checkpoint["history"]["teacher_train"]
    adapter = checkpoint["history"]["adapter_train"]
    heldout = checkpoint["evaluations"]["heldout_route_and_family"]["by_family"]

    assert teacher[-1]["loss"] < teacher[0]["loss"]
    assert adapter[-1]["loss"] < adapter[0]["loss"]
    assert checkpoint["data_attestation"]["teacher_example_families"] == ["clean"]
    assert checkpoint["data_attestation"]["actual_target_tensor_read"] is False
    assert heldout["local_glare"]["teacher_target_mean"] > heldout["clean"][
        "teacher_target_mean"
    ]
    assert (tmp_path / "v3.report.json").exists()
