import copy

import pytest
import torch

from uq_estimator.oracle_adapter_training import (
    ROLE_FAILED_ORACLE,
    ROLE_OFF_FAILURE,
    ROLE_ORACLE_IMITATION,
    OracleAdapterTensorDataset,
    OracleAdapterTrainingError,
    assign_rollout_roles,
    make_mock_samples,
    oracle_adapter_loss,
    prepare_examples,
    split_examples_by_route,
    train_adapter,
)
from uq_estimator.trajectory_adapter import PathRiskTrajectoryAdapter


def prepared_mock():
    samples = make_mock_samples(seed=7)
    roles, role_audit = assign_rollout_roles(samples)
    examples, example_audit = prepare_examples(samples, roles)
    return samples, roles, examples, role_audit, example_audit


def test_only_successful_controlled_stop_rollout_enters_imitation():
    _, roles, examples, role_audit, example_audit = prepared_mock()
    assert role_audit["rollout_role_counts"] == {
        ROLE_OFF_FAILURE: 2,
        ROLE_ORACLE_IMITATION: 2,
    }
    assert example_audit["imitation_examples"] == 8
    assert example_audit["failed_or_off_trajectory_imitation_examples"] == 0
    for example in examples:
        if example.role == ROLE_OFF_FAILURE:
            assert example.imitation_weight == 0.0
            assert example.preservation_weight == 0.0
            assert example.stop_weight == 0.0
        elif example.imitation_weight > 0:
            assert example.role == ROLE_ORACLE_IMITATION
            assert example.path_risk == 1.0


def test_successful_oracle_without_controlled_stop_is_not_positive():
    samples = make_mock_samples(seed=3)
    oracle_rollout = samples[0]["source"]["rollout_id"]
    for sample in samples:
        if sample["source"]["rollout_id"] == oracle_rollout:
            sample["labels"]["stop_required"] = False
    roles, _ = assign_rollout_roles(samples)
    assert roles[oracle_rollout] == ROLE_FAILED_ORACLE
    examples, audit = prepare_examples(samples, roles)
    assert all(
        example.imitation_weight == 0
        for example in examples
        if example.rollout_id == oracle_rollout
    )
    assert audit["failed_or_off_trajectory_imitation_examples"] == 0


def test_route_disjoint_split_has_no_route_overlap():
    _, _, examples, _, _ = prepared_mock()
    split = split_examples_by_route(
        examples, validation_fraction=0.5, seed=11
    )
    assert split.mode == "route_disjoint"
    assert set(split.train_routes).isdisjoint(split.validation_routes)
    assert {
        examples[index].route_key for index in split.train_indices
    } == set(split.train_routes)
    assert {
        examples[index].route_key for index in split.validation_indices
    } == set(split.validation_routes)


def test_single_route_requires_explicit_smoke_mode():
    _, _, examples, _, _ = prepared_mock()
    route = examples[0].route_key
    single_route = [example for example in examples if example.route_key == route]
    with pytest.raises(OracleAdapterTrainingError, match="at least two routes"):
        split_examples_by_route(single_route, validation_fraction=0.2, seed=1)
    split = split_examples_by_route(
        single_route,
        validation_fraction=0.2,
        seed=1,
        allow_single_route_smoke=True,
    )
    assert split.mode == "single_route_smoke_not_route_disjoint"
    assert split.train_routes == split.validation_routes == [route]
    assert any(single_route[index].imitation_weight > 0 for index in split.train_indices)
    assert any(
        single_route[index].imitation_weight > 0
        for index in split.validation_indices
    )


def test_diagnostic_target_cannot_change_loss():
    model = PathRiskTrajectoryAdapter(hidden_dim=8, max_residual_m=1.0)
    base = torch.ones(2, 6, 2)
    batch = {
        "target": base.clone(),
        "mask": torch.ones(2, 6),
        "path_risk": torch.ones(2),
        "stop_target": torch.tensor([1.0, 0.0]),
        "imitation_weight": torch.tensor([1.0, 0.0]),
        "preservation_weight": torch.tensor([0.0, 0.0]),
        "stop_weight": torch.tensor([1.0, 0.0]),
        "recover": torch.zeros(2),
    }
    batch["target"][0] = 0.0
    first = oracle_adapter_loss(
        model(base[:, None], batch["path_risk"][:, None]), batch
    )["total"]
    changed = copy.deepcopy(batch)
    changed["target"][1] = 10000.0
    second = oracle_adapter_loss(
        model(base[:, None], batch["path_risk"][:, None]), changed
    )["total"]
    assert torch.equal(first, second)


def test_mock_training_pipeline_produces_finite_statistics():
    _, _, examples, _, _ = prepared_mock()
    split = split_examples_by_route(
        examples, validation_fraction=0.5, seed=9
    )
    model, history, best = train_adapter(
        examples,
        split,
        hidden_dim=8,
        max_residual_m=1.0,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu",
        seed=9,
        max_steps_per_epoch=2,
    )
    assert isinstance(model, PathRiskTrajectoryAdapter)
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(best))
    assert history[0]["train"]["imitation_points"] > 0
    assert history[0]["validation"]["imitation_points"] > 0


def test_tensor_dataset_uses_base_target_for_off_failure():
    _, _, examples, _, _ = prepared_mock()
    index = next(
        index for index, example in enumerate(examples)
        if example.role == ROLE_OFF_FAILURE
    )
    item = OracleAdapterTensorDataset(examples, [index])[0]
    assert torch.equal(item["base"], item["target"])
    assert item["imitation_weight"] == 0
    assert item["stop_weight"] == 0
