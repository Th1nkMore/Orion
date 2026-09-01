import argparse

import pytest
import torch

from scripts.train_stage2l_v11_identifiable_smoke import (
    OneEventPerStepSampler,
    PROTOCOL_SCHEMA,
    _language_release_checks,
    _load_v101_contextual_relevance,
    _protocol_checks,
    _protocol_input_hashes,
    _task_relevance_rows,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRelevanceMapHead,
    ViewAlignedTaskRelevanceQueryTokenizer,
)


def test_one_event_sampler_round_robins_all_thirteen_events():
    event_groups = {
        "event-%02d" % index: ("g%02da" % index, "g%02db" % index)
        for index in range(13)
    }
    sampler = OneEventPerStepSampler(event_groups, seed=4)
    first_cycle = [sampler.next() for _ in range(13)]
    second_cycle = [sampler.next() for _ in range(13)]
    assert {value[:3] for value in first_cycle} == {"g%02d" % i for i in range(13)}
    assert all(left != right for left, right in zip(first_cycle, second_cycle))


class _Rows:
    def row(self, group_id, variant, family):
        assert group_id == "g"
        assert family == "task_relevance"
        answer = {
            "zero_uq": "none",
            "on_path_uq": "high",
            "off_path_uq": "low",
            "view_shuffled_uq": "low",
        }[variant]
        return {"conversation": [{"value": "same question"}, {"value": answer}]}


def test_task_relevance_candidates_allow_shared_semantic_classes():
    rows = _task_relevance_rows(_Rows(), "g")
    assert set(rows) == {
        "zero_uq",
        "on_path_uq",
        "off_path_uq",
        "view_shuffled_uq",
    }


def test_language_release_requires_dev_preference_and_no_u_delta():
    before = {
        "train": {"mean_target_nll": 4.0},
        "dev": {"mean_target_nll": 4.0},
    }
    after = {
        "train": {"mean_target_nll": 3.0},
        "dev": {
            "mean_target_nll": 3.5,
            "full_conditioning": {"all_passed": True},
            "no_u_ablation": {"overall_preference_fraction": 0.25},
            "full_minus_no_u_preference_fraction": 0.55,
        },
    }
    factorization = {
        split: {"all_release_checks_passed": True}
        for split in ("train", "dev")
    }
    protocol = {
        "language_gates": {
            "maximum_no_u_preference_fraction": 0.5,
            "minimum_full_minus_no_u_fraction": 0.25,
        }
    }
    checks = _language_release_checks(
        before=before,
        after=after,
        factorization=factorization,
        protocol=protocol,
    )
    assert all(checks.values())
    after["dev"]["no_u_ablation"]["overall_preference_fraction"] = 0.75
    assert _language_release_checks(
        before=before,
        after=after,
        factorization=factorization,
        protocol=protocol,
    )["dev_no_u_preference_below_ceiling"] is False


class _ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.base_weight = torch.nn.Parameter(torch.ones(2, 2))


def test_v101_contextual_checkpoint_loads_and_freezes(tmp_path):
    lm = _ToyLM()
    queries = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=8,
        image_feature_dim=6,
        hidden_dim=4,
        grid_hw=(2, 2),
        max_views=3,
    )
    head = TaskRelevanceMapHead(model_dim=8, hidden_dim=4)
    checkpoint = tmp_path / "phase_a.pt"
    torch.save({
        "schema": "orion.stage2l_v101_view_aligned_phase_a.v1",
        "status": "phase_a_failed_gate",
        "optimizer_steps": 120,
        "phase_a_only": True,
        "stage1_uq_loaded": False,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "lora": {"lora_weight": torch.full((2, 2), 3.0)},
        "view_aligned_relevance_queries": queries.state_dict(),
        "relevance_head": head.state_dict(),
    }, checkpoint)
    report = _load_v101_contextual_relevance(
        lm=lm,
        relevance_queries=queries,
        relevance_head=head,
        checkpoint_path=checkpoint,
    )
    assert report["all_parameters_frozen"] is True
    assert torch.equal(lm.lora_weight, torch.full((2, 2), 3.0))


def _args(tmp_path):
    names = (
        "config",
        "checkpoint",
        "dataset_manifest",
        "v11_records",
        "dataset_audit_report",
        "view_feature_cache",
        "u_tokenizer_checkpoint",
        "v101_checkpoint",
        "v101_report",
        "parent_contract",
    )
    values = {}
    for name in names:
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        values[name] = path
    values["output_dir"] = tmp_path / "output"
    return argparse.Namespace(**values)


def test_protocol_checks_lock_every_module_except_k_bridge(tmp_path):
    args = _args(tmp_path)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen_bounded_identifiability_protocol_launch_locked",
        "input_sha256": _protocol_input_hashes(args),
        "output_root": str(args.output_dir.resolve()),
        "architecture": {
            "only_trainable_module": "TaskRiskLanguageBridge",
            "u_enters_relevance_query": False,
            "stage1_trainable": False,
            "u_tokenizer_trainable": False,
            "contextual_relevance_trainable": False,
            "orion_lora_trainable": False,
            "learned_structured_field_head_used": False,
            "trajectory_or_control_loss": False,
            "density_uq_used": False,
            "governor_used": False,
        },
        "training": {"optimizer_steps": 40, "anchors_per_step": 2},
        "launch_locks": {"real_training_allowed": False},
    }
    _protocol_checks(args, protocol)
    protocol["architecture"]["u_enters_relevance_query"] = True
    with pytest.raises(ValueError, match="protocol"):
        _protocol_checks(args, protocol)
