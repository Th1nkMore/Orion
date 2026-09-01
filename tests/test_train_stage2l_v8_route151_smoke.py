import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_stage2l_v8_route151_smoke.py"
PROTOCOL = (
    ROOT
    / "configs"
    / "scenario_factory"
    / "stage2l_training_v8_gradient_routed_structured_qa.json"
)


def test_v8_trainer_is_syntax_valid_and_uses_repaired_primitives():
    source = TRAINER.read_text()
    ast.parse(source)
    assert "build_gradient_routed_conditioning" in source
    assert "dataset_frequency_balanced_stance_loss" in source
    assert "same_family_unique_structured_answers" in source
    assert "generation_semantic_metrics" in source
    assert "detach_relevance_for_language=True" in source
    assert "detach_stance_probabilities_for_language=True" in source
    assert "with torch.no_grad():\n                current_logits" in source


def test_v8_trainer_is_bounded_and_fail_closed():
    source = TRAINER.read_text()
    assert 'default=60' in source
    assert 'args.max_optimizer_steps != 60' in source
    assert 'args.answer_batch_size != 2' in source
    assert "real v8 smoke requires a separate launch amendment" in source
    assert "locked preflight requires an explicit output path" in source
    assert "refusing to overwrite trainer preflight output" in source
    assert '"training_started": False' in source
    assert '"training_authorized": False' in source
    assert 'automatic_retry_or_extension' in source


def test_v8_trainer_preserves_architecture_boundary():
    source = TRAINER.read_text()
    assert '"legacy_density_uq_used": False' in source
    assert '"hard_governor_used": False' in source
    assert '"trajectory_or_control_loss": False' in source
    assert '"ground_truth_stance_enters_forward": False' in source
    assert '"format_prefix_is_release_evidence": False' in source
    assert "DensityUQEstimator" not in source
    assert "RiskGovernor" not in source


def test_v8_protocol_keeps_probe_unapproved():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["future_probe_bound_not_authorized"][
        "proposed_maximum_optimizer_steps"
    ] == 60
    assert protocol["launch_locks"] == {
        "real_orion_smoke_allowed": False,
        "stage2l_pilot_training_allowed": False,
        "stage2p_allowed": False,
        "new_immutable_amendment_required": True,
    }
