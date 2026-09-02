import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v15_protocol_is_task_free_bounded_and_all_candidate():
    protocol = json.loads(
        (
            ROOT
            / "configs/scenario_factory/stage2l_v15_u_language_alignment_pilot_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["training"]["optimizer_steps"] == 720
    assert protocol["training"]["field_objective"] == "all_candidate_cross_entropy"
    assert protocol["training"]["task_agnostic_reconstruction_retention"] is True
    assert protocol["architecture"]["direct_u_tokens_enter_orion"] is True
    assert protocol["architecture"]["u_tokenizer_existing_projection_trainable"] is True
    assert protocol["architecture"]["new_bridge_or_r_input_present"] is False
    assert protocol["architecture"]["route_task_risk_action_present"] is False
    assert protocol["architecture"]["trajectory_or_control_loss"] is False
    assert protocol["evaluation"]["u_states"] == 120
    assert protocol["evaluation"]["all_six_fields"] is True
    assert protocol["resources"] == {
        "partition": "Nvidia_A800",
        "gpus": 1,
        "cpus_per_task": 2,
        "memory": "192G",
        "time_limit": "20:00:00",
        "maximum_submissions": 1,
    }
    assert all(value is False for value in protocol["locks"].values())


def test_v15_submitter_matches_the_bounded_protocol_and_preflights_first():
    source = (
        ROOT / "scripts/submit_stage2l_v15_u_language_alignment_pilot.sh"
    ).read_text(encoding="utf-8")
    assert source.index("--preflight-only") < source.index("sbatch --parsable")
    assert "--optimizer-steps 720" in source
    assert "--answer-batch-size 2" in source
    assert "--gres=gpu:1" in source
    assert "--cpus-per-task=2" in source
    assert "--mem=192G" in source
    assert "--time=20:00:00" in source
    assert 'job_name="s2l_v15_u"' in source
    assert "refusing duplicate active v15 job" in source
    assert "trained-v14-checkpoint" in source


def test_v15_trainer_contains_no_r_bridge_task_or_planning_import():
    source = (
        ROOT / "scripts/train_stage2l_v15_u_language_alignment_pilot.py"
    ).read_text(encoding="utf-8")
    assert "all_candidate_cross_entropy" in source
    assert "stage1_u_tokenizer_pretraining_terms" in source
    assert "TaskRiskBridge" not in source
    assert "TaskRelevanceMapHead" not in source
    assert "loss_plan" not in source
    assert "risk_governor" not in source
