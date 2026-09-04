import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs/scenario_factory/stage2l_v15_2_text_oracle_localization_v1.json"
)
EVALUATOR = ROOT / "scripts/evaluate_stage2l_v15_2_text_oracle_localization.py"
SUBMITTER = ROOT / "scripts/submit_stage2l_v15_2_text_oracle_localization.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_text_oracle_protocol_is_hash_bound_task_free_and_complete():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["training_performed"] is False
    assert protocol["optimizer_steps"] == 0
    assert protocol["architecture"] == {
        "continuous_u_tokens_present": False,
        "text_oracle_is_only_u_input": True,
        "orion_visual_tokens": 529,
        "r_bridge_task_route_risk_action_present": False,
        "trajectory_control_closed_loop_present": False,
    }
    assert protocol["evaluation"]["dev_groups"] == 20
    assert protocol["evaluation"]["u_states"] == 120
    assert protocol["evaluation"]["field_decisions_per_model"] == 720
    assert protocol["evaluation"]["primary_decode"] == "all_candidate_nll"
    assert protocol["evaluation"]["model_controls"] == [
        "original_orion",
        "v15_lora",
    ]
    assert protocol["automatic_retry"] is False
    assert protocol["resources"]["maximum_submissions"] == 1
    assert all(value is False for value in protocol["locks"].values())
    for relative, expected in protocol["implementation_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_evaluator_removes_continuous_u_and_uses_all_states_and_controls():
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "continuous_u_tokens_present\": False" in source
    assert "assets.groups_for_split(\"dev\")" in source
    assert "len(dev_groups) != 20" in source
    assert "field_decision_count\"] != 720" in source
    assert '"original_orion": original' in source
    assert '"v15_lora": v15_text' in source
    assert "_load_v15_lora(lm, v15_state)" in source
    assert "base._answer_nlls_mr1" in source
    assert "optimizer =" not in source
    assert "UQComponentTokenizer" not in source
    assert "TaskRiskBridge" not in source
    assert "TaskRelevanceMapHead" not in source


def test_submitter_preflights_before_exactly_one_bounded_a800_submission():
    source = SUBMITTER.read_text(encoding="utf-8")
    assert source.index("--preflight-only") < source.index("sbatch --parsable")
    assert source.count("sbatch --parsable") == 1
    assert "--gres=gpu:1" in source
    assert "--cpus-per-task=2" in source
    assert "--mem=192G" in source
    assert "--time=12:00:00" in source
    assert "--answer-batch-size 10" in source
    assert 'job_name="s2l_v152_txt"' in source
    assert "refusing duplicate active v15.2 text-oracle job" in source
    assert "refusing to overwrite" in source
