import json
from pathlib import Path

from scripts.preflight_uqformer_alignment import run_preflight


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/scenario_factory/uqformer_alignment_v1.json"


def test_protocol_freezes_task_free_native_9d_boundary_and_no_gpu_authority():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    architecture = protocol["architecture"]
    assert architecture["source_summary_width"] == 9
    assert architecture["latent_width"] == 256
    assert architecture["language_width"] == 4096
    assert (
        architecture["language_width_projection_position"]
        == "final_orion_boundary_only"
    )
    assert architecture["query_layout"] == {
        "view_query_hw": [2, 2],
        "temporal_queries": 3,
        "component_queries": 3,
        "global_queries": 4,
        "tokens_for_six_views": 34,
    }
    assert architecture["task_risk_language_bridge_present"] is False
    assert architecture["task_relevance_is_a_bridge_output"] is False
    assert protocol["resource_authority"] == {
        "cpu_preflight_allowed": True,
        "gpu_job_allowed": False,
        "slurm_submission_allowed": False,
        "orion_load_allowed": False,
        "automatic_retry_allowed": False,
    }
    assert all(value is False for value in protocol["downstream_locks"].values())
    assert set(protocol["forbidden_inputs"]) == {
        "route",
        "actor",
        "task_relevance_R",
        "task_risk_K",
        "TTC",
        "action",
        "collision",
        "planning",
        "corruption_family_label",
    }


def test_cpu_preflight_is_finite_auditable_and_changes_only_language_boundary():
    report = run_preflight(PROTOCOL)
    assert report["status"] == "passed_architecture_and_gradient_preflight_only"
    assert report["device"] == "cpu"
    assert report["shapes"]["source_summary_9d"][-1] == 9
    assert report["shapes"]["source_latent"][-1] == 24
    assert report["shapes"]["compact_latent"] == [2, 34, 24]
    assert report["shapes"]["language_tokens"] == [2, 34, 48]
    assert report["shapes"]["alternate_language_tokens"] == [2, 34, 64]
    assert all(report["checks"].values())
    assert report["gpu_job_submitted"] is False
