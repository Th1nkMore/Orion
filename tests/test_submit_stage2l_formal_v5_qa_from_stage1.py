from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_submitter_is_cpu_workload_with_qos_gpu_request_and_hash_binds_contracts() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "submit_stage2l_formal_v5_qa_from_stage1.sh"
    ).read_text(encoding="utf-8")
    assert "Nvidia_A800" in script
    assert "--gres=gpu:1" in script
    assert "GPU_REQUESTED_FOR_QOS=1" in script
    assert "GPU_USED_BY_WORKLOAD=0" in script
    assert "qa_factory_v5_vlm_task_fields.json" in script
    assert "c942d85a3ac44b551397cbb4b47172d75594d520be8258147dee1aa4a2e9b476" in script
    assert "qa_factory_v2_matched_supervision.json" in script
    assert "2236bbc84bb794abc0ce69fc3b4706b131eaa1282b2942212ef429a3b381471e" in script


def test_submitter_validates_stage1_before_finalization() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "submit_stage2l_formal_v5_qa_from_stage1.sh"
    ).read_text(encoding="utf-8")
    assert "validate_stage2l_formal_stage1_reuse.py" in script
    assert '"${validate_command} && ${finalize_command}"' in script
    assert "STAGE1_REUSED=1" in script
    assert "GPU_REQUIRED_BY_WORKLOAD=0" in script
