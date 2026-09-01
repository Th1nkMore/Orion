from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_multiframe_submitter_binds_v5_qa_factory() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "submit_stage2l_multiframe_event_factory.sh"
    ).read_text(encoding="utf-8")

    assert "qa_factory_v5_vlm_task_fields.json" in script
    assert "c942d85a3ac44b551397cbb4b47172d75594d520be8258147dee1aa4a2e9b476" in script
    assert 'actual_qa_factory_sha256="$(sha256sum' in script
    assert '--qa-factory-config "${qa_factory_config}"' in script
    assert "qa_factory_v2_matched_supervision.json" in script
    assert "2236bbc84bb794abc0ce69fc3b4706b131eaa1282b2942212ef429a3b381471e" in script
    assert '--base-qa-factory-config "${base_qa_factory_config}"' in script


def test_formal_multiframe_submitter_reports_qa_factory_in_dry_run() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "submit_stage2l_multiframe_event_factory.sh"
    ).read_text(encoding="utf-8")

    assert 'echo "QA_FACTORY_CONFIG=${qa_factory_config}"' in script
    assert 'echo "QA_FACTORY_CONFIG_SHA256=${actual_qa_factory_sha256}"' in script
    assert 'echo "BASE_QA_FACTORY_CONFIG=${base_qa_factory_config}"' in script
    assert 'echo "BASE_QA_FACTORY_CONFIG_SHA256=${actual_base_qa_factory_sha256}"' in script
