import hashlib
import json
from pathlib import Path

import pytest

from scripts.write_native_glare_activation_amendment import write_activation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    root = tmp_path / "project"
    source = root / "team_code" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({
        "source_contract": {"team_code/source.py": _sha(source)},
    }))
    staging = root / "staging.json"
    staging.write_text(json.dumps({
        "schema": "orion.scenario_factory.amendment.v1",
        "status": "implementation_complete_remote_activation_deferred_dependency_freeze",
        "dependency_freeze": {"active_job_id": "1112878"},
    }))
    log = root / "tests.log"
    log.write_text(".................. 18 passed in 2.0s\n")
    return root, source, protocol, staging, log


def test_activation_requires_terminal_dependency_and_matching_hashes(tmp_path):
    root, _, protocol, staging, log = _fixture(tmp_path)
    output = root / "activation.json"
    result = write_activation(
        project_root=root,
        staging_amendment=staging,
        protocol_path=protocol,
        platform_test_log=log,
        training_job_id="1112878",
        training_job_state="COMPLETED",
        output=output,
    )
    assert result["status"] == "native_glare_orion_interface_activated"
    assert result["launch_locks"]["route151_native_glare_pair_allowed"]
    assert not result["launch_locks"]["route203_native_glare_submission_allowed"]
    assert output.is_file()


def test_activation_rejects_running_job_or_source_drift(tmp_path):
    root, source, protocol, staging, log = _fixture(tmp_path)
    with pytest.raises(ValueError, match="not terminal"):
        write_activation(
            project_root=root,
            staging_amendment=staging,
            protocol_path=protocol,
            platform_test_log=log,
            training_job_id="1112878",
            training_job_state="RUNNING",
            output=root / "running.json",
        )
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source contract differs"):
        write_activation(
            project_root=root,
            staging_amendment=staging,
            protocol_path=protocol,
            platform_test_log=log,
            training_job_id="1112878",
            training_job_state="COMPLETED",
            output=root / "drift.json",
        )
