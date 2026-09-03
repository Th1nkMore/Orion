import hashlib
import json
from pathlib import Path

import pytest

from scripts.submit_corruption_hardcase_wave2_clean_q1 import (
    CORRUPTION_KEYS,
    build_jobs,
    write_submission_record,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    repository = tmp_path / "repo"
    repository.mkdir()
    lineage = {}
    for name in ("wave1", "candidate_amendment", "candidate", "repair"):
        path = repository / (name + ".json")
        path.write_text("{}\n", encoding="utf-8")
        lineage[name] = {"path": path.name, "sha256": _sha(path)}
    routes = []
    for route_index in (160, 165, 161):
        path = tmp_path / ("route_%d.xml" % route_index)
        path.write_text("<route id='%d'/>\n" % route_index, encoding="utf-8")
        routes.append({
            "route_index": route_index,
            "route_xml": {"path": str(path), "sha256": _sha(path)},
        })
    prereg_path = repository / "prereg.json"
    prereg = {
        "schema": (
            "orion.corruption_hardcase_wave2_clean_qualification_"
            "preregistration.v1"
        ),
        "status": "prepared_paused_no_submission_authority",
        "lineage": lineage,
        "selection": {"routes": [160, 165, 161]},
        "routes": routes,
        "qualification_protocol": {
            "q1": {
                "run_id": "corruption_hardcase_wave2_clean_q1_v1",
                "authorized": False,
            }
        },
        "resources": {
            "partition": "Nvidia_A800",
            "cpus_per_job": 2,
            "memory_per_job": "192G",
            "time_limit": "02:00:00",
            "excluded_nodes": ["gpu5", "gpu2"],
        },
        "execution_locks": {"q1_clean_submission": False},
    }
    prereg_path.write_text(json.dumps(prereg) + "\n", encoding="utf-8")
    activation = {
        "schema": "orion.corruption_hardcase_wave2_clean_q1_activation.v1",
        "status": "authorized_after_user_resume",
        "base_prereg": {
            "path": prereg_path.name,
            "sha256": _sha(prereg_path),
        },
        "scope": {
            "routes": [160, 165, 161],
            "condition": "clean_off",
            "runs_per_route": 1,
            "run_id": "corruption_hardcase_wave2_clean_q1_v1",
        },
        "authorization": {
            "q1_clean_submission": True,
            "user_resume_recorded": True,
            "q2_clean_submission": False,
            "corruption_submission": False,
            "heldout_confirmation": False,
            "severity_change": False,
            "learned_uq_or_governor": False,
            "stage2p": False,
            "formal_200_route_evaluation": False,
        },
    }
    return repository, prereg_path, prereg, activation


def test_builds_exactly_three_clean_jobs_with_exact_frame_speedometer(tmp_path):
    repository, prereg_path, prereg, activation = _fixture(tmp_path)
    jobs = build_jobs(
        prereg=prereg,
        prereg_path=prereg_path,
        activation=activation,
        repository_root=repository,
        asset_root=tmp_path / "assets",
    )
    assert [job["route_index"] for job in jobs] == [160, 165, 161]
    assert all(job["condition"] == "clean_off" for job in jobs)
    assert all(not CORRUPTION_KEYS.intersection(job["environment"]) for job in jobs)
    assert all(
        job["environment"]["ORION_EXACT_FRAME_SPEEDOMETER"] == "1"
        for job in jobs
    )
    assert all(
        job["environment"]["ORION_SENSOR_QUEUE_DIAGNOSTICS"] == "1"
        for job in jobs
    )


def test_rejects_template_or_missing_resume_authority(tmp_path):
    repository, prereg_path, prereg, activation = _fixture(tmp_path)
    activation["schema"] = (
        "orion.corruption_hardcase_wave2_clean_q1_activation_template.v1"
    )
    activation["status"] = "template_not_authorized"
    activation["authorization"]["q1_clean_submission"] = False
    activation["authorization"]["user_resume_recorded"] = False
    with pytest.raises(ValueError, match="absent or is only a template"):
        build_jobs(
            prereg=prereg,
            prereg_path=prereg_path,
            activation=activation,
            repository_root=repository,
            asset_root=tmp_path / "assets",
        )


def test_rejects_scope_expansion_and_changed_prereg_hash(tmp_path):
    repository, prereg_path, prereg, activation = _fixture(tmp_path)
    activation["authorization"]["corruption_submission"] = True
    with pytest.raises(ValueError, match="scope beyond clean Q1"):
        build_jobs(
            prereg=prereg,
            prereg_path=prereg_path,
            activation=activation,
            repository_root=repository,
            asset_root=tmp_path / "assets",
        )
    activation["authorization"]["corruption_submission"] = False
    prereg_path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="lineage hash differs"):
        build_jobs(
            prereg=prereg,
            prereg_path=prereg_path,
            activation=activation,
            repository_root=repository,
            asset_root=tmp_path / "assets",
        )


def test_submission_record_is_atomically_replaced_and_valid_json(tmp_path):
    output = tmp_path / "submission.json"
    first = {
        "status": "submission_in_progress",
        "submission_attempts": [{"job_key": "route160_clean_q1"}],
    }
    write_submission_record(output, first)
    assert json.loads(output.read_text(encoding="utf-8")) == first

    second = {
        "status": "submitted",
        "job_ids": [{"job_key": "route160_clean_q1", "job_id": 1234}],
    }
    write_submission_record(output, second)
    assert json.loads(output.read_text(encoding="utf-8")) == second
    assert list(tmp_path.glob(".submission.json.tmp-*")) == []
