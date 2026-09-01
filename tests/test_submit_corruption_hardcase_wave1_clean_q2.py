import hashlib
import json
from pathlib import Path

import pytest

from scripts.submit_corruption_hardcase_wave1_clean_q2 import (
    CORRUPTION_KEYS,
    build_jobs,
    extend_excluded_nodes,
    select_jobs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    route_files = {}
    routes = []
    for route_index in (158, 185):
        path = tmp_path / ("route_%d.xml" % route_index)
        path.write_text("<route id='%d'/>\n" % route_index)
        route_files[route_index] = path
        routes.append({
            "route_index": route_index,
            "route_xml": {"path": str(path), "sha256": _sha(path)},
        })
    protocol = repo / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "orion.corruption_hardcase_wave1_clean_qualification.v1",
        "routes": routes,
    }))
    q1_result = repo / "q1_result.json"
    q1_result.write_text(json.dumps({"decision": {"q2_exact_scope": [158, 185]}}))
    collection = tmp_path / "collection.json"
    collection.write_text(json.dumps({
        "schema": "orion.corruption_hardcase_clean_qualification_collection.v1",
        "q1_pass_routes": [158, 185],
    }))
    activation = {
        "schema": "orion.corruption_hardcase_wave1_clean_q2_activation.v1",
        "base_protocol": {"path": "protocol.json", "sha256": _sha(protocol)},
        "q1_result": {"path": "q1_result.json", "sha256": _sha(q1_result)},
        "q1_collection": {"path": str(collection), "sha256": _sha(collection)},
        "scope": {"routes": [158, 185], "run_id": "wave1-q2"},
        "resources": {
            "partition": "Nvidia_A800", "cpus_per_job": 2,
            "memory_per_job": "192G", "time_limit": "02:00:00",
            "excluded_nodes": ["gpu5", "gpu2"],
        },
        "execution_authority": {"q2_clean_submission": True},
    }
    return repo, activation


def test_q2_exactly_matches_q1_passers_and_is_clean(tmp_path):
    repo, activation = _fixture(tmp_path)
    jobs = build_jobs(activation=activation, repository_root=repo, asset_root=tmp_path / "assets")
    assert [job["route_index"] for job in jobs] == [158, 185]
    assert all(job["condition"] == "clean_off" for job in jobs)
    assert all(not CORRUPTION_KEYS.intersection(job["environment"]) for job in jobs)
    assert all(job["environment"]["SLURM_EXCLUDE"] == "gpu2,gpu5" for job in jobs)


def test_q2_rejects_scope_not_equal_to_q1_passers(tmp_path):
    repo, activation = _fixture(tmp_path)
    activation["scope"]["routes"] = [158]
    with pytest.raises(ValueError, match="exact Q1 passers"):
        build_jobs(activation=activation, repository_root=repo, asset_root=tmp_path / "assets")


def test_q2_rejects_changed_q1_collection(tmp_path):
    repo, activation = _fixture(tmp_path)
    Path(activation["q1_collection"]["path"]).write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="Q1 collection hash differs"):
        build_jobs(activation=activation, repository_root=repo, asset_root=tmp_path / "assets")


def test_q2_exact_retry_can_select_one_authorized_route(tmp_path):
    repo, activation = _fixture(tmp_path)
    jobs = build_jobs(
        activation=activation, repository_root=repo, asset_root=tmp_path / "assets"
    )
    selected = select_jobs(jobs, [185])
    assert [job["route_index"] for job in selected] == [185]
    assert selected[0]["condition"] == "clean_off"


def test_q2_retry_rejects_route_outside_activation(tmp_path):
    repo, activation = _fixture(tmp_path)
    jobs = build_jobs(
        activation=activation, repository_root=repo, asset_root=tmp_path / "assets"
    )
    with pytest.raises(ValueError, match="unknown Q2 route indices"):
        select_jobs(jobs, [164])


def test_q2_runtime_retry_only_extends_node_exclusions(tmp_path):
    repo, activation = _fixture(tmp_path)
    jobs = build_jobs(
        activation=activation, repository_root=repo, asset_root=tmp_path / "assets"
    )
    selected = select_jobs(jobs, [185])
    extended = extend_excluded_nodes(selected, ["gpu1"])
    assert extended[0]["environment"]["SLURM_EXCLUDE"] == "gpu1,gpu2,gpu5"
    assert extended[0]["environment"]["PILOT_RUN_ID"] == "wave1-q2"
