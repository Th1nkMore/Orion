import hashlib
import json
from pathlib import Path

import pytest

from scripts.submit_corruption_hardcase_wave1_clean_q1 import (
    CORRUPTION_KEYS,
    build_jobs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    repository = tmp_path / "repo"
    repository.mkdir()
    lineage = {}
    for name in ("funnel", "windows", "wave0", "readiness"):
        path = repository / (name + ".json")
        path.write_text("{}\n", encoding="utf-8")
        lineage[name] = {"path": path.name, "sha256": _sha(path)}
    routes = []
    for route_index in (158, 164):
        path = tmp_path / ("route_%d.xml" % route_index)
        path.write_text("<route id='%d'/>\n" % route_index, encoding="utf-8")
        routes.append({
            "route_index": route_index,
            "route_xml": {"path": str(path), "sha256": _sha(path)},
        })
    config = {
        "schema": "orion.corruption_hardcase_wave1_clean_qualification.v1",
        "lineage": lineage,
        "selection": {"routes": [158, 164]},
        "routes": routes,
        "qualification_protocol": {
            "q1": {"authorized": True, "run_id": "wave1-q1"}
        },
        "resources": {
            "partition": "Nvidia_A800",
            "cpus_per_job": 2,
            "memory_per_job": "192G",
            "time_limit": "02:00:00",
            "excluded_nodes": ["gpu5", "gpu2"],
        },
        "execution_locks": {"q1_clean_submission": True},
    }
    return repository, config


def test_builds_only_clean_isolated_jobs(tmp_path):
    repository, config = _fixture(tmp_path)
    jobs = build_jobs(config, repository, tmp_path / "assets")
    assert [job["route_index"] for job in jobs] == [158, 164]
    assert all(job["condition"] == "clean_off" for job in jobs)
    assert all(not CORRUPTION_KEYS.intersection(job["environment"]) for job in jobs)
    assert all(job["environment"]["SLURM_EXCLUDE"] == "gpu2,gpu5" for job in jobs)
    assert all(job["environment"]["ORION_CLOSEDLOOP_UQ_MODE"] == "none" for job in jobs)
    assert all(job["environment"]["ORION_CLOSEDLOOP_RISK_MODE"] == "off" for job in jobs)


def test_rejects_changed_lineage(tmp_path):
    repository, config = _fixture(tmp_path)
    (repository / "funnel.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="lineage hash differs"):
        build_jobs(config, repository, tmp_path / "assets")


def test_rejects_closed_authority(tmp_path):
    repository, config = _fixture(tmp_path)
    config["qualification_protocol"]["q1"]["authorized"] = False
    with pytest.raises(ValueError, match="not authorized"):
        build_jobs(config, repository, tmp_path / "assets")
