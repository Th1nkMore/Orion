from pathlib import Path

import pytest

from scripts.submit_corruption_hardcase_wave1_route158 import (
    CONDITION_NAMES,
    build_jobs,
    extend_excluded_nodes,
    select_jobs,
    verify_environment_isolation,
)


def _activation(tmp_path: Path):
    route = tmp_path / "route_158_hazard.xml"
    route.write_text("<route id='158'/>\n")
    bank = tmp_path / "bank"
    bank.mkdir()
    gate = tmp_path / "gate.json"
    gate.write_text("{}\n")
    activation = {
        "matrix": {
            "run_id": "wave1-route158",
        },
        "route": {
            "route_xml": {
                "path": str(route),
                "sha256": "route-hash",
            },
        },
        "event_window": {
            "start_progress": 0.1331479129200312,
            "duration_seconds": 5.0,
        },
        "resources": {
            "partition": "Nvidia_A800",
            "cpus_per_job": 2,
            "memory_per_job": "192G",
            "time_limit": "02:00:00",
            "excluded_nodes": ["gpu5", "gpu2"],
        },
    }
    validated = {
        "route_path": route,
        "gate_path": gate,
        "bank_path": bank,
    }
    return activation, validated


def test_route158_builds_only_three_approved_corruption_jobs(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    assert [job["condition"] for job in jobs] == list(CONDITION_NAMES)
    assert all(job["route_index"] == 158 for job in jobs)
    assert all(job["environment"]["ORION_CLOSEDLOOP_RISK_MODE"] == "off" for job in jobs)
    assert all(job["environment"]["SLURM_EXCLUDE"] == "gpu2,gpu5" for job in jobs)
    assert jobs[0]["environment"]["ORION_CLOSEDLOOP_CORRUPTION_SEVERITY"] == "2"
    assert jobs[1]["environment"]["ORION_PAIRED_WATERDROP_PROFILE"] == "medium"
    assert jobs[2]["environment"]["ORION_NATIVE_MOTION_BLUR_PROFILE"] == "medium"


def test_route158_screen_has_no_clean_or_control_job(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    assert "clean_off" not in {job["condition"] for job in jobs}
    assert all(job["environment"]["ORION_CLOSEDLOOP_UQ_MODE"] == "none" for job in jobs)
    assert all(job["environment"]["ORION_STAGE2_SPATIAL_UQ_SOURCE"] == "disabled" for job in jobs)


def test_route158_environment_rejects_severity_drift(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    jobs[0]["environment"]["ORION_NATIVE_MOTION_BLUR_PROFILE"] = "heavy"
    with pytest.raises(ValueError, match="corruption environment differs"):
        verify_environment_isolation(jobs)


def test_route158_environment_rejects_risk_mode(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    jobs[1]["environment"]["ORION_CLOSEDLOOP_RISK_MODE"] = "threshold"
    with pytest.raises(ValueError, match="risk mode is not off"):
        verify_environment_isolation(jobs)


def test_route158_runtime_retry_selects_exact_one_job(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    selected = select_jobs(jobs, ["route158_front_stale_transient_off"])
    assert [job["condition"] for job in selected] == ["front_stale_transient_off"]


def test_route158_runtime_retry_only_extends_exclusions(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    selected = select_jobs(
        jobs, ["route158_lens_waterdrop_paired_template_transient_off"]
    )
    extended = extend_excluded_nodes(selected, ["gpu1"])
    assert extended[0]["environment"]["SLURM_EXCLUDE"] == "gpu1,gpu2,gpu5"
    assert extended[0]["environment"]["ORION_PAIRED_WATERDROP_PROFILE"] == "medium"


def test_route158_retry_rejects_unknown_job_key(tmp_path):
    activation, validated = _activation(tmp_path)
    jobs = build_jobs(
        activation=activation,
        repository_root=tmp_path,
        asset_root=tmp_path / "assets",
        validated=validated,
    )
    with pytest.raises(ValueError, match="unknown Route158 job keys"):
        select_jobs(jobs, ["route158_clean_off"])
