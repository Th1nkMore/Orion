import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/submit_corruption_hardcase_online_wave.py"
SPEC = importlib.util.spec_from_file_location("submit_hardcase_wave", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_environment_isolation_accepts_exact_four_condition_contract():
    base = {
        "ORION_CLOSEDLOOP_UQ_MODE": "none",
        "ORION_CLOSEDLOOP_CONDITIONING": "none",
        "ORION_CLOSEDLOOP_RISK_MODE": "off",
        "ORION_PLANNING_RESPONSE_MODE": "off",
    }
    jobs = [
        {"condition": "clean_off", "environment": dict(base)},
        {
            "condition": "front_stale_transient_off",
            "environment": dict(base, **{
                "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS": "0.2",
                "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS": "5",
                "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY": "2",
            }),
        },
        {
            "condition": "lens_waterdrop_paired_template_transient_off",
            "environment": dict(base, **{
                "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS": "0.2",
                "ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS": "5",
                "ORION_PAIRED_WATERDROP_PROFILE": "medium",
                "ORION_PAIRED_WATERDROP_BANK": "/tmp/bank",
            }),
        },
        {
            "condition": "native_motion_blur_off",
            "environment": dict(
                base, ORION_NATIVE_MOTION_BLUR_PROFILE="medium"
            ),
        },
    ]
    MODULE.verify_environment_isolation(jobs)


def test_clean_condition_rejects_leaked_corruption_environment():
    jobs = [{
        "condition": "clean_off",
        "environment": {"ORION_NATIVE_MOTION_BLUR_PROFILE": "medium"},
    }]
    try:
        MODULE.verify_environment_isolation(jobs)
    except ValueError as error:
        assert "inherits corruption variables" in str(error)
    else:
        raise AssertionError("clean environment leak was not rejected")


def test_slurm_exclusion_is_not_treated_as_corruption_leak():
    jobs = [{
        "condition": "clean_off",
        "environment": {"SLURM_EXCLUDE": "gpu5"},
    }]
    MODULE.verify_environment_isolation(jobs)
