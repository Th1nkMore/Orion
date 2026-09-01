import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_corruption_hardcase_orion_screen.py"
SPEC = importlib.util.spec_from_file_location("hardcase_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_front_stale_resolves_physical_delay():
    family, condition, environment = MODULE.resolve_condition(
        pilot_condition="front_stale_transient_off",
        corruption_severity=2,
        paired_waterdrop_profile="",
        native_motion_blur_profile="none",
    )
    assert family == "front_stale"
    assert condition == "delay_ms:200"
    assert environment["ORION_CLOSEDLOOP_CORRUPTION"] == "front_stale"


def test_paired_waterdrop_never_resolves_to_retired_alias():
    family, condition, environment = MODULE.resolve_condition(
        pilot_condition="lens_waterdrop_paired_template_transient_off",
        corruption_severity=1,
        paired_waterdrop_profile="medium",
        native_motion_blur_profile="none",
    )
    assert family == "lens_waterdrop_paired_template"
    assert condition == "profile:medium"
    assert environment["ORION_CLOSEDLOOP_CORRUPTION"] == (
        "lens_waterdrop_paired_template"
    )


def test_retired_waterdrop_condition_is_rejected():
    try:
        MODULE.resolve_condition(
            pilot_condition="lens_waterdrop_transient_off",
            corruption_severity=1,
            paired_waterdrop_profile="",
            native_motion_blur_profile="none",
        )
    except MODULE.VisualApprovalError as error:
        assert "retired failed v1" in str(error)
    else:
        raise AssertionError("retired waterdrop condition unexpectedly resolved")


def test_native_motion_blur_resolves_without_synthetic_corruption():
    family, condition, environment = MODULE.resolve_condition(
        pilot_condition="native_motion_blur_off",
        corruption_severity=1,
        paired_waterdrop_profile="",
        native_motion_blur_profile="medium",
    )
    assert family == "native_motion_blur"
    assert condition == "profile:medium"
    assert environment["ORION_CLOSEDLOOP_CORRUPTION"] == ""
