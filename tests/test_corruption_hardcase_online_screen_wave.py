import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_corruption_hardcase_online_screen_wave.py"
SPEC = importlib.util.spec_from_file_location("online_wave", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_front_stale_condition_maps_to_physical_severity():
    runtime = MODULE.condition_runtime("front_stale", "delay_ms:400")
    assert runtime["PILOT_CONDITION"] == "front_stale_transient_off"
    assert runtime["ORION_CLOSEDLOOP_CORRUPTION_SEVERITY"] == "3"


def test_waterdrop_condition_uses_nonretired_runtime_name():
    runtime = MODULE.condition_runtime(
        "lens_waterdrop_paired_template", "profile:medium"
    )
    assert runtime["PILOT_CONDITION"] == (
        "lens_waterdrop_paired_template_transient_off"
    )


def test_native_motion_blur_condition_is_renderer_profile():
    runtime = MODULE.condition_runtime("native_motion_blur", "profile:medium")
    assert runtime == {
        "PILOT_CONDITION": "native_motion_blur_off",
        "ORION_NATIVE_MOTION_BLUR_PROFILE": "medium",
    }
