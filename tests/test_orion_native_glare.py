import json
from pathlib import Path

import pytest

from team_code.orion_native_glare import (
    install_orion_native_glare_sensor_patch,
    readback_render_condition,
    record_render_condition_readback,
    requested_render_condition,
)


class FakeWrapper:
    @staticmethod
    def _preprocess_sensor_spec(wrapper, sensor_spec):
        del wrapper
        return "sensor.camera.rgb", sensor_spec["id"], object(), {"fov": "100"}


class Point:
    x = 1
    y = 2
    z = 3
    pitch = 4
    yaw = 5
    roll = 6


class Transform:
    location = Point()
    rotation = Point()


class Actor:
    def __init__(self, actor_id, attributes):
        self.id = actor_id
        self.type_id = "sensor.camera.rgb"
        self.attributes = attributes

    def get_transform(self):
        return Transform()


class Interface:
    def __init__(self, front, bev):
        self._sensors_objects = {"CAM_FRONT": front, "bev": bev}


class Weather:
    sun_altitude_angle = 8.0
    sun_azimuth_angle = 180.0


class World:
    def get_weather(self):
        return Weather()


def _attributes(profile, sensor_id):
    return {key: str(value).lower() for key, value in requested_render_condition(profile)["requested"][sensor_id].items()}


def test_frozen_profile_patch_targets_front_and_disables_bev_only():
    class Wrapper(FakeWrapper):
        pass

    assert install_orion_native_glare_sensor_patch(Wrapper, "medium") is True
    _, _, _, front = Wrapper._preprocess_sensor_spec(None, {"id": "CAM_FRONT"})
    _, _, _, bev = Wrapper._preprocess_sensor_spec(None, {"id": "bev"})
    _, _, _, side = Wrapper._preprocess_sensor_spec(None, {"id": "CAM_LEFT"})
    assert front["lens_flare_intensity"] == "0.75"
    assert front["bloom_intensity"] == "1.5"
    assert front["enable_postprocess_effects"] == "true"
    assert bev["enable_postprocess_effects"] == "false"
    assert bev["lens_flare_intensity"] == "0.0"
    assert "lens_flare_intensity" not in side


@pytest.mark.parametrize("profile,lens,bloom", [
    ("clean", 0.0, 0.0), ("medium", 0.75, 1.5), ("heavy", 1.5, 3.0),
])
def test_profiles_are_frozen(profile, lens, bloom):
    request = requested_render_condition(profile)
    assert request["requested"]["CAM_FRONT"]["lens_flare_intensity"] == lens
    assert request["requested"]["CAM_FRONT"]["bloom_intensity"] == bloom
    assert request["requested"]["bev"]["enable_postprocess_effects"] == "false"


def test_active_readback_is_verified_and_mismatch_fails_closed():
    front = Actor(1, _attributes("heavy", "CAM_FRONT"))
    bev = Actor(2, _attributes("heavy", "bev"))
    result = readback_render_condition(Interface(front, bev), World(), "heavy")
    assert result["status"] == "verified"
    assert result["weather"]["sun_altitude_angle"] == 8.0
    front.attributes["lens_flare_intensity"] = "0.0"
    with pytest.raises(RuntimeError, match="readback mismatch"):
        readback_render_condition(Interface(front, bev), World(), "heavy")


def test_manifest_readback_update_is_atomic_and_preserves_other_fields(tmp_path):
    requested = requested_render_condition("medium")
    initial = dict(requested)
    initial["actual_readback"] = {"status": "pending_agent_runtime"}
    manifest = {"route": "151", "render_condition": initial}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    readback = {"schema": "orion.closedloop_render_condition_readback.v1", "status": "verified"}
    record_render_condition_readback(tmp_path, requested, readback)
    updated = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert updated["route"] == "151"
    assert updated["render_condition"]["actual_readback"]["status"] == "verified"
    assert json.loads((tmp_path / "render_condition_readback.json").read_text())["status"] == "verified"


def test_real_agent_and_runner_retain_strict_contracts():
    root = Path(__file__).resolve().parents[1]
    agent = (root / "team_code/orion_b2d_agent.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_closedloop_uq_pilot.sh").read_text(encoding="utf-8")
    assert "install_orion_native_glare_sensor_patch(" in agent
    assert "readback_render_condition(" in agent
    assert "model_input_front_tensor.shape != (640, 640, 3)" in agent
    assert "native glare cannot be mixed with synthetic" in runner
    assert 'none|clean|medium|heavy' in runner
    assert '"team_code/orion_native_glare.py"' in runner
