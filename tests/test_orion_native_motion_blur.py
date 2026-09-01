import pytest

from team_code.orion_native_motion_blur import (
    install_orion_native_motion_blur_sensor_patch,
    readback_native_motion_blur_condition,
    requested_native_motion_blur_condition,
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


def _attributes(profile, sensor_id):
    return {
        key: str(value).lower()
        for key, value in requested_native_motion_blur_condition(profile)[
            "requested"
        ][sensor_id].items()
    }


@pytest.mark.parametrize(
    ("profile", "intensity"),
    [("clean", 0.0), ("light", 0.35), ("medium", 0.70), ("heavy", 1.0)],
)
def test_candidate_profiles_are_frozen(profile, intensity):
    requested = requested_native_motion_blur_condition(profile)
    assert requested["requested"]["CAM_FRONT"]["motion_blur_intensity"] == intensity
    assert requested["requested"]["bev"]["enable_postprocess_effects"] == "false"
    assert requested["severity_status"] == "candidate_pending_visual_bakeoff"


def test_patch_targets_front_and_disables_bev_only():
    class Wrapper(FakeWrapper):
        pass

    assert install_orion_native_motion_blur_sensor_patch(Wrapper, "medium") is True
    _, _, _, front = Wrapper._preprocess_sensor_spec(None, {"id": "CAM_FRONT"})
    _, _, _, bev = Wrapper._preprocess_sensor_spec(None, {"id": "bev"})
    _, _, _, side = Wrapper._preprocess_sensor_spec(None, {"id": "CAM_LEFT"})
    assert front["motion_blur_intensity"] == "0.7"
    assert front["motion_blur_max_distortion"] == "0.45"
    assert bev["enable_postprocess_effects"] == "false"
    assert "motion_blur_intensity" not in side


def test_native_baseline_does_not_patch_camera_blueprint():
    class Wrapper(FakeWrapper):
        pass

    assert install_orion_native_motion_blur_sensor_patch(Wrapper, "none") is False
    _, _, _, front = Wrapper._preprocess_sensor_spec(None, {"id": "CAM_FRONT"})
    assert front == {"fov": "100"}


def test_intensity_only_diagnostic_changes_one_front_attribute():
    class Wrapper(FakeWrapper):
        pass

    requested = requested_native_motion_blur_condition("intensity_zero_only")
    assert requested["kind"] == "carla_camera_attribute_diagnostic"
    assert requested["requested"]["CAM_FRONT"] == {
        "motion_blur_intensity": 0.0
    }
    assert requested["requested"]["bev"] is None
    assert install_orion_native_motion_blur_sensor_patch(
        Wrapper, "intensity_zero_only"
    ) is True
    _, _, _, front = Wrapper._preprocess_sensor_spec(None, {"id": "CAM_FRONT"})
    _, _, _, bev = Wrapper._preprocess_sensor_spec(None, {"id": "bev"})
    assert front == {"fov": "100", "motion_blur_intensity": "0.0"}
    assert bev == {"fov": "100"}


def test_readback_verifies_and_mismatch_fails_closed():
    front = Actor(1, _attributes("heavy", "CAM_FRONT"))
    bev = Actor(2, _attributes("heavy", "bev"))
    result = readback_native_motion_blur_condition(
        Interface(front, bev), "heavy"
    )
    assert result["status"] == "verified"
    front.attributes["motion_blur_intensity"] = "0.0"
    with pytest.raises(RuntimeError, match="readback mismatch"):
        readback_native_motion_blur_condition(Interface(front, bev), "heavy")
