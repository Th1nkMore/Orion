import importlib.util
from pathlib import Path
import sys

import pytest
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "qwen_drive_navsim_images.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_drive_navsim_images_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
NavsimImageResolver = MODULE.NavsimImageResolver
make_reader = MODULE.make_reader


def _write_image(path: Path, color=(12, 34, 56)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (13, 7), color=color).save(path)


def test_clean_reference_preserves_the_original_file(tmp_path):
    relative = "test/log/CAM_F0/front.jpg"
    _write_image(tmp_path / relative)
    resolver = NavsimImageResolver(image_root=tmp_path)

    reference = resolver.reference(relative)

    assert reference == (tmp_path / relative).resolve()
    with Image.open(tmp_path / relative) as expected:
        assert resolver(relative).tobytes() == expected.convert("RGB").tobytes()


def test_front_dropout_changes_only_the_selected_camera(tmp_path):
    front = "test/log/CAM_F0/front.jpg"
    left = "test/log/CAM_L0/left.jpg"
    _write_image(tmp_path / front)
    _write_image(tmp_path / left, color=(80, 90, 100))
    resolver = NavsimImageResolver(
        image_root=tmp_path,
        corruption="camera_dropout",
        cameras=frozenset({"CAM_F0"}),
    )

    front_reference = resolver.reference(front)

    assert callable(front_reference)
    assert front_reference().size == (13, 7)
    assert front_reference().getbbox() is None
    assert resolver.reference(left) == (tmp_path / left).resolve()
    with Image.open(tmp_path / left) as expected:
        assert resolver(left).tobytes() == expected.convert("RGB").tobytes()


def test_path_manifest_can_limit_dropout_to_the_current_frame(tmp_path):
    history = "test/log/CAM_F0/history.jpg"
    current = "test/log/CAM_F0/current.jpg"
    _write_image(tmp_path / history)
    _write_image(tmp_path / current)
    resolver = NavsimImageResolver(
        image_root=tmp_path,
        corruption="camera_dropout",
        selected_paths=frozenset({current}),
    )

    assert resolver.reference(history) == (tmp_path / history).resolve()
    assert callable(resolver.reference(current))


def test_environment_factory_and_path_guard(tmp_path, monkeypatch):
    relative = "test/log/CAM_F0/current.jpg"
    _write_image(tmp_path / relative)
    manifest = tmp_path / "paths.txt"
    manifest.write_text(f"# current front only\n{relative}\n", encoding="utf-8")
    monkeypatch.setenv("ORION_NAVSIM_IMAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("ORION_NAVSIM_CORRUPTION", "camera_dropout")
    monkeypatch.setenv("ORION_NAVSIM_CORRUPTION_CAMERAS", "CAM_F0")
    monkeypatch.setenv("ORION_NAVSIM_CORRUPTION_PATHS", str(manifest))

    resolver = make_reader()

    assert callable(resolver.reference(relative))
    with pytest.raises(ValueError, match="must be relative"):
        resolver.reference("../outside.jpg")
