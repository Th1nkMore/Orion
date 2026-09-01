import hashlib
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.analyze_orion_glare_input_stage import (
    _load_selected,
    _map_normalized_roi,
    _q20_round_trip,
    _tensor_sha256,
    preprocess_orion_front,
)


IDA = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
NORM = {
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
    "to_rgb": True,
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_maps_raw_hazard_roi_through_frozen_orion_crop():
    mapped = _map_normalized_roi((0.35, 0.50, 0.90, 0.85), IDA)
    assert mapped == pytest.approx((0.35, 0.4375, 0.90, 0.83125))


def test_preprocess_produces_finite_640_tensor_and_stable_hash():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[..., 0] = np.arange(1600, dtype=np.uint16)[None, :] % 256
    image[..., 1] = np.arange(900, dtype=np.uint16)[:, None] % 256
    image[..., 2] = 127

    tensor, model_bgr, geometry = preprocess_orion_front(image, ida=IDA, norm=NORM)

    assert tensor.shape == (640, 640, 3)
    assert tensor.dtype == np.float32
    assert model_bgr.shape == (640, 640, 3)
    assert np.isfinite(tensor).all()
    assert geometry["resize_dims_wh"] == [640, 360]
    assert geometry["crop_ltrb"] == [0, 40, 640, 360]
    assert _tensor_sha256(tensor) == _tensor_sha256(tensor.copy())


def test_preprocess_matches_repository_inference_transforms():
    pytest.importorskip("torch")
    try:
        from mmcv.datasets.pipelines.transforms_3d import (
            NormalizeMultiviewImage,
            PadMultiViewImage,
            ResizeCropFlipRotImage,
            ResizeMultiview3D,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip("repository MMCV inference transforms unavailable: %s" % exc)

    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[..., 0] = np.arange(1600, dtype=np.uint16)[None, :] % 256
    image[..., 1] = np.arange(900, dtype=np.uint16)[:, None] % 256
    image[..., 2] = 127
    expected, expected_bgr, _ = preprocess_orion_front(image, ida=IDA, norm=NORM)

    results = {
        "img": [_q20_round_trip(image).astype(np.float32)],
        "cam_intrinsic": [np.eye(4, dtype=np.float32)],
        "lidar2cam": [np.eye(4, dtype=np.float32)],
    }
    results = ResizeCropFlipRotImage(IDA, training=False)(results)
    results = ResizeMultiview3D(
        img_scale=(640, 640), keep_ratio=False, multiscale_mode="value"
    )(results)
    assert np.array_equal(results["img"][0], expected_bgr)
    results = NormalizeMultiviewImage(**NORM)(results)
    results = PadMultiViewImage(size_divisor=32)(results)

    actual = np.ascontiguousarray(results["img"][0], dtype=np.float32)
    assert np.array_equal(actual, expected)


def test_selected_manifest_is_hash_bound_and_capture_aligned(tmp_path):
    profiles = {}
    for profile in ("clean", "light", "medium", "heavy"):
        rows = []
        for index in (1, 2, 3):
            path = tmp_path / ("%s_%d.png" % (profile, index))
            assert cv2.imwrite(str(path), np.full((4, 4, 3), index, np.uint8))
            rows.append(
                {"capture_index": index, "path": str(path), "sha256": _sha(path)}
            )
        profiles[profile] = rows
    manifest = tmp_path / "selected.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "orion.glare_method_selected_images.v1",
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )

    _, resolved = _load_selected(manifest)
    assert [row["capture_index"] for row in resolved["heavy"]] == [1, 2, 3]

    profiles["heavy"][0]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(
            {
                "schema": "orion.glare_method_selected_images.v1",
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash-mismatched"):
        _load_selected(manifest)
