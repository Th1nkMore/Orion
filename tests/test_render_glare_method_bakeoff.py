import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_glare_method_bakeoff import _legacy_white_patch, _load_selected_images


def test_legacy_patch_reproduces_frozen_alpha_blend():
    image = np.full((10, 20, 3), 100, dtype=np.uint8)
    changed = _legacy_white_patch(image, (0.25, 0.2, 0.75, 0.8), 0.6)
    assert np.all(changed[2:8, 5:15] == 193)
    assert np.all(changed[:2] == 100)


def _selected_manifest(tmp_path: Path):
    profiles = {}
    for profile in ("clean", "light", "medium", "heavy"):
        rows = []
        for capture_index in (1, 2, 3):
            image = tmp_path / profile / ("%04d.png" % capture_index)
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(("%s-%d" % (profile, capture_index)).encode("ascii"))
            rows.append(
                {
                    "capture_index": capture_index,
                    "path": str(image.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
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
    return manifest


def test_selected_manifest_is_hash_bound_and_profile_aligned(tmp_path):
    clean, matches = _load_selected_images(_selected_manifest(tmp_path))
    assert [row["capture_index"] for row in clean] == [1, 2, 3]
    assert set(matches) == {"light", "medium", "heavy"}
    assert all(len(rows) == 3 for rows in matches.values())


def test_selected_manifest_rejects_changed_image(tmp_path):
    manifest = _selected_manifest(tmp_path)
    (tmp_path / "clean" / "0001.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_selected_images(manifest)
