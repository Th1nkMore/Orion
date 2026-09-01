import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "assets/waterdrop_patterns/evocargo_ccby4_v1"
PROTOCOL = ROOT / "configs/scenario_factory/lens_waterdrop_v2_source_protocol_v1.json"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_mask_bank_has_frozen_ccby_provenance_and_nonempty_masks():
    metadata = json.loads((BANK / "metadata.json").read_text())
    assert metadata["schema"] == "orion.real_waterdrop_mask_bank.v1"
    assert metadata["source"]["license"] == "CC-BY-4.0"
    assert metadata["source"]["archive_md5"] == "0ea1b373c981f8ed3ecd311d6596ec0f"
    assert len(metadata["assets"]) == 4
    for row in metadata["assets"]:
        path = BANK / row["file"]
        assert _sha256(path) == row["sha256"]
        with Image.open(path) as mask:
            assert mask.size == (1920, 1080)
            assert mask.convert("L").getbbox() is not None


def test_v2_protocol_does_not_mislabel_derived_displacement_as_real():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == (
        "real_mask_source_frozen_direct_silhouette_renderer_failed_visual_review"
    )
    assert protocol["real_mask_bank"]["scene_rgb_redistributed"] is False
    assert protocol["renderer_contract"]["input_resolution"] == [1600, 900]
    assert "calling derived displacement a real ground-truth field" in protocol[
        "renderer_contract"
    ]["forbidden"]
    assert protocol["execution_locks"]["orion_screen"] is False
    assert "failed visual review" in protocol["claim_boundary"]
