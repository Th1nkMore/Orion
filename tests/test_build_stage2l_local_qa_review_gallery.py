import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_stage2l_local_qa_review_gallery import (
    VARIANT_ORDER,
    build_gallery,
)


def _write_queue(tmp_path: Path):
    event_id = "route151_step218"
    root = tmp_path / "contact_sheets"
    variants = {}
    for variant in VARIANT_ORDER:
        payload = (event_id + ":" + variant).encode("utf-8")
        local = root / event_id / "frame_0018" / variant / "sheet.png"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        variants[variant] = {
            "contact_sheet": {
                "path": "/remote/%s/frame_0018/%s/sheet.png"
                % (event_id, variant),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
    queue = {
        "schema": "orion.stage2_l.qa_geometry_review_queue.v1",
        "status": "pending_human_qa_geometry_review",
        "review_order": [
            {
                "event_id": event_id,
                "keyframe_count": 1,
                "required_checks": ["map_text_consistency"],
                "visualizations": [
                    {
                        "selected_saved_frame_index": 18,
                        "variants": variants,
                    }
                ],
            }
        ],
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    return queue_path, root


def test_gallery_verifies_and_renders_all_five_variants(tmp_path):
    queue, root = _write_queue(tmp_path)
    output = tmp_path / "gallery.html"
    counts = build_gallery(
        queue_path=queue,
        contact_sheet_root=root,
        output_path=output,
    )
    rendered = output.read_text(encoding="utf-8")
    assert counts == {"event_count": 1, "frame_count": 1, "image_count": 5}
    assert rendered.count("<img ") == 5
    assert all(variant in rendered for variant in VARIANT_ORDER)


def test_gallery_fails_closed_on_local_hash_mismatch(tmp_path):
    queue, root = _write_queue(tmp_path)
    corrupt = next(root.rglob("sheet.png"))
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_gallery(
            queue_path=queue,
            contact_sheet_root=root,
            output_path=tmp_path / "gallery.html",
        )
