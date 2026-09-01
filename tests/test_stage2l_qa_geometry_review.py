import json

from scripts.build_stage2l_qa_geometry_review_queue import (
    HUMAN_CHECKS,
    build_queue,
    decisions_template,
)
from scripts.freeze_stage2l_qa_geometry_review import freeze_review
from scripts.scenario_factory_lib import sha256_file


def test_hash_bound_multiframe_geometry_review_freezes_acceptance(tmp_path):
    visualizations = []
    variants = ("observed", "zero_uq", "on_path_uq", "off_path_uq", "view_shuffled_uq")
    for frame in (10, 12, 14):
        for variant in variants:
            manifest = tmp_path / ("%d_%s_manifest.json" % (frame, variant))
            manifest.write_text("{}")
            contact = tmp_path / ("%d_%s.png" % (frame, variant))
            contact.write_bytes(b"png")
            visualizations.append({
                "selected_saved_frame_index": frame,
                "variant": variant,
                "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
                "contact_sheet": {"path": str(contact), "sha256": sha256_file(contact)},
            })
    report = tmp_path / "factory.json"
    report.write_text(json.dumps({
        "schema": "orion.uq_relevance_multiframe_event_factory.v1",
        "status": "pending_multiframe_human_geometry_review",
        "event_id": "route1_step20",
        "event_package": {"path": "event.json", "sha256": "a" * 64},
        "keyframe_count": 3,
        "selected_saved_frames": [10, 12, 14],
        "visualizations": visualizations,
    }))
    queue = build_queue([report])
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue))
    decisions = decisions_template(queue, queue_path)
    decisions["reviewer"] = "reviewer"
    decisions["reviewed_at"] = "2026-08-29T13:00:00+08:00"
    decisions["status"] = "reviewed"
    decisions["decisions"][0]["decision"] = "accept"
    decisions["decisions"][0]["checks"] = {name: "pass" for name in HUMAN_CHECKS}
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(decisions))
    frozen = freeze_review(queue_path=queue_path, decisions_path=decisions_path)
    assert frozen["accepted_count"] == 1
    assert frozen["rejected_count"] == 0
    assert frozen["accepted"][0]["event_id"] == "route1_step20"
