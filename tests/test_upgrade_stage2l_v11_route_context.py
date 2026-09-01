import hashlib
import json

from scripts.upgrade_stage2l_v11_route_context import upgrade_records


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_route_context_upgrade_uses_hash_bound_source_meta(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "command": 4,
        "plan": [[0.1, 2.0], [0.2, 4.0]],
        "speed": 3.25,
        "desired_speed": 9.0,
        "route_progress": 0.5,
    }))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({
        "source_meta": {"path": str(meta), "sha256": _sha256(meta)}
    }))
    rows = []
    for variant in ("zero_uq", "on_path_uq"):
        for family in ("task_relevance", "driving_implication"):
            rows.append({
                "schema": "orion.uq_relevance_qa_record.v5",
                "split": "train",
                "event_id": "event-1",
                "question_family": family,
                "counterfactual": {"group_id": "group-1", "variant": variant},
                "model_input": {
                    "route_context": {
                        "payload": {
                            "command": 4,
                            "orion_unmodified_plan_right_forward_m": [
                                [0.1, 2.0], [0.2, 4.0]
                            ],
                        },
                        "sha256": "a" * 64,
                    },
                    "observation": {"observation_sha256": "b" * 64},
                    "stage1_observation_uq": {"checkpoint_sha256": "c" * 64},
                },
                "provenance": {
                    "relevance_supervision": {
                        "sha256": "d" * 64,
                        "geometry_manifest": {
                            "path": str(geometry),
                            "sha256": _sha256(geometry),
                        },
                    }
                },
            })
    source = tmp_path / "records.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = upgrade_records(source, tmp_path / "output")
    assert report["group_count"] == 1
    assert report["output"]["record_count"] == 4
    upgraded = [
        json.loads(line)
        for line in (tmp_path / "output" / "records.jsonl").read_text().splitlines()
    ]
    for row in upgraded:
        route = row["model_input"]["route_context"]
        assert route["schema"] == "orion.route_context.v2"
        assert route["payload"]["ego_state"] == {
            "speedometer_mps": 3.25
        }
        assert "desired_speed" not in route["payload"]
        assert "route_progress" not in route["payload"]
        assert row["provenance"]["route_context_v11_upgrade"][
            "route_context_only_change"
        ] is True
