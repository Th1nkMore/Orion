import hashlib
import json

import numpy as np
import pytest

from scripts.audit_stage1_u_tokenizer_pretraining_sources import audit_sources


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, variant="observed", map_source="frozen_stage1_observation_adapter"):
    event_specs = []
    for index, split in enumerate(("train", "dev")):
        component_path = tmp_path / ("uq_observed_%s.npz" % split)
        np.savez_compressed(
            component_path,
            uncertainty_components=np.linspace(
                0.0, 1.0, 4 * 6 * 40 * 40 * 3, dtype=np.float32
            ).reshape(4, 6, 40, 40, 3) * (1.0 - 0.1 * index),
        )
        records_path = tmp_path / ("records_%s.jsonl" % split)
        row = {
            "counterfactual": {
                "variant": variant,
                "group_id": "%s_event/frame" % split,
            },
            "model_input": {
                "stage1_observation_uq": {
                    "source": map_source,
                    "control_influence": False,
                    "component_names": [
                        "persistent_direction",
                        "persistent_magnitude",
                        "transient_inconsistency",
                    ],
                    "component_key": "uncertainty_components",
                    "component_shape": [4, 6, 40, 40, 3],
                    "checkpoint_sha256": "a" * 64,
                    "path": str(component_path),
                    "sha256": _sha(component_path),
                }
            },
            "target": {"must_not_be_read": "task label"},
            "conversation": [{"must_not_be_read": "QA text"}],
        }
        records_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        event_specs.append({
            "event_id": "%s_event" % split,
            "split": split,
            "keyframe_count": 1,
            "source_records": {
                "path": str(records_path),
                "sha256": _sha(records_path),
            },
        })
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"events": event_specs}), encoding="utf-8")
    return manifest_path


def test_audit_selects_only_unique_observed_u_and_records_no_task_inputs(tmp_path):
    report = audit_sources(_fixture(tmp_path))
    assert report["passed"] is True
    assert report["unique_observed_u_map_count"] == 2
    assert report["train_map_count"] == 1
    assert report["dev_map_count"] == 1
    assert report["forbidden_inputs_consumed"]["task_relevance"] is False
    assert report["forbidden_inputs_consumed"]["qa_text_or_fields"] is False


def test_audit_rejects_task_constructed_variant_as_observed_source(tmp_path):
    manifest = _fixture(tmp_path, variant="on_path_uq")
    with pytest.raises(ValueError, match="coverage"):
        audit_sources(manifest)


def test_audit_rejects_non_stage1_observed_source(tmp_path):
    manifest = _fixture(tmp_path, map_source="controlled_stage1_uq_counterfactual")
    with pytest.raises(ValueError, match="provenance"):
        audit_sources(manifest)
