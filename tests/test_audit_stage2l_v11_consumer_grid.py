import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_stage2l_v11_consumer_grid import audit_consumer_grid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_variant(tmp_path, group_id, variant, center):
    support = np.zeros((6, 40, 40), dtype=np.float32)
    view, y, x = center
    values = {
        (-1, 0): 0.4,
        (0, -1): 0.4,
        (0, 0): 0.9,
        (0, 1): 0.4,
        (1, 0): 0.4,
    }
    for (dy, dx), value in values.items():
        support[view, y + dy, x + dx] = value
    components = np.repeat(support[None, ..., None], 4, axis=0)
    components = np.repeat(components, 3, axis=-1)
    path = tmp_path / (variant + ".npz")
    np.savez_compressed(path, uncertainty_components=components)
    return {
        "split": "dev",
        "question_family": "task_relevance",
        "counterfactual": {"group_id": group_id, "variant": variant},
        "model_input": {
            "stage1_observation_uq": {
                "path": str(path),
                "sha256": _sha256(path),
                "component_key": "uncertainty_components",
            }
        },
    }


def test_raw_match_can_be_destroyed_by_consumer_pooling(tmp_path):
    records = tmp_path / "records.jsonl"
    rows = [
        _write_variant(tmp_path, "group", "on_path_uq", (0, 8, 8)),
        _write_variant(tmp_path, "group", "off_path_uq", (0, 10, 10)),
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit_consumer_grid(records)
    assert result["raw_full_match_count"] == 1
    assert result["consumer_full_match_count"] == 0
    assert result["status"] == "raw_match_destroyed_before_model_consumer"
    assert result["by_split"]["dev"] == {
        "groups": 1,
        "raw_matches": 1,
        "consumer_matches": 0,
    }
