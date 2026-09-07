import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "audit_qwen_visibility_numeric_text_upper_bound.py"
SPEC = importlib.util.spec_from_file_location("qwen_u_text_upper_bound_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_audit_recomputes_targets_predictions_and_metrics(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"model")
    tokens = {}
    for label, value in (("ON_ROUTE", 0.3), ("OFF_ROUTE", 0.1)):
        token = tmp_path / (label + ".npz")
        np.savez_compressed(
            token,
            visibility_tokens_feature_names=np.asarray(["route_weight_mean"]),
            visibility_tokens_frontier=np.asarray([[value]], dtype=np.float32),
            visibility_tokens_frontier_mask=np.asarray([True]),
        )
        tokens[label] = token
    records = []
    examples = []
    rows = []
    for index in range(20):
        sample_id = "sample-%02d" % index
        example_id = "example-%02d" % index
        split = "validation" if index < 10 else "held_out"
        expected = "ON_ROUTE" if index % 10 < 5 else "OFF_ROUTE"
        value = 0.3 if expected == "ON_ROUTE" else 0.1
        records.append(
            {
                "sample_id": sample_id,
                "token_artifact": str(tokens[expected]),
                "query_frontier": "F00",
                "camera_images": ["left.png", "front.png", "right.png"],
            }
        )
        examples.append(
            {
                "example_id": example_id,
                "sample_id": sample_id,
                "query_frontier": "F00",
                "split": split,
                "expected_answer": expected,
            }
        )
        for mode in ("text_only", "rgb_plus_text"):
            rows.append(
                {
                    "mode": mode,
                    "example_id": example_id,
                    "sample_id": sample_id,
                    "split": split,
                    "frontier": "F00",
                    "route_weight_mean": float(np.float32(value)),
                    "expected": expected,
                    "predicted": expected,
                    "correct": True,
                    "target_margin": 1.0,
                    "candidate_nlls": (
                        {"ON_ROUTE": 1.0, "OFF_ROUTE": 2.0}
                        if expected == "ON_ROUTE"
                        else {"ON_ROUTE": 2.0, "OFF_ROUTE": 1.0}
                    ),
                }
            )
    manifest = tmp_path / "manifest.json"
    curriculum = tmp_path / "curriculum.json"
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    curriculum.write_text(
        json.dumps(
            {
                "examples": examples,
                "evaluation_example_ids": [row["example_id"] for row in examples],
            }
        ),
        encoding="utf-8",
    )
    metrics = MODULE.summarize(rows)
    report = {
        "schema": MODULE.SCHEMA,
        "status": "complete",
        "training_performed": False,
        "optimizer_steps": 0,
        "continuous_u_tokens_present": False,
        "planning_expert_loaded": False,
        "system_prompt": MODULE.SYSTEM_PROMPT,
        "thresholds": {
            "minimum_overall_accuracy": 0.9,
            "minimum_each_split_and_label_accuracy": 0.8,
        },
        "passed_by_mode": {"text_only": True, "rgb_plus_text": True},
        "metrics": metrics,
        "rows": rows,
        "inputs": {
            "model": str(model),
            "model_sha256": MODULE.sha256(model / "model.safetensors"),
            "manifest": str(manifest),
            "manifest_sha256": MODULE.sha256(manifest),
            "curriculum": str(curriculum),
            "curriculum_sha256": MODULE.sha256(curriculum),
        },
        "claim_boundary": {
            "language_and_numeric_upper_bound_only": True,
            "continuous_u_alignment_claim_allowed": False,
            "visual_spatial_alignment_claim_allowed": False,
            "planning_or_safety_claim_allowed": False,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = MODULE.audit(report, report_path)
    assert result["status"] == "passed"
    assert result["failures"] == []
    assert result["row_count"] == 40
