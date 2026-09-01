import hashlib
import json

import numpy as np

from scripts.audit_stage2l_v11_identifiability_dataset import audit_dataset


VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(tmp_path, *, include_ego=True):
    route_payload = {
        "command": 4,
        "orion_unmodified_plan_right_forward_m": [[0.0, 2.0]],
    }
    if include_ego:
        route_payload["ego_state"] = {"speedometer_mps": 3.0}
    route_context = {
        "schema": "orion.route_context.v2" if include_ego else None,
        "payload": route_payload,
        "sha256": hashlib.sha256(
            json.dumps(route_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    supports = {
        "observed": {"support_type": "frozen_stage1_observed"},
        "zero_uq": {"support_type": "zero_everywhere"},
        "view_shuffled_uq": {
            "support_type": "cyclic_view_shift",
            "view_shift": 1,
        },
    }
    common = {
        "latest_nonzero_patches": 1,
        "latest_peak": 0.9,
        "latest_spatial_sum": 0.9,
        "radius_patches": 1,
        "matched_on_path_center_view_y_x": [0, 1, 1],
        "matched_off_path_center_view_y_x": [0, 0, 0],
        "same_view_matched_pair": True,
        "support_type": "matched_local_gaussian_region_v1",
    }
    supports["on_path_uq"] = dict(
        common,
        center_view_y_x=[0, 1, 1],
        support_weighted_relevance=0.9,
    )
    supports["off_path_uq"] = dict(
        common,
        center_view_y_x=[0, 0, 0],
        support_weighted_relevance=0.0,
    )
    rows = []
    for variant in VARIANTS:
        uq = np.zeros((2, 2, 2, 2), dtype=np.float32)
        if variant == "on_path_uq":
            uq[-1, 0, 1, 1] = 0.9
        elif variant == "off_path_uq":
            uq[-1, 0, 0, 0] = 0.9
        elif variant in ("observed", "view_shuffled_uq"):
            uq[-1, 1, 1, 1] = 0.5
        components = np.repeat(uq[..., None], 3, axis=-1)
        path = tmp_path / (variant + ".npz")
        np.savez_compressed(path, uncertainty=uq, uncertainty_components=components)
        model_input = {
            "observation": {"observation_sha256": "a" * 64},
            "route_context": route_context,
            "stage1_observation_uq": {
                "path": path.name,
                "sha256": _sha256(path),
                "shape": list(uq.shape),
                "component_key": "uncertainty_components",
                "component_shape": list(components.shape),
                "checkpoint_sha256": "b" * 64,
            },
        }
        for family in ("task_relevance", "driving_implication"):
            rows.append(
                {
                    "event_id": "event-1",
                    "split": "train",
                    "question_family": family,
                    "counterfactual": {
                        "group_id": "group-1",
                        "variant": variant,
                        "spatial_support": supports[variant],
                    },
                    "model_input": model_input,
                    "provenance": {"relevance_supervision": {"sha256": "c" * 64}},
                }
            )
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return records


def test_v11_dataset_audit_requires_metadata_and_tensor_evidence(tmp_path):
    records = _records(tmp_path)
    metadata = audit_dataset(records, verify_tensors=False)
    assert metadata["metadata_passed"] is True
    assert metadata["tensor_passed"] is None
    assert metadata["v11_ready"] is False

    full = audit_dataset(records, verify_tensors=True)
    assert full["metadata_passed"] is True
    assert full["tensor_passed"] is True
    assert full["v11_ready"] is True


def test_v11_dataset_audit_rejects_historical_route_context(tmp_path):
    records = _records(tmp_path, include_ego=False)
    report = audit_dataset(records, verify_tensors=True)
    assert report["metadata_passed"] is False
    assert report["v11_ready"] is False
    assert "not version v2" in report["failed_groups"][0]["errors"][0]


def _v111_support(on_path):
    on_center = [0, 4, 5]
    off_center = [0, 4, 1]
    return {
        "support_type": "matched_token_grid_gaussian_region_v2",
        "center_view_y_x": on_center if on_path else off_center,
        "matched_on_path_center_view_y_x": on_center,
        "matched_off_path_center_view_y_x": off_center,
        "radius_cells": 1,
        "consumer_grid_hw": [10, 10],
        "stored_grid_hw": [40, 40],
        "latest_peak": 0.9,
        "latest_spatial_sum": 72.0,
        "latest_nonzero_patches": 80,
        "consumer_latest_peak": 0.9,
        "consumer_latest_spatial_sum": 4.5,
        "consumer_latest_nonzero_cells": 5,
        "support_weighted_relevance": 0.9 if on_path else 0.0,
        "same_view_matched_pair": True,
        "construction": "consumer_grid_then_exact_block_expand",
    }


def test_v111_support_contract_requires_consumer_grid_match(tmp_path):
    records_path = _records(tmp_path)
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    written = set()
    shapes = {}
    for row in rows:
        variant = row["counterfactual"]["variant"]
        reference = row["model_input"]["stage1_observation_uq"]
        if variant not in written:
            uq = np.zeros((2, 6, 40, 40), dtype=np.float32)
            if variant in {"on_path_uq", "off_path_uq"}:
                center = (0, 4, 5) if variant == "on_path_uq" else (0, 4, 1)
                token_grid = np.zeros((6, 10, 10), dtype=np.float32)
                view, y, x = center
                for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                    token_grid[view, y + dy, x + dx] = 0.9
                uq[-1] = np.repeat(np.repeat(token_grid, 4, axis=1), 4, axis=2)
            elif variant in {"observed", "view_shuffled_uq"}:
                uq[-1, 1, 4:8, 4:8] = 0.5
            components = np.repeat(uq[..., None], 3, axis=-1)
            path = tmp_path / (variant + ".npz")
            np.savez_compressed(path, uncertainty=uq, uncertainty_components=components)
            shapes[variant] = (list(uq.shape), list(components.shape))
            written.add(variant)
        reference.update(
            {
                "sha256": _sha256(tmp_path / (variant + ".npz")),
                "shape": shapes[variant][0],
                "component_shape": shapes[variant][1],
            }
        )
        if variant in {"on_path_uq", "off_path_uq"}:
            row["counterfactual"]["spatial_support"] = _v111_support(
                variant == "on_path_uq"
            )
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit_dataset(records_path, verify_tensors=True)
    assert result["v11_ready"] is True
