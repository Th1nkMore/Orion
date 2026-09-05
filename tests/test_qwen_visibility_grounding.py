import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType("_qwen_visibility_grounding_test_package")
package.__path__ = [str(PROJECT_ROOT / "uq_estimator")]
sys.modules[package.__name__] = package
visibility = _load_local_module(
    package.__name__ + ".qwen_visibility_belief",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py",
)
grounding = _load_local_module(
    package.__name__ + ".qwen_visibility_grounding",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding.py",
)

VISIBILITY_TOKEN_FEATURE_NAMES = visibility.VISIBILITY_TOKEN_FEATURE_NAMES
VISIBILITY_TOKEN_SCHEMA = visibility.VISIBILITY_TOKEN_SCHEMA
GroundingThresholds = grounding.GroundingThresholds
build_route151_grounding_manifest = grounding.build_route151_grounding_manifest
derive_visibility_grounding_target = grounding.derive_visibility_grounding_target
deterministic_frontier_permutation = grounding.deterministic_frontier_permutation
permute_frontier_rows = grounding.permute_frontier_rows


NAMES = VISIBILITY_TOKEN_FEATURE_NAMES
INDEX = {name: index for index, name in enumerate(NAMES)}


def _frontiers(rows):
    tokens = np.zeros((32, len(NAMES)), dtype=np.float32)
    mask = np.zeros(32, dtype=bool)
    for index, values in enumerate(rows):
        mask[index] = True
        tokens[index, INDEX["token_is_frontier"]] = 1.0
        for name, value in values.items():
            tokens[index, INDEX[name]] = value
    return tokens, mask


def _write_token(path: Path, step: int, rows):
    global_tokens = np.zeros((16, len(NAMES)), dtype=np.float32)
    global_tokens[:, INDEX["token_is_global"]] = 1.0
    frontier_tokens, frontier_mask = _frontiers(rows)
    metadata = {
        "schema": VISIBILITY_TOKEN_SCHEMA,
        "control": "true_u",
        "feature_names": list(NAMES),
    }
    provenance = {
        "source_oracle_depth": True,
        "source_used_by_qwen": False,
        "source_step": step,
    }
    np.savez_compressed(
        path,
        visibility_tokens_global=global_tokens,
        visibility_tokens_frontier=frontier_tokens,
        visibility_tokens_global_mask=np.ones(16, dtype=bool),
        visibility_tokens_frontier_mask=frontier_mask,
        visibility_tokens_feature_names=np.asarray(NAMES),
        visibility_tokens_metadata_json=np.asarray(json.dumps(metadata)),
        provenance_json=np.asarray(json.dumps(provenance)),
    )


def test_frontier_permutation_removes_f00_selection_shortcut():
    tokens, mask = _frontiers(
        [
            {"frontier_selection_score": 0.9, "route_weight_mean": 0.5},
            {"frontier_selection_score": 0.1, "route_weight_mean": 0.5},
            {"frontier_selection_score": 0.2, "route_weight_mean": 0.5},
        ]
    )
    permutation = np.asarray([2, 0, 1])
    target = derive_visibility_grounding_target(
        tokens, mask, NAMES, permutation
    )
    assert target.frontier == "F01"
    assert target.original_frontier_index == 0
    assert json.loads(target.canonical_answer())["frontier"] == "F01"
    permuted, permuted_mask = permute_frontier_rows(tokens, mask, permutation)
    assert np.array_equal(permuted_mask, mask)
    assert permuted[1, INDEX["frontier_selection_score"]] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "route_weight,margin,urgency,expected",
    [
        (0.19, -0.2, 1.0, ("OFF_ROUTE", "INSIDE", "KEEP")),
        (0.20, -0.001, 0.0, ("ON_ROUTE", "INSIDE", "STOP")),
        (0.50, 5.0 / 60.0, 0.0, ("ON_ROUTE", "NEAR", "SLOW")),
        (0.50, 0.20, 0.10, ("ON_ROUTE", "CLEAR", "SLOW")),
        (0.50, 0.20, 0.099, ("ON_ROUTE", "CLEAR", "KEEP")),
    ],
)
def test_grounding_target_threshold_boundaries(
    route_weight, margin, urgency, expected
):
    tokens, mask = _frontiers(
        [
            {
                "frontier_selection_score": 0.7,
                "route_weight_mean": route_weight,
                "frontier_stopping_margin_normalized": margin,
                "urgency_max": urgency,
            }
        ]
    )
    target = derive_visibility_grounding_target(
        tokens,
        mask,
        NAMES,
        np.asarray([0]),
        thresholds=GroundingThresholds(),
    )
    assert (target.route, target.margin, target.action) == expected


def test_permutation_is_seeded_and_covers_valid_rows():
    first = deterministic_frontier_permutation(32, 91)
    second = deterministic_frontier_permutation(32, 91)
    different = deterministic_frontier_permutation(32, 92)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert sorted(first.tolist()) == list(range(32))


def test_sparse_manifest_is_hashed_nonreportable_and_refuses_overwrite(tmp_path):
    token_root = tmp_path / "tokens"
    audit_root = tmp_path / "audit"
    token_root.mkdir()
    step_root = audit_root / "step_000260"
    step_root.mkdir(parents=True)
    token_path = token_root / "step_000260.npz"
    _write_token(
        token_path,
        260,
        [
            {
                "frontier_selection_score": 0.8,
                "route_weight_mean": 0.6,
                "frontier_stopping_margin_normalized": -0.01,
                "urgency_max": 0.7,
            },
            {"frontier_selection_score": 0.2},
        ],
    )
    for name in (
        "CAM_FRONT_rgb.png",
        "CAM_FRONT_LEFT_rgb.png",
        "CAM_FRONT_RIGHT_rgb.png",
    ):
        (step_root / name).write_bytes((name + "\n").encode())
    output = tmp_path / "manifest.json"
    manifest = build_route151_grounding_manifest(
        token_root, audit_root, output, permutation_seed=10
    )
    assert manifest["record_count"] == 1
    assert manifest["reportable_generalization"] is False
    assert manifest["hidden_actor_labels_used"] is False
    assert manifest["controls_used_for_optimizer"] is False
    record = manifest["records"][0]
    assert record["token_sha256"] == hashlib.sha256(token_path.read_bytes()).hexdigest()
    assert record["target"]["route"] == "ON_ROUTE"
    assert record["target"]["action"] == "STOP"
    assert json.loads(record["canonical_answer"]) == record["target"]
    with pytest.raises(FileExistsError):
        build_route151_grounding_manifest(token_root, audit_root, output)


def test_invalid_feature_order_fails_closed():
    tokens, mask = _frontiers([{"frontier_selection_score": 1.0}])
    wrong = list(NAMES)
    wrong[2], wrong[3] = wrong[3], wrong[2]
    with pytest.raises(ValueError, match="exact v1 feature order"):
        derive_visibility_grounding_target(tokens, mask, wrong, np.asarray([0]))
