"""Deterministic structured targets for Qwen visibility-token grounding.

This module is NumPy-only.  It defines what the VLM is asked to read from the
physical tokens, but it never imports Qwen, Torch, CARLA, or a planning model.
The target is a description of an observed visibility frontier and its
deterministic route/stopping exposure; it is not a hidden-actor label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .qwen_visibility_belief import (
    VISIBILITY_TOKEN_FEATURE_NAMES,
    VISIBILITY_TOKEN_SCHEMA,
)


VISIBILITY_GROUNDING_SCHEMA = "orion.qwen-visibility-grounding/v1"
VISIBILITY_GROUNDING_MANIFEST_SCHEMA = (
    "orion.qwen-visibility-grounding-manifest/v1"
)
GROUNDING_SYSTEM_PROMPT = (
    "You ground an explicit metric visibility-belief input for driving. "
    "The continuous tokens describe observed free/occupied space, occluded "
    "unknown space, observation age, and deterministic route/stopping "
    "exposure. They do not assert that a hidden actor exists."
)
GROUNDING_QUESTION = (
    "Read the continuous visibility-belief tokens inserted after the camera "
    "block. Select the frontier token with the largest "
    "frontier_selection_score. Report whether that frontier intersects the "
    "route, its stopping-margin bucket, and the conservative longitudinal "
    "response. Reply with exactly one compact JSON object with keys "
    "frontier, route, margin, action. Allowed values are F00..F31, "
    "ON_ROUTE/OFF_ROUTE, INSIDE/NEAR/CLEAR, and KEEP/SLOW/STOP."
)


@dataclass(frozen=True)
class GroundingThresholds:
    """Versioned deterministic thresholds used only to create VLM targets."""

    route_weight_on: float = 0.2
    near_margin_m: float = 5.0
    max_range_m: float = 60.0
    slow_urgency_max: float = 0.1

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.route_weight_on,
                self.near_margin_m,
                self.max_range_m,
                self.slow_urgency_max,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("grounding thresholds must be finite")
        if not 0.0 < float(self.route_weight_on) <= 1.0:
            raise ValueError("route_weight_on must lie in (0,1]")
        if float(self.near_margin_m) <= 0.0:
            raise ValueError("near_margin_m must be positive")
        if float(self.max_range_m) <= float(self.near_margin_m):
            raise ValueError("max_range_m must exceed near_margin_m")
        if not 0.0 <= float(self.slow_urgency_max) <= 1.0:
            raise ValueError("slow_urgency_max must lie in [0,1]")

    @property
    def near_margin_normalized(self) -> float:
        return float(self.near_margin_m / self.max_range_m)

    def as_dict(self) -> Dict[str, float]:
        return {
            "route_weight_on": float(self.route_weight_on),
            "near_margin_m": float(self.near_margin_m),
            "max_range_m": float(self.max_range_m),
            "near_margin_normalized": self.near_margin_normalized,
            "slow_urgency_max": float(self.slow_urgency_max),
        }


@dataclass(frozen=True)
class VisibilityGroundingTarget:
    """Canonical structured answer for one permuted physical-token scene."""

    frontier_index: int
    original_frontier_index: int
    route: str
    margin: str
    action: str
    route_weight: float
    stopping_margin_normalized: float
    urgency_max: float
    frontier_selection_score: float

    def __post_init__(self) -> None:
        if not 0 <= int(self.frontier_index) < 32:
            raise ValueError("frontier_index must lie in [0,31]")
        if int(self.original_frontier_index) < 0:
            raise ValueError("original_frontier_index must be non-negative")
        if self.route not in {"ON_ROUTE", "OFF_ROUTE"}:
            raise ValueError("invalid route target")
        if self.margin not in {"INSIDE", "NEAR", "CLEAR"}:
            raise ValueError("invalid stopping-margin target")
        if self.action not in {"KEEP", "SLOW", "STOP"}:
            raise ValueError("invalid longitudinal-action target")
        values = np.asarray(
            [
                self.route_weight,
                self.stopping_margin_normalized,
                self.urgency_max,
                self.frontier_selection_score,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("grounding evidence must be finite")

    @property
    def frontier(self) -> str:
        return "F%02d" % int(self.frontier_index)

    def answer_dict(self) -> Dict[str, str]:
        return {
            "frontier": self.frontier,
            "route": self.route,
            "margin": self.margin,
            "action": self.action,
        }

    def canonical_answer(self) -> str:
        return json.dumps(self.answer_dict(), separators=(",", ":"))

    def evidence_dict(self) -> Dict[str, object]:
        return {
            "target_frontier_sequence_index": int(self.frontier_index),
            "target_frontier_original_index": int(self.original_frontier_index),
            "route_weight_mean": float(self.route_weight),
            "frontier_stopping_margin_normalized": float(
                self.stopping_margin_normalized
            ),
            "urgency_max": float(self.urgency_max),
            "frontier_selection_score": float(self.frontier_selection_score),
        }


def _feature_index(feature_names: Sequence[str]) -> Dict[str, int]:
    names = tuple(str(value) for value in feature_names)
    if names != tuple(VISIBILITY_TOKEN_FEATURE_NAMES):
        raise ValueError("visibility grounding requires the exact v1 feature order")
    return {name: index for index, name in enumerate(names)}


def deterministic_frontier_permutation(count: int, seed: int) -> np.ndarray:
    """Return ``new_index -> old_index`` for valid frontier rows."""

    if isinstance(count, bool) or int(count) != count or not 0 < int(count) <= 32:
        raise ValueError("valid frontier count must lie in [1,32]")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("permutation seed must be an integer")
    return np.random.default_rng(int(seed)).permutation(int(count)).astype(np.int64)


def permute_frontier_rows(
    frontier_tokens: np.ndarray,
    frontier_mask: np.ndarray,
    permutation: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a recorded valid-row permutation without touching padded rows."""

    tokens = np.asarray(frontier_tokens, dtype=np.float32)
    mask = np.asarray(frontier_mask, dtype=bool)
    if tokens.ndim != 2 or tokens.shape[1] != len(VISIBILITY_TOKEN_FEATURE_NAMES):
        raise ValueError("frontier tokens have an invalid feature shape")
    if mask.shape != (tokens.shape[0],):
        raise ValueError("frontier mask has an invalid shape")
    valid_count = int(mask.sum())
    if not np.all(mask[:valid_count]) or np.any(mask[valid_count:]):
        raise ValueError("valid frontier rows must precede padding")
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (valid_count,) or sorted(order.tolist()) != list(
        range(valid_count)
    ):
        raise ValueError("frontier permutation must cover every valid row once")
    result = tokens.copy()
    result[:valid_count] = tokens[order]
    return result, mask.copy()


def derive_visibility_grounding_target(
    frontier_tokens: np.ndarray,
    frontier_mask: np.ndarray,
    feature_names: Sequence[str],
    permutation: Sequence[int],
    thresholds: GroundingThresholds = GroundingThresholds(),
) -> VisibilityGroundingTarget:
    """Derive the four accepted labels from one true-U token set.

    The selection happens after a deterministic row permutation, preventing the
    original descending-score serialization from making the answer always F00.
    """

    index = _feature_index(feature_names)
    permuted, mask = permute_frontier_rows(
        frontier_tokens, frontier_mask, permutation
    )
    valid = permuted[mask]
    scores = valid[:, index["frontier_selection_score"]]
    selected = int(np.argmax(scores))
    old_index = int(np.asarray(permutation, dtype=np.int64)[selected])
    row = valid[selected]
    route_weight = float(row[index["route_weight_mean"]])
    margin_value = float(row[index["frontier_stopping_margin_normalized"]])
    urgency_max = float(row[index["urgency_max"]])
    selection_score = float(row[index["frontier_selection_score"]])

    route = (
        "ON_ROUTE"
        if route_weight >= float(thresholds.route_weight_on)
        else "OFF_ROUTE"
    )
    # O3 artifacts are float32; tolerate only their representation error at
    # exact physical bucket boundaries (for example 5 / 60 metres).
    boundary_tolerance = 1e-6
    if margin_value <= 0.0:
        margin = "INSIDE"
    elif margin_value <= thresholds.near_margin_normalized + boundary_tolerance:
        margin = "NEAR"
    else:
        margin = "CLEAR"

    if route == "OFF_ROUTE":
        action = "KEEP"
    elif margin == "INSIDE":
        action = "STOP"
    elif margin == "NEAR" or urgency_max >= float(thresholds.slow_urgency_max):
        action = "SLOW"
    else:
        action = "KEEP"

    return VisibilityGroundingTarget(
        frontier_index=selected,
        original_frontier_index=old_index,
        route=route,
        margin=margin,
        action=action,
        route_weight=route_weight,
        stopping_margin_normalized=margin_value,
        urgency_max=urgency_max,
        frontier_selection_score=selection_score,
    )


def load_true_visibility_token_artifact(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...], dict]:
    """Load and fail closed on a pickle-free oracle true-U O3 artifact."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as artifact:
        global_tokens = np.asarray(
            artifact["visibility_tokens_global"], dtype=np.float32
        )
        frontier_tokens = np.asarray(
            artifact["visibility_tokens_frontier"], dtype=np.float32
        )
        global_mask = np.asarray(
            artifact["visibility_tokens_global_mask"], dtype=bool
        )
        frontier_mask = np.asarray(
            artifact["visibility_tokens_frontier_mask"], dtype=bool
        )
        feature_names = tuple(
            str(value) for value in artifact["visibility_tokens_feature_names"].tolist()
        )
        token_metadata = json.loads(
            str(artifact["visibility_tokens_metadata_json"])
        )
        provenance = json.loads(str(artifact["provenance_json"]))
    _feature_index(feature_names)
    if token_metadata.get("schema") != VISIBILITY_TOKEN_SCHEMA:
        raise ValueError("unexpected visibility token schema")
    if token_metadata.get("control") != "true_u":
        raise ValueError("grounding supervision accepts true-U artifacts only")
    if (
        provenance.get("source_oracle_depth") is not True
        or provenance.get("source_used_by_qwen") is not False
    ):
        raise ValueError("grounding source must be unused oracle U")
    if global_tokens.shape != (16, len(feature_names)):
        raise ValueError("expected the accepted 16-token global raster")
    if frontier_tokens.shape != (32, len(feature_names)):
        raise ValueError("expected the accepted 32-token frontier budget")
    if global_mask.shape != (16,) or not bool(global_mask.all()):
        raise ValueError("global token mask contract changed")
    if frontier_mask.shape != (32,) or not bool(frontier_mask.any()):
        raise ValueError("frontier token mask contract changed")
    if not np.isfinite(global_tokens).all() or not np.isfinite(
        frontier_tokens
    ).all():
        raise ValueError("visibility token artifact contains non-finite values")
    return (
        global_tokens,
        frontier_tokens,
        global_mask,
        frontier_mask,
        feature_names,
        provenance,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_route151_grounding_manifest(
    token_root: Path,
    audit_root: Path,
    output_path: Path,
    permutation_seed: int = 20260906,
    thresholds: GroundingThresholds = GroundingThresholds(),
) -> dict:
    """Build the explicitly non-reportable sparse Route-151 V1 plumbing set."""

    token_root = Path(token_root).resolve()
    audit_root = Path(audit_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite grounding manifest: %s" % output_path)
    if not token_root.is_dir() or not audit_root.is_dir():
        raise FileNotFoundError("token_root and audit_root must be directories")
    if isinstance(permutation_seed, bool) or not isinstance(
        permutation_seed, (int, np.integer)
    ):
        raise ValueError("permutation_seed must be an integer")

    camera_names = (
        "CAM_FRONT_rgb.png",
        "CAM_FRONT_LEFT_rgb.png",
        "CAM_FRONT_RIGHT_rgb.png",
    )
    records = []
    for audit_step in sorted(audit_root.glob("step_*")):
        if not audit_step.is_dir():
            continue
        try:
            step = int(audit_step.name.split("_")[-1])
        except ValueError as error:
            raise ValueError("invalid audit step directory: %s" % audit_step) from error
        token_path = token_root / ("step_%06d.npz" % step)
        image_paths = [audit_step / name for name in camera_names]
        if not token_path.is_file() or not all(path.is_file() for path in image_paths):
            raise FileNotFoundError("step %d lacks tokens or all three RGB images" % step)
        (
            _,
            frontier_tokens,
            _,
            frontier_mask,
            feature_names,
            provenance,
        ) = load_true_visibility_token_artifact(token_path)
        valid_count = int(frontier_mask.sum())
        sample_seed = int(permutation_seed) + step
        permutation = deterministic_frontier_permutation(valid_count, sample_seed)
        target = derive_visibility_grounding_target(
            frontier_tokens,
            frontier_mask,
            feature_names,
            permutation,
            thresholds=thresholds,
        )
        source_step = int(provenance.get("source_step", -1))
        if source_step != step:
            raise ValueError("token/audit step mismatch at %d" % step)
        records.append(
            {
                "sample_id": "route151-step-%06d" % step,
                "source_step": step,
                "split": "plumbing_overfit_train_eval",
                "token_artifact": str(token_path),
                "token_sha256": _sha256(token_path),
                "camera_images": [str(path) for path in image_paths],
                "camera_sha256": [_sha256(path) for path in image_paths],
                "frontier_permutation_new_to_old": permutation.tolist(),
                "frontier_permutation_seed": sample_seed,
                "target": target.answer_dict(),
                "canonical_answer": target.canonical_answer(),
                "target_evidence": target.evidence_dict(),
            }
        )
    if not records:
        raise FileNotFoundError("audit root contains no complete step directories")

    counts = {
        field: {
            value: sum(record["target"][field] == value for record in records)
            for value in sorted({record["target"][field] for record in records})
        }
        for field in ("frontier", "route", "margin", "action")
    }
    manifest = {
        "schema": VISIBILITY_GROUNDING_MANIFEST_SCHEMA,
        "grounding_schema": VISIBILITY_GROUNDING_SCHEMA,
        "purpose": "Route 151 structured-grounding plumbing overfit only",
        "reportable_generalization": False,
        "oracle_depth": True,
        "hidden_actor_labels_used": False,
        "controls_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "image_profile": "three native 1600x900 current RGB views; official Qwen current-image preprocessing",
        "token_root": str(token_root),
        "audit_root": str(audit_root),
        "permutation_seed": int(permutation_seed),
        "thresholds": thresholds.as_dict(),
        "system_prompt": GROUNDING_SYSTEM_PROMPT,
        "question": GROUNDING_QUESTION,
        "record_count": len(records),
        "target_counts": counts,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
