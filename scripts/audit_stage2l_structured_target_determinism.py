#!/usr/bin/env python3
"""Audit whether Stage2-L structured QA targets are deterministic from U/R/K.

This is a data/architecture audit, not model evaluation.  It independently
reconstructs every structured summary from frozen Stage1 scalar uncertainty U,
the task-relevance supervision R, and K=U*R.  A complete match demonstrates
that a learned classifier for risk view, region, level, or stance is not needed
to define those fields; the learned problem can remain the VLM-owned R map and
language interpretation of the resulting deterministic semantics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_structured_target_determinism_audit.v1"
EXPECTED_VARIANTS = {
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
}
EXPECTED_FAMILIES = {
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
}
REAR_VIEWS = {"CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _level(value: float, thresholds: Mapping[str, float]) -> str:
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    return "low"


def _region(row: int, column: int, height: int, width: int) -> str:
    row_name = ("upper", "middle", "lower")[min(2, (3 * row) // height)]
    column_name = ("left", "center", "right")[
        min(2, (3 * column) // width)
    ]
    return row_name + "_" + column_name


def reconstruct_summary(
    uncertainty: np.ndarray,
    relevance: np.ndarray,
    *,
    camera_order: Sequence[str],
    level_thresholds: Mapping[str, float],
    stance_thresholds: Mapping[str, float],
    rearward_high_risk_stance_cap: str,
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    if uncertainty.ndim != 4 or relevance.ndim != 3:
        raise ValueError("expected U[T,V,H,W] and R[V,H,W]")
    if uncertainty.shape[1:] != relevance.shape:
        raise ValueError("U and R spatial shapes differ")
    if uncertainty.shape[1] != len(camera_order):
        raise ValueError("camera order does not match U/R views")
    if not (
        np.isfinite(uncertainty).all()
        and np.isfinite(relevance).all()
        and (uncertainty >= 0.0).all()
        and (uncertainty <= 1.0).all()
        and (relevance >= 0.0).all()
        and (relevance <= 1.0).all()
    ):
        raise ValueError("U/R must be finite and lie in [0,1]")

    latest = uncertainty[-1]
    task_risk = latest * relevance
    uq_flat = int(np.argmax(latest))
    uq_index = tuple(map(int, np.unravel_index(uq_flat, latest.shape)))
    uq_score = float(latest[uq_index])
    risk_flat = int(np.argmax(task_risk))
    risk_index = tuple(map(int, np.unravel_index(risk_flat, task_risk.shape)))
    risk_score = float(task_risk[risk_index])
    height, width = latest.shape[-2:]

    if uq_score <= epsilon:
        observation_view = "none"
        observation_region = "none"
        temporal_delta = 0.0
        temporal_trend = "stable"
        relevance_score = 0.0
        relevance_level = "not_applicable"
    else:
        observation_view = str(camera_order[uq_index[0]])
        observation_region = _region(
            uq_index[1], uq_index[2], height, width
        )
        temporal = uncertainty[:, uq_index[0], uq_index[1], uq_index[2]]
        temporal_delta = float(temporal[-1] - temporal[0])
        temporal_trend = (
            "rising"
            if temporal_delta > 0.05
            else "falling"
            if temporal_delta < -0.05
            else "stable"
        )
        relevance_score = float(relevance[uq_index])
        relevance_level = _level(relevance_score, level_thresholds)

    if risk_score <= epsilon:
        risk_level = "none"
        risk_view = "none"
        risk_region = "none"
        risk_bearing = "none"
        stance = "maintain"
    else:
        risk_level = _level(risk_score, level_thresholds)
        risk_view = str(camera_order[risk_index[0]])
        risk_region = _region(
            risk_index[1], risk_index[2], height, width
        )
        risk_bearing = (
            "rearward" if risk_view in REAR_VIEWS else "forward_or_crossing"
        )
        if risk_score >= float(stance_thresholds["prepare_to_yield"]):
            stance = (
                str(rearward_high_risk_stance_cap)
                if risk_view in REAR_VIEWS
                else "prepare_to_yield"
            )
        elif risk_score >= float(stance_thresholds["caution"]):
            stance = "caution"
        else:
            stance = "maintain"

    return {
        "observation_uncertainty": {
            "level": _level(uq_score, level_thresholds),
            "peak_score": uq_score,
            "peak_view": observation_view,
            "peak_region": observation_region,
            "temporal_trend": temporal_trend,
            "temporal_peak_region_delta": temporal_delta,
            "temporal_summary_scope": "latest_peak_patch_across_time",
        },
        "relevance_at_most_uncertain_region": {
            "level": relevance_level,
            "score": relevance_score,
        },
        "task_risk": {
            "level": risk_level,
            "peak_score": risk_score,
            "peak_view": risk_view,
            "peak_region": risk_region,
        },
        "planning_implication": {
            "stance": stance,
            "risk_bearing": risk_bearing,
            "is_direct_control_command": False,
        },
    }


def _compare(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            errors.append("%s keys/type differ" % path)
            return
        for key in sorted(expected):
            _compare(expected[key], actual[key], path + "." + str(key), errors)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            errors.append("%s differs: %r != %r" % (path, expected, actual))
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (
            math.isfinite(float(expected))
            and math.isfinite(float(actual))
            and math.isclose(
                float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6
            )
        ):
            errors.append("%s differs: %r != %r" % (path, expected, actual))
        return
    if expected != actual:
        errors.append("%s differs: %r != %r" % (path, expected, actual))


def audit(
    *, manifest_path: Path, qa_config_path: Path, max_errors: int = 50
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    config = _load_json(qa_config_path)
    records_ref = manifest["records"]
    records_path = _resolve(records_ref["path"], manifest_path.parent)
    if sha256_file(records_path) != records_ref["sha256"]:
        raise ValueError("dataset records are absent or stale")
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest["record_count"]):
        raise ValueError("manifest and record counts differ")

    grouped: Dict[Tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["counterfactual"]["group_id"]),
            str(row["counterfactual"]["variant"]),
        )
        grouped.setdefault(key, []).append(row)
    errors: list[str] = []
    reconstructed_count = 0
    component_scalar_checks = 0
    sidecar_product_checks = 0
    family_sets = set()
    variant_sets: Dict[str, set[str]] = {}
    for (group_id, variant), group_rows in sorted(grouped.items()):
        families = {str(row["question_family"]) for row in group_rows}
        family_sets.add(tuple(sorted(families)))
        variant_sets.setdefault(group_id, set()).add(variant)
        if families != EXPECTED_FAMILIES or len(group_rows) != len(EXPECTED_FAMILIES):
            errors.append("%s/%s question-family group is incomplete" % (group_id, variant))
            continue
        event_id = str(group_rows[0]["event_id"])
        target_summaries = [row["target"]["structured_summary"] for row in group_rows]
        if any(value != target_summaries[0] for value in target_summaries[1:]):
            errors.append("%s/%s changes summary across QA families" % (group_id, variant))
            continue
        row = group_rows[0]
        uq_ref = row["model_input"]["stage1_observation_uq"]
        uq_path = _resolve(uq_ref["path"], records_path.parent)
        sidecar_ref = row["target"]["map_sidecar"]
        sidecar_path = _resolve(sidecar_ref["path"], records_path.parent)
        if sha256_file(uq_path) != uq_ref["sha256"]:
            errors.append("%s/%s U artifact hash differs" % (group_id, variant))
            continue
        if sha256_file(sidecar_path) != sidecar_ref["sha256"]:
            errors.append("%s/%s sidecar hash differs" % (group_id, variant))
            continue
        with np.load(uq_path, allow_pickle=False) as archive:
            uncertainty = np.asarray(archive["uncertainty"], dtype=np.float32)
            components = np.asarray(
                archive[uq_ref["component_key"]], dtype=np.float32
            )
        if not np.allclose(uncertainty, components.mean(axis=-1), atol=1e-5):
            errors.append("%s/%s component mean does not equal scalar U" % (group_id, variant))
            continue
        component_scalar_checks += 1
        with np.load(sidecar_path, allow_pickle=False) as archive:
            relevance = np.asarray(
                archive[sidecar_ref["relevance_key"]], dtype=np.float32
            )
            stored_task_risk = np.asarray(
                archive[sidecar_ref["task_risk_key"]], dtype=np.float32
            )
        if not np.allclose(stored_task_risk, uncertainty[-1] * relevance, atol=1e-6):
            errors.append("%s/%s sidecar violates K=U*R" % (group_id, variant))
            continue
        sidecar_product_checks += 1
        camera_order = sidecar_ref["metadata"]["camera_order"]
        reconstructed = reconstruct_summary(
            uncertainty,
            relevance,
            camera_order=camera_order,
            level_thresholds=config["level_thresholds"],
            stance_thresholds=config["planning_stance_thresholds"],
            rearward_high_risk_stance_cap=config[
                "rearward_high_risk_stance_cap"
            ],
        )
        current_errors: list[str] = []
        _compare(
            target_summaries[0],
            reconstructed,
            "%s/%s" % (event_id, group_id + "/" + variant),
            current_errors,
        )
        errors.extend(current_errors)
        reconstructed_count += int(not current_errors)
        if len(errors) >= max_errors:
            break

    complete_variant_groups = all(
        variants == EXPECTED_VARIANTS for variants in variant_sets.values()
    )
    checks = {
        "record_count_matches_manifest": len(rows) == int(manifest["record_count"]),
        "every_group_has_all_five_counterfactual_variants": complete_variant_groups,
        "every_variant_has_all_four_qa_families": family_sets
        == {tuple(sorted(EXPECTED_FAMILIES))},
        "scalar_u_equals_mean_normalized_components": (
            component_scalar_checks == len(grouped)
        ),
        "stored_task_risk_equals_latest_u_times_r": (
            sidecar_product_checks == len(grouped)
        ),
        "every_structured_summary_reconstructed_exactly": (
            reconstructed_count == len(grouped) and not errors
        ),
    }
    passed = all(checks.values()) and not errors
    return {
        "schema": SCHEMA,
        "status": (
            "structured_targets_are_deterministic_from_u_r_k"
            if passed
            else "structured_target_determinism_audit_failed"
        ),
        "passed": passed,
        "record_count": len(rows),
        "matched_group_count": len(variant_sets),
        "group_variant_count": len(grouped),
        "reconstructed_summary_count": reconstructed_count,
        "checks": checks,
        "errors": errors[:max_errors],
        "architecture_evidence": {
            "stage1_u_remains_task_agnostic": True,
            "task_relevance_r_remains_vlm_owned": True,
            "task_risk_is_fixed_k_equals_u_times_r": True,
            "risk_view_region_and_level_are_deterministic_from_k": passed,
            "relevance_level_is_deterministic_from_u_and_r": passed,
            "stance_is_deterministic_from_k_thresholds_and_view": passed,
            "learned_structured_field_classifier_required_to_define_targets": False,
            "language_understanding_still_requires_separate_evaluation": True,
        },
        "sources": {
            "dataset_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "records": {
                "path": str(records_path),
                "sha256": sha256_file(records_path),
            },
            "qa_config": {
                "path": str(qa_config_path.resolve()),
                "sha256": sha256_file(qa_config_path),
            },
        },
        "claim_boundary": (
            "This audit establishes target determinism and architectural "
            "redundancy only. It does not establish learned-R accuracy, QA "
            "understanding, planning, closed-loop behavior, generalization, or safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--qa-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite determinism audit")
    result = audit(
        manifest_path=args.dataset_manifest.resolve(),
        qa_config_path=args.qa_config.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
