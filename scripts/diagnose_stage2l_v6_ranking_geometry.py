#!/usr/bin/env python3
"""Diagnose whether frozen U/R geometry can satisfy the Stage2-L risk margin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


VARIANTS = ("on_path_uq", "off_path_uq")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is absent: %s" % (name, path))
    if _sha256(path) != reference.get("sha256"):
        raise ValueError("%s has a SHA-256 mismatch" % name)
    return path


def _pool_40_to_10(value: np.ndarray) -> np.ndarray:
    if value.shape[-2:] != (40, 40):
        raise ValueError("diagnostic expects a 40x40 spatial grid")
    leading = value.shape[:-2]
    reshaped = value.reshape(*leading, 10, 4, 10, 4)
    return reshaped.mean(axis=(-3, -1))


def _peak(value: np.ndarray) -> Dict[str, Any]:
    flat = int(np.argmax(value))
    view, y, x = np.unravel_index(flat, value.shape)
    return {"value": float(value[view, y, x]), "view": int(view), "y": int(y), "x": int(x)}


def diagnose(records_path: Path, margin: float = 0.2) -> Dict[str, Any]:
    if margin < 0.0:
        raise ValueError("ranking margin must be non-negative")
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {}
    for row in rows:
        if row.get("question_family") != "task_relevance":
            continue
        variant = str(row["counterfactual"]["variant"])
        if variant not in VARIANTS:
            continue
        group = str(row["counterfactual"]["group_id"])
        key = (group, variant)
        if key in selected:
            raise ValueError("duplicate task-relevance row: %s/%s" % key)
        selected[key] = row
    groups = sorted({group for group, _ in selected})
    if not groups or set(selected) != {
        (group, variant) for group in groups for variant in VARIANTS
    }:
        raise ValueError("records do not contain complete on/off task-relevance pairs")

    diagnostics = []
    for group in groups:
        relevance = None
        uncertainty = {}
        for variant in VARIANTS:
            row = selected[(group, variant)]
            uq_ref = row["model_input"]["stage1_observation_uq"]
            uq_path = _resolve(
                uq_ref, records_path.parent, "%s/%s UQ" % (group, variant)
            )
            with np.load(uq_path) as archive:
                components = archive[str(uq_ref["component_key"])].astype(
                    np.float64
                )
            if components.shape != (4, 6, 40, 40, 3):
                raise ValueError("unexpected Stage1 component shape")
            uncertainty[variant] = _pool_40_to_10(
                components[-1].mean(axis=-1)
            )

            sidecar_ref = row["target"]["map_sidecar"]
            sidecar_path = _resolve(
                sidecar_ref,
                records_path.parent,
                "%s/%s R sidecar" % (group, variant),
            )
            with np.load(sidecar_path) as archive:
                current = archive[str(sidecar_ref["relevance_key"])].astype(
                    np.float64
                )
            current = _pool_40_to_10(current)
            if current.shape != (6, 10, 10):
                raise ValueError("unexpected pooled R shape")
            if relevance is None:
                relevance = current
            elif not np.array_equal(relevance, current):
                raise ValueError("matched on/off variants change the R target")

        risks = {
            variant: uncertainty[variant] * relevance for variant in VARIANTS
        }
        on_peak = float(risks["on_path_uq"].max())
        off_peak = float(risks["off_path_uq"].max())
        difference = on_peak - off_peak
        diagnostics.append(
            {
                "group_id": group,
                "target_r": {
                    "max": float(relevance.max()),
                    "mean": float(relevance.mean()),
                    "peak": _peak(relevance),
                },
                "on_path_uq": {
                    "max": float(uncertainty["on_path_uq"].max()),
                    "mean": float(uncertainty["on_path_uq"].mean()),
                    "peak": _peak(uncertainty["on_path_uq"]),
                    "target_risk_peak": _peak(risks["on_path_uq"]),
                    "target_risk_mean": float(risks["on_path_uq"].mean()),
                },
                "off_path_uq": {
                    "max": float(uncertainty["off_path_uq"].max()),
                    "mean": float(uncertainty["off_path_uq"].mean()),
                    "peak": _peak(uncertainty["off_path_uq"]),
                    "target_risk_peak": _peak(risks["off_path_uq"]),
                    "target_risk_mean": float(risks["off_path_uq"].mean()),
                },
                "on_minus_off_target_risk_peak": difference,
                "ranking_loss_under_target_r": float(max(0.0, margin - difference)),
                "margin_feasible_under_target_r": bool(difference >= margin),
            }
        )
    differences = [row["on_minus_off_target_risk_peak"] for row in diagnostics]
    losses = [row["ranking_loss_under_target_r"] for row in diagnostics]
    return {
        "schema": "orion.stage2l_v6_ranking_geometry_diagnostic.v1",
        "records": {"path": str(records_path.resolve()), "sha256": _sha256(records_path)},
        "margin": float(margin),
        "groups": diagnostics,
        "summary": {
            "group_count": len(diagnostics),
            "mean_on_minus_off_target_risk_peak": float(np.mean(differences)),
            "minimum_on_minus_off_target_risk_peak": float(np.min(differences)),
            "maximum_on_minus_off_target_risk_peak": float(np.max(differences)),
            "mean_ranking_loss_under_target_r": float(np.mean(losses)),
            "margin_feasible_group_count": int(
                sum(row["margin_feasible_under_target_r"] for row in diagnostics)
            ),
        },
        "interpretation_boundary": "Uses frozen U and target R, not learned R logits. It diagnoses label/objective compatibility and is not a model-performance result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = diagnose(args.records.resolve(), margin=args.margin)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite ranking diagnostic")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
