#!/usr/bin/env python3
"""Audit matched Stage2-L U controls at the grid consumed by the model.

The frame-bundle audit operates on the stored Stage-1 grid.  The active
U-tokenizer first average-pools every component map before deriving scalar U.
Matched controls must therefore remain matched *after* that deterministic
transformation.  This audit reports both levels and fails closed when a raw
match is destroyed by pooling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "orion.stage2l_v11_consumer_grid_audit.v1"
VARIANTS = ("on_path_uq", "off_path_uq")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_value: str, records_path: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (records_path.parent / path).resolve()


def _rows(path: Path) -> list[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("records JSONL is empty")
    return rows


def _selected_rows(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    selected: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        counterfactual = row.get("counterfactual") or {}
        variant = str(counterfactual.get("variant", ""))
        if variant not in VARIANTS or row.get("question_family") != "task_relevance":
            continue
        group_id = str(counterfactual.get("group_id", ""))
        key = (group_id, variant)
        if not group_id or key in selected:
            raise ValueError("duplicate or empty group/variant task-relevance row")
        selected[key] = row
    groups = sorted({group_id for group_id, _ in selected})
    if not groups or any((group_id, variant) not in selected for group_id in groups for variant in VARIANTS):
        raise ValueError("records do not contain complete on/off task-relevance pairs")
    return selected


def _load_latest_scalar(
    row: Mapping[str, Any], records_path: Path
) -> np.ndarray:
    reference = row["model_input"]["stage1_observation_uq"]
    path = _resolve(reference["path"], records_path)
    if not path.is_file() or _sha256(path) != reference.get("sha256"):
        raise ValueError("component tensor is missing or hash-stale: %s" % path)
    with np.load(path, allow_pickle=False) as archive:
        components = np.asarray(
            archive[reference["component_key"]], dtype=np.float32
        )
    if components.ndim != 5 or components.shape[-1] <= 0:
        raise ValueError("components must have shape [T,V,H,W,C]")
    if not np.all(np.isfinite(components)) or np.any(components < 0.0) or np.any(components > 1.0):
        raise ValueError("components are not finite in [0,1]")
    return components[-1].mean(axis=-1)


def _area_pool(value: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
    if value.ndim != 3:
        raise ValueError("scalar U must have shape [V,H,W]")
    views, height, width = value.shape
    grid_h, grid_w = map(int, grid_hw)
    if min(grid_h, grid_w) <= 0 or height % grid_h or width % grid_w:
        raise ValueError("consumer grid must evenly divide the stored U grid")
    block_h, block_w = height // grid_h, width // grid_w
    return value.reshape(
        views, grid_h, block_h, grid_w, block_w
    ).mean(axis=(2, 4))


def _metrics(value: np.ndarray) -> Dict[str, Any]:
    return {
        "mass": float(value.sum()),
        "peak": float(value.max()),
        "support_count": int(np.count_nonzero(value)),
    }


def _matched(
    on: Mapping[str, Any], off: Mapping[str, Any], *, rtol: float, atol: float
) -> bool:
    return bool(
        np.isclose(on["mass"], off["mass"], rtol=rtol, atol=atol)
        and np.isclose(on["peak"], off["peak"], rtol=rtol, atol=atol)
        and on["support_count"] == off["support_count"]
    )


def _summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def audit_consumer_grid(
    records_path: Path,
    *,
    grid_hw: Tuple[int, int] = (10, 10),
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    records_path = records_path.resolve()
    selected = _selected_rows(_rows(records_path))
    groups = sorted({group_id for group_id, _ in selected})
    per_group = []
    raw_pass = 0
    consumer_pass = 0
    by_split: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"groups": 0, "raw_matches": 0, "consumer_matches": 0}
    )
    peak_relative_differences = []
    mass_relative_differences = []
    support_count_differences = []
    for group_id in groups:
        on_raw = _load_latest_scalar(selected[(group_id, "on_path_uq")], records_path)
        off_raw = _load_latest_scalar(selected[(group_id, "off_path_uq")], records_path)
        if on_raw.shape != off_raw.shape:
            raise ValueError("on/off stored U shapes differ")
        on_consumer = _area_pool(on_raw, grid_hw)
        off_consumer = _area_pool(off_raw, grid_hw)
        raw = {"on_path_uq": _metrics(on_raw), "off_path_uq": _metrics(off_raw)}
        consumer = {
            "on_path_uq": _metrics(on_consumer),
            "off_path_uq": _metrics(off_consumer),
        }
        raw_matched = _matched(raw["on_path_uq"], raw["off_path_uq"], rtol=rtol, atol=atol)
        consumer_matched = _matched(
            consumer["on_path_uq"], consumer["off_path_uq"], rtol=rtol, atol=atol
        )
        split = str(selected[(group_id, "on_path_uq")].get("split", ""))
        raw_pass += int(raw_matched)
        consumer_pass += int(consumer_matched)
        by_split[split]["groups"] += 1
        by_split[split]["raw_matches"] += int(raw_matched)
        by_split[split]["consumer_matches"] += int(consumer_matched)
        on_peak = consumer["on_path_uq"]["peak"]
        off_peak = consumer["off_path_uq"]["peak"]
        on_mass = consumer["on_path_uq"]["mass"]
        off_mass = consumer["off_path_uq"]["mass"]
        peak_relative = abs(on_peak - off_peak) / max(on_peak, off_peak, 1e-12)
        mass_relative = abs(on_mass - off_mass) / max(on_mass, off_mass, 1e-12)
        support_difference = abs(
            consumer["on_path_uq"]["support_count"]
            - consumer["off_path_uq"]["support_count"]
        )
        peak_relative_differences.append(float(peak_relative))
        mass_relative_differences.append(float(mass_relative))
        support_count_differences.append(float(support_difference))
        per_group.append(
            {
                "group_id": group_id,
                "split": split,
                "raw": raw,
                "consumer": consumer,
                "raw_matched": raw_matched,
                "consumer_matched": consumer_matched,
                "consumer_peak_relative_difference": float(peak_relative),
                "consumer_mass_relative_difference": float(mass_relative),
                "consumer_support_count_absolute_difference": int(support_difference),
            }
        )
    group_count = len(groups)
    return {
        "schema": SCHEMA,
        "status": (
            "consumer_grid_match_passed"
            if consumer_pass == group_count
            else "raw_match_destroyed_before_model_consumer"
        ),
        "records_path": str(records_path),
        "records_sha256": _sha256(records_path),
        "consumer_grid_hw": list(map(int, grid_hw)),
        "group_count": group_count,
        "raw_full_match_count": raw_pass,
        "consumer_full_match_count": consumer_pass,
        "raw_match_fraction": raw_pass / group_count,
        "consumer_match_fraction": consumer_pass / group_count,
        "by_split": dict(sorted(by_split.items())),
        "consumer_peak_relative_difference": _summary(peak_relative_differences),
        "consumer_mass_relative_difference": _summary(mass_relative_differences),
        "consumer_support_count_absolute_difference": _summary(support_count_differences),
        "per_group": per_group,
        "claim_boundary": (
            "Deterministic input-transformation audit only. It does not evaluate "
            "R, language learning, learned Stage1 U, planning or safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--grid-width", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite consumer-grid audit")
    result = audit_consumer_grid(
        args.records, grid_hw=(args.grid_height, args.grid_width)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "groups": result["group_count"],
                "raw_matches": result["raw_full_match_count"],
                "consumer_matches": result["consumer_full_match_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
