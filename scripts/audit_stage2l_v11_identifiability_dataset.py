#!/usr/bin/env python3
"""Audit matched Stage2-L records before an identifiable U/R bridge run.

The audit separates metadata evidence from tensor evidence.  A metadata-only
run is useful for finding stale v1 route contexts, but it never marks a dataset
v11-ready.  Full readiness requires loading and hashing every U tensor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "orion.stage2l_v11_identifiability_dataset_audit.v1"
REQUIRED_VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)
FORBIDDEN_ROUTE_KEYS = frozenset(
    (
        "ttc",
        "collision",
        "collision_outcome",
        "desired_speed",
        "route_progress",
        "scenario_family",
        "corruption_family",
        "severity",
        "ground_truth_stance",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("records JSONL is empty or malformed")
    return rows


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(values: Iterable[Any]) -> List[Any]:
    by_encoding = {_canonical(value): value for value in values}
    return [by_encoding[key] for key in sorted(by_encoding)]


def _resolve(path_value: str, records_path: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (records_path.parent / path).resolve()


def _one_variant_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    model_inputs = _unique(row.get("model_input") for row in rows)
    supports = _unique(row.get("counterfactual") for row in rows)
    relevance = _unique(
        row.get("provenance", {}).get("relevance_supervision") for row in rows
    )
    if len(model_inputs) != 1 or len(supports) != 1 or len(relevance) != 1:
        raise ValueError("question families change a variant's bound inputs")
    return rows[0]


def _route_context_check(route_context: Mapping[str, Any]) -> Tuple[bool, str]:
    if route_context.get("schema") != "orion.route_context.v2":
        return False, "route_context is not version v2"
    payload = route_context.get("payload")
    if not isinstance(payload, Mapping):
        return False, "route_context payload is not an object"
    forbidden = sorted(FORBIDDEN_ROUTE_KEYS.intersection(payload))
    if forbidden:
        return False, "route_context contains forbidden keys: %s" % forbidden
    ego = payload.get("ego_state")
    if not isinstance(ego, Mapping) or set(ego) != {"speedometer_mps"}:
        return False, "route_context ego_state must contain only speedometer_mps"
    try:
        speed = float(ego["speedometer_mps"])
    except (TypeError, ValueError):
        return False, "route_context ego speed is not numeric"
    if not math.isfinite(speed):
        return False, "route_context ego speed is invalid"
    encoded = _canonical(payload).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != route_context.get("sha256"):
        return False, "route_context payload hash differs"
    return True, ""


def _matched_support_check(
    on: Mapping[str, Any], off: Mapping[str, Any]
) -> Tuple[bool, str]:
    support_type = on.get("support_type")
    if support_type != off.get("support_type"):
        return False, "on/off support types differ"
    if support_type == "matched_local_gaussian_region_v1":
        required_equal = (
            "latest_nonzero_patches",
            "radius_patches",
            "matched_on_path_center_view_y_x",
            "matched_off_path_center_view_y_x",
        )
        numeric_equal = ("latest_peak", "latest_spatial_sum")
    elif support_type == "matched_token_grid_gaussian_region_v2":
        required_equal = (
            "latest_nonzero_patches",
            "consumer_latest_nonzero_cells",
            "radius_cells",
            "consumer_grid_hw",
            "stored_grid_hw",
            "matched_on_path_center_view_y_x",
            "matched_off_path_center_view_y_x",
        )
        numeric_equal = (
            "latest_peak",
            "latest_spatial_sum",
            "consumer_latest_peak",
            "consumer_latest_spatial_sum",
        )
        if on.get("consumer_grid_hw") != [10, 10] or on.get("stored_grid_hw") != [
            40,
            40,
        ]:
            return False, "token-grid support dimensions differ from v11.1"
        if on.get("construction") != "consumer_grid_then_exact_block_expand":
            return False, "token-grid support construction is not exact"
    else:
        return False, "unsupported on/off support type"
    for key in required_equal:
        if on.get(key) != off.get(key):
            return False, "on/off support metadata differs at %s" % key
    for key in numeric_equal:
        try:
            matched = math.isclose(
                float(on[key]), float(off[key]), rel_tol=1e-5, abs_tol=1e-6
            )
        except (KeyError, TypeError, ValueError):
            matched = False
        if not matched:
            return False, "on/off support metadata differs at %s" % key
    if (
        on.get("same_view_matched_pair") is not True
        or off.get("same_view_matched_pair") is not True
    ):
        return False, "on/off support is not attested as a same-view pair"
    if on.get("center_view_y_x") == off.get("center_view_y_x"):
        return False, "on/off support centers are identical"
    if float(on.get("support_weighted_relevance", 0.0)) <= float(
        off.get("support_weighted_relevance", 0.0)
    ):
        return False, "on-path support is not more relevant than off-path support"
    return True, ""


def _load_u_tensor(
    row: Mapping[str, Any], records_path: Path
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    reference = row["model_input"]["stage1_observation_uq"]
    path = _resolve(reference["path"], records_path)
    if not path.is_file():
        raise FileNotFoundError("U tensor is unavailable: %s" % path)
    if _sha256(path) != reference.get("sha256"):
        raise ValueError("U tensor SHA-256 differs: %s" % path)
    with np.load(path, allow_pickle=False) as archive:
        uq = np.asarray(archive["uncertainty"], dtype=np.float32)
        component_key = reference.get("component_key")
        components = (
            np.asarray(archive[component_key], dtype=np.float32)
            if component_key is not None
            else None
        )
    if tuple(reference.get("shape", [])) != uq.shape:
        raise ValueError("U tensor declared shape differs")
    if not np.all(np.isfinite(uq)) or np.any(uq < 0.0) or np.any(uq > 1.0):
        raise ValueError("U tensor is not finite in [0,1]")
    if components is not None:
        if tuple(reference.get("component_shape", [])) != components.shape:
            raise ValueError("U component declared shape differs")
        if not np.allclose(uq, components.mean(axis=-1), atol=1e-5):
            raise ValueError("scalar U is not the component mean")
    return uq, components


def _tensor_group_check(
    selected: Mapping[str, Mapping[str, Any]], records_path: Path
) -> Tuple[bool, List[str]]:
    errors = []
    tensors = {}
    components = {}
    for variant in REQUIRED_VARIANTS:
        try:
            tensors[variant], components[variant] = _load_u_tensor(
                selected[variant], records_path
            )
        except Exception as error:
            errors.append("%s: %s" % (variant, error))
    if errors:
        return False, errors
    zero = tensors["zero_uq"]
    on = tensors["on_path_uq"][-1]
    off = tensors["off_path_uq"][-1]
    if np.any(zero != 0.0):
        errors.append("zero_uq tensor is not exactly zero")
    for name, left, right in (
        ("mass", float(on.sum()), float(off.sum())),
        ("peak", float(on.max()), float(off.max())),
        ("support count", int(np.count_nonzero(on)), int(np.count_nonzero(off))),
    ):
        if not np.isclose(left, right, rtol=1e-5, atol=1e-6):
            errors.append("on/off tensor %s differs" % name)
    if np.array_equal(on, off):
        errors.append("on/off tensor support is not spatially distinct")
    support_type = selected["on_path_uq"]["counterfactual"]["spatial_support"].get(
        "support_type"
    )
    if support_type == "matched_token_grid_gaussian_region_v2":
        if on.shape[-2:] != (40, 40):
            errors.append("v11.1 stored U grid is not 40x40")
        else:
            on_consumer = on.reshape(6, 10, 4, 10, 4).mean(axis=(2, 4))
            off_consumer = off.reshape(6, 10, 4, 10, 4).mean(axis=(2, 4))
            for name, left, right in (
                ("consumer mass", float(on_consumer.sum()), float(off_consumer.sum())),
                ("consumer peak", float(on_consumer.max()), float(off_consumer.max())),
                (
                    "consumer support count",
                    int(np.count_nonzero(on_consumer)),
                    int(np.count_nonzero(off_consumer)),
                ),
            ):
                if not np.isclose(left, right, rtol=1e-5, atol=1e-6):
                    errors.append("on/off tensor %s differs" % name)
            if np.array_equal(on_consumer, off_consumer):
                errors.append("on/off consumer support is not spatially distinct")
    return not errors, errors


def audit_dataset(
    records_path: Path,
    *,
    verify_tensors: bool,
) -> Dict[str, Any]:
    rows = _rows(records_path)
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        counterfactual = row.get("counterfactual") or {}
        grouped[str(counterfactual.get("group_id", ""))][
            str(counterfactual.get("variant", ""))
        ].append(row)
    group_results = []
    for group_id in sorted(grouped):
        variants = grouped[group_id]
        errors = []
        missing = sorted(set(REQUIRED_VARIANTS) - set(variants))
        if missing:
            errors.append("missing variants: %s" % missing)
            group_results.append(
                {"group_id": group_id, "metadata_passed": False, "errors": errors}
            )
            continue
        try:
            selected = {
                variant: _one_variant_row(variants[variant])
                for variant in REQUIRED_VARIANTS
            }
        except Exception as error:
            group_results.append(
                {
                    "group_id": group_id,
                    "metadata_passed": False,
                    "errors": [str(error)],
                }
            )
            continue
        observations = {
            row["model_input"]["observation"]["observation_sha256"]
            for row in selected.values()
        }
        if len(observations) != 1:
            errors.append("matched variants change visual observation")
        route_contexts = _unique(
            row["model_input"]["route_context"] for row in selected.values()
        )
        if len(route_contexts) != 1:
            errors.append("matched variants change route/ego context")
        else:
            route_ok, route_error = _route_context_check(route_contexts[0])
            if not route_ok:
                errors.append(route_error)
        relevance = {
            row["provenance"]["relevance_supervision"]["sha256"]
            for row in selected.values()
        }
        if len(relevance) != 1:
            errors.append("matched variants change R supervision")
        checkpoints = {
            row["model_input"]["stage1_observation_uq"]["checkpoint_sha256"]
            for row in selected.values()
        }
        if len(checkpoints) != 1:
            errors.append("matched variants change Stage1 checkpoint")
        splits = {str(row.get("split")) for row in selected.values()}
        events = {str(row.get("event_id")) for row in selected.values()}
        if len(splits) != 1 or len(events) != 1:
            errors.append("matched variants cross split or event")
        support_ok, support_error = _matched_support_check(
            selected["on_path_uq"]["counterfactual"]["spatial_support"],
            selected["off_path_uq"]["counterfactual"]["spatial_support"],
        )
        if not support_ok:
            errors.append(support_error)
        tensor_passed = None
        tensor_errors: List[str] = []
        if verify_tensors:
            tensor_passed, tensor_errors = _tensor_group_check(selected, records_path)
        group_results.append(
            {
                "group_id": group_id,
                "event_id": next(iter(events)) if len(events) == 1 else None,
                "split": next(iter(splits)) if len(splits) == 1 else None,
                "metadata_passed": not errors,
                "tensor_verified": bool(verify_tensors),
                "tensor_passed": tensor_passed,
                "errors": errors + tensor_errors,
            }
        )

    metadata_passed = all(row["metadata_passed"] for row in group_results)
    tensor_passed = (
        all(row["tensor_passed"] is True for row in group_results)
        if verify_tensors
        else None
    )
    failed = [row for row in group_results if row["errors"]]
    return {
        "schema": SCHEMA,
        "records_path": str(records_path.resolve()),
        "records_sha256": _sha256(records_path),
        "record_count": len(rows),
        "group_count": len(group_results),
        "metadata_passed": metadata_passed,
        "tensor_verification_requested": bool(verify_tensors),
        "tensor_passed": tensor_passed,
        "v11_ready": bool(metadata_passed and tensor_passed is True),
        "failed_group_count": len(failed),
        "failed_groups": failed,
        "claim_boundary": (
            "Dataset identifiability preflight only. It does not establish "
            "learned-U validity, VLM semantics, planning or closed-loop safety."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-tensors", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit_dataset(
        args.records.resolve(), verify_tensors=bool(args.verify_tensors)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "metadata_passed": report["metadata_passed"],
                "tensor_passed": report["tensor_passed"],
                "v11_ready": report["v11_ready"],
                "failed_group_count": report["failed_group_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 1 if args.require_pass and not report["v11_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
