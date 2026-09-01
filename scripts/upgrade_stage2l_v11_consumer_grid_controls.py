#!/usr/bin/env python3
"""Build v11.1 matched-U controls on the tokenizer's consumed 10x10 grid.

The v11 controls were matched on their stored 40x40 grid but ceased to be
matched after the frozen U-tokenizer's 40x40 -> 10x10 area pooling.  This
CPU-only upgrade replaces only the on/off-path controlled U tensors and their
derived sidecars/QA summaries.  Observation, route/ego input, Stage-1 lineage,
relevance supervision, split and event identity remain unchanged.

The source dataset is never edited.  Output is built in a private sibling
directory and atomically renamed only after the v5 QA audit passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uq_relevance_qa_factory_lib import summarize_maps  # noqa: E402
from scripts.upgrade_stage2l_v9_qa_records import (  # noqa: E402
    TASK_RELEVANCE_FIELDS,
    audit_records as audit_v5_records,
    expected_semantic_fields,
    expected_task_field_targets,
    normalize_structured_summary,
    render_structured_answer,
)


SCHEMA = "orion.stage2l_v11_consumer_grid_control_upgrade.v1"
SUPPORT_SCHEMA = "matched_token_grid_gaussian_region_v2"
CONTROL_SOURCE = "controlled_stage1_uq_counterfactual_consumer_grid_v2"
CONTROL_VARIANTS = ("on_path_uq", "off_path_uq")
EXPECTED_VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)
CONSUMER_HW = (10, 10)
STORED_HW = (40, 40)
BLOCK_HW = (4, 4)
RADIUS_CELLS = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("source records are empty or malformed")
    return rows


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, List[Mapping[str, Any]]]]:
    groups: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for row in rows:
        group_id = str(row["counterfactual"]["group_id"])
        variant = str(row["counterfactual"]["variant"])
        groups.setdefault(group_id, {}).setdefault(variant, []).append(row)
    for group_id, variants in groups.items():
        if set(variants) != set(EXPECTED_VARIANTS):
            raise ValueError(
                "group %s does not contain exactly five variants" % group_id
            )
        for variant, variant_rows in variants.items():
            if len(variant_rows) != 4:
                raise ValueError(
                    "group %s variant %s does not contain four QA families"
                    % (group_id, variant)
                )
    return groups


def _load_npz_reference(
    reference: Mapping[str, Any], key: str
) -> Tuple[np.ndarray, Path]:
    path = Path(str(reference.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError("referenced tensor is missing: %s" % path)
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("referenced tensor SHA-256 differs: %s" % path)
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive[key], dtype=np.float32)
    return value, path


def pool_40_to_10(value: np.ndarray) -> np.ndarray:
    """Exact area average used by the frozen U tokenizer for a [6,40,40] map."""

    if value.shape != (6, *STORED_HW):
        raise ValueError("consumer pooling requires shape [6,40,40]")
    return value.reshape(6, 10, 4, 10, 4).mean(axis=(2, 4))


def _gaussian_template(radius: int = RADIUS_CELLS) -> np.ndarray:
    size = 2 * radius + 1
    template = np.zeros((size, size), dtype=np.float32)
    sigma = max(1.0, radius / 1.5)
    for y in range(size):
        for x in range(size):
            dy = y - radius
            dx = x - radius
            distance_squared = dy * dy + dx * dx
            if distance_squared <= radius * radius:
                template[y, x] = math.exp(-distance_squared / (2.0 * sigma * sigma))
    template /= float(template.max())
    return template


def _candidate_score(
    relevance: np.ndarray,
    center: Tuple[int, int, int],
    template: np.ndarray,
) -> float:
    view, y, x = center
    radius = template.shape[0] // 2
    patch = relevance[
        view,
        y - radius : y + radius + 1,
        x - radius : x + radius + 1,
    ]
    return float(np.sum(patch * template) / np.sum(template))


def select_consumer_grid_centers(
    relevance_10: np.ndarray,
    *,
    radius: int = RADIUS_CELLS,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], float, float]:
    """Select deterministic same-view, disjoint high/low-relevance centers."""

    if relevance_10.shape != (6, *CONSUMER_HW):
        raise ValueError("consumer relevance must have shape [6,10,10]")
    if not np.all(np.isfinite(relevance_10)):
        raise ValueError("consumer relevance is not finite")
    template = _gaussian_template(radius)
    candidates: List[Tuple[float, int, int, int]] = []
    for view in range(relevance_10.shape[0]):
        for y in range(radius, relevance_10.shape[1] - radius):
            for x in range(radius, relevance_10.shape[2] - radius):
                center = (view, y, x)
                candidates.append(
                    (_candidate_score(relevance_10, center, template), view, y, x)
                )
    if not candidates:
        raise ValueError("no interior consumer-grid support exists")
    # score descending, then lexicographic view/y/x for deterministic ties.
    on_item = min(candidates, key=lambda item: (-item[0], item[1:]))
    on_center = (on_item[1], on_item[2], on_item[3])
    off_candidates = [
        item
        for item in candidates
        if item[1] == on_center[0]
        and math.hypot(item[2] - on_center[1], item[3] - on_center[2]) >= 2 * radius + 1
    ]
    if not off_candidates:
        raise ValueError("no disjoint same-view off-path support exists")
    off_item = min(off_candidates, key=lambda item: (item[0], item[1:]))
    off_center = (off_item[1], off_item[2], off_item[3])
    if not on_item[0] > off_item[0] + 1e-8:
        raise ValueError("consumer-grid relevance does not distinguish on/off support")
    return on_center, off_center, float(on_item[0]), float(off_item[0])


def _support_grid(
    center: Tuple[int, int, int], *, radius: int = RADIUS_CELLS
) -> np.ndarray:
    support = np.zeros((6, *CONSUMER_HW), dtype=np.float32)
    view, y, x = center
    template = _gaussian_template(radius)
    support[
        view,
        y - radius : y + radius + 1,
        x - radius : x + radius + 1,
    ] = template
    return support


def build_consumer_grid_pair(
    relevance_40: np.ndarray,
    *,
    time_steps: int,
    component_count: int,
    peak: float,
) -> Dict[str, Dict[str, Any]]:
    """Construct exactly matched on/off tensors on the consumed grid."""

    if relevance_40.shape != (6, *STORED_HW):
        raise ValueError("relevance supervision must have shape [6,40,40]")
    if time_steps < 1 or component_count < 1:
        raise ValueError("time/component dimensions must be positive")
    if not 0.0 < peak <= 1.0:
        raise ValueError("counterfactual peak must lie in (0,1]")
    relevance_10 = pool_40_to_10(relevance_40)
    on_center, off_center, on_score, off_score = select_consumer_grid_centers(
        relevance_10
    )
    temporal = np.linspace(max(0.2, peak * 0.4), peak, time_steps, dtype=np.float32)
    result: Dict[str, Dict[str, Any]] = {}
    for variant, center, weighted_relevance in (
        ("on_path_uq", on_center, on_score),
        ("off_path_uq", off_center, off_score),
    ):
        consumer_support = _support_grid(center)
        stored_support = np.repeat(
            np.repeat(consumer_support, BLOCK_HW[0], axis=1),
            BLOCK_HW[1],
            axis=2,
        )
        uq = temporal[:, None, None, None] * stored_support[None, ...]
        uq = np.asarray(uq, dtype=np.float32)
        components = np.repeat(uq[..., None], component_count, axis=-1)
        latest_consumer = pool_40_to_10(uq[-1])
        support = {
            "support_type": SUPPORT_SCHEMA,
            "center_view_y_x": list(center),
            "matched_on_path_center_view_y_x": list(on_center),
            "matched_off_path_center_view_y_x": list(off_center),
            "radius_cells": RADIUS_CELLS,
            "consumer_grid_hw": list(CONSUMER_HW),
            "stored_grid_hw": list(STORED_HW),
            "latest_peak": float(uq[-1].max()),
            "latest_spatial_sum": float(uq[-1].sum()),
            "latest_nonzero_patches": int(np.count_nonzero(uq[-1])),
            "consumer_latest_peak": float(latest_consumer.max()),
            "consumer_latest_spatial_sum": float(latest_consumer.sum()),
            "consumer_latest_nonzero_cells": int(np.count_nonzero(latest_consumer)),
            "support_weighted_relevance": weighted_relevance,
            "same_view_matched_pair": on_center[0] == off_center[0],
            "construction": "consumer_grid_then_exact_block_expand",
        }
        result[variant] = {
            "uq": uq,
            "components": components,
            "support": support,
            "consumer_uq": np.stack([pool_40_to_10(frame) for frame in uq], axis=0),
            "consumer_relevance": relevance_10,
        }
    on = result["on_path_uq"]
    off = result["off_path_uq"]
    for key in ("latest_peak", "latest_spatial_sum", "latest_nonzero_patches"):
        if not np.isclose(
            float(on["support"][key]),
            float(off["support"][key]),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise AssertionError("stored-grid control mismatch at %s" % key)
    for key in (
        "consumer_latest_peak",
        "consumer_latest_spatial_sum",
        "consumer_latest_nonzero_cells",
    ):
        if not np.isclose(
            float(on["support"][key]),
            float(off["support"][key]),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise AssertionError("consumer-grid control mismatch at %s" % key)
    return result


def _field_targets(
    family: str, variant: str, summary: Mapping[str, Any]
) -> Dict[str, str]:
    task_fields = expected_task_field_targets(summary)
    if family == "task_relevance":
        return {key: task_fields[key] for key in TASK_RELEVANCE_FIELDS}
    if family == "driving_implication" and variant in {
        "zero_uq",
        "on_path_uq",
        "off_path_uq",
    }:
        return {"stance": task_fields["stance"]}
    return {}


def normalize_v5_structured_summary(
    source: Mapping[str, Any], *, relevance_high_threshold: float
) -> Dict[str, Any]:
    """Apply explicit absence rules and the binary v5 relevance vocabulary."""

    summary = normalize_structured_summary(source)
    relevance = summary["relevance_at_most_uncertain_region"]
    if relevance["level"] != "not_applicable":
        relevance["level"] = (
            "high"
            if float(relevance["score"]) >= float(relevance_high_threshold)
            else "low"
        )
    return summary


def _safe_group_name(group_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in group_id
    )
    if not safe:
        raise ValueError("group id cannot be converted to a file name")
    return safe


def _source_control_contract(
    variants: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Tuple[int, int, float]:
    shapes = []
    peaks = []
    for variant in CONTROL_VARIANTS:
        row = variants[variant][0]
        reference = row["model_input"]["stage1_observation_uq"]
        shape = tuple(int(value) for value in reference["shape"])
        component_shape = tuple(int(value) for value in reference["component_shape"])
        if shape != component_shape[:-1] or shape[1:] != (6, *STORED_HW):
            raise ValueError("source controlled U shape differs from v11")
        old_uq, _ = _load_npz_reference(reference, "uncertainty")
        if old_uq.shape != shape:
            raise ValueError("source controlled U declared shape differs")
        shapes.append((shape[0], component_shape[-1]))
        peaks.append(float(row["counterfactual"]["spatial_support"]["latest_peak"]))
    if len(set(shapes)) != 1 or not math.isclose(
        peaks[0], peaks[1], rel_tol=1e-6, abs_tol=1e-7
    ):
        raise ValueError("source on/off temporal, component or peak contract differs")
    return shapes[0][0], shapes[0][1], peaks[0]


def _assert_only_authorized_changes(
    source: Mapping[str, Any], upgraded: Mapping[str, Any]
) -> None:
    """Fail closed if the upgrade changes anything outside its contract."""

    source_copy = copy.deepcopy(source)
    upgraded_copy = copy.deepcopy(upgraded)
    upgraded_copy["provenance"].pop("consumer_grid_control_v11_1_upgrade")
    variant = str(source["counterfactual"]["variant"])
    if variant not in CONTROL_VARIANTS:
        if canonical_sha256(source_copy) != canonical_sha256(upgraded_copy):
            raise AssertionError("non-control record changed outside provenance")
        return
    for value in (source_copy, upgraded_copy):
        value["model_input"].pop("stage1_observation_uq")
        value["counterfactual"].pop("spatial_support")
        for key in (
            "structured_summary",
            "semantic_fields",
            "vlm_task_field_targets",
            "rendered_answer",
            "map_sidecar",
        ):
            value["target"].pop(key)
        value["conversation"][1].pop("value")
    if canonical_sha256(source_copy) != canonical_sha256(upgraded_copy):
        raise AssertionError("control record changed outside authorized fields")


def _write_control_files(
    *,
    group_id: str,
    variant: str,
    bundle: Mapping[str, Any],
    relevance_40: np.ndarray,
    build_dir: Path,
    final_dir: Path,
    sidecar_template: Mapping[str, Any],
) -> Dict[str, Any]:
    safe_group = _safe_group_name(group_id)
    tensor_relative = Path("u_tensors") / safe_group / (variant + ".npz")
    tensor_build = build_dir / tensor_relative
    tensor_build.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tensor_build,
        uncertainty=bundle["uq"],
        uncertainty_components=bundle["components"],
    )
    task_risk_40 = np.asarray(bundle["uq"][-1] * relevance_40, dtype=np.float32)
    sidecar_relative = Path("map_sidecars") / (safe_group + "_" + variant + ".npz")
    sidecar_build = build_dir / sidecar_relative
    sidecar_build.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar_build,
        task_relevance=relevance_40,
        task_risk=task_risk_40,
    )
    sidecar = copy.deepcopy(sidecar_template)
    sidecar.update(
        {
            "path": str((final_dir / sidecar_relative).resolve()),
            "sha256": sha256_file(sidecar_build),
            "relevance_key": "task_relevance",
            "task_risk_key": "task_risk",
        }
    )
    metadata = sidecar.setdefault("metadata", {})
    metadata.update(
        {
            "shape": [6, 40, 40],
            "consumer_grid_hw": [10, 10],
            "summary_semantics_grid": "consumer_10x10",
            "control_construction": "consumer_grid_then_exact_block_expand",
        }
    )
    return {
        "tensor_path": str((final_dir / tensor_relative).resolve()),
        "tensor_sha256": sha256_file(tensor_build),
        "sidecar": sidecar,
    }


def _cleanup_private_build_dir(build_dir: Path) -> None:
    """Best-effort cleanup for short NFS directory-entry visibility delays."""

    for attempt in range(5):
        if not build_dir.exists():
            return
        shutil.rmtree(build_dir, ignore_errors=True)
        if not build_dir.exists():
            return
        time.sleep(0.2 * (attempt + 1))


def upgrade_dataset(
    *,
    source_records: Path,
    output_dir: Path,
    qa_config_path: Path,
) -> Dict[str, Any]:
    """Write one immutable v11.1 dataset and return its upgrade report."""

    source_records = source_records.resolve()
    output_dir = output_dir.resolve()
    qa_config_path = qa_config_path.resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite output directory: %s" % output_dir)
    build_dir = output_dir.parent / ("." + output_dir.name + ".building")
    if build_dir.exists():
        raise FileExistsError("stale private build directory exists: %s" % build_dir)
    qa_config = json.loads(qa_config_path.read_text(encoding="utf-8"))
    rows = _load_records(source_records)
    groups = _group_rows(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir()
    replacement_by_group: Dict[str, Dict[str, Dict[str, Any]]] = {}
    group_diagnostics: List[Dict[str, Any]] = []
    try:
        for group_id in sorted(groups):
            variants = groups[group_id]
            relevance_reference = variants["observed"][0]["provenance"][
                "relevance_supervision"
            ]
            relevance_40, _ = _load_npz_reference(relevance_reference, "relevance")
            if relevance_40.shape != (6, *STORED_HW):
                raise ValueError("group %s relevance is not [6,40,40]" % group_id)
            time_steps, component_count, peak = _source_control_contract(variants)
            pair = build_consumer_grid_pair(
                relevance_40,
                time_steps=time_steps,
                component_count=component_count,
                peak=peak,
            )
            replacement_by_group[group_id] = {}
            for variant in CONTROL_VARIANTS:
                row = variants[variant][0]
                files = _write_control_files(
                    group_id=group_id,
                    variant=variant,
                    bundle=pair[variant],
                    relevance_40=relevance_40,
                    build_dir=build_dir,
                    final_dir=output_dir,
                    sidecar_template=row["target"]["map_sidecar"],
                )
                consumer_summary = summarize_maps(
                    pair[variant]["consumer_uq"],
                    pair[variant]["consumer_relevance"],
                    config=qa_config,
                )
                consumer_summary.pop("task_risk_map", None)
                replacement_by_group[group_id][variant] = {
                    **files,
                    "bundle": pair[variant],
                    "summary": normalize_v5_structured_summary(
                        consumer_summary,
                        relevance_high_threshold=float(
                            qa_config["level_thresholds"]["high"]
                        ),
                    ),
                }
            group_diagnostics.append(
                {
                    "group_id": group_id,
                    "event_id": str(variants["observed"][0]["event_id"]),
                    "split": str(variants["observed"][0]["split"]),
                    "on_path_support_weighted_relevance": pair["on_path_uq"]["support"][
                        "support_weighted_relevance"
                    ],
                    "off_path_support_weighted_relevance": pair["off_path_uq"][
                        "support"
                    ]["support_weighted_relevance"],
                    "on_center_view_y_x": pair["on_path_uq"]["support"][
                        "center_view_y_x"
                    ],
                    "off_center_view_y_x": pair["off_path_uq"]["support"][
                        "center_view_y_x"
                    ],
                }
            )

        upgraded: List[Dict[str, Any]] = []
        changed_records = 0
        for source_row in rows:
            row = copy.deepcopy(source_row)
            group_id = str(row["counterfactual"]["group_id"])
            variant = str(row["counterfactual"]["variant"])
            row.setdefault("provenance", {})["consumer_grid_control_v11_1_upgrade"] = {
                "schema": SCHEMA,
                "source_record_sha256": canonical_sha256(source_row),
                "source_dataset_sha256": sha256_file(source_records),
                "controlled_u_and_derived_targets_changed": variant in CONTROL_VARIANTS,
                "observation_unchanged": True,
                "route_and_ego_context_unchanged": True,
                "stage1_checkpoint_lineage_unchanged": True,
                "relevance_supervision_unchanged": True,
                "split_event_and_group_identity_unchanged": True,
                "consumer_grid_hw": [10, 10],
                "stored_grid_hw": [40, 40],
            }
            if variant in CONTROL_VARIANTS:
                changed_records += 1
                replacement = replacement_by_group[group_id][variant]
                bundle = replacement["bundle"]
                reference = row["model_input"]["stage1_observation_uq"]
                reference.update(
                    {
                        "path": replacement["tensor_path"],
                        "sha256": replacement["tensor_sha256"],
                        "shape": list(bundle["uq"].shape),
                        "component_shape": list(bundle["components"].shape),
                        "source": CONTROL_SOURCE,
                    }
                )
                row["counterfactual"]["spatial_support"] = bundle["support"]
                summary = replacement["summary"]
                family = str(row["question_family"])
                answer = render_structured_answer(family, summary)
                row["target"]["structured_summary"] = summary
                row["target"]["semantic_fields"] = expected_semantic_fields(
                    family, summary
                )
                row["target"]["vlm_task_field_targets"] = _field_targets(
                    family, variant, summary
                )
                row["target"]["rendered_answer"] = answer
                row["target"]["map_sidecar"] = replacement["sidecar"]
                row["conversation"][1]["value"] = answer
            _assert_only_authorized_changes(source_row, row)
            upgraded.append(row)

        qa_audit = audit_v5_records(upgraded)
        if not qa_audit["passed"]:
            raise ValueError("upgraded records failed the v5 QA audit")
        records_build = build_dir / "records.jsonl"
        records_build.write_text(
            "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
                for row in upgraded
            ),
            encoding="utf-8",
        )
        qa_audit_path = build_dir / "qa_audit.json"
        qa_audit_path.write_text(
            json.dumps(qa_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        split_groups: Dict[str, set] = {}
        events = set()
        for row in upgraded:
            split_groups.setdefault(str(row["split"]), set()).add(
                str(row["counterfactual"]["group_id"])
            )
            events.add(str(row["event_id"]))
        margins = [
            float(item["on_path_support_weighted_relevance"])
            - float(item["off_path_support_weighted_relevance"])
            for item in group_diagnostics
        ]
        report = {
            "schema": SCHEMA,
            "status": "complete_qa_audited_requires_independent_full_tensor_preflight",
            "source": {
                "path": str(source_records),
                "sha256": sha256_file(source_records),
                "record_count": len(rows),
            },
            "output": {
                "path": str((output_dir / "records.jsonl").resolve()),
                "sha256": sha256_file(records_build),
                "record_count": len(upgraded),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "qa_config": {
                "path": str(qa_config_path),
                "sha256": sha256_file(qa_config_path),
            },
            "qa_audit": {
                "path": str((output_dir / "qa_audit.json").resolve()),
                "sha256": sha256_file(qa_audit_path),
                "passed": True,
            },
            "event_count": len(events),
            "group_count": len(groups),
            "groups_by_split": {
                split: len(group_ids)
                for split, group_ids in sorted(split_groups.items())
            },
            "changed_control_record_count": changed_records,
            "unchanged_noncontrol_record_count": len(upgraded) - changed_records,
            "consumer_grid_contract": {
                "consumer_grid_hw": [10, 10],
                "stored_grid_hw": [40, 40],
                "exact_block_expand_hw": [4, 4],
                "radius_cells": RADIUS_CELLS,
                "relevance_level_vocabulary": ["low", "high"],
                "relevance_high_threshold": float(
                    qa_config["level_thresholds"]["high"]
                ),
                "on_minus_off_weighted_relevance_minimum": min(margins),
                "on_minus_off_weighted_relevance_maximum": max(margins),
                (
                    "all_pairs_spatially_distinct_and_"
                    "magnitude_matched_by_construction"
                ): True,
            },
            "checks": {
                "source_record_count_preserved": len(rows) == len(upgraded),
                "only_on_off_control_tensors_and_derived_targets_changed": True,
                "all_noncontrol_semantics_and_inputs_bitwise_canonical_unchanged": True,
                "observation_route_ego_r_stage1_split_event_identity_preserved": True,
                "v5_qa_audit_passed": True,
                "source_dataset_was_not_modified": True,
            },
            "group_diagnostics": group_diagnostics,
            "training_started": False,
            "gpu_job_authorized": False,
            "claim_boundary": (
                "CPU-only repair of the controlled-U consumer-grid contract. "
                "It does not establish R generalization, language use of U, "
                "learned-U validity, planning benefit or closed-loop safety."
            ),
        }
        # Recheck the source after all reads/writes; this is deliberately not
        # inferred from the fact that output uses a different directory.
        report["checks"]["source_dataset_was_not_modified"] = (
            sha256_file(source_records) == report["source"]["sha256"]
        )
        report_path = build_dir / "upgrade_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        build_dir.rename(output_dir)
        return report
    except Exception:
        _cleanup_private_build_dir(build_dir)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qa-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/scenario_factory/qa_factory_v2_matched_supervision.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = upgrade_dataset(
        source_records=args.source_records,
        output_dir=args.output_dir,
        qa_config_path=args.qa_config,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "event_count": report["event_count"],
                "group_count": report["group_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
