#!/usr/bin/env python3
"""Diagnose clean-reference score tails without training or held-out access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    EVIDENCE_COMPONENTS,
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    _exact_quantiles_1d,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_route_shard_records_selective,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-clean-tail-diagnostic/v1"
REPORT_SCHEMA_VERSION = "orion.counterfactual-evidence-clean-tail-report/v1"
DIAGNOSTIC_FAMILIES = ("local_blur", "local_dark")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_id(row: Mapping[str, object]) -> str:
    route_ids = row.get("route_ids")
    if not isinstance(route_ids, list) or len(route_ids) != 1:
        raise RuntimeError("route shard must own exactly one route")
    return str(route_ids[0])


def _town(route_id: str) -> str:
    match = re.search(r"_(Town\d+(?:HD)?)_", route_id)
    if match is None:
        raise RuntimeError("route Town cannot be parsed: %s" % route_id)
    return match.group(1)


def _summary(values: torch.Tensor, threshold: float) -> Dict[str, float]:
    values = values.detach().cpu().float().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise RuntimeError("diagnostic summary values are empty or non-finite")
    q50, q90, q95, q99 = _exact_quantiles_1d(values, (0.5, 0.9, 0.95, 0.99))
    return {
        "cell_count": int(values.numel()),
        "mean": float(values.mean()),
        "p50": float(q50),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "maximum": float(values.max()),
        "fraction_above_frozen_gate_threshold": float((values > threshold).float().mean()),
    }


def _group_summary(
    rows: Sequence[Mapping[str, object]], key: str, value_key: str, threshold: float
) -> Dict[str, object]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    frame_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        name = str(row[key])
        grouped[name].append(row[value_key])
        frame_counts[name] += 1
    return {
        name: {
            "frame_count": frame_counts[name],
            **_summary(torch.cat(values), threshold),
        }
        for name, values in sorted(grouped.items())
    }


def _tail_composition(
    rows: Sequence[Mapping[str, object]], tail_threshold: float
) -> Dict[str, object]:
    total_tail = 0
    invalid_tail = 0
    total_cells = 0
    invalid_cells = 0
    component_tail = {name: 0 for name in EVIDENCE_COMPONENTS}
    view_tail: Dict[str, int] = defaultdict(int)
    for row in rows:
        score = row["score"]
        mask = score >= tail_threshold
        tail_count = int(mask.sum())
        cell_count = int(score.numel())
        total_tail += tail_count
        total_cells += cell_count
        if not bool(row["previous_valid"]):
            invalid_tail += tail_count
            invalid_cells += cell_count
        for index, name in enumerate(EVIDENCE_COMPONENTS):
            component_tail[name] += int(mask[..., index].sum())
        for index, name in enumerate(row["camera_view_names"]):
            view_tail[str(name)] += int(mask[index].sum())
    invalid_base = invalid_cells / total_cells
    invalid_tail_fraction = invalid_tail / max(total_tail, 1)
    return {
        "definition": "score >= the exact overall clean-reference p95",
        "threshold": float(tail_threshold),
        "tail_cell_count": total_tail,
        "previous_invalid_tail_cell_count": invalid_tail,
        "previous_invalid_tail_fraction": invalid_tail_fraction,
        "previous_invalid_all_cell_fraction": invalid_base,
        "previous_invalid_tail_enrichment": (
            invalid_tail_fraction / invalid_base if invalid_base > 0 else math.nan
        ),
        "by_component_cell_count": component_tail,
        "by_camera_cell_count": dict(sorted(view_tail.items())),
    }


def _frame_row(
    *,
    model: ObservationEvidenceHurdleAdapter,
    record: object,
    camera_names: Sequence[str],
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:
    current = record.reference_current.unsqueeze(0).to(device)
    previous = record.reference_previous.unsqueeze(0).to(device)
    valid = torch.tensor([record.previous_valid], dtype=torch.bool, device=device)
    with torch.no_grad():
        parts = model.predict_parts(current, previous, valid)
    score = parts.score[0].detach().cpu().float()
    presence = parts.presence_probability[0].detach().cpu().float()
    magnitude = parts.conditional_magnitude[0].detach().cpu().float()
    if score.shape[0] != len(camera_names):
        raise RuntimeError("camera metadata/model view count differs")
    return {
        "pair_id": record.pair_id,
        "route_id": record.route_id,
        "town": _town(record.route_id),
        "frame_idx": int(record.frame_idx),
        "previous_valid": bool(record.previous_valid),
        "previous_valid_group": "valid" if record.previous_valid else "invalid",
        "camera_view_names": list(camera_names),
        "score": score,
        "presence_probability": presence,
        "conditional_magnitude": magnitude,
        "score_summary": _summary(score, threshold),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)

    if args.output.exists():
        raise SystemExit("refusing to overwrite clean-tail diagnostic")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA clean-tail diagnostic requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected clean-tail diagnostic config schema")
    input_paths = {
        "dataset_manifest_sha256": args.dataset_manifest,
        "checkpoint_sha256": args.checkpoint,
        "training_report_sha256": args.training_report,
    }
    hashes = {name: _sha256(path) for name, path in input_paths.items()}
    for name, value in hashes.items():
        if value != config["inputs"][name]:
            raise RuntimeError("clean-tail diagnostic input hash changed: %s" % name)

    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    if training_report.get("schema_version") != config["inputs"]["training_report_schema"]:
        raise RuntimeError("training report schema differs")
    original_mean = float(training_report["route_validation"]["reference_prediction_mean"])
    original_p95 = float(training_report["route_validation"]["reference_prediction_p95"])

    manifest = load_fp16_dataset_manifest(args.dataset_manifest, verify_shards=False)
    if (
        manifest.get("schema_version") != FP16_DIRECT_DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("expanded FP16 dataset contract differs")
    validation_rows = sorted(
        [dict(row) for row in manifest["shards"] if row.get("split") == "validation"],
        key=_route_id,
    )
    expected = config["population"]
    if len(validation_rows) != int(expected["route_count"]):
        raise RuntimeError("validation route count differs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != config["inputs"]["checkpoint_schema"]:
        raise RuntimeError("checkpoint schema differs")
    model = ObservationEvidenceHurdleAdapter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["student_state"])
    device = torch.device(args.device)
    model.to(device).eval()

    camera_names = manifest.get("source", {}).get("camera_view_names")
    if not isinstance(camera_names, list) or not camera_names:
        raise RuntimeError("dataset camera names are absent")
    if any(row.get("camera_view_names") != camera_names for row in validation_rows):
        raise RuntimeError("validation route camera order differs")
    threshold = float(config["frozen_reference_gate_threshold"])
    dataset_root = args.dataset_manifest.parent
    frame_rows: List[Dict[str, object]] = []
    for route_index, row in enumerate(validation_rows, start=1):
        shard_path = dataset_root / str(row["file"])
        if not shard_path.is_file() or shard_path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError("validation shard file/size differs: %s" % shard_path)
        records = load_fp16_route_shard_records_selective(
            shard_path, families=DIAGNOSTIC_FAMILIES
        )
        unique = {}
        for record in records:
            unique.setdefault(record.pair_id, record)
        if len(unique) != int(expected["frames_per_route"]):
            raise RuntimeError("unique clean frame count differs: %s" % _route_id(row))
        for pair_id in sorted(unique, key=lambda value: unique[value].frame_idx):
            frame_rows.append(
                _frame_row(
                    model=model,
                    record=unique[pair_id],
                    camera_names=camera_names,
                    device=device,
                    threshold=threshold,
                )
            )
        print(
            "[CleanTail] route=%d/%d id=%s"
            % (route_index, len(validation_rows), _route_id(row)),
            flush=True,
        )

    if len(frame_rows) != int(expected["clean_frame_count"]):
        raise RuntimeError("clean diagnostic frame count differs")
    invalid_frames = sum(not bool(row["previous_valid"]) for row in frame_rows)
    if invalid_frames != int(expected["previous_invalid_frame_count"]):
        raise RuntimeError("previous-invalid frame count differs")

    all_scores = torch.cat([row["score"].reshape(-1) for row in frame_rows])
    all_presence = torch.cat(
        [row["presence_probability"].reshape(-1) for row in frame_rows]
    )
    all_magnitude = torch.cat(
        [row["conditional_magnitude"].reshape(-1) for row in frame_rows]
    )
    overall = _summary(all_scores, threshold)
    p95_tolerance = float(config["reproduction_absolute_tolerance"])
    reproduced = (
        abs(overall["mean"] - original_mean) <= p95_tolerance
        and abs(overall["p95"] - original_p95) <= p95_tolerance
    )

    valid_scores = torch.cat(
        [row["score"].reshape(-1) for row in frame_rows if row["previous_valid"]]
    )
    invalid_scores = torch.cat(
        [row["score"].reshape(-1) for row in frame_rows if not row["previous_valid"]]
    )
    valid_summary = _summary(valid_scores, threshold)
    invalid_summary = _summary(invalid_scores, threshold)
    if not reproduced:
        decision = "invalid_diagnostic_reference_metric_not_reproduced"
    elif valid_summary["p95"] <= threshold < invalid_summary["p95"]:
        decision = "temporal_boundary_sufficient_to_explain_frozen_p95_failure"
    else:
        decision = "general_clean_false_positive_persists"

    component_rows = []
    camera_rows = []
    for row in frame_rows:
        for index, name in enumerate(EVIDENCE_COMPONENTS):
            component_rows.append(
                {"component": name, "values": row["score"][..., index]}
            )
        for index, name in enumerate(camera_names):
            camera_rows.append({"camera": name, "values": row["score"][index]})

    top_frames = sorted(
        (
            {
                "pair_id": row["pair_id"],
                "route_id": row["route_id"],
                "town": row["town"],
                "frame_idx": row["frame_idx"],
                "previous_valid": row["previous_valid"],
                **row["score_summary"],
            }
            for row in frame_rows
        ),
        key=lambda row: (row["p95"], row["mean"]),
        reverse=True,
    )[: int(config["top_frame_count"])]

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inputs": {
            **hashes,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        },
        "population": {
            "route_count": len(validation_rows),
            "town_count": len({row["town"] for row in frame_rows}),
            "towns": sorted({row["town"] for row in frame_rows}),
            "clean_frame_count": len(frame_rows),
            "previous_valid_frame_count": len(frame_rows) - invalid_frames,
            "previous_invalid_frame_count": invalid_frames,
            "previous_invalid_frame_fraction": invalid_frames / len(frame_rows),
            "camera_view_names": camera_names,
            "component_names": list(EVIDENCE_COMPONENTS),
        },
        "reproduction": {
            "passed": reproduced,
            "absolute_tolerance": p95_tolerance,
            "training_report_reference_prediction_mean": original_mean,
            "diagnostic_reference_prediction_mean": overall["mean"],
            "mean_absolute_difference": abs(overall["mean"] - original_mean),
            "training_report_reference_prediction_p95": original_p95,
            "diagnostic_reference_prediction_p95": overall["p95"],
            "p95_absolute_difference": abs(overall["p95"] - original_p95),
        },
        "score": {
            "overall": overall,
            "by_previous_valid": {
                "valid": {
                    "frame_count": len(frame_rows) - invalid_frames,
                    **valid_summary,
                },
                "invalid": {"frame_count": invalid_frames, **invalid_summary},
            },
            "by_component": _group_summary(
                component_rows, "component", "values", threshold
            ),
            "by_camera": _group_summary(camera_rows, "camera", "values", threshold),
            "by_route": _group_summary(frame_rows, "route_id", "score", threshold),
            "by_town": _group_summary(frame_rows, "town", "score", threshold),
            "tail_composition": _tail_composition(frame_rows, overall["p95"]),
            "top_frames_by_score_p95": top_frames,
        },
        "hurdle_heads": {
            "presence_probability": _summary(all_presence, threshold),
            "conditional_magnitude": _summary(all_magnitude, threshold),
            "note": "Head summaries are diagnostic only; the frozen gate applies to their product score.",
        },
        "frozen_gate": {
            "metric": "reference_prediction_p95",
            "threshold": threshold,
            "original_status": "failed",
            "threshold_changed": False,
            "amendment_made": False,
        },
        "decision": decision,
        "scope_attestation": {
            "training_performed": False,
            "checkpoint_weights_changed": False,
            "optimizer_families_tensor_values_accessed": list(DIAGNOSTIC_FAMILIES),
            "validation_co_sharded_glare_metadata_visible": True,
            "validation_co_sharded_glare_tensor_values_accessed": False,
            "heldout_split_tensor_values_accessed": False,
            "native_weather_read": False,
            "orion_finetuning": False,
            "stage_b": False,
            "selected_validation_shard_hashes_recomputed": False,
            "selected_validation_shard_sizes_checked": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reproduced": reproduced,
                "overall_p95": overall["p95"],
                "valid_p95": valid_summary["p95"],
                "invalid_p95": invalid_summary["p95"],
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("COUNTERFACTUAL_EVIDENCE_CLEAN_TAIL_DIAGNOSTIC_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
