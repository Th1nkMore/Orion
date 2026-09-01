#!/usr/bin/env python3
"""Audit whether a frozen low-dimensional projection preserves paired targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_compaction import (  # noqa: E402
    COMPACT_PROJECTION_SCHEMA_VERSION,
    deterministic_rademacher_projection,
    project_feature_grid,
    projection_sha256,
)
from uq_estimator.counterfactual_evidence import (  # noqa: E402
    EVIDENCE_COMPONENTS,
    counterfactual_evidence_target,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    CounterfactualEvidenceRecord,
    _responsive_top_fraction_labels,
    records_from_counterfactual_shard,
    select_records,
)
from uq_estimator.observation_uq_shard import load_feature_shard  # noqa: E402
from uq_estimator.observation_uq_v3 import _binary_auc, _spearman  # noqa: E402


SCHEMA_VERSION = "orion.counterfactual-compact-projection-audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_index(seed: int, key: str, count: int) -> int:
    payload = "%d|%s" % (int(seed), key)
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value % int(count)


def _sample_route_conditions(
    records: list[CounterfactualEvidenceRecord], seed: int
) -> list[CounterfactualEvidenceRecord]:
    groups = defaultdict(list)
    for record in records:
        groups[(record.split, record.route_id, record.family, record.severity)].append(
            record
        )
    selected = []
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row.frame_idx)
        transition = [
            row for row in ordered if row.previous_valid and row.frame_idx % 4 == 0
        ]
        stable = [
            row for row in ordered if row.previous_valid and row.frame_idx % 4 != 0
        ]
        if not transition or not stable:
            raise RuntimeError("route condition lacks transition/stable compact samples")
        text_key = "|".join(map(str, key))
        selected.append(transition[_stable_index(seed, text_key + "|transition", len(transition))])
        selected.append(stable[_stable_index(seed, text_key + "|stable", len(stable))])
    return selected


def _project_record(
    record: CounterfactualEvidenceRecord,
    projection: torch.Tensor,
    device: torch.device,
    patch_batch_size: int,
    cache: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def compact(value: torch.Tensor) -> torch.Tensor:
        key = int(value.data_ptr())
        if key not in cache:
            cache[key] = project_feature_grid(
                value,
                projection,
                device,
                output_dtype=torch.float16,
                patch_batch_size=patch_batch_size,
            )
        return cache[key]

    return (
        compact(record.reference_current),
        compact(record.observed_current),
        compact(record.reference_previous),
        compact(record.observed_previous),
    )


def _paired_target(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    previous_valid: bool,
    device: torch.device,
):
    reference, observed, reference_previous, observed_previous = (
        value.unsqueeze(0).to(device) for value in tensors
    )
    return counterfactual_evidence_target(
        reference,
        observed,
        reference_previous,
        observed_previous,
        torch.tensor([previous_valid], dtype=torch.bool, device=device),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-shard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite compact projection audit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA compact projection audit requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected compact projection audit schema")
    feature_sha = _sha256(args.feature_shard)
    if feature_sha != config["inputs"]["feature_shard_sha256"]:
        raise RuntimeError("feature shard changed after compact audit freeze")

    device = torch.device(args.device)
    payload = load_feature_shard(args.feature_shard)
    records = select_records(
        records_from_counterfactual_shard(payload),
        config["sampling"]["splits"],
        config["sampling"]["families"],
    )
    selected = _sample_route_conditions(records, int(config["sampling"]["seed"]))
    counts = {
        "total": len(selected),
        "train": sum(record.split == "train" for record in selected),
        "validation": sum(record.split == "validation" for record in selected),
        "routes": len({record.route_id for record in selected}),
        "route_conditions": len(
            {
                (record.split, record.route_id, record.family, record.severity)
                for record in selected
            }
        ),
        "transition": sum(record.frame_idx % 4 == 0 for record in selected),
        "stable": sum(record.frame_idx % 4 != 0 for record in selected),
    }
    if counts != config["sampling"]["expected_counts"]:
        raise RuntimeError("compact audit sample counts changed")

    projection_config = config["projection"]
    projection = deterministic_rademacher_projection(
        int(projection_config["input_dim"]),
        int(projection_config["output_dim"]),
        int(projection_config["seed"]),
    )
    projection_hash = projection_sha256(projection)
    cache: dict[int, torch.Tensor] = {}
    original_components = [[] for _ in EVIDENCE_COMPONENTS]
    compact_components = [[] for _ in EVIDENCE_COMPONENTS]
    original_combined = []
    compact_combined = []
    record_rows = []

    for index, record in enumerate(selected):
        original = _paired_target(
            (
                record.reference_current,
                record.observed_current,
                record.reference_previous,
                record.observed_previous,
            ),
            record.previous_valid,
            device,
        )
        compact = _paired_target(
            _project_record(
                record,
                projection,
                device,
                int(projection_config["patch_batch_size"]),
                cache,
            ),
            record.previous_valid,
            device,
        )
        original_values = original.values[0].cpu()
        compact_values = compact.values[0].cpu()
        validity = original.component_valid[0].cpu()
        if not torch.equal(validity, compact.component_valid[0].cpu()):
            raise RuntimeError("compact projection changed target validity")
        for component in range(len(EVIDENCE_COMPONENTS)):
            valid = validity[..., component]
            original_components[component].append(original_values[..., component][valid])
            compact_components[component].append(compact_values[..., component][valid])
        valid_count = validity.sum(dim=-1).clamp_min(1)
        original_score = (
            original_values * validity.to(original_values.dtype)
        ).sum(dim=-1) / valid_count
        compact_score = (
            compact_values * validity.to(compact_values.dtype)
        ).sum(dim=-1) / valid_count
        original_combined.append(original_score.reshape(-1))
        compact_combined.append(compact_score.reshape(-1))
        persistent_mass = original_values[..., :2].sum(dim=(1, 2, 3))
        view = int(torch.argmax(persistent_mass))
        original_view = original_score[view].reshape(-1)
        compact_view = compact_score[view].reshape(-1)
        record_rows.append(
            {
                "split": record.split,
                "route_id": record.route_id,
                "family": record.family,
                "severity": record.severity,
                "frame_idx": record.frame_idx,
                "window_state": "transition" if record.frame_idx % 4 == 0 else "stable",
                "within_view_spearman": _spearman(compact_view, original_view),
                "within_view_original_top20_auroc": _binary_auc(
                    compact_view, _responsive_top_fraction_labels(original_view)
                ),
            }
        )
        if (index + 1) % 40 == 0:
            print("[CompactAudit] processed=%d/%d" % (index + 1, len(selected)), flush=True)

    component_metrics = {}
    component_spearman = []
    for name, original_rows, compact_rows in zip(
        EVIDENCE_COMPONENTS, original_components, compact_components
    ):
        original_values = torch.cat(original_rows)
        compact_values = torch.cat(compact_rows)
        spearman = _spearman(compact_values, original_values)
        component_spearman.append(spearman)
        component_metrics[name] = {
            "spearman": spearman,
            "original_top20_auroc": _binary_auc(
                compact_values, _responsive_top_fraction_labels(original_values)
            ),
            "mean_ratio": float(
                compact_values.mean() / original_values.mean().clamp_min(1e-12)
            ),
        }
    original_flat = torch.cat(original_combined)
    compact_flat = torch.cat(compact_combined)
    metrics = {
        "components": component_metrics,
        "minimum_component_spearman": min(component_spearman),
        "combined_spearman": _spearman(compact_flat, original_flat),
        "combined_original_top20_auroc": _binary_auc(
            compact_flat, _responsive_top_fraction_labels(original_flat)
        ),
        "median_record_within_view_spearman": float(
            torch.tensor([row["within_view_spearman"] for row in record_rows]).median()
        ),
        "median_record_within_view_original_top20_auroc": float(
            torch.tensor(
                [row["within_view_original_top20_auroc"] for row in record_rows]
            ).median()
        ),
    }
    gates = config["gates"]
    checks = [
        ("minimum_component_spearman", metrics["minimum_component_spearman"]),
        ("combined_spearman", metrics["combined_spearman"]),
        ("combined_original_top20_auroc", metrics["combined_original_top20_auroc"]),
        ("median_record_within_view_spearman", metrics["median_record_within_view_spearman"]),
        (
            "median_record_within_view_original_top20_auroc",
            metrics["median_record_within_view_original_top20_auroc"],
        ),
    ]
    gate_rows = [
        {
            "metric": name,
            "value": float(value),
            "threshold": float(gates[name]),
            "passed": float(value) >= float(gates[name]),
        }
        for name, value in checks
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "feature_shard_sha256": feature_sha,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "projection": {
            "schema_version": COMPACT_PROJECTION_SCHEMA_VERSION,
            **projection_config,
            "matrix_sha256": projection_hash,
        },
        "sample_counts": counts,
        "metrics": metrics,
        "gate": {"passed": all(row["passed"] for row in gate_rows), "checks": gate_rows},
        "scope_attestation": {
            "optimizer_steps": 0,
            "output_feature_shard_written": False,
            "spatial_grid_changed": False,
            "corruption_mask_read": False,
            "heldout_glare_evaluated": False,
            "native_weather_evaluated": False,
            "actual_target_read": False,
            "orion_finetuning": False,
            "stage_b": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}, indent=2))
    print("COUNTERFACTUAL_COMPACT_PROJECTION_AUDIT_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
