#!/usr/bin/env python3
"""Stream frozen ORION features directly into resumable FP16 route shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.extract_paired_spatial_features import (  # noqa: E402
    _build_real_backbone,
    _extract_tokens,
    _stable_sample_seed,
)
from uq_estimator.corruptions import corrupt_multiview_images_with_metadata  # noqa: E402
from uq_estimator.counterfactual_evidence_extraction import (  # noqa: E402
    COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V3,
    deterministic_window_balanced_view,
    load_counterfactual_protocol,
    projected_feature_counts,
    split_interventions,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    write_direct_fp16_route_shard,
)
from uq_estimator.observation_uq_v3 import route_splits_from_manifest  # noqa: E402
from uq_estimator.paired_feature_extraction import (  # noqa: E402
    build_info_identity_index,
    camera_view_names_from_info,
    exact_mask_to_patch_coverage,
    find_info_for_image_meta,
    resolve_route_frame_identity,
    select_contiguous_route_balanced_infos,
)


DIRECT_EXTRACTION_CONTRACT_SCHEMA = "orion.counterfactual-direct-fp16-contract/v1"
DIRECT_EXTRACTION_PROGRESS_SCHEMA = "orion.counterfactual-direct-fp16-progress/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage(path: Path, kind: str) -> Dict[str, Any]:
    path = Path(path)
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / (path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _to_feature_grid(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise RuntimeError("unexpected EVAViT token shape")
    return tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1]).half().cpu()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--train-routes", type=int, default=70)
    parser.add_argument("--validation-routes", type=int, default=10)
    parser.add_argument("--heldout-routes", type=int, default=10)
    parser.add_argument("--frames-per-route", type=int, default=16)
    parser.add_argument("--target-batch-size", type=int, default=2)
    parser.add_argument("--max-output-gb", type=float, default=150.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_progress(output_dir: Path, fingerprint: str) -> list[Dict[str, Any]]:
    progress_path = output_dir / "manifest.partial.json"
    if not progress_path.exists():
        return []
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DIRECT_EXTRACTION_PROGRESS_SCHEMA:
        raise RuntimeError("direct FP16 progress schema differs")
    if payload.get("direct_extraction_fingerprint") != fingerprint:
        raise RuntimeError("direct FP16 progress lineage differs")
    rows = payload.get("shards")
    if not isinstance(rows, list):
        raise RuntimeError("direct FP16 progress has no shard rows")
    route_ids = []
    for row in rows:
        route_ids.extend(str(value) for value in row.get("route_ids", []))
        shard_path = output_dir / str(row.get("file", ""))
        if not shard_path.is_file() or _sha256(shard_path) != row.get("sha256"):
            raise RuntimeError("completed direct FP16 shard hash differs")
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("direct FP16 progress contains duplicate routes")
    return rows


def _save_progress(
    output_dir: Path, fingerprint: str, shard_rows: Sequence[Mapping[str, Any]]
) -> None:
    _write_json_atomic(
        output_dir / "manifest.partial.json",
        {
            "schema_version": DIRECT_EXTRACTION_PROGRESS_SCHEMA,
            "direct_extraction_fingerprint": fingerprint,
            "status": "resumable_incomplete_not_for_training",
            "written_route_count": len(shard_rows),
            "written_clean_count": sum(int(row["clean_count"]) for row in shard_rows),
            "written_observed_count": sum(
                int(row["observed_count"]) for row in shard_rows
            ),
            "shards": list(shard_rows),
            "adapter_training_performed": False,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.batch_size <= 0
        or args.workers < 0
        or args.frames_per_route <= 0
        or args.frames_per_route % args.batch_size
        or args.target_batch_size <= 0
        or args.max_output_gb <= 0
    ):
        raise SystemExit("direct FP16 extraction guard values are invalid")
    if not torch.cuda.is_available():
        raise SystemExit("direct FP16 feature extraction requires CUDA")
    route_quotas = {
        "train": args.train_routes,
        "validation": args.validation_routes,
        "held_out": args.heldout_routes,
    }
    if any(value <= 0 for value in route_quotas.values()):
        raise SystemExit("direct FP16 route quotas must be positive")
    counts = projected_feature_counts(route_quotas, args.frames_per_route)

    protocol = load_counterfactual_protocol(args.protocol)
    if not protocol["schema_version"].endswith(("/v2", "/v3")):
        raise RuntimeError("direct route extraction requires the window-cycle schedule")
    view_schedule = protocol["intervention_split"]["view_schedule"]
    input_lineage = {
        "config": _lineage(args.config, "orion_config"),
        "checkpoint": _lineage(args.checkpoint, "orion_checkpoint"),
        "ann_file": _lineage(args.ann_file, "expanded_b2d_infos"),
        "route_manifest": _lineage(args.route_manifest, "frozen_route_manifest"),
        "protocol": _lineage(args.protocol, "counterfactual_protocol"),
        "extractor": _lineage(Path(__file__), "direct_fp16_extractor"),
        "storage_module": _lineage(
            REPOSITORY_ROOT / "uq_estimator/counterfactual_sharded_dataset.py",
            "direct_fp16_storage_module",
        ),
    }
    contract_core = {
        "schema_version": DIRECT_EXTRACTION_CONTRACT_SCHEMA,
        "extraction_schema_version": COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V3,
        "seed": args.seed,
        "route_quotas": route_quotas,
        "frames_per_route": args.frames_per_route,
        "batch_size": args.batch_size,
        "target_batch_size": args.target_batch_size,
        "expected_counts": counts,
        "view_schedule": view_schedule,
        "input_lineage": input_lineage,
        "feature_storage_dtype": "float16",
        "target_storage_dtype": "float32",
        "monolithic_intermediate_created": False,
        "corruption_mask_is_primary_target": False,
        "exact_nonzero_presence_label_authorized": False,
        "adapter_training_performed": False,
    }
    extraction_fingerprint = _fingerprint(contract_core)
    contract = dict(contract_core)
    contract["direct_extraction_fingerprint"] = extraction_fingerprint

    output_dir = args.output_dir
    contract_path = output_dir / "extraction_contract.json"
    final_manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume only for matching progress")
        if final_manifest_path.exists():
            raise RuntimeError("direct FP16 dataset is already complete")
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise RuntimeError("resume contract differs from frozen direct extraction")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(contract_path, contract)
    shard_rows = _load_progress(output_dir, extraction_fingerprint)
    completed_routes = {
        str(row["route_ids"][0]) for row in shard_rows
    }

    manifest_payload = json.loads(args.route_manifest.read_text(encoding="utf-8"))
    route_splits = route_splits_from_manifest(manifest_payload)
    cfg, backbone, backbone_metadata = _build_real_backbone(args)
    from mmcv.datasets import build_dataloader, build_dataset

    cfg.data.test.ann_file = str(args.ann_file)
    dataset = build_dataset(cfg.data.test)
    selected_infos = select_contiguous_route_balanced_infos(
        dataset.data_infos,
        manifest_payload,
        route_quotas,
        args.frames_per_route,
    )
    if len(selected_infos) != counts["reference"]:
        raise RuntimeError("direct FP16 reference selection count changed")
    selected_route_order = []
    for info in selected_infos:
        route_id = str(info.get("folder", info.get("route_folder", ""))).strip()
        if not selected_route_order or selected_route_order[-1] != route_id:
            selected_route_order.append(route_id)
    if (
        len(selected_route_order) != sum(route_quotas.values())
        or len(selected_route_order) != len(set(selected_route_order))
    ):
        raise RuntimeError("direct FP16 selected route order differs")
    if not completed_routes.issubset(selected_route_order):
        raise RuntimeError("resume progress contains a route outside frozen selection")
    selected_position = {route_id: index for index, route_id in enumerate(selected_route_order)}
    shard_rows.sort(key=lambda row: selected_position[str(row["route_ids"][0])])
    remaining_infos = [
        info
        for info in selected_infos
        if str(info.get("folder", info.get("route_folder", ""))).strip()
        not in completed_routes
    ]
    dataset.data_infos = remaining_infos
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    info_index = build_info_identity_index(dataset.data_infos) if remaining_infos else {}
    loader = (
        build_dataloader(
            dataset,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=args.workers,
            dist=False,
            shuffle=False,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        if remaining_infos
        else ()
    )

    expected_camera_names = None
    projected_bytes = None
    current: Optional[Dict[str, Any]] = None

    def flush_current() -> None:
        nonlocal current, shard_rows
        if current is None:
            return
        split = current["split"]
        expected_observed = args.frames_per_route * len(
            split_interventions(split, protocol)
        )
        if len(current["clean_features"]) != args.frames_per_route or len(
            current["observed_features"]
        ) != expected_observed:
            raise RuntimeError("route buffer is incomplete")
        unique_views = {
            key: sorted(values) for key, values in current["schedule_route_views"].items()
        }
        if not unique_views or min(len(values) for values in unique_views.values()) < 4:
            raise RuntimeError("route failed four-view-per-condition schedule gate")
        row = write_direct_fp16_route_shard(
            output_dir,
            route_id=current["route_id"],
            clean_features=current["clean_features"],
            clean_items=current["clean_items"],
            observed_features=current["observed_features"],
            observed_items=current["observed_items"],
            direct_extraction_fingerprint=extraction_fingerprint,
            extraction_schema_version=COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V3,
            target_batch_size=args.target_batch_size,
            target_device=torch.device("cuda"),
        )
        row["family_counts"] = dict(sorted(current["family_counts"].items()))
        row["view_schedule_counts"] = dict(sorted(current["schedule_counts"].items()))
        row["view_schedule_unique_views"] = unique_views
        row["camera_view_names"] = list(expected_camera_names or ())
        shard_rows.append(row)
        shard_rows.sort(key=lambda value: selected_position[str(value["route_ids"][0])])
        _save_progress(output_dir, extraction_fingerprint, shard_rows)
        print(
            "[DirectFP16] route=%d/%d clean=%d observed=%d file=%s"
            % (
                len(shard_rows),
                len(selected_route_order),
                row["clean_count"],
                row["observed_count"],
                row["file"],
            ),
            flush=True,
        )
        current = None

    for data in loader:
        images = data["img"][0].data.cuda(non_blocking=True)
        image_metas = data["img_metas"][0]
        if len(image_metas) != images.shape[0]:
            raise RuntimeError("img_metas/image batch count mismatch")
        identities = []
        splits = []
        for image_meta in image_metas:
            info = find_info_for_image_meta(info_index, image_meta)
            identity = resolve_route_frame_identity(info, image_meta)
            camera_names = camera_view_names_from_info(info)
            if expected_camera_names is None:
                expected_camera_names = camera_names
            elif expected_camera_names != camera_names:
                raise RuntimeError("camera order changes inside direct FP16 extraction")
            identities.append(identity)
            splits.append(route_splits[identity.route_id])
        batch_routes = {identity.route_id for identity in identities}
        if len(batch_routes) != 1 or len(set(splits)) != 1:
            raise RuntimeError("one direct FP16 batch crossed route/split boundaries")
        route_id = identities[0].route_id
        split = splits[0]
        if current is not None and current["route_id"] != route_id:
            flush_current()
        if current is None:
            current = {
                "route_id": route_id,
                "split": split,
                "clean_features": [],
                "clean_items": [],
                "observed_features": [],
                "observed_items": [],
                "family_counts": {},
                "schedule_counts": {},
                "schedule_route_views": {},
            }
        if current["split"] != split:
            raise RuntimeError("one route changed split during extraction")

        with torch.inference_mode():
            clean_tokens, patch_height, patch_width = _extract_tokens(backbone, images)
        if projected_bytes is None:
            bytes_per_feature = clean_tokens[0].numel() * 2
            projected_bytes = bytes_per_feature * counts["total"]
            if projected_bytes > int(args.max_output_gb * (1024**3)):
                raise RuntimeError(
                    "projected FP16 features %.2f GiB exceed --max-output-gb=%g"
                    % (projected_bytes / (1024**3), args.max_output_gb)
                )
            print(
                "[DirectFP16] reference=%d observed=%d projected_features=%.2f GiB"
                % (
                    counts["reference"],
                    counts["observed"],
                    projected_bytes / (1024**3),
                ),
                flush=True,
            )

        clean_indices = []
        for sample_index, identity in enumerate(identities):
            clean_index = len(current["clean_features"])
            clean_indices.append(clean_index)
            current["clean_features"].append(
                _to_feature_grid(clean_tokens[sample_index], patch_height, patch_width)
            )
            current["clean_items"].append(
                {
                    "clean_index": clean_index,
                    "sample_id": identity.sample_token,
                    "route_id": identity.route_id,
                    "town": identity.town,
                    "frame_idx": identity.frame_idx,
                    "split": split,
                    "reference_semantics": "unintervened_mixed_weather",
                }
            )

        for family, severity in split_interventions(split, protocol):
            corrupt_images = []
            exact_masks = []
            metadata_rows = []
            for sample_index, identity in enumerate(identities):
                view_index = deterministic_window_balanced_view(
                    identity.route_id,
                    identity.frame_idx,
                    family,
                    severity,
                    images.shape[1],
                    args.seed,
                    window_frames=int(view_schedule["window_frames"]),
                )
                result = corrupt_multiview_images_with_metadata(
                    images[sample_index],
                    corruption=family,
                    severity=severity,
                    view_indices=[view_index],
                    seed=_stable_sample_seed(
                        args.seed, identity.sample_token, family, severity
                    ),
                )
                corrupt_images.append(result.images)
                exact_masks.append(result.mask)
                metadata = result.metadata.to_dict()
                metadata["view_schedule"] = view_schedule["version"]
                metadata_rows.append(metadata)
                schedule_key = "%s/severity_%d/view_%d" % (
                    family,
                    severity,
                    view_index,
                )
                current["schedule_counts"][schedule_key] = (
                    current["schedule_counts"].get(schedule_key, 0) + 1
                )
                condition_key = "%s/severity_%d" % (family, severity)
                current["schedule_route_views"].setdefault(condition_key, set()).add(
                    view_index
                )
            corrupt_batch = torch.stack(corrupt_images)
            exact_mask = torch.cat(exact_masks)
            with torch.inference_mode():
                observed_tokens, observed_h, observed_w = _extract_tokens(
                    backbone, corrupt_batch
                )
            if (observed_h, observed_w) != (patch_height, patch_width):
                raise RuntimeError("reference/observed patch grids differ")
            coverage = exact_mask_to_patch_coverage(
                exact_mask, patch_height, patch_width
            ).reshape(images.shape[0], images.shape[1], patch_height, patch_width)
            for sample_index, identity in enumerate(identities):
                observed_index = len(current["observed_features"])
                current["observed_features"].append(
                    _to_feature_grid(
                        observed_tokens[sample_index], patch_height, patch_width
                    )
                )
                current["observed_items"].append(
                    {
                        "observed_index": observed_index,
                        "clean_index": clean_indices[sample_index],
                        "sample_id": "%s/%s/severity_%d"
                        % (identity.sample_token, family, severity),
                        "route_id": identity.route_id,
                        "town": identity.town,
                        "frame_idx": identity.frame_idx,
                        "split": split,
                        "family": family,
                        "severity": float(severity),
                        "corruption_mask": coverage[sample_index].half().cpu(),
                        "corruption_metadata": metadata_rows[sample_index],
                        "target_role": "counterfactual_evidence_intervention_only",
                    }
                )
                family_key = "%s/%s/severity_%d" % (split, family, severity)
                current["family_counts"][family_key] = (
                    current["family_counts"].get(family_key, 0) + 1
                )
        if len(current["clean_features"]) == args.frames_per_route:
            flush_current()
        elif len(current["clean_features"]) > args.frames_per_route:
            raise RuntimeError("route buffer exceeded frozen frame count")

    flush_current()
    completed_routes = {str(row["route_ids"][0]) for row in shard_rows}
    if completed_routes != set(selected_route_order):
        raise RuntimeError("direct FP16 extraction ended with incomplete route coverage")
    if (
        sum(int(row["clean_count"]) for row in shard_rows) != counts["reference"]
        or sum(int(row["observed_count"]) for row in shard_rows) != counts["observed"]
    ):
        raise RuntimeError("direct FP16 extraction final counts differ")
    feature_shapes = {tuple(row["feature_shape"]) for row in shard_rows}
    if len(feature_shapes) != 1:
        raise RuntimeError("direct FP16 route feature shapes differ globally")
    camera_orders = {tuple(row["camera_view_names"]) for row in shard_rows}
    if len(camera_orders) != 1 or not next(iter(camera_orders)):
        raise RuntimeError("direct FP16 camera order differs globally")
    family_counts: Dict[str, int] = {}
    for row in shard_rows:
        for key, value in row["family_counts"].items():
            family_counts[key] = family_counts.get(key, 0) + int(value)
    final_manifest = {
        "schema_version": FP16_DIRECT_DATASET_SCHEMA_VERSION,
        "status": "complete",
        "source": {
            "direct_extraction_fingerprint": extraction_fingerprint,
            "extraction_contract": str(contract_path.resolve()),
            "extraction_contract_sha256": _sha256(contract_path),
            "route_count": len(selected_route_order),
            "clean_count": counts["reference"],
            "observed_count": counts["observed"],
            "feature_shape": list(next(iter(feature_shapes))),
            "camera_view_names": list(next(iter(camera_orders))),
            "selected_route_order": selected_route_order,
            "backbone": dict(backbone_metadata),
        },
        "storage_contract": {
            "whole_routes_per_shard": True,
            "feature_dtype": "float16",
            "direct_from_frozen_backbone": True,
            "monolithic_intermediate_created": False,
            "targets_computed_from_route_buffer_fp16": True,
            "target_dtype": "float32",
            "corruption_mask_is_optimizer_target": False,
            "resumable_at_verified_route_boundaries": True,
        },
        "route_quotas": route_quotas,
        "frames_per_route": args.frames_per_route,
        "family_counts": dict(sorted(family_counts.items())),
        "written_route_count": len(shard_rows),
        "written_clean_count": counts["reference"],
        "written_observed_count": counts["observed"],
        "total_size_bytes": sum(int(row["size_bytes"]) for row in shard_rows),
        "shards": shard_rows,
        "claim_boundary": {
            "counterfactual_target_is_unique_uncertainty_truth": False,
            "corruption_mask_is_primary_target": False,
            "adapter_training_performed": False,
            "supports_closed_loop_safety_claim": False,
        },
    }
    _write_json_atomic(final_manifest_path, final_manifest)
    (output_dir / "manifest.partial.json").unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output_dir.resolve()),
                "route_count": len(shard_rows),
                "clean_count": counts["reference"],
                "observed_count": counts["observed"],
                "total_size_bytes": final_manifest["total_size_bytes"],
                "manifest_sha256": _sha256(final_manifest_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("COUNTERFACTUAL_DIRECT_FP16_ROUTE_EXTRACTION_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
