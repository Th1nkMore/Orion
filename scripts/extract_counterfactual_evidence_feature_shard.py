#!/usr/bin/env python3
"""Extract the frozen 560-frame counterfactual-evidence supervision shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

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
    COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION,
    COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V2,
    deterministic_condition_view,
    deterministic_window_balanced_view,
    load_counterfactual_protocol,
    projected_feature_counts,
    split_interventions,
)
from uq_estimator.observation_uq_shard import (  # noqa: E402
    FEATURE_SHARD_SCHEMA_VERSION,
    save_feature_shard,
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_feature_grid(tokens, height, width):
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise RuntimeError("unexpected EVAViT token shape")
    return tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1]).half().cpu()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference-feature-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-output-gb", type=float, default=75.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if args.batch_size <= 0 or args.workers < 0 or args.max_output_gb <= 0:
        raise SystemExit("batch/workers/output guard values are invalid")
    if not torch.cuda.is_available():
        raise SystemExit("counterfactual feature extraction requires CUDA")
    protocol_bytes = args.protocol.read_bytes()
    protocol = load_counterfactual_protocol(args.protocol)
    is_v2 = protocol["schema_version"].endswith("/v2")
    view_schedule = protocol["intervention_split"].get("view_schedule", {})
    reference_sha = _sha256(args.reference_feature_shard)
    if reference_sha != protocol["reference_data"]["reference_feature_sha256"]:
        raise RuntimeError("reference feature lineage differs from frozen protocol")
    route_quotas = {"train": 35, "validation": 5, "held_out": 5}
    frames_per_route = 16
    counts = projected_feature_counts(route_quotas, frames_per_route)

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
        frames_per_route,
    )
    if len(selected_infos) != counts["reference"]:
        raise RuntimeError("counterfactual reference selection count changed")
    dataset.data_infos = selected_infos
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    info_index = build_info_identity_index(dataset.data_infos)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    clean_features = []
    clean_items = []
    observed_features = []
    observed_items = []
    expected_camera_names = None
    projected_bytes = None
    family_counts = {}
    schedule_counts = {}
    schedule_route_views = {}
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
                raise RuntimeError("camera order changes inside supervision shard")
            identities.append(identity)
            splits.append(route_splits[identity.route_id])
        if len(set(splits)) != 1:
            raise RuntimeError("one extraction batch crossed route-disjoint splits")
        split = splits[0]
        interventions = split_interventions(split, protocol)

        with torch.inference_mode():
            clean_tokens, patch_height, patch_width = _extract_tokens(backbone, images)
        if projected_bytes is None:
            bytes_per_feature = clean_tokens[0].numel() * 2
            projected_bytes = bytes_per_feature * counts["total"]
            limit = int(args.max_output_gb * (1024**3))
            if projected_bytes > limit:
                raise RuntimeError(
                    "projected FP16 shard %.2f GiB exceeds --max-output-gb=%g"
                    % (projected_bytes / (1024**3), args.max_output_gb)
                )
            print(
                "[CounterfactualShard] reference=%d observed=%d projected=%.2f GiB"
                % (
                    counts["reference"],
                    counts["observed"],
                    projected_bytes / (1024**3),
                ),
                flush=True,
            )

        clean_indices = []
        for sample_index, (identity, owner_split) in enumerate(zip(identities, splits)):
            clean_index = len(clean_features)
            clean_indices.append(clean_index)
            clean_features.append(
                _to_feature_grid(clean_tokens[sample_index], patch_height, patch_width)
            )
            clean_items.append(
                {
                    "clean_index": clean_index,
                    "sample_id": identity.sample_token,
                    "route_id": identity.route_id,
                    "town": identity.town,
                    "frame_idx": identity.frame_idx,
                    "split": owner_split,
                    "reference_semantics": "unintervened_mixed_weather",
                }
            )

        for family, severity in interventions:
            corrupt_images = []
            exact_masks = []
            metadata_rows = []
            for sample_index, identity in enumerate(identities):
                if is_v2:
                    view_index = deterministic_window_balanced_view(
                        identity.route_id,
                        identity.frame_idx,
                        family,
                        severity,
                        images.shape[1],
                        args.seed,
                        window_frames=int(view_schedule["window_frames"]),
                    )
                else:
                    view_index = deterministic_condition_view(
                        identity.route_id,
                        family,
                        severity,
                        images.shape[1],
                        args.seed,
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
                row = result.metadata.to_dict()
                row["view_schedule"] = (
                    view_schedule["version"]
                    if is_v2
                    else "route_condition_hash_single/v1"
                )
                schedule_key = "%s/%s/severity_%d/view_%d" % (
                    split, family, severity, view_index
                )
                schedule_counts[schedule_key] = schedule_counts.get(schedule_key, 0) + 1
                route_key = "%s|%s|%s|%d" % (
                    split, identity.route_id, family, severity
                )
                schedule_route_views.setdefault(route_key, set()).add(view_index)
                metadata_rows.append(row)
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
            ).reshape(
                images.shape[0], images.shape[1], patch_height, patch_width
            )
            for sample_index, identity in enumerate(identities):
                observed_index = len(observed_features)
                observed_features.append(
                    _to_feature_grid(
                        observed_tokens[sample_index], patch_height, patch_width
                    )
                )
                observed_items.append(
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
                key = "%s/%s/severity_%d" % (split, family, severity)
                family_counts[key] = family_counts.get(key, 0) + 1
        if len(clean_features) % 80 == 0 or len(clean_features) == counts["reference"]:
            print(
                "[CounterfactualShard] progress reference=%d/%d observed=%d/%d"
                % (
                    len(clean_features),
                    counts["reference"],
                    len(observed_features),
                    counts["observed"],
                ),
                flush=True,
            )

    if len(clean_features) != counts["reference"] or len(observed_features) != counts["observed"]:
        raise RuntimeError("counterfactual extraction counts are incomplete")
    schedule_unique_views = {
        key: len(views) for key, views in sorted(schedule_route_views.items())
    }
    if is_v2 and (
        not schedule_unique_views or min(schedule_unique_views.values()) < 4
    ):
        raise RuntimeError("v2 schedule failed four-view-per-route-condition gate")
    payload = {
        "schema_version": FEATURE_SHARD_SCHEMA_VERSION,
        "clean_features": clean_features,
        "clean_items": clean_items,
        "observed_features": observed_features,
        "observed_items": observed_items,
        "provenance": {
            "extraction_schema_version": (
                COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION_V2
                if is_v2
                else COUNTERFACTUAL_EXTRACTION_SCHEMA_VERSION
            ),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "ann_file": str(args.ann_file.resolve()),
            "route_manifest": str(args.route_manifest.resolve()),
            "reference_feature_lineage": str(args.reference_feature_shard.resolve()),
            "reference_feature_lineage_sha256": reference_sha,
            "route_quotas": route_quotas,
            "frames_per_route": frames_per_route,
            "counts": counts,
            "family_counts": family_counts,
            "camera_view_names": list(expected_camera_names or ()),
            "backbone": dict(backbone_metadata),
            "storage_dtype": "float16",
            "reference_semantics": "unintervened_mixed_weather",
            "corruption_mask_is_primary_target": False,
            "actual_target_read": False,
            "optimizer_family_leakage": False,
            "view_schedule": (
                dict(view_schedule)
                if is_v2
                else {"version": "route_condition_hash_single/v1"}
            ),
            "view_schedule_counts": schedule_counts,
            "view_schedule_unique_views_per_route_condition": schedule_unique_views,
            "exact_nonzero_presence_label_authorized": False if is_v2 else None,
        },
    }
    summary = save_feature_shard(payload, args.output)
    summary.update(
        {
            "output": str(args.output.resolve()),
            "projected_gib": projected_bytes / (1024**3),
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "family_counts": family_counts,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("COUNTERFACTUAL_EVIDENCE_FEATURE_SHARD_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
