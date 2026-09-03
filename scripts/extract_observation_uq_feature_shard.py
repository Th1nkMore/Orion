#!/usr/bin/env python3
"""Extract a deduplicated FP16 clean-first EVAViT feature shard.

Clean features are stored once.  Diagnostic observations are generated only
for declared non-training splits, so a larger clean Teacher corpus does not
multiply storage by the number of corruption families or severities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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


def _quota(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("quota must use SPLIT=COUNT")
    split, raw_count = value.split("=", 1)
    try:
        count = int(raw_count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("quota count must be an integer") from error
    if not split.strip() or count <= 0:
        raise argparse.ArgumentTypeError("quota split/count must be non-empty and positive")
    return split.strip(), count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="adzoo/orion/configs/orion_stage3_agent.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-route-quota",
        action="append",
        type=_quota,
        default=[],
        help="Repeat SPLIT=COUNT; default train=35 validation=5 held_out=5.",
    )
    parser.add_argument("--samples-per-route", type=int, default=16)
    parser.add_argument(
        "--diagnostic-split",
        action="append",
        default=[],
        help="Splits receiving held-out observations; default validation+held_out.",
    )
    parser.add_argument(
        "--diagnostic-corruption",
        action="append",
        choices=("local_blur", "local_dark", "local_glare", "local_occlusion"),
        default=[],
    )
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--view-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-output-gb", type=float, default=25.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.split_route_quotas = dict(
        args.split_route_quota
        or [("train", 35), ("validation", 5), ("held_out", 5)]
    )
    if len(args.split_route_quotas) != len(
        args.split_route_quota or args.split_route_quotas
    ):
        raise SystemExit("split route quotas must be unique")
    args.diagnostic_splits = tuple(
        args.diagnostic_split or ("validation", "held_out")
    )
    args.diagnostic_corruptions = tuple(
        args.diagnostic_corruption or ("local_glare",)
    )
    if not args.diagnostic_splits or len(set(args.diagnostic_splits)) != len(
        args.diagnostic_splits
    ):
        raise SystemExit("diagnostic splits must be non-empty and unique")
    if any(split == "train" for split in args.diagnostic_splits):
        raise SystemExit("Teacher viability shard prohibits train-split corrupt observations")
    if len(set(args.diagnostic_corruptions)) != len(args.diagnostic_corruptions):
        raise SystemExit("diagnostic corruptions must be unique")
    if args.samples_per_route <= 1:
        raise SystemExit("samples-per-route must exceed one for temporal context")
    if not args.severities or any(value not in (1, 2, 3) for value in args.severities):
        raise SystemExit("severities must contain only 1, 2, and/or 3")
    if len(set(args.severities)) != len(args.severities):
        raise SystemExit("severities must be unique")
    if not args.view_indices or len(set(args.view_indices)) != len(args.view_indices):
        raise SystemExit("view indices must be non-empty and unique")
    if args.batch_size <= 0 or args.workers < 0 or args.max_output_gb <= 0:
        raise SystemExit("batch/workers/output guard values are invalid")
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if not torch.cuda.is_available():
        raise SystemExit("real feature shard extraction requires CUDA")


def _to_feature_grid(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise RuntimeError("unexpected EVAViT token shape")
    return tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1]).half().cpu()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    manifest_payload = json.loads(args.route_manifest.read_text(encoding="utf-8"))
    route_splits = route_splits_from_manifest(manifest_payload)

    from mmcv.datasets import build_dataloader, build_dataset

    cfg, backbone, backbone_metadata = _build_real_backbone(args)
    cfg.data.test.ann_file = str(args.ann_file)
    dataset = build_dataset(cfg.data.test)
    selected_infos = select_contiguous_route_balanced_infos(
        dataset.data_infos,
        manifest_payload,
        args.split_route_quotas,
        args.samples_per_route,
    )
    expected_clean_count = sum(args.split_route_quotas.values()) * args.samples_per_route
    if len(selected_infos) != expected_clean_count:
        raise RuntimeError("route-balanced clean selection count is inconsistent")
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
                raise RuntimeError("camera order changes inside selected shard")
            if len(camera_names) != images.shape[1]:
                raise RuntimeError("camera metadata count does not match image tensor")
            identities.append(identity)
            splits.append(route_splits[identity.route_id])

        with torch.inference_mode():
            clean_tokens, patch_height, patch_width = _extract_tokens(backbone, images)
        if projected_bytes is None:
            bytes_per_feature = (
                clean_tokens[0].numel() * torch.tensor([], dtype=torch.float16).element_size()
            )
            diagnostic_clean_count = sum(
                args.split_route_quotas[split] * args.samples_per_route
                for split in args.diagnostic_splits
            )
            expected_observed = (
                diagnostic_clean_count
                * len(args.diagnostic_corruptions)
                * len(args.severities)
            )
            projected_bytes = bytes_per_feature * (expected_clean_count + expected_observed)
            limit = int(args.max_output_gb * (1024**3))
            if projected_bytes > limit:
                raise RuntimeError(
                    "projected FP16 shard %.2f GiB exceeds --max-output-gb=%g"
                    % (projected_bytes / (1024**3), args.max_output_gb)
                )
            print(
                "[ObservationUQShard] clean=%d observed=%d projected=%.2f GiB"
                % (expected_clean_count, expected_observed, projected_bytes / (1024**3)),
                flush=True,
            )

        for sample_index, (identity, split) in enumerate(zip(identities, splits)):
            clean_index = len(clean_features)
            clean_grid = _to_feature_grid(
                clean_tokens[sample_index], patch_height, patch_width
            )
            clean_features.append(clean_grid)
            clean_items.append(
                {
                    "clean_index": clean_index,
                    "sample_id": identity.sample_token,
                    "route_id": identity.route_id,
                    "town": identity.town,
                    "frame_idx": identity.frame_idx,
                    "split": split,
                }
            )
            if split not in args.diagnostic_splits:
                continue
            for family in args.diagnostic_corruptions:
                for severity in args.severities:
                    corruption = corrupt_multiview_images_with_metadata(
                        images[sample_index],
                        corruption=family,
                        severity=severity,
                        view_indices=args.view_indices,
                        seed=_stable_sample_seed(
                            args.seed, identity.sample_token, family, severity
                        ),
                    )
                    with torch.inference_mode():
                        observed_tokens, observed_h, observed_w = _extract_tokens(
                            backbone, corruption.images.unsqueeze(0)
                        )
                    if (observed_h, observed_w) != (patch_height, patch_width):
                        raise RuntimeError("clean/observed patch grids differ")
                    coverage = exact_mask_to_patch_coverage(
                        corruption.mask, patch_height, patch_width
                    )[0].reshape(images.shape[1], patch_height, patch_width)
                    observed_index = len(observed_features)
                    observed_features.append(
                        _to_feature_grid(observed_tokens[0], patch_height, patch_width)
                    )
                    observed_items.append(
                        {
                            "observed_index": observed_index,
                            "clean_index": clean_index,
                            "sample_id": "%s/%s/severity_%d"
                            % (identity.sample_token, family, severity),
                            "route_id": identity.route_id,
                            "town": identity.town,
                            "frame_idx": identity.frame_idx,
                            "split": split,
                            "family": family,
                            "severity": float(severity),
                            "corruption_mask": coverage.half().cpu(),
                            "corruption_metadata": corruption.metadata.to_dict(),
                        }
                    )
        if len(clean_features) % 50 == 0 or len(clean_features) == expected_clean_count:
            print(
                "[ObservationUQShard] progress clean=%d/%d observed=%d"
                % (len(clean_features), expected_clean_count, len(observed_features)),
                flush=True,
            )
    if len(clean_features) != expected_clean_count:
        raise RuntimeError("incomplete clean feature extraction")
    payload = {
        "schema_version": FEATURE_SHARD_SCHEMA_VERSION,
        "clean_features": clean_features,
        "clean_items": clean_items,
        "observed_features": observed_features,
        "observed_items": observed_items,
        "provenance": {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "ann_file": str(args.ann_file.resolve()),
            "route_manifest": str(args.route_manifest.resolve()),
            "route_quotas": dict(args.split_route_quotas),
            "samples_per_route": args.samples_per_route,
            "diagnostic_splits": list(args.diagnostic_splits),
            "diagnostic_corruptions": list(args.diagnostic_corruptions),
            "severities": list(args.severities),
            "view_indices": list(args.view_indices),
            "camera_view_names": list(expected_camera_names or ()),
            "backbone": dict(backbone_metadata),
            "clean_token_deduplicated": True,
            "storage_dtype": "float16",
            "synthetic_labels_are_targets": False,
        },
    }
    summary = save_feature_shard(payload, args.output)
    summary.update(
        {
            "schema_version": FEATURE_SHARD_SCHEMA_VERSION,
            "output": str(args.output.resolve()),
            "projected_gib": projected_bytes / (1024**3),
            "writes_performed": True,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
