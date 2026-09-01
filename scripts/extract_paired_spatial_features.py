#!/usr/bin/env python3
"""Extract paired clean/corrupt EVAViT patch features for Stage-1 UQ.

The output uses ``PairedSpatialFeatureRecord`` v2 and retains every camera and
patch token.  Its target is explicitly limited to paired cosine
representation-error severity proxy supervision. It intentionally has no
failure-event probability target. No perception-failure, semantic-UQ,
closed-loop-safety, or LLM-understanding claim follows from this dataset.

Real extraction requires CUDA because it runs ORION's frozen EVAViT backbone.
``--mock`` and ``--mock --dry-run`` provide dependency-free CPU schema smoke
tests and never import MMCV, ORION, CARLA, or Transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.corruptions import (  # noqa: E402
    corrupt_multiview_images_with_metadata,
)
from uq_estimator.paired_feature_extraction import (  # noqa: E402
    PAIRED_EXTRACTION_SCHEMA_VERSION,
    RouteFrameIdentity,
    build_info_identity_index,
    camera_view_names_from_info,
    exact_mask_to_patch_coverage,
    feature_map_to_patch_tokens,
    find_info_for_image_meta,
    make_representation_proxy_record,
    resolve_route_frame_identity,
    select_contiguous_route_balanced_infos,
)
from uq_estimator.spatial_training import (  # noqa: E402
    TARGET_REPRESENTATION_PROXY,
    PairedSpatialFeatureRecord,
    load_paired_feature_records,
    save_paired_feature_records,
)


SUPPORTED_CORRUPTIONS = (
    "local_blur",
    "local_dark",
    "local_glare",
    "local_occlusion",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="adzoo/orion/configs/orion_stage3_agent.py"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ann-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--corruption",
        action="append",
        choices=SUPPORTED_CORRUPTIONS,
        dest="corruptions",
        help="Repeat for multiple local corruption families.",
    )
    parser.add_argument(
        "--severities", nargs="+", type=int, default=[1, 2, 3]
    )
    parser.add_argument(
        "--view-indices",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "Camera indices to corrupt; default is index 0. The resolved camera "
            "name/order is recorded from data_infos and is never assumed to be front."
        ),
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=float,
        metavar=("TOP", "LEFT", "BOTTOM", "RIGHT"),
        help="Optional fixed normalized region; omitted regions are seeded per frame.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Required for real extraction to bound the unpooled feature cache.",
    )
    parser.add_argument(
        "--route-manifest",
        type=Path,
        help="Optional v1 manifest for deterministic route-balanced selection.",
    )
    parser.add_argument(
        "--split-route-quota",
        action="append",
        metavar="SPLIT=COUNT",
        help=(
            "Repeat with route quotas, e.g. train=4 validation=2 held_out=2. "
            "Requires --route-manifest and --samples-per-route."
        ),
    )
    parser.add_argument("--samples-per-route", type=int)
    parser.add_argument(
        "--max-output-gb",
        type=float,
        default=5.0,
        help=(
            "Fail before bulk extraction when the first real batch projects a "
            "larger in-memory tensor payload (default: 5 GiB). Increase only "
            "after checking shared quota; this record format duplicates clean tokens."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-samples", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.corruptions = args.corruptions or ["local_blur"]
    if len(set(args.corruptions)) != len(args.corruptions):
        raise SystemExit("--corruption values must not contain duplicates")
    if not args.severities or any(value not in (1, 2, 3) for value in args.severities):
        raise SystemExit("--severities must contain only 1, 2, and/or 3")
    if len(set(args.severities)) != len(args.severities):
        raise SystemExit("--severities must not contain duplicates")
    if not args.view_indices or len(set(args.view_indices)) != len(args.view_indices):
        raise SystemExit("--view-indices must be non-empty and unique")
    if any(value < 0 for value in args.view_indices):
        raise SystemExit("--view-indices must be non-negative")
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("--batch-size must be positive and --workers non-negative")
    if args.mock_samples <= 0:
        raise SystemExit("--mock-samples must be positive")
    if args.max_output_gb <= 0:
        raise SystemExit("--max-output-gb must be positive")
    balanced_options = (
        args.route_manifest is not None,
        bool(args.split_route_quota),
        args.samples_per_route is not None,
    )
    if any(balanced_options) and not all(balanced_options):
        raise SystemExit(
            "route-balanced selection requires --route-manifest, "
            "--split-route-quota, and --samples-per-route together"
        )
    args.split_route_quotas = {}
    for item in args.split_route_quota or []:
        if "=" not in item:
            raise SystemExit("--split-route-quota must use SPLIT=COUNT")
        split, raw_count = item.split("=", 1)
        split = split.strip()
        try:
            count = int(raw_count)
        except ValueError:
            raise SystemExit("--split-route-quota COUNT must be an integer")
        if not split or count <= 0 or split in args.split_route_quotas:
            raise SystemExit("split route quotas must be unique and positive")
        args.split_route_quotas[split] = count
    if args.samples_per_route is not None and args.samples_per_route <= 0:
        raise SystemExit("--samples-per-route must be positive")
    if not args.dry_run and args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    if not args.mock:
        if args.checkpoint is None or args.ann_file is None:
            raise SystemExit("real extraction requires --checkpoint and --ann-file")
        if args.max_samples is None or args.max_samples <= 0:
            raise SystemExit(
                "real extraction requires a positive --max-samples because unpooled "
                "paired tokens are large"
            )
        if not torch.cuda.is_available():
            raise SystemExit("real EVAViT extraction requires CUDA; use --mock for CPU smoke")
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")


def _stable_sample_seed(
    base_seed: int,
    sample_token: str,
    corruption: str,
    severity: int,
) -> int:
    payload = f"{base_seed}|{sample_token}|{corruption}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _first_feature_map(backbone_output: Any) -> torch.Tensor:
    output = backbone_output[0] if isinstance(backbone_output, (tuple, list)) else backbone_output
    if not torch.is_tensor(output) or output.ndim != 4:
        raise RuntimeError("EVAViT must return a 4D feature map as its first output")
    return output


def _extract_tokens(
    backbone: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B,V,3,H,W]")
    batch, views = images.shape[:2]
    feature_map = _first_feature_map(backbone(images.flatten(0, 1)))
    patch_height, patch_width = feature_map.shape[-2:]
    return (
        feature_map_to_patch_tokens(feature_map, batch, views),
        patch_height,
        patch_width,
    )


def _records_for_batch(
    *,
    backbone: torch.nn.Module,
    images: torch.Tensor,
    identities: Sequence[RouteFrameIdentity],
    corruptions: Sequence[str],
    severities: Sequence[int],
    view_indices: Sequence[int],
    region: Optional[Sequence[float]],
    seed: int,
    backbone_metadata: Mapping[str, Any],
) -> list[PairedSpatialFeatureRecord]:
    if len(identities) != images.shape[0]:
        raise ValueError("identity count must equal image batch size")
    with torch.inference_mode():
        clean_tokens, patch_height, patch_width = _extract_tokens(backbone, images)
        records: list[PairedSpatialFeatureRecord] = []
        for corruption in corruptions:
            for severity in severities:
                corrupt_images = []
                pixel_masks = []
                metadata_items = []
                for sample_index, identity in enumerate(identities):
                    result = corrupt_multiview_images_with_metadata(
                        images[sample_index],
                        corruption=corruption,
                        severity=severity,
                        view_indices=view_indices,
                        seed=_stable_sample_seed(
                            seed, identity.sample_token, corruption, severity
                        ),
                        region=region,
                    )
                    corrupt_images.append(result.images)
                    pixel_masks.append(result.mask)
                    metadata_items.append(result.metadata.to_dict())
                corrupt_batch = torch.stack(corrupt_images, dim=0)
                exact_masks = torch.cat(pixel_masks, dim=0)
                corrupt_tokens, corrupt_h, corrupt_w = _extract_tokens(
                    backbone, corrupt_batch
                )
                if (corrupt_h, corrupt_w) != (patch_height, patch_width):
                    raise RuntimeError("clean/corrupt EVAViT patch grids differ")
                coverage = exact_mask_to_patch_coverage(
                    exact_masks, patch_height, patch_width
                )
                for sample_index, identity in enumerate(identities):
                    item_metadata = dict(metadata_items[sample_index])
                    item_metadata["exact_pixel_mask"] = {
                        "shape": list(exact_masks[sample_index].shape),
                        "true_pixel_count": int(exact_masks[sample_index].sum().item()),
                        "reconstructible_from": (
                            "view_indices + parameters.pixel_region + input_image_shape"
                        ),
                    }
                    records.append(
                        make_representation_proxy_record(
                            identity=identity,
                            corruption=corruption,
                            severity=severity,
                            clean_patch_features=clean_tokens[sample_index],
                            corrupt_patch_features=corrupt_tokens[sample_index],
                            patch_corruption_coverage=coverage[sample_index],
                            corruption_metadata=item_metadata,
                            backbone_metadata=backbone_metadata,
                        )
                    )
    return records


class _MockPatchBackbone(torch.nn.Module):
    """Deterministic CPU stand-in that preserves a real patch grid contract."""

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        pooled = F.avg_pool2d(images, kernel_size=4, stride=4)
        # Six deterministic channels, enough to catch accidental pooling or
        # camera-axis collapse in schema tests.
        features = torch.cat((pooled, pooled.square()), dim=1)
        return [features]


def _mock_records(args: argparse.Namespace) -> list[PairedSpatialFeatureRecord]:
    generator = torch.Generator().manual_seed(args.seed)
    images = torch.randn(args.mock_samples, 2, 3, 16, 20, generator=generator)
    identities = [
        RouteFrameIdentity(
            route_id=f"mock_route_{index:03d}",
            town=f"Town{1 + index % 5:02d}",
            frame_idx=index,
            sample_token=f"mock_route_{index:03d}__frame_{index:06d}",
        )
        for index in range(args.mock_samples)
    ]
    if any(view >= images.shape[1] for view in args.view_indices):
        raise SystemExit("mock mode has two views; --view-indices must be 0 or 1")
    return _records_for_batch(
        backbone=_MockPatchBackbone().eval(),
        images=images,
        identities=identities,
        corruptions=args.corruptions,
        severities=args.severities,
        view_indices=args.view_indices,
        region=args.region,
        seed=args.seed,
        backbone_metadata={
            "type": "MockPatchBackbone",
            "frozen": True,
            "real_orion_weights": False,
            "camera_view_names": ["MOCK_CAM_0", "MOCK_CAM_1"],
        },
    )


def _build_real_backbone(args: argparse.Namespace):
    # Lazy imports keep --mock usable in a plain CPU PyTorch environment.
    from mmcv.models.backbones.eva_vit import EVAViT
    from mmcv.utils import Config

    cfg = Config.fromfile(str(args.config))
    backbone_cfg = cfg.model.img_backbone.copy()
    backbone_cfg.pop("type", None)
    backbone_cfg.pop("pretrained", None)
    backbone = EVAViT(**backbone_cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("checkpoint must contain a mapping at 'state_dict'")
    prefix = "img_backbone."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise RuntimeError("checkpoint contains no img_backbone.* tensors")
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    del checkpoint, state_dict, backbone_state
    backbone.requires_grad_(False).cuda().eval()
    metadata = {
        "type": "EVAViT",
        "source": "ORION checkpoint img_backbone.*",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(Path(args.config).resolve()),
        "frozen": True,
        "real_orion_weights": True,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "global_pooling": False,
    }
    return cfg, backbone, metadata


def _real_records(args: argparse.Namespace) -> list[PairedSpatialFeatureRecord]:
    from mmcv.datasets import build_dataloader, build_dataset

    cfg, backbone, backbone_metadata = _build_real_backbone(args)
    cfg.data.test.ann_file = str(args.ann_file)
    dataset = build_dataset(cfg.data.test)
    if args.route_manifest is not None:
        manifest = json.loads(args.route_manifest.read_text(encoding="utf-8"))
        selected_infos = select_contiguous_route_balanced_infos(
            dataset.data_infos,
            manifest,
            args.split_route_quotas,
            args.samples_per_route,
        )
        if len(selected_infos) != args.max_samples:
            raise RuntimeError(
                "balanced selection produced %d frames, but --max-samples=%d; "
                "set max-samples to sum(route quotas) * samples-per-route"
                % (len(selected_infos), args.max_samples)
            )
        dataset.data_infos = selected_infos
    elif len(dataset) < args.max_samples:
        raise RuntimeError(
            f"requested --max-samples={args.max_samples}, dataset has only {len(dataset)}"
        )
    else:
        dataset.data_infos = dataset.data_infos[: args.max_samples]
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

    records: list[PairedSpatialFeatureRecord] = []
    seen_tokens: set[str] = set()
    expected_camera_names: Optional[tuple[str, ...]] = None
    projected_tensor_bytes: Optional[int] = None
    for data in loader:
        image_container = data["img"][0]
        images = image_container.data.cuda(non_blocking=True)
        image_metas = data["img_metas"][0]
        if len(image_metas) != images.shape[0]:
            raise RuntimeError("img_metas count does not equal image batch size")
        identities = []
        for image_meta in image_metas:
            info = find_info_for_image_meta(info_index, image_meta)
            identity = resolve_route_frame_identity(info, image_meta)
            camera_names = camera_view_names_from_info(info)
            if len(camera_names) != images.shape[1]:
                raise RuntimeError(
                    "data_infos camera count does not equal image view dimension"
                )
            if expected_camera_names is None:
                expected_camera_names = camera_names
            elif camera_names != expected_camera_names:
                raise RuntimeError(
                    "camera view order changes across data_infos records"
                )
            if identity.sample_token in seen_tokens:
                raise RuntimeError(
                    f"dataloader yielded duplicate sample token {identity.sample_token!r}"
                )
            seen_tokens.add(identity.sample_token)
            identities.append(identity)
        batch_backbone_metadata = dict(backbone_metadata)
        batch_backbone_metadata["camera_view_names"] = list(expected_camera_names)
        batch_records = _records_for_batch(
            backbone=backbone,
            images=images,
            identities=identities,
            corruptions=args.corruptions,
            severities=args.severities,
            view_indices=args.view_indices,
            region=args.region,
            seed=args.seed,
            backbone_metadata=batch_backbone_metadata,
        )
        if projected_tensor_bytes is None:
            first = batch_records[0]
            bytes_per_record = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    first.observed_patch_features,
                    first.clean_patch_features,
                    first.corruption_mask,
                )
            )
            variants = len(args.corruptions) * len(args.severities)
            projected_tensor_bytes = bytes_per_record * args.max_samples * variants
            limit_bytes = int(args.max_output_gb * (1024**3))
            if projected_tensor_bytes > limit_bytes:
                raise RuntimeError(
                    "projected paired tensor payload is "
                    f"{projected_tensor_bytes / (1024**3):.2f} GiB, above "
                    f"--max-output-gb={args.max_output_gb:g}; use online generation "
                    "or a deduplicated shard format instead of bulk caching"
                )
            print(
                "[PairedExtraction] projected tensor payload "
                f"{projected_tensor_bytes / (1024**3):.2f} GiB "
                f"(limit {args.max_output_gb:g} GiB)",
                flush=True,
            )
        records.extend(batch_records)
    if len(seen_tokens) != args.max_samples:
        raise RuntimeError(
            f"incomplete extraction: saw {len(seen_tokens)}/{args.max_samples} frames"
        )
    return records


def _summary(
    records: Sequence[PairedSpatialFeatureRecord],
    args: argparse.Namespace,
    writes_performed: bool,
) -> dict[str, Any]:
    shapes = sorted({tuple(record.observed_patch_features.shape) for record in records})
    return {
        "schema_version": PAIRED_EXTRACTION_SCHEMA_VERSION,
        "record_schema_version": records[0].schema_version,
        "record_count": len(records),
        "route_count": len({record.route_id for record in records}),
        "pair_count": len({record.pair_id for record in records}),
        "feature_shapes": [list(shape) for shape in shapes],
        "target_provenance": sorted({record.target_provenance for record in records}),
        "representation_error_proxy_only": True,
        "semantic_uq_claim": False,
        "global_pooling": False,
        "mock": bool(args.mock),
        "dry_run": bool(args.dry_run),
        "writes_performed": writes_performed,
        "output": str(args.output.resolve()) if writes_performed else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    records = _mock_records(args) if args.mock else _real_records(args)
    if not records:
        raise RuntimeError("extraction produced no paired records")
    if {record.target_provenance for record in records} != {
        TARGET_REPRESENTATION_PROXY
    }:
        raise RuntimeError("extractor may output only representation-error proxy records")

    writes_performed = False
    if not args.dry_run:
        save_paired_feature_records(args.output, records)
        # Immediate schema round-trip catches unsupported metadata/tensor types.
        loaded = load_paired_feature_records(args.output)
        if len(loaded) != len(records):
            raise RuntimeError("saved paired dataset failed record-count round-trip")
        writes_performed = True
    print(json.dumps(_summary(records, args, writes_performed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
