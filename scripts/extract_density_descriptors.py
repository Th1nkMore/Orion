"""Extract compact Density-UQ descriptors directly from ORION images.

This avoids writing the full per-patch EVAViT token cache, which would occupy
hundreds of gigabytes for the Bench2Drive validation subset.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models.backbones.eva_vit import EVAViT
from mmcv.utils import Config
from scripts.extract_orion_features import get_scene_id_and_type
from uq_estimator.density import compute_view_moments


WEATHER_RE = re.compile(r"_Weather(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="adzoo/orion/configs/orion_stage3_agent.py"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_backbone(cfg: Config, checkpoint_path: str) -> EVAViT:
    backbone_cfg = cfg.model.img_backbone.copy()
    backbone_cfg.pop("type", None)
    backbone_cfg.pop("pretrained", None)
    backbone = EVAViT(**backbone_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    prefix = "img_backbone."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    print(
        f"[Descriptors] loaded {len(backbone_state)} backbone tensors; "
        f"missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )
    del checkpoint, state_dict, backbone_state
    return backbone.cuda().eval()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    cfg = Config.fromfile(args.config)
    cfg.data.test.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.test)
    if args.max_samples is not None:
        dataset.data_infos = dataset.data_infos[: args.max_samples]
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f"[Descriptors] dataset size={len(dataset)}", flush=True)
    backbone = build_backbone(cfg, args.checkpoint)

    output_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    descriptors = None
    filenames: list[str] = []
    routes: list[str] = []
    scene_types: list[str] = []
    weather_ids = torch.empty(len(dataset), dtype=torch.int16)
    offset = 0

    with torch.inference_mode():
        for data in tqdm(loader, desc="Extracting Density-UQ descriptors"):
            img_container = data["img"][0]
            images = img_container.data.cuda(non_blocking=True)
            metas = data["img_metas"][0]
            batch_size, views, channels, height, width = images.shape

            features = backbone(images.flatten(0, 1))[0]
            _, dim, feat_height, feat_width = features.shape
            patch_tokens = (
                features.reshape(batch_size, views, dim, feat_height, feat_width)
                .permute(0, 1, 3, 4, 2)
                .reshape(batch_size, views, feat_height * feat_width, dim)
            )
            batch_descriptors = compute_view_moments(patch_tokens).cpu().to(output_dtype)
            if descriptors is None:
                descriptors = torch.empty(
                    (len(dataset), batch_descriptors.shape[-1]), dtype=output_dtype
                )
                print(
                    f"[Descriptors] shape={tuple(descriptors.shape)}, "
                    f"dtype={descriptors.dtype}",
                    flush=True,
                )
            descriptors[offset : offset + batch_size].copy_(batch_descriptors)

            for index, meta in enumerate(metas):
                scene_id, scene_type = get_scene_id_and_type(meta, offset + index)
                match = WEATHER_RE.search(scene_id.rsplit("__", 1)[0])
                if match is None:
                    raise ValueError(f"Cannot parse weather id from {scene_id}")
                route = scene_id.rsplit("__", 1)[0]
                filenames.append(f"{scene_id}.pt")
                routes.append(route)
                scene_types.append(scene_type)
                weather_ids[offset + index] = int(match.group(1))
            offset += batch_size

    if descriptors is None or offset != len(dataset):
        raise RuntimeError(f"Descriptor extraction incomplete: {offset}/{len(dataset)}")
    if any(scene_type == "unknown" for scene_type in scene_types):
        raise ValueError("At least one sample has an unknown scene type")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "descriptors": descriptors,
            "filenames": filenames,
            "routes": routes,
            "weather_ids": weather_ids,
            "scene_types": scene_types,
            "descriptor": "per_view_patch_mean_std",
            "source": "direct_evavit_extraction",
        },
        output,
    )
    size_mb = output.stat().st_size / 1e6
    print(
        f"[Descriptors] saved {offset} descriptors to {output} ({size_mb:.1f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
