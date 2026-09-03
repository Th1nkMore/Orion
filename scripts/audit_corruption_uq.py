"""Audit whether deterministic image corruptions increase Density UQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.utils import Config, load_checkpoint, set_random_seed

from scripts.eval_risk_qa import prepare_visual_context
from scripts.train_film import custom_wrap_fp16_model
from scripts.train_uq_token import filter_dataset_by_split, sample_id_from_batch
from uq_estimator.corruptions import corrupt_batch_images
from uq_estimator.density import get_uq_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="adzoo/orion/configs/orion_stage3_infer.py"
    )
    parser.add_argument("--checkpoint", default="ckpts/Orion.pth")
    parser.add_argument(
        "--density-checkpoint", default="checkpoints/density_uq/best.pt"
    )
    parser.add_argument("--ann-file", default="data/infos/b2d_infos_val.pkl")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(records: list[dict], key: str) -> dict:
    clean = np.asarray([record["clean"] for record in records])
    corrupt = np.asarray([record[key] for record in records])
    delta = corrupt - clean
    return {
        "count": len(records),
        "clean_mean": float(clean.mean()),
        "corrupted_mean": float(corrupt.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "increase_rate": float((delta > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)
    density_payload = torch.load(
        args.density_checkpoint, map_location="cpu", weights_only=True
    )
    assignment = density_payload["split_assignment"]
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.frozen = True
    cfg.model.use_lora = True
    cfg.model.use_uq_token = True
    cfg.model.pts_bbox_head.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = args.density_checkpoint
    cfg.model.pts_bbox_head.transformer.use_uncertainty = False
    cfg.data.test.ann_file = args.ann_file

    dataset = build_dataset(cfg.data.test)
    filter_dataset_by_split(dataset, assignment, args.split)
    dataset.data_infos = dataset.data_infos[:args.max_samples]
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        custom_wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.pts_bbox_head.uq_estimator.load_state_dict(
        get_uq_state_dict(density_payload), strict=False
    )
    model.cuda().eval()
    model.pts_bbox_head.with_dn = False
    model.pts_bbox_head.uq_estimator.eval()

    settings = [
        (name, severity)
        for name in ("blur", "dark", "camera_dropout")
        for severity in (1, 2, 3)
    ]
    records = []
    with torch.no_grad():
        for batch in loader:
            record = {"sample_id": sample_id_from_batch(batch)}
            record["clean"] = float(prepare_visual_context(model, batch)[2])
            for name, severity in settings:
                corrupted = corrupt_batch_images(batch, name, severity)
                record[f"{name}_{severity}"] = float(
                    prepare_visual_context(model, corrupted)[2]
                )
            records.append(record)
            print(f"[CorruptionAudit] evaluated {record['sample_id']}")

    summary = {
        f"{name}_{severity}": summarize(
            records, f"{name}_{severity}"
        )
        for name, severity in settings
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print(f"[CorruptionAudit] saved {output}")


if __name__ == "__main__":
    main()
