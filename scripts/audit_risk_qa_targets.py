"""Audit structured Risk QA targets before model training."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from transformers import AutoTokenizer

from uq_estimator.density import DensityUQEstimator
from uq_estimator.risk_qa import (
    RISK_QA_QUESTION,
    build_risk_qa_answer,
    parse_risk_qa_answer,
    render_risk_qa_answer,
    select_critical_objects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--descriptor-cache", required=True)
    parser.add_argument("--density-checkpoint", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sample_id(info: dict) -> str:
    route = Path(info["folder"]).name
    return f"{route}__{int(info['frame_idx']):05d}.pt"


def main() -> None:
    args = parse_args()
    with open(args.ann_file, "rb") as handle:
        infos = pickle.load(handle)
    cache = torch.load(
        args.descriptor_cache, map_location="cpu", weights_only=True
    )
    density_payload = torch.load(
        args.density_checkpoint, map_location="cpu", weights_only=True
    )
    assignment = density_payload["split_assignment"]
    density = DensityUQEstimator.from_checkpoint(density_payload).eval()

    descriptor_by_name = {
        filename: cache["descriptors"][index].float()
        for index, filename in enumerate(cache["filenames"])
    }
    selected = [
        info for info in infos
        if assignment.get(Path(info["folder"]).name) == args.split
        and sample_id(info) in descriptor_by_name
    ][:args.max_samples]
    if not selected:
        raise RuntimeError(f"No samples matched split {args.split!r}")

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=False
        )

    records = []
    with torch.no_grad():
        for info in selected:
            name = sample_id(info)
            _, score, _, _ = density.encode_descriptor(
                descriptor_by_name[name].unsqueeze(0)
            )
            objects = select_critical_objects(
                info["gt_boxes"],
                info["gt_names"],
            )
            answer = build_risk_qa_answer(float(score.item()), objects)
            rendered = render_risk_qa_answer(answer)
            parsed = parse_risk_qa_answer(rendered)
            text = f"{RISK_QA_QUESTION}\n{rendered}"
            token_count = (
                len(tokenizer(text, add_special_tokens=True).input_ids)
                if tokenizer is not None else None
            )
            records.append(
                {
                    "sample_id": name,
                    "uq_score": float(score.item()),
                    "target": rendered,
                    "parsed_percentile": parsed.reliability_percentile,
                    "critical_object_count": len(objects),
                    "token_count": token_count,
                }
            )

    object_coverage = sum(
        item["critical_object_count"] > 0 for item in records
    ) / len(records)
    token_counts = [
        item["token_count"] for item in records
        if item["token_count"] is not None
    ]
    summary = {
        "split": args.split,
        "count": len(records),
        "parse_success_rate": 1.0,
        "critical_object_coverage": object_coverage,
        "max_token_count": max(token_counts) if token_counts else None,
        "mean_token_count": (
            sum(token_counts) / len(token_counts) if token_counts else None
        ),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "records"},
        indent=2,
    ))


if __name__ == "__main__":
    main()

