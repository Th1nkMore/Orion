#!/usr/bin/env python3
"""Verify Qwen NAVSIM pair membership and report paired trajectory changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records


def _by_unique_token(records: list[dict], path: Path) -> dict[str, dict]:
    indexed = {}
    for record in records:
        token = str(record.get("token", ""))
        if not token:
            raise ValueError(f"record without a token in {path}")
        if token in indexed:
            raise ValueError(f"duplicate token {token!r} in {path}")
        indexed[token] = record
    return indexed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(scene_path: Path, clean_path: Path, corrupted_path: Path) -> dict:
    scenes = _read_jsonl(scene_path)
    clean = _by_unique_token(_read_jsonl(clean_path), clean_path)
    corrupted = _by_unique_token(_read_jsonl(corrupted_path), corrupted_path)
    scene_tokens = [str(record.get("meta_info", {}).get("token", "")) for record in scenes]
    if any(not token for token in scene_tokens):
        raise ValueError("scene file contains a record without meta_info.token")
    if len(scene_tokens) != len(set(scene_tokens)):
        raise ValueError("scene file contains duplicate tokens")
    expected = set(scene_tokens)
    if set(clean) != expected or set(corrupted) != expected:
        raise ValueError(
            "pair token mismatch: "
            f"scenes={len(expected)}, clean={len(clean)}, corrupted={len(corrupted)}, "
            f"missing_clean={len(expected - set(clean))}, "
            f"missing_corrupted={len(expected - set(corrupted))}"
        )

    token_rows = []
    for token in scene_tokens:
        clean_trajectory = np.asarray(clean[token]["trajectories"], dtype=np.float32)
        corrupted_trajectory = np.asarray(
            corrupted[token]["trajectories"], dtype=np.float32
        )
        if clean_trajectory.shape != corrupted_trajectory.shape:
            raise ValueError(f"trajectory shape mismatch for token {token}")
        if clean_trajectory.shape[0] != 1:
            raise ValueError(
                f"expected exactly one trajectory sample for token {token}, "
                f"got shape {clean_trajectory.shape}"
            )
        delta = np.linalg.norm(
            corrupted_trajectory[0, :, :2] - clean_trajectory[0, :, :2], axis=1
        )
        token_rows.append(
            {
                "token": token,
                "mean_xy_delta_m": float(np.mean(delta)),
                "max_xy_delta_m": float(np.max(delta)),
                "endpoint_xy_delta_m": float(delta[-1]),
            }
        )

    return {
        "schema": "orion.qwen-drive-navsim-pair-audit/v1",
        "pair_integrity": "pass",
        "num_tokens": len(scene_tokens),
        "num_samples_per_token": 1,
        "scene_sha256": _sha256(scene_path),
        "clean_predictions_sha256": _sha256(clean_path),
        "corrupted_predictions_sha256": _sha256(corrupted_path),
        "mean_of_token_mean_xy_delta_m": float(
            np.mean([row["mean_xy_delta_m"] for row in token_rows])
        ),
        "mean_endpoint_xy_delta_m": float(
            np.mean([row["endpoint_xy_delta_m"] for row in token_rows])
        ),
        "tokens": token_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--corrupted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.scenes, args.clean, args.corrupted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "tokens"}, indent=2))


if __name__ == "__main__":
    main()
