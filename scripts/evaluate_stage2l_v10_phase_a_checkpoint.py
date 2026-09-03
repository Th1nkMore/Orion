#!/usr/bin/env python3
"""Zero-optimizer-step replay of a failed Stage2-L v10 Phase-A checkpoint.

The evaluator restores the exact v10 LoRA/R-map state, replays every frozen
17-event visual context, and exports threshold-free spatial diagnostics.  It
never constructs an optimizer and cannot unlock Phase B, Phase C, formal
Stage2-L, Stage2-P, or closed loop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "orion.stage2l_v10_phase_a_checkpoint_replay.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v10_phase_a_replay_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v10_phase_a_replay_preflight.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
VIEW_NAMES = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _support_mask(target: np.ndarray, support_fraction: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    if target.ndim < 2 or not np.isfinite(target).all():
        raise ValueError("target must be a finite batched dense array")
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support fraction must lie in (0,1)")
    flat = target.reshape(target.shape[0], -1)
    peaks = flat.max(axis=1)
    if np.any(peaks <= 0.0):
        raise ValueError("every target requires positive support")
    shape = (target.shape[0],) + (1,) * (target.ndim - 1)
    return target >= (peaks.reshape(shape) * float(support_fraction))


def _average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError("scores/labels are malformed")
    positives = int(truth.sum())
    if positives == 0:
        raise ValueError("average precision requires positives")
    order = np.argsort(-scores, kind="mergesort")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


def _threshold_row(
    probabilities: np.ndarray,
    foreground: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = np.asarray(probabilities) >= float(threshold)
    foreground = np.asarray(foreground, dtype=bool)
    if predicted.shape != foreground.shape:
        raise ValueError("prediction/support shapes differ")
    background = ~foreground
    tp = int((predicted & foreground).sum())
    fp = int((predicted & background).sum())
    fn = int((~predicted & foreground).sum())
    tn = int((~predicted & background).sum())
    return {
        "threshold": float(threshold),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "background_fpr": float(fp / (fp + tn)),
    }


def spatial_diagnostics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    support_fraction: float,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape or not np.isfinite(probabilities).all():
        raise ValueError("probabilities/targets are malformed")
    foreground = _support_mask(targets, support_fraction)
    background = ~foreground
    foreground_values = probabilities[foreground]
    background_values = probabilities[background]
    frozen_thresholds = targets.reshape(targets.shape[0], -1).max(axis=1)
    frozen_thresholds *= float(support_fraction)
    threshold_shape = (targets.shape[0],) + (1,) * (targets.ndim - 1)
    frozen_prediction = probabilities >= frozen_thresholds.reshape(threshold_shape)
    tp = int((frozen_prediction & foreground).sum())
    fp = int((frozen_prediction & background).sum())
    fn = int((~frozen_prediction & foreground).sum())
    tn = int((~frozen_prediction & background).sum())

    def quantile_payload(values: np.ndarray) -> dict[str, float]:
        return {
            ("q%03d" % round(q * 100)): float(np.quantile(values, q))
            for q in QUANTILES
        }

    return {
        "cell_count": int(probabilities.size),
        "foreground_cell_count": int(foreground.sum()),
        "background_cell_count": int(background.sum()),
        "average_precision": _average_precision(probabilities, foreground),
        "probability_quantiles": {
            "foreground": quantile_payload(foreground_values),
            "background": quantile_payload(background_values),
        },
        "frozen_target_relative_threshold": {
            "support_fraction_of_peak": float(support_fraction),
            "mean_absolute_threshold": float(frozen_thresholds.mean()),
            "recall": float(tp / (tp + fn)),
            "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
            "background_fpr": float(fp / (fp + tn)),
        },
        "absolute_threshold_sweep": [
            _threshold_row(probabilities, foreground, threshold)
            for threshold in thresholds
        ],
    }


def _render_group_maps(
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    support_fraction: float,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = np.asarray(probabilities)[0]
    targets = np.asarray(targets)[0]
    support = _support_mask(targets[None], support_fraction)[0]
    figure, axes = plt.subplots(3, 6, figsize=(14.4, 7.4), constrained_layout=True)
    for view in range(6):
        axes[0, view].imshow(probabilities[view], vmin=0.0, vmax=1.0, cmap="magma")
        axes[1, view].imshow(targets[view], vmin=0.0, vmax=1.0, cmap="viridis")
        overlay = np.zeros((*support[view].shape, 3), dtype=np.float32)
        overlay[..., 0] = support[view].astype(np.float32)
        overlay[..., 1] = probabilities[view].astype(np.float32)
        axes[2, view].imshow(overlay, vmin=0.0, vmax=1.0)
        axes[0, view].set_title(VIEW_NAMES[view], fontsize=8)
        for row in range(3):
            axes[row, view].set_xticks([])
            axes[row, view].set_yticks([])
    for row, label in enumerate(("predicted R probability", "continuous R target", "red=support, green=predicted R")):
        axes[row, 0].set_ylabel(label, fontsize=8)
    figure.suptitle(title, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _validated_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "orion_config_sha256": _sha256(args.config.resolve()),
        "orion_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "u_tokenizer_checkpoint_sha256": _sha256(args.u_tokenizer_checkpoint.resolve()),
        "phase_a_checkpoint_sha256": _sha256(args.phase_a_checkpoint.resolve()),
        "v10_report_sha256": _sha256(args.v10_report.resolve()),
    }


def _validate_static_inputs(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    expected = _validated_inputs(args)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported replay protocol")
    if protocol.get("validated_inputs") != expected:
        raise ValueError("replay protocol input hashes differ")
    if protocol.get("optimizer_steps") != 0:
        raise ValueError("checkpoint replay must keep optimizer_steps=0")
    locks = protocol.get("locks", {})
    required_false = (
        "checkpoint_update",
        "phase_b",
        "phase_c",
        "formal_stage2l",
        "stage2p",
        "closed_loop",
        "route203_native_glare_submission",
    )
    if any(locks.get(key) is not False for key in required_false):
        raise ValueError("replay protocol expands a locked scope")


def _preflight(args: argparse.Namespace, protocol: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = __import__("torch").load(args.phase_a_checkpoint, map_location="cpu")
    report = _read_json(args.v10_report)
    if (
        checkpoint.get("schema") != "orion.stage2l_v10_staged_smoke.v1"
        or checkpoint.get("status") != "phase_a_failed_gate"
        or checkpoint.get("completed_phases") != []
        or checkpoint.get("uq_tokenizer_frozen") is not True
        or checkpoint.get("formal_stage2l_ready") is not False
        or checkpoint.get("stage2p_ready") is not False
        or report.get("status") != "stopped_after_phase_a_failed_gate"
    ):
        raise ValueError("expected the scientifically stopped v10 Phase-A lineage")
    tensor_sections = ("uq_tokenizer", "relevance_queries", "relevance_head", "risk_bridge", "lora")
    counts = {key: len(checkpoint.get(key, {})) for key in tensor_sections}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Phase-A checkpoint tensor section is empty")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "phase_a_replay_preflight_pass_evaluation_locked",
        "passed": True,
        "optimizer_steps": 0,
        "gpu_used": False,
        "training_started": False,
        "validated_inputs": _validated_inputs(args),
        "protocol_sha256": _sha256(args.protocol.resolve()),
        "checkpoint_tensor_counts": counts,
        "output_root": str(args.output_dir.resolve()),
        "locks": dict(protocol["locks"]),
    }


def _validate_launch(args: argparse.Namespace, preflight: Mapping[str, Any]) -> None:
    amendment = _read_json(args.launch_amendment.resolve())
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("optimizer_steps") != 0
        or preflight.get("validated_inputs") != _validated_inputs(args)
        or preflight.get("protocol_sha256") != _sha256(args.protocol.resolve())
        or preflight.get("output_root") != str(args.output_dir.resolve())
    ):
        raise ValueError("Phase-A replay preflight is stale")
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("status") != "immutable_evaluation_only_authorization"
        or amendment.get("validated_inputs") != _validated_inputs(args)
        or amendment.get("protocol_sha256") != _sha256(args.protocol.resolve())
        or amendment.get("preflight_sha256") != _sha256(args.preflight.resolve())
        or amendment.get("authorized_run", {}).get("optimizer_steps") != 0
        or amendment.get("authorized_run", {}).get("maximum_submissions") != 1
        or amendment.get("authorized_run", {}).get("automatic_retry") is not False
        or amendment.get("authorized_run", {}).get("output_root")
        != str(args.output_dir.resolve())
    ):
        raise ValueError("Phase-A replay amendment is absent or stale")


def _load_exact_state(module, state: Mapping[str, Any], *, name: str) -> None:
    current = module.state_dict()
    if set(current) != set(state):
        missing = sorted(set(current) - set(state))
        extra = sorted(set(state) - set(current))
        raise ValueError("%s state keys differ missing=%s extra=%s" % (name, missing[:5], extra[:5]))
    module.load_state_dict(state, strict=True)


def _load_exact_lora(lm, state: Mapping[str, Any]) -> None:
    current = {key for key in lm.state_dict() if "lora_" in key}
    if current != set(state):
        raise ValueError("Phase-A LoRA keys differ from the current ORION graph")
    result = lm.load_state_dict(state, strict=False)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Phase-A LoRA restore was incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-a-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prerequisites = (
        args.config,
        args.checkpoint,
        args.dataset_manifest,
        args.u_tokenizer_checkpoint,
        args.phase_a_checkpoint,
        args.v10_report,
        args.protocol,
    )
    if not all(path.is_file() for path in prerequisites):
        raise FileNotFoundError("Phase-A replay prerequisite is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Phase-A replay output")
    protocol = _read_json(args.protocol.resolve())
    _validate_static_inputs(args, protocol)
    if args.preflight_only:
        if args.preflight is not None or args.launch_amendment is not None:
            raise ValueError("preflight-only mode cannot consume launch artifacts")
        if args.preflight_output is None or args.preflight_output.exists():
            raise ValueError("preflight-only mode requires a fresh output")
        value = _preflight(args, protocol)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.preflight_output is not None or args.preflight is None or args.launch_amendment is None:
        raise ValueError("real replay requires preflight and launch amendment")
    _validate_launch(args, _read_json(args.preflight.resolve()))

    import torch
    from mmcv.utils import set_random_seed
    from uq_estimator.uq_relevance_tokenizer import (
        SpatialTaskRelevanceQueryTokenizer,
        TaskRelevanceMapHead,
    )
    import scripts.train_stage2l_v10_staged_smoke as trainer

    if not torch.cuda.is_available():
        raise RuntimeError("real Phase-A replay requires CUDA")
    set_random_seed(20260831, deterministic=True)
    trainer._load_base(require_real_agent=True)
    trainer._configure_assets()
    assets = trainer.base.MultiRouteAssets(args.dataset_manifest.resolve())
    uq_tokenizer = trainer._load_frozen_u_tokenizer(args.u_tokenizer_checkpoint.resolve())
    lm, text_tokenizer = trainer.base._load_orion_lm(args.config.resolve(), args.checkpoint.resolve())
    checkpoint = torch.load(args.phase_a_checkpoint.resolve(), map_location="cpu")
    if checkpoint["frozen_u_tokenizer_sha256"] != _sha256(args.u_tokenizer_checkpoint.resolve()):
        raise ValueError("Phase-A checkpoint references a different U tokenizer")
    _load_exact_state(uq_tokenizer, checkpoint["uq_tokenizer"], name="uq_tokenizer")
    relevance_queries = SpatialTaskRelevanceQueryTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10), max_views=6
    )
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256)
    _load_exact_state(relevance_queries, checkpoint["relevance_queries"], name="relevance_queries")
    _load_exact_state(relevance_head, checkpoint["relevance_head"], name="relevance_head")
    _load_exact_lora(lm, checkpoint["lora"])
    modules = (lm, uq_tokenizer, relevance_queries, relevance_head)
    for module in modules:
        module.requires_grad_(False)
        module.cuda().eval()

    support_fraction = float(protocol["support_fraction_of_peak"])
    thresholds = np.linspace(0.0, 1.0, int(protocol["absolute_threshold_points"])).tolist()
    raw: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    with torch.inference_mode():
        for split in ("train", "dev"):
            for group_id in assets.groups_for_split(split):
                logits, target, _ = trainer._map_logits(
                    lm=lm,
                    text_tokenizer=text_tokenizer,
                    relevance_queries=relevance_queries,
                    relevance_head=relevance_head,
                    assets=assets,
                    group_id=group_id,
                )
                probabilities = logits.sigmoid().float().cpu().numpy()
                targets = target.float().cpu().numpy()
                event_id = assets.group_event[group_id]
                grouped[event_id].append(group_id)
                raw[group_id] = {
                    "split": split,
                    "event_id": event_id,
                    "probabilities": probabilities,
                    "targets": targets,
                }
                _render_group_maps(
                    probabilities=probabilities,
                    targets=targets,
                    support_fraction=support_fraction,
                    output=args.output_dir / "maps" / event_id / (group_id + ".png"),
                    title="%s | %s | %s" % (split, event_id, group_id),
                )

    per_group = {}
    per_event = {}
    for group_id, row in sorted(raw.items()):
        per_group[group_id] = {
            "split": row["split"],
            "event_id": row["event_id"],
            "diagnostics": spatial_diagnostics(
                row["probabilities"], row["targets"],
                support_fraction=support_fraction, thresholds=thresholds,
            ),
            "map_png": str((args.output_dir / "maps" / row["event_id"] / (group_id + ".png")).resolve()),
        }
    for event_id, group_ids in sorted(grouped.items()):
        probabilities = np.concatenate([raw[value]["probabilities"] for value in group_ids], axis=0)
        targets = np.concatenate([raw[value]["targets"] for value in group_ids], axis=0)
        per_event[event_id] = {
            "split": raw[group_ids[0]]["split"],
            "group_ids": sorted(group_ids),
            "diagnostics": spatial_diagnostics(
                probabilities, targets,
                support_fraction=support_fraction, thresholds=thresholds,
            ),
        }
    per_split = {}
    for split in ("train", "dev"):
        group_ids = [key for key, value in raw.items() if value["split"] == split]
        probabilities = np.concatenate([raw[value]["probabilities"] for value in group_ids], axis=0)
        targets = np.concatenate([raw[value]["targets"] for value in group_ids], axis=0)
        per_split[split] = spatial_diagnostics(
            probabilities, targets,
            support_fraction=support_fraction, thresholds=thresholds,
        )
    report = {
        "schema": SCHEMA,
        "status": "phase_a_checkpoint_replay_complete_evaluation_only",
        "optimizer_steps": 0,
        "checkpoint_updated": False,
        "group_count": len(raw),
        "event_count": len(grouped),
        "per_split": per_split,
        "per_event": per_event,
        "per_group": per_group,
        "provenance": {
            "validated_inputs": _validated_inputs(args),
            "protocol_sha256": _sha256(args.protocol.resolve()),
            "preflight_sha256": _sha256(args.preflight.resolve()),
            "launch_amendment_sha256": _sha256(args.launch_amendment.resolve()),
        },
        "locks": dict(protocol["locks"]),
        "claim_boundary": (
            "Evaluation-only diagnosis of a failed engineering checkpoint; not a "
            "gate revision, training result, formal generalization result, planning "
            "result, closed-loop result, or safety claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "event_count": report["event_count"],
        "group_count": report["group_count"],
        "dev_average_precision": report["per_split"]["dev"]["average_precision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
