#!/usr/bin/env python3
"""Train the standalone path-risk trajectory adapter on oracle rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.oracle_adapter_training import (
    assign_rollout_roles,
    load_exported_samples,
    make_mock_samples,
    prepare_examples,
    split_examples_by_route,
    train_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train PathRiskTrajectoryAdapter; failed/off rollout trajectories "
            "are diagnostic-only and cannot enter imitation"
        )
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-residual-m", type=float, default=2.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--lambda-imitation", type=float, default=1.0)
    parser.add_argument("--lambda-preservation", type=float, default=0.5)
    parser.add_argument("--lambda-stop", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one-epoch pipeline check; allows an explicitly non-disjoint one-route split",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use a deterministic two-route in-memory mock dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock and args.dataset_dir is not None:
        raise SystemExit("--mock and --dataset-dir are mutually exclusive")
    if not args.mock and args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required unless --mock is used")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    samples = make_mock_samples(args.seed) if args.mock else load_exported_samples(
        args.dataset_dir
    )
    roles, role_audit = assign_rollout_roles(samples)
    examples, example_audit = prepare_examples(samples, roles)
    split = split_examples_by_route(
        examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        allow_single_route_smoke=args.smoke,
    )
    epochs = 1 if args.smoke else args.epochs
    max_steps = 3 if args.smoke else None
    model, history, best_validation = train_adapter(
        examples,
        split,
        hidden_dim=args.hidden_dim,
        max_residual_m=args.max_residual_m,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_imitation=args.lambda_imitation,
        lambda_preservation=args.lambda_preservation,
        lambda_stop=args.lambda_stop,
        device=device,
        seed=args.seed,
        max_steps_per_epoch=max_steps,
    )
    payload = {
        "format_version": "orion.path_risk_trajectory_adapter.v1",
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "model_config": {
            "context_dim": 0,
            "hidden_dim": args.hidden_dim,
            "max_residual_m": args.max_residual_m,
        },
        "training_config": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lambda_imitation": args.lambda_imitation,
            "lambda_preservation": args.lambda_preservation,
            "lambda_stop": args.lambda_stop,
            "seed": args.seed,
            "device": device,
            "smoke": bool(args.smoke),
            "mock": bool(args.mock),
        },
        "role_policy": {
            "positive_imitation": (
                "eligible completed zero-collision hazard/on-path oracle rollout "
                "with a controlled-stop label; only active path-risk frames"
            ),
            "failed_or_off": "diagnostic-only; zero trajectory and stop loss weight",
            "preservation": "base ORION trajectory target; never executed failure trajectory",
        },
        "role_audit": role_audit,
        "example_audit": example_audit,
        "split": {
            "mode": split.mode,
            "train_routes": split.train_routes,
            "validation_routes": split.validation_routes,
            "train_examples": len(split.train_indices),
            "validation_examples": len(split.validation_indices),
        },
        "best_validation_loss": best_validation,
        "history": history,
        "interpretation": (
            "Pipeline/training artifact only. Offline loss reduction is not evidence "
            "of closed-loop safety or learned-UQ validity."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary = {key: value for key, value in payload.items() if key != "model_state"}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.out),
                "summary": str(summary_path),
                "split_mode": split.mode,
                "best_validation_loss": best_validation,
                "role_audit": role_audit,
                "example_audit": example_audit,
                "final": history[-1],
                "claim_allowed": False if args.smoke or args.mock else "closed_loop_gate_required",
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

