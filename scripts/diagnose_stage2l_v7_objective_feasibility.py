#!/usr/bin/env python3
"""CPU-only feasibility diagnostic for corrected Stage2-L v7 objectives.

This deliberately does not load ORION or any visual-language weights.  It
optimizes one explicit R logit map per Route151 keyframe and a tiny shared
stance head on the resulting K summaries.  Passing proves only that the
supervision geometry, calibrated losses and balanced gates are mutually
attainable on the five smoke groups.  It is a prerequisite for, not evidence
from, another real-ORION smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.scenario_factory_lib import sha256_file
from uq_estimator.stage2l_calibrated_objective import (
    SCHEMA as OBJECTIVE_SCHEMA,
    class_balanced_matched_stance_loss,
    foreground_balanced_relevance_terms,
    geometry_normalized_task_risk_ranking_terms,
    matched_stance_metrics,
    relevance_support_metrics,
)
from uq_estimator.stage2l_matched_objective import (
    HARD_STANCE_VARIANTS,
    partition_complete_matched_groups,
)


SCHEMA = "orion.stage2l_v7_objective_feasibility.v1"
EXPECTED_EVENT_ID = "route151_step218"


def _load_records(path: Path) -> Tuple[Tuple[Mapping[str, Any], ...], ...]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = partition_complete_matched_groups(rows)
    if len(groups) != 5:
        raise ValueError("Route151 objective diagnostic requires exactly five groups")
    if {str(row.get("event_id", "")) for row in rows} != {EXPECTED_EVENT_ID}:
        raise ValueError("objective diagnostic is bound to Route151")
    return groups


def _resolve(reference: Mapping[str, Any], base_dir: Path) -> Path:
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(reference["sha256"]):
        raise ValueError("artifact SHA-256 mismatch: %s" % path)
    return path


def _row(
    group: Tuple[Mapping[str, Any], ...], variant: str, family: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in group
        if str(row["counterfactual"]["variant"]) == variant
        and str(row["question_family"]) == family
    ]
    if len(matches) != 1:
        raise ValueError("group lacks a unique variant/family record")
    return matches[0]


def _pooled_scalar_uq(reference: Mapping[str, Any], base_dir: Path) -> torch.Tensor:
    path = _resolve(reference, base_dir)
    with np.load(path, allow_pickle=False) as archive:
        components = np.asarray(
            archive[str(reference["component_key"])], dtype=np.float32
        )
    if components.shape != (4, 6, 40, 40, 3):
        raise ValueError("unexpected Stage1 component shape")
    tensor = torch.from_numpy(components)
    time, views, height, width, component_dim = tensor.shape
    pooled = tensor.permute(0, 1, 4, 2, 3).reshape(
        time * views, component_dim, height, width
    )
    pooled = F.adaptive_avg_pool2d(pooled, (10, 10))
    pooled = pooled.reshape(time, views, component_dim, 10, 10).permute(
        0, 1, 3, 4, 2
    )
    return pooled[-1].mean(dim=-1)


def _pooled_relevance(reference: Mapping[str, Any], base_dir: Path) -> torch.Tensor:
    path = _resolve(reference, base_dir)
    with np.load(path, allow_pickle=False) as archive:
        target = np.asarray(
            archive[str(reference["relevance_key"])], dtype=np.float32
        )
    if target.shape != (6, 40, 40):
        raise ValueError("unexpected task-relevance target shape")
    return F.adaptive_avg_pool2d(torch.from_numpy(target).unsqueeze(0), (10, 10))[0]


def _load_assets(
    records_path: Path,
) -> Tuple[List[str], torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[str]]]:
    groups = _load_records(records_path)
    group_ids = []
    targets = []
    uq_by_variant: Dict[str, List[torch.Tensor]] = {
        variant: [] for variant in HARD_STANCE_VARIANTS
    }
    target_stances: Dict[str, List[str]] = {
        variant: [] for variant in HARD_STANCE_VARIANTS
    }
    for group in groups:
        group_id = str(group[0]["counterfactual"]["group_id"])
        group_ids.append(group_id)
        relevance_row = _row(group, "observed", "task_relevance")
        targets.append(
            _pooled_relevance(
                relevance_row["target"]["map_sidecar"], records_path.parent
            )
        )
        for variant in HARD_STANCE_VARIANTS:
            row = _row(group, variant, "driving_implication")
            uq_by_variant[variant].append(
                _pooled_scalar_uq(
                    row["model_input"]["stage1_observation_uq"],
                    records_path.parent,
                )
            )
            stance = str(
                row["target"]["structured_summary"]["planning_implication"][
                    "stance"
                ]
            )
            target_stances[variant].append(stance)
    return (
        group_ids,
        torch.stack(targets),
        {variant: torch.stack(values) for variant, values in uq_by_variant.items()},
        target_stances,
    )


def _raw_k_features(task_risk: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if task_risk.ndim != 4:
        raise ValueError("task risk must have shape [B,V,H,W]")
    batch, views, height, width = task_risk.shape
    y = torch.linspace(-1.0, 1.0, height, dtype=task_risk.dtype)
    x = torch.linspace(-1.0, 1.0, width, dtype=task_risk.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mass_by_view = task_risk.sum(dim=(-2, -1))
    denominator = mass_by_view + epsilon
    soft_y = (task_risk * yy).sum(dim=(-2, -1)) / denominator
    soft_x = (task_risk * xx).sum(dim=(-2, -1)) / denominator
    view_coordinate = torch.linspace(
        -1.0, 1.0, views, dtype=task_risk.dtype
    ).expand(batch, -1)
    global_mass = mass_by_view.sum(dim=-1)
    global_denominator = global_mass + epsilon
    global_soft_y = (soft_y * mass_by_view).sum(dim=-1) / global_denominator
    global_soft_x = (soft_x * mass_by_view).sum(dim=-1) / global_denominator
    global_soft_view = (
        view_coordinate * mass_by_view
    ).sum(dim=-1) / global_denominator
    return torch.stack(
        (
            task_risk.flatten(1).amax(dim=-1),
            task_risk.flatten(1).mean(dim=-1),
            (task_risk.flatten(1).square().mean(dim=-1) + epsilon).sqrt(),
            global_soft_y,
            global_soft_x,
            global_soft_view,
        ),
        dim=-1,
    )


class RawKStanceHead(nn.Module):
    """Tiny diagnostic head; not a replacement for the ORION bottleneck."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, raw_k_features: torch.Tensor) -> torch.Tensor:
        return self.net(raw_k_features)


def _evaluate(
    *,
    relevance_logits: torch.Tensor,
    relevance_target: torch.Tensor,
    uq_by_variant: Mapping[str, torch.Tensor],
    target_stances: Mapping[str, List[str]],
    stance_head: RawKStanceHead,
    required_oracle_fraction: float,
) -> Dict[str, Any]:
    support = relevance_support_metrics(relevance_logits, relevance_target)
    ranking = geometry_normalized_task_risk_ranking_terms(
        uq_by_variant["on_path_uq"],
        uq_by_variant["off_path_uq"],
        relevance_logits,
        relevance_target,
        required_oracle_fraction=required_oracle_fraction,
    )
    probabilities = relevance_logits.sigmoid()
    logits_by_variant = {
        variant: stance_head(_raw_k_features(uq * probabilities))
        for variant, uq in uq_by_variant.items()
    }
    stance_metrics = matched_stance_metrics(
        {
            variant: [value[index : index + 1] for index in range(value.shape[0])]
            for variant, value in logits_by_variant.items()
        },
        {
            variant: target_stances[variant]
            for variant, value in logits_by_variant.items()
        },
    )
    return {
        "relevance": support,
        "ranking": {
            "learned_gap": ranking.learned_gap.detach().tolist(),
            "oracle_gap": ranking.oracle_gap.detach().tolist(),
            "attained_fraction": ranking.attained_fraction.detach().tolist(),
            "minimum_attained_fraction": float(
                ranking.attained_fraction.min().item()
            ),
            "positive_order_fraction": float(
                ranking.learned_gap.gt(0.0).float().mean().item()
            ),
        },
        "stance": stance_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--required-oracle-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0.0:
        raise ValueError("diagnostic bounds must be positive")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite objective diagnostic")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    records_path = args.records.resolve()
    group_ids, target, uq_by_variant, target_stances = _load_assets(records_path)
    relevance_logits = nn.Parameter(torch.zeros_like(target))
    stance_head = RawKStanceHead()
    optimizer = torch.optim.Adam(
        [relevance_logits, *stance_head.parameters()], lr=args.learning_rate
    )

    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        relevance = foreground_balanced_relevance_terms(
            relevance_logits, target
        )
        ranking = geometry_normalized_task_risk_ranking_terms(
            uq_by_variant["on_path_uq"],
            uq_by_variant["off_path_uq"],
            relevance_logits,
            target,
            required_oracle_fraction=args.required_oracle_fraction,
        )
        learned_r = relevance_logits.sigmoid()
        stance_logits = {
            variant: stance_head(_raw_k_features(uq * learned_r))
            for variant, uq in uq_by_variant.items()
        }
        stance = class_balanced_matched_stance_loss(
            stance_logits, target_stances
        )
        loss = 2.0 * relevance.loss + ranking.loss + 2.0 * stance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [relevance_logits, *stance_head.parameters()], 5.0
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "balanced_relevance": float(relevance.loss.item()),
                    "foreground_brier": float(relevance.foreground_brier.item()),
                    "background_brier": float(relevance.background_brier.item()),
                    "ranking": float(ranking.loss.item()),
                    "stance": float(stance.item()),
                    "minimum_attained_fraction": float(
                        ranking.attained_fraction.min().item()
                    ),
                }
            )

    with torch.no_grad():
        metrics = _evaluate(
            relevance_logits=relevance_logits,
            relevance_target=target,
            uq_by_variant=uq_by_variant,
            target_stances=target_stances,
            stance_head=stance_head,
            required_oracle_fraction=args.required_oracle_fraction,
        )
    checks = {
        "foreground_recall_gte_0_95": (
            metrics["relevance"]["foreground_recall"] >= 0.95
        ),
        "background_fpr_lte_0_05": (
            metrics["relevance"]["background_false_positive_rate"] <= 0.05
        ),
        "all_groups_positive_on_off_order": (
            metrics["ranking"]["positive_order_fraction"] == 1.0
        ),
        "all_groups_attain_0_8_oracle_gap": (
            metrics["ranking"]["minimum_attained_fraction"]
            >= args.required_oracle_fraction
        ),
        "zero_uq_correct": (
            metrics["stance"]["per_variant_accuracy"]["zero_uq"] == 1.0
        ),
        "off_path_uq_correct": (
            metrics["stance"]["per_variant_accuracy"]["off_path_uq"] == 1.0
        ),
        "on_path_uq_correct": (
            metrics["stance"]["per_variant_accuracy"]["on_path_uq"] == 1.0
        ),
        "minimum_stance_target_probability_gte_0_5": (
            metrics["stance"]["minimum_target_probability"] >= 0.5
        ),
    }
    passed = all(checks.values())
    report = {
        "schema": SCHEMA,
        "status": (
            "objective_feasibility_pass" if passed else "objective_feasibility_failed"
        ),
        "passed": passed,
        "event_id": EXPECTED_EVENT_ID,
        "group_ids": group_ids,
        "optimizer_steps": args.steps,
        "objective_schema": OBJECTIVE_SCHEMA,
        "target_stance_by_variant": target_stances,
        "loss_weights": {
            "foreground_balanced_relevance": 2.0,
            "geometry_normalized_ranking": 1.0,
            "class_balanced_stance": 2.0,
        },
        "required_oracle_fraction": args.required_oracle_fraction,
        "metrics": metrics,
        "checks": checks,
        "history": history,
        "provenance": {
            "records": {
                "path": str(records_path),
                "sha256": sha256_file(records_path),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "claim_boundary": (
            "CPU-only objective/label feasibility on five Route151 groups. "
            "No ORION/VLM learning, language generation, held-out, planning, "
            "closed-loop, generalization or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
