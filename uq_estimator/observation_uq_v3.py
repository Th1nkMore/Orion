"""Generator-independent observation uncertainty prototype (v3).

The module deliberately separates three roles:

* a conditional prediction teacher is fitted on clean sequences only;
* its conditional feature surprise is distilled into a single-pass adapter;
* corruption metadata is used only for split construction and evaluation.

Neither the teacher nor the adapter accepts a corruption family, severity, or
corruption mask in ``forward``.  This makes the main supervision path unable to
silently turn a synthetic corruption label into an uncertainty target.

This is a deliberately small first implementation.  It predicts image-space
patch uncertainty with one previous-frame context.  Cross-view geometry and
task relevance remain outside this module.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


OBSERVATION_UQ_SCHEMA_VERSION = "orion.observation-uq/v3.1"
OBSERVATION_UQ_CHECKPOINT_VERSION = "orion.observation-uq-checkpoint/v3.1"

# v3 used a four-phase single-patch lattice.  That leaked the same location
# from the previous frame and most of a spatially coherent corruption through
# adjacent current-frame patches.  v3.1 predicts blocks on a 3x3 phase grid and
# removes a small halo from both current and previous context.  Every patch is
# still scored exactly once, but the target cannot be copied from its temporal
# counterpart or its immediate spatial neighbourhood.
MASK_PHASE_GRID = 3
MASK_PHASE_COUNT = MASK_PHASE_GRID * MASK_PHASE_GRID
DEFAULT_MASK_BLOCK_SIZE = 4
DEFAULT_MASK_HALO = 2

CLAIM_BOUNDARY = {
    "learned_quantity": (
        "clean-conditional feature surprise distilled to a single-pass spatial "
        "observation-insufficiency score"
    ),
    "corruption_is_ground_truth": False,
    "corruption_metadata_used_by_teacher": False,
    "corruption_metadata_used_by_adapter": False,
    "actual_orion_failure_is_primary_label": False,
    "actual_orion_failure_role": "independent diagnostic or optional later auxiliary",
    "task_relevance_is_adapter_output": False,
    "closed_loop_safety_claim_supported_by_stage1_alone": False,
    "model_independence_claimed": False,
}


class ObservationUQError(ValueError):
    """Raised when a v3 data or split invariant is violated."""


def _require_feature_grid(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise ObservationUQError("%s must be a tensor" % name)
    if value.ndim != 4:
        raise ObservationUQError("%s must have shape [V,H,W,D]" % name)
    if min(value.shape) <= 0:
        raise ObservationUQError("%s must be non-empty" % name)
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ObservationUQError("%s must contain finite floating-point values" % name)


@dataclass(frozen=True)
class ObservationUQExample:
    """One model input plus audit-only family/mask metadata.

    Model code consumes only ``current``, ``previous``, and ``previous_valid``.
    ``family``, ``severity``, and ``corruption_mask`` are deliberately retained
    outside the model input for split enforcement and post-hoc metrics.
    """

    sample_id: str
    route_id: str
    split: str
    family: str
    severity: float
    current: torch.Tensor
    previous: torch.Tensor
    previous_valid: bool
    corruption_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "route_id", "split", "family"):
            if not str(getattr(self, name)).strip():
                raise ObservationUQError("%s must be non-empty" % name)
        if not math.isfinite(float(self.severity)) or self.severity < 0:
            raise ObservationUQError("severity must be finite and non-negative")
        _require_feature_grid(self.current, "current")
        _require_feature_grid(self.previous, "previous")
        if self.previous.shape != self.current.shape:
            raise ObservationUQError("current and previous feature grids must match")
        if self.corruption_mask is not None:
            if self.corruption_mask.shape != self.current.shape[:-1]:
                raise ObservationUQError(
                    "corruption_mask must have shape [V,H,W]"
                )
            if self.corruption_mask.dtype == torch.bool:
                mask = self.corruption_mask.float()
            else:
                mask = self.corruption_mask
            if not mask.is_floating_point() or not bool(torch.isfinite(mask).all()):
                raise ObservationUQError("corruption_mask must be finite")
            if bool((mask < 0).any()) or bool((mask > 1).any()):
                raise ObservationUQError("corruption_mask must lie in [0,1]")


def mask_phase(
    height: int,
    width: int,
    phase: int,
    device: torch.device,
    block_size: int = 1,
    phase_grid: int = MASK_PHASE_GRID,
) -> torch.Tensor:
    """Return one phase of a block lattice that partitions the patch grid."""

    if height <= 0 or width <= 0 or block_size <= 0 or phase_grid <= 0:
        raise ObservationUQError("mask grid dimensions must be positive")
    phase_count = phase_grid * phase_grid
    if phase < 0 or phase >= phase_count:
        raise ObservationUQError(
            "mask phase must be in [0,%d]" % (phase_count - 1)
        )
    row_phase, col_phase = divmod(int(phase), phase_grid)
    rows = (
        torch.div(torch.arange(height, device=device), block_size, rounding_mode="floor")
        .remainder(phase_grid)
        .eq(row_phase)
    )
    cols = (
        torch.div(torch.arange(width, device=device), block_size, rounding_mode="floor")
        .remainder(phase_grid)
        .eq(col_phase)
    )
    return rows[:, None] & cols[None, :]


def _phase_masks(
    height: int,
    width: int,
    phase: int,
    device: torch.device,
    requested_block_size: int,
    requested_halo: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build target and context masks, adapting only for tiny test grids."""

    if min(height, width) < MASK_PHASE_GRID:
        raise ObservationUQError(
            "spatial grid must be at least %dx%d" % (MASK_PHASE_GRID, MASK_PHASE_GRID)
        )
    block_size = min(
        int(requested_block_size), max(1, min(height, width) // MASK_PHASE_GRID)
    )
    halo = min(int(requested_halo), block_size // 2)
    target = mask_phase(
        height,
        width,
        phase,
        device,
        block_size=block_size,
        phase_grid=MASK_PHASE_GRID,
    )
    if not bool(target.any()):
        raise ObservationUQError("mask phase unexpectedly selected no target patches")
    context = target
    if halo > 0:
        context = (
            F.max_pool2d(
                target[None, None].float(),
                kernel_size=2 * halo + 1,
                stride=1,
                padding=halo,
            )[0, 0]
            > 0
        )
    return target, context, block_size, halo


def _check_batched_features(
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
    previous_valid: Optional[torch.Tensor],
    feature_dim: int,
    max_views: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if current.ndim != 5:
        raise ObservationUQError("current must have shape [B,V,H,W,D]")
    if current.shape[-1] != feature_dim:
        raise ObservationUQError("input feature dimension does not match model")
    if current.shape[1] > max_views:
        raise ObservationUQError("input view count exceeds configured max_views")
    if not current.is_floating_point() or not bool(torch.isfinite(current).all()):
        raise ObservationUQError("current features must be finite floating point")
    if previous is None:
        previous = torch.zeros_like(current)
    if previous.shape != current.shape:
        raise ObservationUQError("previous must match current feature shape")
    if previous_valid is None:
        previous_valid = torch.zeros(
            current.shape[0], dtype=torch.bool, device=current.device
        )
    if previous_valid.shape != (current.shape[0],):
        raise ObservationUQError("previous_valid must have shape [B]")
    return previous, previous_valid.to(device=current.device, dtype=torch.bool)


class CleanConditionalTeacher(nn.Module):
    """Predict withheld clean patch features from spatial/temporal context."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        max_views: int = 6,
        mask_block_size: int = DEFAULT_MASK_BLOCK_SIZE,
        mask_halo: int = DEFAULT_MASK_HALO,
    ):
        super().__init__()
        if (
            feature_dim <= 0
            or hidden_dim <= 0
            or max_views <= 0
            or mask_block_size <= 0
            or mask_halo < 0
        ):
            raise ObservationUQError("model dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.mask_block_size = int(mask_block_size)
        self.mask_halo = int(mask_halo)
        self.mask_phase_count = MASK_PHASE_COUNT
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.current_projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.previous_projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.coordinate_projection = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.view_embedding = nn.Embedding(max_views, hidden_dim)
        self.masked_token = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.previous_missing = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4
            ),
            nn.GELU(),
        )
        self.output_projection = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)

    def forward(
        self,
        current: torch.Tensor,
        target_mask: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        previous_valid: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        previous, previous_valid = _check_batched_features(
            current, previous, previous_valid, self.feature_dim, self.max_views
        )
        batch, views, height, width, _ = current.shape
        if target_mask.shape != (batch, views, height, width):
            raise ObservationUQError("target_mask must have shape [B,V,H,W]")
        target_mask = target_mask.to(device=current.device, dtype=torch.bool)
        if context_mask is None:
            context_mask = target_mask
        if context_mask.shape != target_mask.shape:
            raise ObservationUQError("context_mask must match target_mask")
        context_mask = context_mask.to(device=current.device, dtype=torch.bool)
        if bool((target_mask & ~context_mask).any()):
            raise ObservationUQError("context_mask must include every target patch")
        current_n = self.feature_norm(current).permute(0, 1, 4, 2, 3)
        previous_n = self.feature_norm(previous).permute(0, 1, 4, 2, 3)
        current_n = current_n.reshape(batch * views, self.feature_dim, height, width)
        previous_n = previous_n.reshape(batch * views, self.feature_dim, height, width)
        hidden = self.current_projection(current_n)
        mask_flat = context_mask.reshape(batch * views, 1, height, width)
        hidden = torch.where(mask_flat, self.masked_token.to(hidden.dtype), hidden)

        temporal = self.previous_projection(previous_n)
        valid = previous_valid[:, None].expand(batch, views).reshape(
            batch * views, 1, 1, 1
        )
        temporal = torch.where(
            valid,
            temporal,
            self.previous_missing.to(dtype=temporal.dtype),
        )
        # A valid prior frame must not reveal the withheld location either.
        temporal = torch.where(
            mask_flat,
            self.previous_missing.to(dtype=temporal.dtype),
            temporal,
        )
        y = torch.linspace(-1.0, 1.0, height, device=current.device)
        x = torch.linspace(-1.0, 1.0, width, device=current.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=0).to(dtype=hidden.dtype)
        coordinates = coordinates[None].expand(batch * views, -1, -1, -1)
        spatial_position = self.coordinate_projection(coordinates)
        view_ids = torch.arange(views, device=current.device)
        view_ids = view_ids[None, :].expand(batch, views).reshape(-1)
        view_context = self.view_embedding(view_ids).view(
            batch * views, self.hidden_dim, 1, 1
        )
        prediction = self.output_projection(
            self.context(hidden + temporal + view_context + spatial_position)
        )
        return prediction.reshape(
            batch, views, self.feature_dim, height, width
        ).permute(0, 1, 3, 4, 2)


class ObservationUQAdapter(nn.Module):
    """Single-pass spatiotemporal adapter distilled from teacher surprise."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, max_views: int = 6):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or max_views <= 0:
            raise ObservationUQError("model dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.current_projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.previous_projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.delta_projection = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.view_embedding = nn.Embedding(max_views, hidden_dim)
        self.previous_missing = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        previous_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        previous, previous_valid = _check_batched_features(
            current, previous, previous_valid, self.feature_dim, self.max_views
        )
        batch, views, height, width, _ = current.shape
        current_n = self.feature_norm(current)
        previous_n = self.feature_norm(previous)
        current_chw = current_n.permute(0, 1, 4, 2, 3).reshape(
            batch * views, self.feature_dim, height, width
        )
        previous_chw = previous_n.permute(0, 1, 4, 2, 3).reshape(
            batch * views, self.feature_dim, height, width
        )
        hidden = self.current_projection(current_chw)
        temporal = self.previous_projection(previous_chw)
        valid = previous_valid[:, None].expand(batch, views).reshape(
            batch * views, 1, 1, 1
        )
        temporal = torch.where(
            valid,
            temporal,
            self.previous_missing.to(dtype=temporal.dtype),
        )
        delta = 1.0 - F.cosine_similarity(
            current_n, previous_n, dim=-1, eps=1e-6
        )
        delta = delta.reshape(batch * views, 1, height, width)
        delta = torch.where(valid, delta, torch.zeros_like(delta))
        view_ids = torch.arange(views, device=current.device)
        view_ids = view_ids[None, :].expand(batch, views).reshape(-1)
        view_context = self.view_embedding(view_ids).view(
            batch * views, self.hidden_dim, 1, 1
        )
        logits = self.context(
            hidden + temporal + self.delta_projection(delta) + view_context
        )
        # A non-negative operational score; it is not a failure probability.
        score = F.softplus(logits).reshape(batch, views, height, width)
        return score


def conditional_prediction_loss(
    teacher: CleanConditionalTeacher,
    current: torch.Tensor,
    previous: torch.Tensor,
    previous_valid: torch.Tensor,
    phase: int,
) -> torch.Tensor:
    """Cosine reconstruction loss evaluated only at withheld patches."""

    batch, views, height, width, _ = current.shape
    phase_grid, context_grid, _, _ = _phase_masks(
        height,
        width,
        phase,
        current.device,
        teacher.mask_block_size,
        teacher.mask_halo,
    )
    target_mask = phase_grid[None, None].expand(batch, views, height, width)
    context_mask = context_grid[None, None].expand(batch, views, height, width)
    prediction = teacher(
        current, target_mask, previous, previous_valid, context_mask=context_mask
    )
    error = 1.0 - F.cosine_similarity(
        prediction, current, dim=-1, eps=1e-6
    ).clamp(-1.0, 1.0)
    return error[target_mask].mean()


def conditional_surprise(
    teachers: Sequence[CleanConditionalTeacher],
    current: torch.Tensor,
    previous: torch.Tensor,
    previous_valid: torch.Tensor,
    disagreement_weight: float = 0.25,
) -> torch.Tensor:
    """Score each patch while withholding that patch from every teacher.

    Only observation tensors enter this function.  Synthetic masks and labels
    cannot influence the produced target.
    """

    if not teachers:
        raise ObservationUQError("conditional surprise requires at least one teacher")
    if disagreement_weight < 0:
        raise ObservationUQError("disagreement_weight must be non-negative")
    batch, views, height, width, _ = current.shape
    output = current.new_zeros(batch, views, height, width)
    reference = teachers[0]
    for teacher in teachers[1:]:
        if (
            teacher.mask_block_size != reference.mask_block_size
            or teacher.mask_halo != reference.mask_halo
            or teacher.mask_phase_count != reference.mask_phase_count
        ):
            raise ObservationUQError("Teacher ensemble masking configurations differ")
    for phase in range(reference.mask_phase_count):
        phase_grid, context_grid, _, _ = _phase_masks(
            height,
            width,
            phase,
            current.device,
            reference.mask_block_size,
            reference.mask_halo,
        )
        target_mask = phase_grid[None, None].expand(batch, views, height, width)
        context_mask = context_grid[None, None].expand(
            batch, views, height, width
        )
        predictions = torch.stack(
            [
                teacher(
                    current,
                    target_mask,
                    previous,
                    previous_valid,
                    context_mask=context_mask,
                )
                for teacher in teachers
            ],
            dim=0,
        )
        mean_prediction = predictions.mean(dim=0)
        residual = 1.0 - F.cosine_similarity(
            mean_prediction, current, dim=-1, eps=1e-6
        ).clamp(-1.0, 1.0)
        if len(teachers) > 1:
            normalized = F.normalize(predictions, dim=-1, eps=1e-6)
            normalized_mean = F.normalize(mean_prediction, dim=-1, eps=1e-6)
            disagreement = 1.0 - (
                normalized * normalized_mean.unsqueeze(0)
            ).sum(dim=-1).clamp(-1.0, 1.0)
            residual = residual + disagreement_weight * disagreement.mean(dim=0)
        output[target_mask] = residual[target_mask]
    return output


def _collate(
    examples: Sequence[ObservationUQExample], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ObservationUQError("cannot collate an empty example list")
    # Feature shards are stored as FP16 to avoid duplicating tens of GiB on
    # disk.  The small trainable heads intentionally optimize in FP32.
    current = torch.stack([item.current for item in examples]).to(
        device=device, dtype=torch.float32
    )
    previous = torch.stack([item.previous for item in examples]).to(
        device=device, dtype=torch.float32
    )
    valid = torch.tensor(
        [item.previous_valid for item in examples], dtype=torch.bool, device=device
    )
    return current, previous, valid


def _batches(
    examples: Sequence[ObservationUQExample],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[List[ObservationUQExample]]:
    if batch_size <= 0:
        raise ObservationUQError("batch_size must be positive")
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def train_clean_teacher_epoch(
    teachers: Sequence[CleanConditionalTeacher],
    examples: Sequence[ObservationUQExample],
    optimizers: Sequence[torch.optim.Optimizer],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Fit all teachers exclusively on examples whose family is ``clean``."""

    if len(teachers) != len(optimizers) or not teachers:
        raise ObservationUQError("teacher/optimizer counts must match and be non-zero")
    if any(item.family != "clean" for item in examples):
        raise ObservationUQError("teacher training is fail-closed to clean examples")
    metrics = []
    for member_index, (teacher, optimizer) in enumerate(zip(teachers, optimizers)):
        teacher.train()
        total = 0.0
        count = 0
        for batch_index, batch in enumerate(
            _batches(examples, batch_size, True, seed + 1009 * member_index)
        ):
            current, previous, valid = _collate(batch, device)
            phase = (
                seed + member_index + batch_index
            ) % teacher.mask_phase_count
            optimizer.zero_grad(set_to_none=True)
            loss = conditional_prediction_loss(
                teacher, current, previous, valid, phase
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        metrics.append(total / max(count, 1))
    return {
        "loss": float(sum(metrics) / len(metrics)),
        "member_min_loss": float(min(metrics)),
        "member_max_loss": float(max(metrics)),
    }


@torch.no_grad()
def evaluate_clean_teacher_prediction_loss(
    teachers: Sequence[CleanConditionalTeacher],
    examples: Sequence[ObservationUQExample],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate clean reconstruction only; safe for checkpoint selection."""

    if not teachers or not examples:
        raise ObservationUQError("clean Teacher validation requires data and models")
    if any(item.family != "clean" for item in examples):
        raise ObservationUQError("clean Teacher validation received a corruption")
    metrics = []
    for teacher in teachers:
        teacher.eval()
        weighted_total = 0.0
        weighted_count = 0
        for batch in _batches(examples, batch_size, False, 0):
            current, previous, valid = _collate(batch, device)
            for phase in range(teacher.mask_phase_count):
                loss = conditional_prediction_loss(
                    teacher, current, previous, valid, phase
                )
                weighted_total += float(loss) * len(batch)
                weighted_count += len(batch)
        metrics.append(weighted_total / max(weighted_count, 1))
    return {
        "loss": float(sum(metrics) / len(metrics)),
        "member_min_loss": float(min(metrics)),
        "member_max_loss": float(max(metrics)),
    }


@torch.no_grad()
def estimate_clean_scale(
    teachers: Sequence[CleanConditionalTeacher],
    clean_examples: Sequence[ObservationUQExample],
    batch_size: int,
    device: torch.device,
    disagreement_weight: float,
    quantile: float = 0.95,
) -> float:
    if not 0.5 <= quantile < 1.0:
        raise ObservationUQError("clean scale quantile must be in [0.5,1)")
    values = []
    for teacher in teachers:
        teacher.eval()
    for batch in _batches(clean_examples, batch_size, False, 0):
        current, previous, valid = _collate(batch, device)
        values.append(
            conditional_surprise(
                teachers, current, previous, valid, disagreement_weight
            ).reshape(-1).cpu()
        )
    if not values:
        raise ObservationUQError("clean calibration requires clean examples")
    scale = torch.quantile(torch.cat(values), quantile)
    return float(scale.clamp_min(1e-4))


def train_adapter_epoch(
    adapter: ObservationUQAdapter,
    teachers: Sequence[CleanConditionalTeacher],
    examples: Sequence[ObservationUQExample],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    seed: int,
    clean_scale: float,
    disagreement_weight: float,
    target_cache: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    if clean_scale <= 0:
        raise ObservationUQError("clean_scale must be positive")
    for teacher in teachers:
        teacher.eval()
        teacher.requires_grad_(False)
    adapter.train()
    total = 0.0
    count = 0
    for batch in _batches(examples, batch_size, True, seed):
        current, previous, valid = _collate(batch, device)
        with torch.no_grad():
            if target_cache is None:
                target = conditional_surprise(
                    teachers, current, previous, valid, disagreement_weight
                ) / clean_scale
            else:
                target = torch.stack(
                    [target_cache[item.sample_id] for item in batch]
                ).to(device=device, dtype=current.dtype)
        prediction = adapter(current, previous, valid)
        per_patch = F.smooth_l1_loss(
            torch.log1p(prediction), torch.log1p(target), reduction="none"
        )
        # Rare high-surprise patches must not be drowned by the much larger
        # clean background.  The weight is derived exclusively from the
        # generator-independent teacher target, never from a corruption mask.
        target_weight = 1.0 + 3.0 * target / (1.0 + target)
        loss = (per_patch * target_weight).sum() / target_weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch)
        count += len(batch)
    return {"loss": total / max(count, 1)}


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    flattened = values.float().reshape(-1)
    sorted_values, order = torch.sort(flattened)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(0).to(dtype=torch.float32)
    starts = ends - counts.to(dtype=torch.float32)
    average_ranks = (starts + ends - 1.0) * 0.5
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or float(a.std(unbiased=False)) == 0.0 or float(b.std(unbiased=False)) == 0.0:
        return float("nan")
    ar = _rankdata(a.float())
    br = _rankdata(b.float())
    ar = ar - ar.mean()
    br = br - br.mean()
    denominator = torch.sqrt(ar.square().sum() * br.square().sum()).clamp_min(1e-12)
    return float((ar * br).sum() / denominator)


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    # Exact Mann-Whitney AUROC in O(N log N) time and O(N) memory.  The prior
    # pairwise implementation allocated O(N_pos*N_neg) and was killed during
    # the first 320-observation real evaluation.
    sorted_scores, order = torch.sort(scores.float().reshape(-1))
    sorted_labels = labels.reshape(-1)[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).to(dtype=torch.float64)
    starts = ends - counts.to(dtype=torch.float64) + 1.0
    average_ranks = (starts + ends) * 0.5
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_labels].sum()
    positive_baseline = positive_count * (positive_count + 1) * 0.5
    auc = (positive_rank_sum - positive_baseline) / (
        float(positive_count) * float(negative_count)
    )
    return float(auc)


@torch.no_grad()
def evaluate_adapter(
    adapter: ObservationUQAdapter,
    teachers: Sequence[CleanConditionalTeacher],
    examples: Sequence[ObservationUQExample],
    batch_size: int,
    device: torch.device,
    clean_scale: float,
    disagreement_weight: float,
    target_cache: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    for teacher in teachers:
        teacher.eval()
    adapter.eval()
    predictions = []
    targets = []
    masks = []
    mask_predictions = []
    family_rows = defaultdict(lambda: {"pred": [], "target": [], "severity": []})
    for batch in _batches(examples, batch_size, False, 0):
        current, previous, valid = _collate(batch, device)
        if target_cache is None:
            target = conditional_surprise(
                teachers, current, previous, valid, disagreement_weight
            ) / clean_scale
        else:
            target = torch.stack(
                [target_cache[item.sample_id] for item in batch]
            ).to(device=device, dtype=current.dtype)
        prediction = adapter(current, previous, valid)
        predictions.append(prediction.reshape(-1).cpu())
        targets.append(target.reshape(-1).cpu())
        for index, item in enumerate(batch):
            family_rows[item.family]["pred"].append(float(prediction[index].mean()))
            family_rows[item.family]["target"].append(float(target[index].mean()))
            family_rows[item.family]["severity"].append(float(item.severity))
            if item.corruption_mask is not None and item.family != "clean":
                masks.append((item.corruption_mask.reshape(-1) >= 0.5).cpu())
                mask_predictions.append(prediction[index].reshape(-1).cpu())
    if not predictions:
        raise ObservationUQError("evaluation requires at least one example")
    prediction_all = torch.cat(predictions)
    target_all = torch.cat(targets)
    by_family = {}
    for family, rows in sorted(family_rows.items()):
        pred = torch.tensor(rows["pred"])
        target = torch.tensor(rows["target"])
        severity = torch.tensor(rows["severity"])
        by_family[family] = {
            "example_count": int(len(pred)),
            "prediction_mean": float(pred.mean()),
            "teacher_target_mean": float(target.mean()),
            "severity_prediction_spearman": _spearman(severity, pred),
            "severity_target_spearman": _spearman(severity, target),
        }
    spatial_auc = float("nan")
    if masks:
        spatial_auc = _binary_auc(torch.cat(mask_predictions), torch.cat(masks))
    clean_mean = by_family.get("clean", {}).get("prediction_mean")
    if clean_mean is not None:
        for family, row in by_family.items():
            if family != "clean":
                row["prediction_uplift_over_clean"] = float(
                    row["prediction_mean"] - clean_mean
                )
    return {
        "example_count": len(examples),
        "distillation_mae": float((prediction_all - target_all).abs().mean()),
        "distillation_spearman": _spearman(prediction_all, target_all),
        "corruption_mask_patch_auroc_diagnostic_only": spatial_auc,
        "by_family": by_family,
    }


@torch.no_grad()
def evaluate_teacher_surprise(
    teachers: Sequence[CleanConditionalTeacher],
    examples: Sequence[ObservationUQExample],
    batch_size: int,
    device: torch.device,
    clean_scale: float,
    disagreement_weight: float,
) -> Dict[str, Any]:
    """Evaluate Teacher itself before an adapter is allowed to train."""

    for teacher in teachers:
        teacher.eval()
    family_rows = defaultdict(lambda: {"score": [], "severity": []})
    mask_scores = []
    mask_labels = []
    all_scores = []
    for batch in _batches(examples, batch_size, False, 0):
        current, previous, valid = _collate(batch, device)
        score = conditional_surprise(
            teachers, current, previous, valid, disagreement_weight
        ) / clean_scale
        all_scores.append(score.reshape(-1).cpu())
        for index, item in enumerate(batch):
            family_rows[item.family]["score"].append(float(score[index].mean()))
            family_rows[item.family]["severity"].append(float(item.severity))
            if item.corruption_mask is not None and item.family != "clean":
                mask_scores.append(score[index].reshape(-1).cpu())
                mask_labels.append(
                    (item.corruption_mask.reshape(-1) >= 0.5).cpu()
                )
    if not all_scores:
        raise ObservationUQError("teacher evaluation requires examples")
    by_family = {}
    for family, rows in sorted(family_rows.items()):
        scores = torch.tensor(rows["score"])
        severity = torch.tensor(rows["severity"])
        by_family[family] = {
            "example_count": len(scores),
            "teacher_score_mean": float(scores.mean()),
            "teacher_score_std": float(scores.std(unbiased=False)),
            "severity_teacher_score_spearman": _spearman(severity, scores),
        }
    clean_mean = by_family.get("clean", {}).get("teacher_score_mean")
    if clean_mean is not None:
        for family, row in by_family.items():
            if family != "clean":
                row["teacher_score_uplift_over_clean"] = float(
                    row["teacher_score_mean"] - clean_mean
                )
    mask_auc = float("nan")
    if mask_scores:
        mask_auc = _binary_auc(torch.cat(mask_scores), torch.cat(mask_labels))
    return {
        "example_count": len(examples),
        "patch_score_mean": float(torch.cat(all_scores).mean()),
        "corruption_mask_patch_auroc_diagnostic_only": mask_auc,
        "by_family": by_family,
    }


@torch.no_grad()
def precompute_surprise_targets(
    teachers: Sequence[CleanConditionalTeacher],
    examples: Sequence[ObservationUQExample],
    batch_size: int,
    device: torch.device,
    clean_scale: float,
    disagreement_weight: float,
) -> Dict[str, torch.Tensor]:
    """Cache compact [V,H,W] targets once after teacher fitting."""

    if clean_scale <= 0:
        raise ObservationUQError("clean_scale must be positive")
    result = {}
    for teacher in teachers:
        teacher.eval()
    for batch in _batches(examples, batch_size, False, 0):
        current, previous, valid = _collate(batch, device)
        target = conditional_surprise(
            teachers, current, previous, valid, disagreement_weight
        ) / clean_scale
        for index, item in enumerate(batch):
            if item.sample_id in result:
                raise ObservationUQError("duplicate sample_id in target cache")
            result[item.sample_id] = target[index].detach().cpu().float()
    if len(result) != len(examples):
        raise ObservationUQError("incomplete surprise target cache")
    return result


def validate_family_protocol(
    train_families: Sequence[str], heldout_families: Sequence[str]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    train = tuple(sorted({str(value).strip() for value in train_families if str(value).strip()}))
    heldout = tuple(
        sorted({str(value).strip() for value in heldout_families if str(value).strip()})
    )
    if not train or not heldout:
        raise ObservationUQError("train and held-out corruption families must be non-empty")
    overlap = set(train) & set(heldout)
    if overlap:
        raise ObservationUQError(
            "corruption family leakage across train/held-out: %s" % sorted(overlap)
        )
    if "clean" in set(train) | set(heldout):
        raise ObservationUQError("clean is reserved and is not a corruption family")
    return train, heldout


def _reshape_record_features(
    features: torch.Tensor, patch_height: int, patch_width: int
) -> torch.Tensor:
    if features.ndim != 3:
        raise ObservationUQError("paired record features must have shape [V,P,D]")
    if features.shape[1] != patch_height * patch_width:
        raise ObservationUQError(
            "patch grid does not match record token count: %d != %dx%d"
            % (features.shape[1], patch_height, patch_width)
        )
    return features.reshape(
        features.shape[0], patch_height, patch_width, features.shape[-1]
    ).float()


def _record_identity(record: Any) -> Tuple[str, int]:
    metadata = record.metadata
    source = metadata.get("source_identity", {}) if isinstance(metadata, Mapping) else {}
    token = source.get("sample_token")
    frame = source.get("frame_idx")
    if token is None or frame is None:
        raise ObservationUQError(
            "v3 conversion requires source_identity.sample_token and frame_idx"
        )
    return str(token), int(frame)


def _record_family(record: Any) -> str:
    metadata = record.metadata
    corruption = metadata.get("corruption", {}) if isinstance(metadata, Mapping) else {}
    family = corruption.get("corruption") if isinstance(corruption, Mapping) else None
    if family is None:
        family = metadata.get("corruption_family") if isinstance(metadata, Mapping) else None
    if family is None:
        raise ObservationUQError("paired record has no corruption family metadata")
    return str(family)


def route_splits_from_manifest(payload: Mapping[str, Any]) -> Dict[str, str]:
    if payload.get("schema_version") != "spatial-uq-route-manifest/v1":
        raise ObservationUQError("unsupported route manifest schema")
    result = {}
    for split, split_payload in payload.get("splits", {}).items():
        route_ids = split_payload.get("route_ids", [])
        for route_id in route_ids:
            if route_id in result:
                raise ObservationUQError("route appears in multiple manifest splits")
            result[str(route_id)] = str(split)
    if not result:
        raise ObservationUQError("route manifest contains no route ids")
    return result


def examples_from_paired_records(
    records: Sequence[Any],
    route_splits: Mapping[str, str],
    patch_height: int,
    patch_width: int,
) -> List[ObservationUQExample]:
    """Convert the existing paired cache without using any old target tensor."""

    if not records:
        raise ObservationUQError("paired conversion requires records")
    clean_by_sample = {}
    frame_by_sample = {}
    route_by_sample = {}
    observed_by_key = {}
    mask_by_key = {}
    for record in records:
        if record.route_id not in route_splits:
            raise ObservationUQError(
                "paired route %s is absent from route manifest" % record.route_id
            )
        sample_token, frame_idx = _record_identity(record)
        clean = _reshape_record_features(
            record.clean_patch_features, patch_height, patch_width
        )
        if sample_token in clean_by_sample and not torch.equal(
            clean_by_sample[sample_token], clean
        ):
            raise ObservationUQError("duplicated clean feature tensors disagree")
        clean_by_sample[sample_token] = clean
        frame_by_sample[sample_token] = frame_idx
        route_by_sample[sample_token] = record.route_id
        family = _record_family(record)
        key = (record.route_id, frame_idx, family, float(record.severity))
        if key in observed_by_key:
            raise ObservationUQError("duplicate route/frame/family/severity record")
        observed_by_key[key] = _reshape_record_features(
            record.observed_patch_features, patch_height, patch_width
        )
        if record.corruption_mask is not None:
            mask_by_key[key] = record.corruption_mask.reshape(
                record.corruption_mask.shape[0], patch_height, patch_width
            ).float()

    sample_by_route_frame = {
        (route_by_sample[token], frame_by_sample[token]): token
        for token in clean_by_sample
    }
    examples = []
    for sample_token in sorted(clean_by_sample):
        route_id = route_by_sample[sample_token]
        frame_idx = frame_by_sample[sample_token]
        previous_token = sample_by_route_frame.get((route_id, frame_idx - 1))
        previous_valid = previous_token is not None
        previous_clean = (
            clean_by_sample[previous_token]
            if previous_token is not None
            else torch.zeros_like(clean_by_sample[sample_token])
        )
        examples.append(
            ObservationUQExample(
                sample_id=sample_token + "/clean",
                route_id=route_id,
                split=route_splits[route_id],
                family="clean",
                severity=0.0,
                current=clean_by_sample[sample_token],
                previous=previous_clean,
                previous_valid=previous_valid,
                corruption_mask=torch.zeros(
                    clean_by_sample[sample_token].shape[:-1], dtype=torch.float32
                ),
            )
        )

    for key in sorted(observed_by_key):
        route_id, frame_idx, family, severity = key
        previous_key = (route_id, frame_idx - 1, family, severity)
        previous = observed_by_key.get(previous_key)
        examples.append(
            ObservationUQExample(
                sample_id="%s/frame_%06d/%s/severity_%g"
                % (route_id, frame_idx, family, severity),
                route_id=route_id,
                split=route_splits[route_id],
                family=family,
                severity=severity,
                current=observed_by_key[key],
                previous=(previous if previous is not None else torch.zeros_like(observed_by_key[key])),
                previous_valid=previous is not None,
                corruption_mask=mask_by_key.get(key),
            )
        )
    return examples


def _mock_clean_feature(
    route_index: int,
    frame_index: int,
    views: int,
    height: int,
    width: int,
    mix: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    rows = torch.linspace(-1.0, 1.0, height)
    cols = torch.linspace(-1.0, 1.0, width)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    view_grid = torch.arange(views).float()[:, None, None] / max(views - 1, 1)
    basis = torch.stack(
        (
            xx.expand(views, -1, -1),
            yy.expand(views, -1, -1),
            torch.sin(math.pi * xx).expand(views, -1, -1),
            torch.cos(math.pi * yy).expand(views, -1, -1),
            view_grid.expand(-1, height, width),
            torch.full((views, height, width), route_index / 10.0),
            torch.full((views, height, width), frame_index / 20.0),
            (xx * yy).expand(views, -1, -1),
        ),
        dim=-1,
    )
    clean = basis @ mix
    return clean + 0.005 * torch.randn(clean.shape, generator=generator)


def _mock_mask(
    views: int, height: int, width: int, route: int, frame: int, severity: int
) -> torch.Tensor:
    box_h = max(2, height // 4 + severity - 1)
    box_w = max(2, width // 4 + severity - 1)
    top = (route + frame) % max(height - box_h + 1, 1)
    left = (2 * route + frame) % max(width - box_w + 1, 1)
    mask = torch.zeros(views, height, width)
    mask[0, top : top + box_h, left : left + box_w] = 1.0
    return mask


def _mock_corrupt(
    clean: torch.Tensor,
    family: str,
    severity: int,
    mask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    output = clean.clone()
    selected = mask.bool().unsqueeze(-1)
    strength = 0.35 + 0.25 * severity
    if family == "local_blur":
        chw = clean.permute(0, 3, 1, 2)
        blurred = F.avg_pool2d(chw, 3, stride=1, padding=1).permute(0, 2, 3, 1)
        replacement = blurred + 0.03 * torch.randn(clean.shape, generator=generator)
    elif family == "local_dark":
        direction = torch.linspace(-1.0, -0.2, clean.shape[-1])
        replacement = clean + strength * direction
    elif family == "local_glare":
        direction = torch.sin(torch.arange(clean.shape[-1]).float() * 1.7)
        replacement = (1.0 - 0.35 * severity) * clean + strength * direction
    elif family == "local_occlusion":
        replacement = torch.zeros_like(clean)
    else:
        raise ObservationUQError("unsupported mock family %s" % family)
    output = torch.where(selected, replacement, output)
    return output


def make_mock_examples(
    feature_dim: int = 16,
    routes: int = 12,
    frames_per_route: int = 5,
    views: int = 2,
    height: int = 8,
    width: int = 8,
    families: Sequence[str] = ("local_blur", "local_dark", "local_glare"),
    severities: Sequence[int] = (1, 2, 3),
    seed: int = 0,
) -> List[ObservationUQExample]:
    """Generate predictable clean fields plus structurally distinct families."""

    if routes < 6 or frames_per_route < 2:
        raise ObservationUQError("mock convergence needs >=6 routes and >=2 frames")
    generator = torch.Generator().manual_seed(seed)
    mix = torch.randn(8, feature_dim, generator=generator)
    clean_by_key = {}
    for route_index in range(routes):
        for frame_index in range(frames_per_route):
            clean_by_key[(route_index, frame_index)] = _mock_clean_feature(
                route_index,
                frame_index,
                views,
                height,
                width,
                mix,
                generator,
            )
    split_boundaries = (routes - 4, routes - 2)
    examples = []
    corrupt_by_key = {}
    for route_index in range(routes):
        split = (
            "train"
            if route_index < split_boundaries[0]
            else "validation"
            if route_index < split_boundaries[1]
            else "held_out"
        )
        route_id = "mock_route_%03d" % route_index
        for frame_index in range(frames_per_route):
            clean = clean_by_key[(route_index, frame_index)]
            previous = clean_by_key.get((route_index, frame_index - 1))
            examples.append(
                ObservationUQExample(
                    sample_id="%s/frame_%03d/clean" % (route_id, frame_index),
                    route_id=route_id,
                    split=split,
                    family="clean",
                    severity=0.0,
                    current=clean,
                    previous=(previous if previous is not None else torch.zeros_like(clean)),
                    previous_valid=previous is not None,
                    corruption_mask=torch.zeros(views, height, width),
                )
            )
            for family in families:
                for severity in severities:
                    mask = _mock_mask(
                        views, height, width, route_index, frame_index, severity
                    )
                    corrupt = _mock_corrupt(clean, family, severity, mask, generator)
                    corrupt_by_key[(route_index, frame_index, family, severity)] = corrupt
                    previous_corrupt = corrupt_by_key.get(
                        (route_index, frame_index - 1, family, severity)
                    )
                    examples.append(
                        ObservationUQExample(
                            sample_id="%s/frame_%03d/%s/%d"
                            % (route_id, frame_index, family, severity),
                            route_id=route_id,
                            split=split,
                            family=family,
                            severity=float(severity),
                            current=corrupt,
                            previous=(
                                previous_corrupt
                                if previous_corrupt is not None
                                else torch.zeros_like(corrupt)
                            ),
                            previous_valid=previous_corrupt is not None,
                            corruption_mask=mask,
                        )
                    )
    return examples


def split_examples_for_training(
    examples: Sequence[ObservationUQExample],
    train_families: Sequence[str],
    heldout_families: Sequence[str],
) -> Dict[str, List[ObservationUQExample]]:
    train_families, heldout_families = validate_family_protocol(
        train_families, heldout_families
    )
    observed_families = {item.family for item in examples if item.family != "clean"}
    missing = (set(train_families) | set(heldout_families)) - observed_families
    if missing:
        raise ObservationUQError("requested corruption families absent: %s" % sorted(missing))
    result = {
        "teacher_train": [
            item for item in examples if item.split == "train" and item.family == "clean"
        ],
        "student_train": [
            item
            for item in examples
            if item.split == "train"
            and (item.family == "clean" or item.family in train_families)
        ],
        "validation_seen": [
            item
            for item in examples
            if item.split == "validation"
            and (item.family == "clean" or item.family in train_families)
        ],
        "validation_heldout_family": [
            item
            for item in examples
            if item.split == "validation"
            and (item.family == "clean" or item.family in heldout_families)
        ],
        "heldout_route_and_family": [
            item
            for item in examples
            if item.split == "held_out"
            and (item.family == "clean" or item.family in heldout_families)
        ],
    }
    empty = [name for name, values in result.items() if not values]
    if empty:
        raise ObservationUQError("empty required v3 data splits: %s" % empty)
    trained_nonclean = {
        item.family for item in result["student_train"] if item.family != "clean"
    }
    evaluated_heldout = {
        item.family
        for item in result["heldout_route_and_family"]
        if item.family != "clean"
    }
    if trained_nonclean & evaluated_heldout:
        raise ObservationUQError("held-out family leaked into student training")
    return result


def run_observation_uq_training(
    examples: Sequence[ObservationUQExample],
    train_families: Sequence[str],
    heldout_families: Sequence[str],
    output_path: Path,
    feature_dim: int,
    hidden_dim: int = 64,
    teacher_members: int = 2,
    teacher_epochs: int = 10,
    adapter_epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    disagreement_weight: float = 0.25,
    mask_block_size: int = DEFAULT_MASK_BLOCK_SIZE,
    mask_halo: int = DEFAULT_MASK_HALO,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run the bounded v3 prototype and save a fully auditable checkpoint."""

    if teacher_members <= 0 or teacher_epochs <= 0 or adapter_epochs <= 0:
        raise ObservationUQError("member and epoch counts must be positive")
    train_families, heldout_families = validate_family_protocol(
        train_families, heldout_families
    )
    splits = split_examples_for_training(examples, train_families, heldout_families)
    torch.manual_seed(seed)
    random.seed(seed)
    torch_device = torch.device(device)
    max_views = max(item.current.shape[0] for item in examples)
    if any(item.current.shape[-1] != feature_dim for item in examples):
        raise ObservationUQError("feature_dim does not match all examples")
    teachers = [
        CleanConditionalTeacher(
            feature_dim,
            hidden_dim,
            max_views,
            mask_block_size=mask_block_size,
            mask_halo=mask_halo,
        ).to(torch_device)
        for _ in range(teacher_members)
    ]
    teacher_optimizers = [
        torch.optim.AdamW(model.parameters(), lr=learning_rate) for model in teachers
    ]
    teacher_history = []
    for epoch in range(teacher_epochs):
        teacher_history.append(
            train_clean_teacher_epoch(
                teachers,
                splits["teacher_train"],
                teacher_optimizers,
                batch_size,
                torch_device,
                seed + epoch,
            )
        )
    clean_scale = estimate_clean_scale(
        teachers,
        splits["teacher_train"],
        batch_size,
        torch_device,
        disagreement_weight,
    )
    target_cache = precompute_surprise_targets(
        teachers,
        examples,
        batch_size,
        torch_device,
        clean_scale,
        disagreement_weight,
    )
    adapter = ObservationUQAdapter(feature_dim, hidden_dim, max_views).to(torch_device)
    adapter_optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    adapter_history = []
    validation_history = []
    for epoch in range(adapter_epochs):
        adapter_history.append(
            train_adapter_epoch(
                adapter,
                teachers,
                splits["student_train"],
                adapter_optimizer,
                batch_size,
                torch_device,
                seed + 10000 + epoch,
                clean_scale,
                disagreement_weight,
                target_cache,
            )
        )
        validation_history.append(
            evaluate_adapter(
                adapter,
                teachers,
                splits["validation_heldout_family"],
                batch_size,
                torch_device,
                clean_scale,
                disagreement_weight,
                target_cache,
            )
        )
    evaluations = {
        name: evaluate_adapter(
            adapter,
            teachers,
            values,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        )
        for name, values in splits.items()
        if name not in {"teacher_train", "student_train"}
    }
    checkpoint = {
        "schema_version": OBSERVATION_UQ_CHECKPOINT_VERSION,
        "observation_uq_schema_version": OBSERVATION_UQ_SCHEMA_VERSION,
        "teacher_states": [model.state_dict() for model in teachers],
        "adapter_state": adapter.state_dict(),
        "model_config": {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "max_views": max_views,
            "teacher_members": teacher_members,
            "temporal_context": "one_previous_frame_when_available",
            "temporal_target_access": "same_context_mask_as_current",
            "mask_block_size": mask_block_size,
            "mask_halo": mask_halo,
            "mask_phase_count": MASK_PHASE_COUNT,
            "cross_view_geometry": "not_implemented_in_v3_mvp",
        },
        "training_config": {
            "teacher_epochs": teacher_epochs,
            "adapter_epochs": adapter_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "disagreement_weight": disagreement_weight,
            "seed": seed,
            "train_families": list(train_families),
            "heldout_families": list(heldout_families),
        },
        "clean_scale": clean_scale,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "data_attestation": {
            "teacher_example_families": sorted(
                {item.family for item in splits["teacher_train"]}
            ),
            "student_training_families": sorted(
                {item.family for item in splits["student_train"]}
            ),
            "heldout_route_evaluation_families": sorted(
                {item.family for item in splits["heldout_route_and_family"]}
            ),
            "corruption_metadata_used_as_target": False,
            "actual_target_tensor_read": False,
            "teacher_targets_precomputed_once_after_teacher_fit": True,
        },
        "history": {
            "teacher_train": teacher_history,
            "adapter_train": adapter_history,
            "validation_heldout_family": validation_history,
        },
        "evaluations": evaluations,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in checkpoint.items()
                if key not in {"teacher_states", "adapter_state"}
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


def run_teacher_viability_training(
    examples: Sequence[ObservationUQExample],
    heldout_families: Sequence[str],
    output_path: Path,
    feature_dim: int,
    hidden_dim: int = 64,
    teacher_members: int = 2,
    teacher_epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    disagreement_weight: float = 0.25,
    mask_block_size: int = DEFAULT_MASK_BLOCK_SIZE,
    mask_halo: int = DEFAULT_MASK_HALO,
    validation_interval: int = 4,
    resume_path: Optional[Path] = None,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train only on many clean frames, then gate on unseen-route interventions."""

    heldout = tuple(
        sorted({str(value).strip() for value in heldout_families if str(value).strip()})
    )
    if not heldout or "clean" in heldout:
        raise ObservationUQError("heldout_families must contain non-clean families")
    if teacher_members <= 0 or teacher_epochs <= 0 or validation_interval <= 0:
        raise ObservationUQError("teacher member and epoch counts must be positive")
    clean_train = [
        item for item in examples if item.split == "train" and item.family == "clean"
    ]
    clean_validation = [
        item
        for item in examples
        if item.split == "validation" and item.family == "clean"
    ]
    clean_heldout = [
        item for item in examples if item.split == "held_out" and item.family == "clean"
    ]
    validation_eval = [
        item
        for item in examples
        if item.split == "validation"
        and (item.family == "clean" or item.family in heldout)
    ]
    heldout_eval = [
        item
        for item in examples
        if item.split == "held_out"
        and (item.family == "clean" or item.family in heldout)
    ]
    required = {
        "clean_train": clean_train,
        "clean_validation": clean_validation,
        "clean_heldout": clean_heldout,
        "validation_eval": validation_eval,
        "heldout_eval": heldout_eval,
    }
    empty = [name for name, values in required.items() if not values]
    if empty:
        raise ObservationUQError("empty Teacher viability splits: %s" % empty)
    if any(item.family != "clean" for item in clean_train):
        raise ObservationUQError("Teacher optimizer input leaked a corruption")
    evaluated_families = {
        item.family for item in validation_eval + heldout_eval if item.family != "clean"
    }
    if not set(heldout).issubset(evaluated_families):
        raise ObservationUQError("held-out family is absent from Teacher evaluation")

    torch.manual_seed(seed)
    random.seed(seed)
    torch_device = torch.device(device)
    max_views = max(item.current.shape[0] for item in examples)
    if any(item.current.shape[-1] != feature_dim for item in examples):
        raise ObservationUQError("feature_dim does not match examples")
    teachers = [
        CleanConditionalTeacher(
            feature_dim,
            hidden_dim,
            max_views,
            mask_block_size=mask_block_size,
            mask_halo=mask_halo,
        ).to(torch_device)
        for _ in range(teacher_members)
    ]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=learning_rate) for model in teachers
    ]
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_states = None
    start_epoch = 0
    if resume_path is not None:
        resume_path = Path(resume_path)
        progress = torch.load(resume_path, map_location="cpu")
        if not isinstance(progress, Mapping) or progress.get("schema_version") != (
            "orion.observation-uq-teacher-progress/v3.1"
        ):
            raise ObservationUQError("unsupported Teacher progress checkpoint")
        expected_model_config = {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "max_views": max_views,
            "teacher_members": teacher_members,
            "mask_block_size": mask_block_size,
            "mask_halo": mask_halo,
            "mask_phase_count": MASK_PHASE_COUNT,
        }
        if progress.get("model_config") != expected_model_config:
            raise ObservationUQError("resume Teacher model configuration mismatch")
        expected_training_config = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        if progress.get("training_config") != expected_training_config:
            raise ObservationUQError("resume Teacher optimizer configuration mismatch")
        saved_teacher_states = progress.get("teacher_states")
        saved_optimizer_states = progress.get("optimizer_states")
        if (
            not isinstance(saved_teacher_states, list)
            or len(saved_teacher_states) != len(teachers)
            or not isinstance(saved_optimizer_states, list)
            or len(saved_optimizer_states) != len(optimizers)
        ):
            raise ObservationUQError("resume Teacher member state count mismatch")
        for model, state in zip(teachers, saved_teacher_states):
            model.load_state_dict(state)
        for optimizer, state in zip(optimizers, saved_optimizer_states):
            optimizer.load_state_dict(state)
        history = list(progress.get("history", []))
        start_epoch = int(progress.get("completed_epochs", -1))
        if start_epoch != len(history) or start_epoch < 0:
            raise ObservationUQError("resume Teacher epoch/history mismatch")
        best_epoch = int(progress.get("best_epoch", 0))
        best_validation_loss = float(
            progress.get("best_validation_loss", float("inf"))
        )
        best_states = progress.get("best_teacher_states")
        if start_epoch >= teacher_epochs:
            raise ObservationUQError(
                "resume checkpoint already reached requested teacher_epochs"
            )
        print(
            "[ObservationUQTeacher] resumed_from=%s completed_epochs=%d"
            % (resume_path, start_epoch),
            flush=True,
        )
    progress_path = Path(output_path).with_suffix(".progress.pt")
    for epoch in range(start_epoch, teacher_epochs):
        epoch_metrics = train_clean_teacher_epoch(
                teachers,
                clean_train,
                optimizers,
                batch_size,
                torch_device,
                seed + epoch,
            )
        should_validate = (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == teacher_epochs
        )
        validation_metrics = None
        if should_validate:
            validation_metrics = evaluate_clean_teacher_prediction_loss(
                teachers,
                clean_validation,
                batch_size,
                torch_device,
            )
            epoch_metrics["clean_validation_loss"] = validation_metrics["loss"]
        history.append(epoch_metrics)
        if (
            validation_metrics is not None
            and validation_metrics["loss"] < best_validation_loss
        ):
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch + 1
            best_states = [
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                for model in teachers
            ]
        if validation_metrics is not None:
            progress_payload = {
                "schema_version": "orion.observation-uq-teacher-progress/v3.1",
                "completed_epochs": epoch + 1,
                "teacher_states": [
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    for model in teachers
                ],
                "optimizer_states": [
                    optimizer.state_dict() for optimizer in optimizers
                ],
                "best_teacher_states": best_states,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "history": history,
                "model_config": {
                    "feature_dim": feature_dim,
                    "hidden_dim": hidden_dim,
                    "max_views": max_views,
                    "teacher_members": teacher_members,
                    "mask_block_size": mask_block_size,
                    "mask_halo": mask_halo,
                    "mask_phase_count": MASK_PHASE_COUNT,
                },
                "training_config": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                },
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_progress_path = progress_path.with_suffix(".tmp")
            torch.save(progress_payload, temporary_progress_path)
            temporary_progress_path.replace(progress_path)
        validation_text = (
            "%.6f" % validation_metrics["loss"]
            if validation_metrics is not None
            else "not_run"
        )
        print(
            "[ObservationUQTeacher] epoch=%d/%d loss=%.6f clean_val=%s "
            "member_min=%.6f member_max=%.6f"
            % (
                epoch + 1,
                teacher_epochs,
                epoch_metrics["loss"],
                validation_text,
                epoch_metrics["member_min_loss"],
                epoch_metrics["member_max_loss"],
            ),
            flush=True,
        )
    if best_states is None:
        raise ObservationUQError("Teacher checkpoint selection produced no state")
    for model, state in zip(teachers, best_states):
        model.load_state_dict(state)
    print(
        "[ObservationUQTeacher] selected_epoch=%d clean_validation_loss=%.6f"
        % (best_epoch, best_validation_loss),
        flush=True,
    )
    clean_scale = estimate_clean_scale(
        teachers,
        clean_train,
        batch_size,
        torch_device,
        disagreement_weight,
    )
    evaluations = {
        "clean_train": evaluate_teacher_surprise(
            teachers, clean_train, batch_size, torch_device, clean_scale, disagreement_weight
        ),
        "clean_validation": evaluate_teacher_surprise(
            teachers, clean_validation, batch_size, torch_device, clean_scale, disagreement_weight
        ),
        "clean_heldout": evaluate_teacher_surprise(
            teachers, clean_heldout, batch_size, torch_device, clean_scale, disagreement_weight
        ),
        "validation_heldout_family": evaluate_teacher_surprise(
            teachers, validation_eval, batch_size, torch_device, clean_scale, disagreement_weight
        ),
        "heldout_route_and_family": evaluate_teacher_surprise(
            teachers, heldout_eval, batch_size, torch_device, clean_scale, disagreement_weight
        ),
    }
    for name, evaluation in evaluations.items():
        print(
            "[ObservationUQTeacher] evaluation=%s patch_mean=%.6f mask_auc=%s"
            % (
                name,
                evaluation["patch_score_mean"],
                evaluation["corruption_mask_patch_auroc_diagnostic_only"],
            ),
            flush=True,
        )
    gate_inputs = {}
    for split in ("validation_heldout_family", "heldout_route_and_family"):
        evaluation = evaluations[split]
        clean_mean = evaluation["by_family"]["clean"]["teacher_score_mean"]
        family_gates = {}
        for family in heldout:
            row = evaluation["by_family"][family]
            family_gates[family] = {
                "positive_uplift": row["teacher_score_mean"] > clean_mean,
                "positive_severity_spearman": (
                    math.isfinite(row["severity_teacher_score_spearman"])
                    and row["severity_teacher_score_spearman"] > 0
                ),
            }
        gate_inputs[split] = {
            "mask_auroc_above_random": (
                math.isfinite(evaluation["corruption_mask_patch_auroc_diagnostic_only"])
                and evaluation["corruption_mask_patch_auroc_diagnostic_only"] > 0.5
            ),
            "families": family_gates,
        }
    checkpoint = {
        "schema_version": "orion.observation-uq-teacher-checkpoint/v3.1",
        "observation_uq_schema_version": OBSERVATION_UQ_SCHEMA_VERSION,
        "teacher_states": [model.state_dict() for model in teachers],
        "model_config": {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "max_views": max_views,
            "teacher_members": teacher_members,
            "temporal_context": "one_previous_frame_when_available",
            "temporal_target_access": "same_context_mask_as_current",
            "mask_block_size": mask_block_size,
            "mask_halo": mask_halo,
            "mask_phase_count": MASK_PHASE_COUNT,
            "context_receptive_field": "dilated_convolution_radius_7",
        },
        "training_config": {
            "teacher_epochs": teacher_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "disagreement_weight": disagreement_weight,
            "validation_interval": validation_interval,
            "seed": seed,
            "heldout_families": list(heldout),
        },
        "clean_scale": clean_scale,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "data_attestation": {
            "teacher_optimizer_example_count": len(clean_train),
            "teacher_optimizer_families": sorted({item.family for item in clean_train}),
            "teacher_optimizer_route_count": len({item.route_id for item in clean_train}),
            "corruption_metadata_used_as_target": False,
            "actual_target_tensor_read": False,
            "adapter_trained": False,
        },
        "history": {"teacher_train": history},
        "checkpoint_selection": {
            "metric": "clean_validation_prediction_loss",
            "selected_epoch": best_epoch,
            "best_value": best_validation_loss,
            "corruption_diagnostic_used_for_selection": False,
            "resumed_from": str(resume_path) if resume_path is not None else None,
            "progress_checkpoint": str(progress_path),
        },
        "evaluations": evaluations,
        "gate_inputs": gate_inputs,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    output_path.with_suffix(".report.json").write_text(
        json.dumps(
            {key: value for key, value in checkpoint.items() if key != "teacher_states"},
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


def run_clean_only_adapter_training(
    examples: Sequence[ObservationUQExample],
    teacher_checkpoint: Mapping[str, Any],
    output_path: Path,
    adapter_epochs: int = 24,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    resume_path: Optional[Path] = None,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Distil a passed v3.1 Teacher using clean optimizer inputs only.

    Diagnostic corruptions remain evaluation-only.  This is intentionally the
    most generator-independent first adapter gate; intervention inputs can be
    added later only if clean-only extrapolation proves insufficient.
    """

    if teacher_checkpoint.get("schema_version") != (
        "orion.observation-uq-teacher-checkpoint/v3.1"
    ):
        raise ObservationUQError("adapter requires a v3.1 Teacher checkpoint")
    if adapter_epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ObservationUQError("adapter training parameters must be positive")
    model_config = teacher_checkpoint.get("model_config", {})
    training_config = teacher_checkpoint.get("training_config", {})
    feature_dim = int(model_config.get("feature_dim", 0))
    hidden_dim = int(model_config.get("hidden_dim", 0))
    max_views = int(model_config.get("max_views", 0))
    teacher_members = int(model_config.get("teacher_members", 0))
    mask_block_size = int(model_config.get("mask_block_size", 0))
    mask_halo = int(model_config.get("mask_halo", -1))
    teacher_states = teacher_checkpoint.get("teacher_states")
    if (
        min(feature_dim, hidden_dim, max_views, teacher_members, mask_block_size) <= 0
        or mask_halo < 0
        or not isinstance(teacher_states, list)
        or len(teacher_states) != teacher_members
    ):
        raise ObservationUQError("invalid v3.1 Teacher model metadata")
    heldout = tuple(training_config.get("heldout_families", ()))
    if not heldout:
        raise ObservationUQError("Teacher checkpoint has no held-out family")
    clean_scale = float(teacher_checkpoint.get("clean_scale", 0.0))
    disagreement_weight = float(training_config.get("disagreement_weight", 0.25))
    if clean_scale <= 0 or disagreement_weight < 0:
        raise ObservationUQError("invalid Teacher score calibration")
    clean_train = [
        item for item in examples if item.split == "train" and item.family == "clean"
    ]
    clean_validation = [
        item
        for item in examples
        if item.split == "validation" and item.family == "clean"
    ]
    clean_heldout = [
        item for item in examples if item.split == "held_out" and item.family == "clean"
    ]
    validation_eval = [
        item
        for item in examples
        if item.split == "validation"
        and (item.family == "clean" or item.family in heldout)
    ]
    heldout_eval = [
        item
        for item in examples
        if item.split == "held_out"
        and (item.family == "clean" or item.family in heldout)
    ]
    required = {
        "clean_train": clean_train,
        "clean_validation": clean_validation,
        "clean_heldout": clean_heldout,
        "validation_eval": validation_eval,
        "heldout_eval": heldout_eval,
    }
    empty = [name for name, values in required.items() if not values]
    if empty:
        raise ObservationUQError("empty clean-only adapter splits: %s" % empty)
    if any(item.current.shape[-1] != feature_dim for item in examples):
        raise ObservationUQError("adapter feature dimension mismatch")

    torch.manual_seed(seed)
    random.seed(seed)
    torch_device = torch.device(device)
    teachers = [
        CleanConditionalTeacher(
            feature_dim,
            hidden_dim,
            max_views,
            mask_block_size=mask_block_size,
            mask_halo=mask_halo,
        ).to(torch_device)
        for _ in range(teacher_members)
    ]
    for teacher, state in zip(teachers, teacher_states):
        teacher.load_state_dict(state)
        teacher.eval()
        teacher.requires_grad_(False)
    target_cache = precompute_surprise_targets(
        teachers,
        examples,
        batch_size,
        torch_device,
        clean_scale,
        disagreement_weight,
    )
    adapter = ObservationUQAdapter(feature_dim, hidden_dim, max_views).to(torch_device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    history = []
    best_validation_mae = float("inf")
    best_epoch = 0
    best_state = None
    output_path = Path(output_path)
    progress_path = output_path.with_suffix(".progress.pt")
    start_epoch = 0
    if resume_path is not None:
        resume_path = Path(resume_path)
        progress = torch.load(resume_path, map_location="cpu")
        if not isinstance(progress, Mapping) or progress.get("schema_version") != (
            "orion.observation-uq-adapter-progress/v3.1"
        ):
            raise ObservationUQError("unsupported adapter progress checkpoint")
        expected_model_config = {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "max_views": max_views,
        }
        expected_training_config = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        if progress.get("model_config") != expected_model_config:
            raise ObservationUQError("resume adapter model configuration mismatch")
        if progress.get("training_config") != expected_training_config:
            raise ObservationUQError("resume adapter optimizer configuration mismatch")
        adapter.load_state_dict(progress["adapter_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        history = list(progress.get("history", []))
        start_epoch = int(progress.get("completed_epochs", -1))
        if start_epoch != len(history) or start_epoch < 0:
            raise ObservationUQError("resume adapter epoch/history mismatch")
        best_epoch = int(progress.get("best_epoch", 0))
        best_validation_mae = float(
            progress.get("best_validation_mae", float("inf"))
        )
        best_state = progress.get("best_adapter_state")
        if start_epoch >= adapter_epochs:
            raise ObservationUQError(
                "resume checkpoint already reached requested adapter_epochs"
            )
        print(
            "[ObservationUQAdapter] resumed_from=%s completed_epochs=%d"
            % (resume_path, start_epoch),
            flush=True,
        )
    for epoch in range(start_epoch, adapter_epochs):
        train_metrics = train_adapter_epoch(
            adapter,
            teachers,
            clean_train,
            optimizer,
            batch_size,
            torch_device,
            seed + epoch,
            clean_scale,
            disagreement_weight,
            target_cache,
        )
        validation_metrics = evaluate_adapter(
            adapter,
            teachers,
            clean_validation,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        )
        row = {
            "loss": train_metrics["loss"],
            "clean_validation_distillation_mae": validation_metrics[
                "distillation_mae"
            ],
            "clean_validation_distillation_spearman": validation_metrics[
                "distillation_spearman"
            ],
        }
        history.append(row)
        if row["clean_validation_distillation_mae"] < best_validation_mae:
            best_validation_mae = row["clean_validation_distillation_mae"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
        progress_payload = {
            "schema_version": "orion.observation-uq-adapter-progress/v3.1",
            "completed_epochs": epoch + 1,
            "adapter_state": {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "best_adapter_state": best_state,
            "best_epoch": best_epoch,
            "best_validation_mae": best_validation_mae,
            "history": history,
            "model_config": {
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "max_views": max_views,
            },
            "training_config": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
            },
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_progress_path = progress_path.with_suffix(".tmp")
        torch.save(progress_payload, temporary_progress_path)
        temporary_progress_path.replace(progress_path)
        print(
            "[ObservationUQAdapter] epoch=%d/%d loss=%.6f clean_val_mae=%.6f "
            "clean_val_spearman=%.6f"
            % (
                epoch + 1,
                adapter_epochs,
                row["loss"],
                row["clean_validation_distillation_mae"],
                row["clean_validation_distillation_spearman"],
            ),
            flush=True,
        )
    if best_state is None:
        raise ObservationUQError("adapter checkpoint selection produced no state")
    adapter.load_state_dict(best_state)
    evaluations = {
        "clean_train": evaluate_adapter(
            adapter,
            teachers,
            clean_train,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        ),
        "clean_validation": evaluate_adapter(
            adapter,
            teachers,
            clean_validation,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        ),
        "clean_heldout": evaluate_adapter(
            adapter,
            teachers,
            clean_heldout,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        ),
        "validation_development_diagnostic": evaluate_adapter(
            adapter,
            teachers,
            validation_eval,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        ),
        "heldout_route_development_diagnostic": evaluate_adapter(
            adapter,
            teachers,
            heldout_eval,
            batch_size,
            torch_device,
            clean_scale,
            disagreement_weight,
            target_cache,
        ),
    }
    checkpoint = {
        "schema_version": "orion.observation-uq-adapter-checkpoint/v3.1",
        "observation_uq_schema_version": OBSERVATION_UQ_SCHEMA_VERSION,
        "adapter_state": adapter.state_dict(),
        "model_config": {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "max_views": max_views,
        },
        "training_config": {
            "adapter_epochs": adapter_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "optimizer_input_families": ["clean"],
            "teacher_target_only": True,
        },
        "teacher_provenance": {
            "schema_version": teacher_checkpoint["schema_version"],
            "selected_epoch": teacher_checkpoint["checkpoint_selection"][
                "selected_epoch"
            ],
            "clean_scale": clean_scale,
        },
        "checkpoint_selection": {
            "metric": "clean_validation_distillation_mae",
            "selected_epoch": best_epoch,
            "best_value": best_validation_mae,
            "corruption_diagnostic_used_for_selection": False,
            "progress_checkpoint": str(progress_path),
            "resumed_from": str(resume_path) if resume_path is not None else None,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "data_attestation": {
            "adapter_optimizer_example_count": len(clean_train),
            "adapter_optimizer_families": ["clean"],
            "adapter_optimizer_route_count": len(
                {item.route_id for item in clean_train}
            ),
            "corruption_metadata_used_as_target": False,
            "corruption_observation_used_by_optimizer": False,
            "actual_target_tensor_read": False,
            "driving_gradient_used": False,
        },
        "history": {"adapter_train": history},
        "evaluations": evaluations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    output_path.with_suffix(".report.json").write_text(
        json.dumps(
            {key: value for key, value in checkpoint.items() if key != "adapter_state"},
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


__all__ = [
    "OBSERVATION_UQ_SCHEMA_VERSION",
    "OBSERVATION_UQ_CHECKPOINT_VERSION",
    "CLAIM_BOUNDARY",
    "ObservationUQError",
    "ObservationUQExample",
    "CleanConditionalTeacher",
    "ObservationUQAdapter",
    "mask_phase",
    "conditional_prediction_loss",
    "conditional_surprise",
    "train_clean_teacher_epoch",
    "train_adapter_epoch",
    "evaluate_adapter",
    "evaluate_teacher_surprise",
    "precompute_surprise_targets",
    "validate_family_protocol",
    "route_splits_from_manifest",
    "examples_from_paired_records",
    "make_mock_examples",
    "split_examples_for_training",
    "run_observation_uq_training",
    "run_teacher_viability_training",
    "run_clean_only_adapter_training",
]
