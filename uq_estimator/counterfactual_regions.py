"""Select matched on-path/off-path local corruption regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MatchedCounterfactualRegions:
    """Equal-area normalized regions with audited route-corridor overlap."""

    on_path_region: tuple[float, float, float, float]
    off_path_region: tuple[float, float, float, float]
    on_path_overlap: float
    off_path_overlap: float
    region_hw: tuple[int, int]


def _normalized_box(
    top: int,
    left: int,
    box_h: int,
    box_w: int,
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    return (
        top / height,
        left / width,
        (top + box_h) / height,
        (left + box_w) / width,
    )


def select_matched_counterfactual_regions(
    route_corridor: torch.Tensor,
    region_hw: tuple[int, int],
    max_off_path_overlap_fraction: float = 0.0,
) -> MatchedCounterfactualRegions:
    """Choose same-size boxes that maximize and minimize path overlap.

    Args:
        route_corridor: non-negative route membership ``[H, W]``.
        region_hw: region height and width on that grid.
        max_off_path_overlap_fraction: maximum permitted off-path window sum as
            a fraction of the on-path sum.  Zero requires no overlap.

    The on-path box is the first row-major maximum-overlap window.  Among
    eligible off-path boxes, the farthest box center from the on-path center is
    selected deterministically.  The returned normalized regions can be passed
    directly to the metadata corruption API at any image resolution.
    """
    if route_corridor.ndim != 2:
        raise ValueError("route_corridor must have shape [H, W]")
    if not route_corridor.is_floating_point() and route_corridor.dtype != torch.bool:
        raise TypeError("route_corridor must be floating point or boolean")
    corridor = route_corridor.float()
    if not torch.isfinite(corridor).all() or torch.any(corridor < 0):
        raise ValueError("route_corridor must be finite and non-negative")
    height, width = corridor.shape
    box_h, box_w = int(region_hw[0]), int(region_hw[1])
    if box_h <= 0 or box_w <= 0 or box_h > height or box_w > width:
        raise ValueError("region_hw must fit inside route_corridor")
    if not 0.0 <= max_off_path_overlap_fraction <= 1.0:
        raise ValueError("max_off_path_overlap_fraction must lie in [0, 1]")
    if not corridor.any():
        raise ValueError("route_corridor must contain at least one path cell")

    kernel = corridor.new_ones((1, 1, box_h, box_w))
    overlap = F.conv2d(corridor[None, None], kernel).squeeze(0).squeeze(0)
    flat = overlap.reshape(-1)
    on_index = int(torch.argmax(flat))
    windows_w = overlap.shape[1]
    on_top, on_left = divmod(on_index, windows_w)
    on_overlap = float(flat[on_index].item())
    if on_overlap <= 0:
        raise ValueError("no candidate region overlaps the route corridor")

    threshold = on_overlap * max_off_path_overlap_fraction + 1e-8
    eligible = overlap <= threshold
    # A counterfactual region must not be the same rectangle even when the
    # caller allows a high off-path overlap fraction.
    eligible[on_top, on_left] = False
    if not eligible.any():
        raise ValueError(
            "no matched off-path region satisfies the overlap constraint"
        )

    rows = torch.arange(overlap.shape[0], device=overlap.device)[:, None]
    cols = torch.arange(overlap.shape[1], device=overlap.device)[None, :]
    distance_sq = (rows - on_top).square() + (cols - on_left).square()
    score = torch.where(
        eligible,
        distance_sq.to(torch.float64),
        torch.full_like(distance_sq, -1, dtype=torch.float64),
    )
    off_index = int(torch.argmax(score.reshape(-1)))
    off_top, off_left = divmod(off_index, windows_w)
    off_overlap = float(overlap[off_top, off_left].item())

    return MatchedCounterfactualRegions(
        on_path_region=_normalized_box(
            on_top, on_left, box_h, box_w, height, width
        ),
        off_path_region=_normalized_box(
            off_top, off_left, box_h, box_w, height, width
        ),
        on_path_overlap=on_overlap,
        off_path_overlap=off_overlap,
        region_hw=(box_h, box_w),
    )


__all__ = [
    "MatchedCounterfactualRegions",
    "select_matched_counterfactual_regions",
]

