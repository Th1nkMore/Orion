"""Project candidate route corridors into multi-camera patch grids.

The projection is deterministic and contains no learned parameters.  It keeps
route relevance outside the spatial UQ head and exposes coverage explicitly so
an invisible route cannot be mistaken for a low-risk route.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProjectedPathCorridor:
    """Projected route-corridor weights and visibility diagnostics.

    Shapes:
        mask: ``[B, M, V, H_patch, W_patch]``.
        point_visible: ``[B, M, T, V]``.
        patch_xy: ``[B, M, T, V, 2]`` in ``(x, y)`` patch coordinates.
        depth: ``[B, M, T, V]`` camera projection depth.
        coverage: ``[B, M]`` visible route-point fraction across cameras.
    """

    mask: torch.Tensor
    point_visible: torch.Tensor
    patch_xy: torch.Tensor
    depth: torch.Tensor
    coverage: torch.Tensor


def project_path_corridor_to_patches(
    path_xyz: torch.Tensor,
    lidar2img: torch.Tensor,
    image_hw: tuple[int, int],
    patch_hw: tuple[int, int],
    corridor_radius_patches: float = 1.5,
    soft: bool = True,
    min_depth: float = 1e-4,
) -> ProjectedPathCorridor:
    """Project local-lidar candidate paths into every camera patch grid.

    Args:
        path_xyz: ``[B, M, T, 2 or 3]`` route/candidate points in lidar frame.
            Two-dimensional inputs are placed on ``z=0``.
        lidar2img: augmented projection matrices ``[B, V, 4, 4]`` mapping
            homogeneous lidar coordinates into image homogeneous coordinates.
        image_hw: augmented image height and width.
        patch_hw: spatial height and width of the visual feature grid.
        corridor_radius_patches: disk/Gaussian radius on the patch grid.
        soft: use Gaussian weights when true, otherwise a hard disk.
        min_depth: minimum positive projection depth.
    """
    if path_xyz.ndim != 4 or path_xyz.shape[-1] not in (2, 3):
        raise ValueError("path_xyz must have shape [B, M, T, 2 or 3]")
    if lidar2img.ndim != 4 or tuple(lidar2img.shape[-2:]) != (4, 4):
        raise ValueError("lidar2img must have shape [B, V, 4, 4]")
    if path_xyz.shape[0] != lidar2img.shape[0]:
        raise ValueError("path_xyz and lidar2img batch sizes must match")
    if not path_xyz.is_floating_point() or not lidar2img.is_floating_point():
        raise TypeError("path_xyz and lidar2img must be floating point")
    if not torch.isfinite(path_xyz).all() or not torch.isfinite(lidar2img).all():
        raise ValueError("path_xyz and lidar2img must contain finite values")
    image_h, image_w = (int(image_hw[0]), int(image_hw[1]))
    patch_h, patch_w = (int(patch_hw[0]), int(patch_hw[1]))
    if min(image_h, image_w, patch_h, patch_w) <= 0:
        raise ValueError("image_hw and patch_hw entries must be positive")
    if corridor_radius_patches <= 0:
        raise ValueError("corridor_radius_patches must be positive")
    if min_depth <= 0:
        raise ValueError("min_depth must be positive")

    batch, modes, steps, coordinate_dim = path_xyz.shape
    views = lidar2img.shape[1]
    if coordinate_dim == 2:
        xyz = torch.cat(
            (path_xyz, torch.zeros_like(path_xyz[..., :1])), dim=-1
        )
    else:
        xyz = path_xyz
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1)

    # [B, V, M, T, 4], then move V behind T for the public contract.
    projected = torch.einsum("bvij,bmtj->bvmti", lidar2img, homogeneous)
    depth_bvmt = projected[..., 2]
    safe_depth = torch.where(
        depth_bvmt.abs() >= min_depth,
        depth_bvmt,
        torch.full_like(depth_bvmt, min_depth),
    )
    pixel_x_bvmt = projected[..., 0] / safe_depth
    pixel_y_bvmt = projected[..., 1] / safe_depth

    visible_bvmt = (
        (depth_bvmt > min_depth)
        & (pixel_x_bvmt >= 0.0)
        & (pixel_x_bvmt <= image_w - 1)
        & (pixel_y_bvmt >= 0.0)
        & (pixel_y_bvmt <= image_h - 1)
    )

    scale_x = (patch_w - 1) / max(image_w - 1, 1)
    scale_y = (patch_h - 1) / max(image_h - 1, 1)
    patch_x_bvmt = pixel_x_bvmt * scale_x
    patch_y_bvmt = pixel_y_bvmt * scale_y

    grid_y, grid_x = torch.meshgrid(
        torch.arange(patch_h, device=path_xyz.device, dtype=path_xyz.dtype),
        torch.arange(patch_w, device=path_xyz.device, dtype=path_xyz.dtype),
        indexing="ij",
    )
    distance_sq = (
        grid_x.view(1, 1, 1, 1, patch_h, patch_w)
        - patch_x_bvmt[..., None, None]
    ).square()
    distance_sq = distance_sq + (
        grid_y.view(1, 1, 1, 1, patch_h, patch_w)
        - patch_y_bvmt[..., None, None]
    ).square()

    radius_sq = float(corridor_radius_patches) ** 2
    if soft:
        point_masks = torch.exp(-0.5 * distance_sq / radius_sq)
        # Truncation keeps an off-path Gaussian tail from producing non-zero
        # overlap everywhere in the image.
        point_masks = torch.where(
            distance_sq <= 9.0 * radius_sq,
            point_masks,
            torch.zeros_like(point_masks),
        )
    else:
        point_masks = (distance_sq <= radius_sq).to(path_xyz.dtype)
    point_masks = point_masks * visible_bvmt[..., None, None].to(path_xyz.dtype)

    # Merge time points by maximum corridor membership: [B, V, M, H, W].
    mask_bvmhw = point_masks.amax(dim=3)
    mask = mask_bvmhw.permute(0, 2, 1, 3, 4).contiguous()

    point_visible = visible_bvmt.permute(0, 2, 3, 1).contiguous()
    patch_xy = torch.stack((patch_x_bvmt, patch_y_bvmt), dim=-1)
    patch_xy = patch_xy.permute(0, 2, 3, 1, 4).contiguous()
    depth = depth_bvmt.permute(0, 2, 3, 1).contiguous()

    # A route point is covered if at least one camera sees it.  Report the
    # fraction across path time steps so callers can refuse unsupported risk.
    covered_points = point_visible.any(dim=-1)  # [B, M, T]
    coverage = covered_points.to(path_xyz.dtype).mean(dim=-1)  # [B, M]
    return ProjectedPathCorridor(
        mask=mask,
        point_visible=point_visible,
        patch_xy=patch_xy,
        depth=depth,
        coverage=coverage,
    )


__all__ = ["ProjectedPathCorridor", "project_path_corridor_to_patches"]

