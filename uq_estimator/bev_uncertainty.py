"""BEV uncertainty heatmap: spatial-aware uncertainty for mode selection.

Two paths:

1. **Attention-based** (requires QT-Former cross-attention weights):
   compute_patch_quality → compute_bev_uncertainty

2. **Attention-free / IPM-based** (geometry only, no model needed):
   compute_patch_quality → compute_bev_uncertainty_ipm

   Uses Inverse Perspective Mapping (IPM): each image patch centre is
   projected to the ground plane (z=0 in ego frame) via known camera
   intrinsics + extrinsics.  This measures *perceptual coverage quality*
   — which parts of the ground plane are well captured by the cameras.
   Low patch quality (blurry, low-contrast) → high BEV uncertainty in
   the corresponding ground region.  No model or attention weights needed.

All other functions are parameter-free (no learnable weights).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ── ① Patch Quality ─────────────────────────────────────────────────


def compute_patch_quality(
    images: torch.Tensor,
    patch_size: int = 16,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-patch image quality (higher = better).

    Args:
        images:     [B, N_views, 3, H, W] float tensor in [0, 255] or [0, 1].
        patch_size: spatial size of each patch (default 16, matching ViT).
        normalize:  if True (default), per-frame min-max normalisation to [0,1].
                    Set to False to return raw quality scores for cross-batch
                    comparisons (sharpness + texture + contrast, unnormalised).

    Returns:
        patch_quality: [B, N_views, N_patches].  In [0,1] when normalize=True.
    """
    B, N_views, C, H, W = images.shape
    device = images.device

    # Convert to grayscale: [B*N_views, 1, H, W]
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                device=device, dtype=images.dtype)
    flat = images.reshape(B * N_views, C, H, W)
    gray = (flat * gray_weights[None, :, None, None]).sum(dim=1, keepdim=True)
    # gray: [B*N_views, 1, H, W]

    # Laplacian kernel for sharpness
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=device, dtype=images.dtype,
    ).reshape(1, 1, 3, 3)
    laplacian = F.conv2d(gray, lap_kernel, padding=1)  # [B*N_views, 1, H, W]

    # Sobel kernels for gradient magnitude
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=device, dtype=images.dtype,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=device, dtype=images.dtype,
    ).reshape(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = (gx ** 2 + gy ** 2).sqrt()  # [B*N_views, 1, H, W]

    # Unfold into patches: [B*N_views, 1, H_p, W_p, P, P]
    H_p, W_p = H // patch_size, W // patch_size

    def _patch_stat(x: torch.Tensor) -> torch.Tensor:
        """Compute per-patch statistics. x: [B*N_views, 1, H, W] → [B*N_views, N_patches]."""
        # Crop to exact patch grid
        x_crop = x[:, :, :H_p * patch_size, :W_p * patch_size]
        # Reshape to patches
        x_p = x_crop.reshape(B * N_views, 1, H_p, patch_size, W_p, patch_size)
        x_p = x_p.permute(0, 1, 2, 4, 3, 5)  # [B*Nv, 1, H_p, W_p, P, P]
        x_p = x_p.reshape(B * N_views, H_p * W_p, patch_size * patch_size)
        return x_p

    # 1. Laplacian variance (sharpness)
    lap_patches = _patch_stat(laplacian.abs())
    sharpness = lap_patches.var(dim=-1)  # [B*N_views, N_patches]

    # 2. Sobel gradient mean (texture richness)
    grad_patches = _patch_stat(grad_mag)
    texture = grad_patches.mean(dim=-1)  # [B*N_views, N_patches]

    # 3. Local contrast (patch std of grayscale)
    gray_patches = _patch_stat(gray)
    contrast = gray_patches.std(dim=-1)  # [B*N_views, N_patches]

    # Equal-weight average
    quality = (sharpness + texture + contrast) / 3.0  # [B*N_views, N_patches]

    if normalize:
        # Per-frame min-max normalisation to [0, 1]
        q_min = quality.min(dim=-1, keepdim=True)[0]
        q_max = quality.max(dim=-1, keepdim=True)[0]
        quality = (quality - q_min) / (q_max - q_min + 1e-8)

    return quality.reshape(B, N_views, H_p * W_p)  # [B, N_views, N_patches]


# ── ② BEV Uncertainty ───────────────────────────────────────────────


def compute_bev_uncertainty(
    attn_weights: torch.Tensor,
    patch_quality: torch.Tensor,
) -> torch.Tensor:
    """Weighted-sum of patch uncertainty via attention weights.

    Args:
        attn_weights: [B, N_q, N_views * N_patches] (QT-Former cross-attn).
        patch_quality: [B, N_views, N_patches] in [0, 1].

    Returns:
        bev_unc: [B, N_q] in ~[0, 1], high = uncertain.
    """
    B, N_views, N_patches = patch_quality.shape
    # patch_unc: [B, N_views * N_patches]
    patch_unc = 1.0 - patch_quality.reshape(B, N_views * N_patches)

    # Normalise attention to sum=1 per query (should already be, but be safe)
    attn_norm = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

    # Weighted sum: [B, N_q, N_total] @ [B, N_total, 1] → [B, N_q]
    bev_unc = torch.bmm(attn_norm, patch_unc.unsqueeze(-1)).squeeze(-1)

    return bev_unc  # [B, N_q]


# ── ③ Trajectory Cost ───────────────────────────────────────────────


def compute_trajectory_cost(
    bev_unc: torch.Tensor,
    ref_xy: torch.Tensor,
    waypoints: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Interpolate BEV uncertainty along each trajectory mode.

    Args:
        bev_unc: [B, N_q] per-query uncertainty.
        ref_xy: [N_q, 2] BEV reference point coordinates (metres).
        waypoints: [B, N_modes, N_steps, 2] trajectory waypoints (metres).
        k: number of nearest BEV queries for inverse-distance weighting.

    Returns:
        traj_cost: [B, N_modes] average uncertainty along each trajectory.
    """
    B, N_modes, N_steps, _ = waypoints.shape
    N_q = ref_xy.shape[0]
    device = waypoints.device

    # Time weights: later steps matter more
    # [0.5, 0.5, 1.0, 1.0, 1.5, 1.5] for N_steps=6
    time_weights = torch.ones(N_steps, device=device)
    third = N_steps // 3
    time_weights[:third] = 0.5
    time_weights[-third:] = 1.5
    time_weights = time_weights / time_weights.sum()  # normalise

    # Flatten waypoints: [B * N_modes * N_steps, 2]
    wp_flat = waypoints.reshape(-1, 2)
    ref = ref_xy.unsqueeze(0).expand(wp_flat.shape[0], -1, -1)  # [BMS, N_q, 2]

    # Pairwise distances: [BMS, N_q]
    dist = torch.cdist(wp_flat.unsqueeze(1), ref).squeeze(1)  # [BMS, N_q]

    # k-NN inverse distance weighting
    k_actual = min(k, N_q)
    topk_dist, topk_idx = dist.topk(k_actual, dim=-1, largest=False)  # [BMS, k]
    inv_dist = 1.0 / (topk_dist + 1e-6)  # [BMS, k]
    inv_dist_norm = inv_dist / inv_dist.sum(dim=-1, keepdim=True)  # [BMS, k]

    # Gather BEV uncertainty at nearest queries
    # bev_unc: [B, N_q] → expand for each mode*step
    bev_unc_expand = bev_unc.unsqueeze(1).unsqueeze(1).expand(
        B, N_modes, N_steps, N_q).reshape(-1, N_q)  # [BMS, N_q]
    topk_unc = torch.gather(bev_unc_expand, 1, topk_idx)  # [BMS, k]

    # Weighted uncertainty per waypoint
    wp_unc = (inv_dist_norm * topk_unc).sum(dim=-1)  # [BMS]
    wp_unc = wp_unc.reshape(B, N_modes, N_steps)  # [B, N_modes, N_steps]

    # Time-weighted average → [B, N_modes]
    traj_cost = (wp_unc * time_weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    return traj_cost  # [B, N_modes]


# ── ④ Mode Score Adjustment ─────────────────────────────────────────


def adjust_mode_scores(
    poses_cls: torch.Tensor,
    traj_cost: torch.Tensor,
    score: torch.Tensor,
    lambda_val: float = 0.5,
    method: str = "multiplicative",
) -> torch.Tensor:
    """Adjust mode selection scores based on spatial uncertainty cost.

    Args:
        poses_cls: [B, N_modes] original mode logits/scores.
        traj_cost: [B, N_modes] per-mode uncertainty cost.
        score: [B, 1] global uncertainty score (from UQEstimator).
        lambda_val: injection strength.
        method: "multiplicative" or "logit".

    Returns:
        adjusted: [B, N_modes] adjusted scores.
    """
    if score is None:
        return poses_cls

    s = score  # [B, 1]
    penalty = lambda_val * s * traj_cost  # [B, N_modes]

    if method == "multiplicative":
        adjusted = poses_cls * (1.0 - penalty.clamp(max=1.0))
    elif method == "logit":
        # Clamp poses_cls to valid sigmoid input range
        eps = 1e-6
        p_clamped = poses_cls.clamp(eps, 1.0 - eps)
        logits = torch.log(p_clamped / (1.0 - p_clamped))
        adjusted = torch.sigmoid(logits - penalty)
    else:
        raise ValueError(f"Unknown method: {method}")

    return adjusted  # [B, N_modes]


# ── ⑤ BEV Heatmap Rendering ────────────────────────────────────────


def render_bev_heatmap(
    bev_unc: torch.Tensor,
    ref_xy: torch.Tensor,
    waypoints: Optional[torch.Tensor] = None,
    img_size: int = 512,
    bev_range: float = 51.2,
    cmap_name: str = "RdYlGn_r",
) -> np.ndarray:
    """Render BEV uncertainty as a heatmap image.

    Args:
        bev_unc: [N_q] per-query uncertainty (single sample).
        ref_xy: [N_q, 2] BEV coordinates in metres.
        waypoints: optional [N_modes, N_steps, 2] to overlay.
        img_size: output image resolution.
        bev_range: half-range of BEV in metres (default ±51.2m).
        cmap_name: matplotlib colormap name.

    Returns:
        img: [img_size, img_size, 3] uint8 numpy array.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bev_np = bev_unc.detach().cpu().numpy()
    xy_np = ref_xy.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=img_size // 6)
    scatter = ax.scatter(
        xy_np[:, 0], xy_np[:, 1],
        c=bev_np, cmap=cmap_name, vmin=0, vmax=1,
        s=8, alpha=0.8, edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="Uncertainty")

    if waypoints is not None:
        wp_np = waypoints.detach().cpu().numpy()
        if wp_np.ndim == 3:  # [N_modes, N_steps, 2]
            for mode_i in range(wp_np.shape[0]):
                ax.plot(wp_np[mode_i, :, 0], wp_np[mode_i, :, 1],
                        "o-", markersize=3, linewidth=1.5, alpha=0.7)

    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("BEV Uncertainty Heatmap")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    return img


# ── ⑥ IPM-based BEV Uncertainty (attention-free) ───────────────────


def _build_patch_centers(img_h: int, img_w: int, patch_size: int) -> torch.Tensor:
    """Return pixel coords of patch centres. → [N_patches, 2] (u, v)."""
    H_p = img_h // patch_size
    W_p = img_w // patch_size
    ys = torch.arange(H_p, dtype=torch.float32) * patch_size + patch_size / 2.0
    xs = torch.arange(W_p, dtype=torch.float32) * patch_size + patch_size / 2.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [N_p, 2]
    return centers  # (u=col, v=row)


def _ipm_project(
    centers_uv: torch.Tensor,
    K: torch.Tensor,
    cam2ego: torch.Tensor,
    bev_range: float,
) -> torch.Tensor:
    """Project pixel centres to ground plane (z=0 in ego frame) via IPM.

    Args:
        centers_uv: [N_p, 2] pixel centres (u, v).
        K:          [3, 3] camera intrinsic matrix.
        cam2ego:    [4, 4] camera-to-ego rigid transform (CARLA convention:
                    camera z=forward, y=down, x=right).
        bev_range:  BEV half-extent in metres (only used to filter invalid points).

    Returns:
        xy_bev:  [N_p, 2] ground-plane coords in metres (ego frame), NaN for
                 pixels that don't hit the ground (facing sky / behind camera).
    """
    N_p = centers_uv.shape[0]
    device = centers_uv.device
    K = K.to(device)
    cam2ego = cam2ego.to(device)

    # ─── Ray direction in camera frame ───
    # CARLA camera: x=right, y=down, z=forward
    uv1 = torch.cat([centers_uv, torch.ones(N_p, 1, device=device)], dim=1)  # [N_p, 3]
    K_inv = torch.linalg.inv(K.float())
    rays_cam = (K_inv @ uv1.T).T  # [N_p, 3], normalised direction in camera coords

    # ─── Transform ray to ego frame ───
    R = cam2ego[:3, :3].float()   # [3, 3] rotation
    t = cam2ego[:3, 3].float()    # [3]   translation (camera position in ego)

    # Camera origin in ego
    origin = t  # [3]

    # Ray directions in ego frame: [N_p, 3]
    rays_ego = (R @ rays_cam.T).T  # [N_p, 3]

    # ─── Intersect with z=0 plane ───
    # origin_z + lambda * rays_ego_z = 0  →  lambda = -origin_z / rays_ego_z
    # Valid only when rays_ego_z < 0 (ray going downward) and lambda > 0
    dz = rays_ego[:, 2]  # [N_p]
    with torch.no_grad():
        lam = -origin[2] / (dz + 1e-12)  # [N_p]

    x_bev = origin[0] + lam * rays_ego[:, 0]  # [N_p]
    y_bev = origin[1] + lam * rays_ego[:, 1]  # [N_p]

    # Mask invalid: ray going up or lambda negative (behind camera)
    valid = (dz < -1e-4) & (lam > 0.05) & (lam < 200.0)
    # Also mask points outside BEV range
    valid &= (x_bev.abs() < bev_range) & (y_bev.abs() < bev_range)

    xy_bev = torch.stack([x_bev, y_bev], dim=1)  # [N_p, 2]
    xy_bev[~valid] = float("nan")
    return xy_bev


def compute_bev_uncertainty_ipm(
    images: torch.Tensor,
    cam_intrinsics: list[torch.Tensor],
    cam2egos: list[torch.Tensor],
    patch_size: int = 16,
    bev_range: float = 51.2,
    bev_resolution: float = 0.4,
    sigma: float = 1.5,
    patch_quality: Optional[torch.Tensor] = None,
    normalize_output: bool = True,
) -> torch.Tensor:
    """Compute BEV uncertainty heatmap using Inverse Perspective Mapping.

    No model or attention weights required — uses camera geometry only.

    For each camera, per-patch image quality is computed and then projected
    to the ground plane via IPM.  Uncertainty = 1 − quality.  A Gaussian
    kernel splatts each patch's uncertainty onto the BEV grid.

    Args:
        images:          [B, N_views, 3, H, W] float tensor (any pixel range).
        cam_intrinsics:  list of N_views tensors, each [3, 3].
        cam2egos:        list of N_views tensors, each [4, 4] cam→ego.
        patch_size:      ViT patch size in pixels (default 16).
        bev_range:       half-extent of BEV grid in metres (default ±51.2 m).
        bev_resolution:  metres per BEV pixel (default 0.4 m/px → 256×256 grid).
        sigma:           Gaussian splat σ in BEV pixels (default 1.5).
        patch_quality:   optional pre-computed [B, N_views, N_patches] quality
                         tensor.  When provided, images is only used for shape.
                         Pass normalize=False quality for cross-condition comparison.
        normalize_output: if True (default), per-sample min-max normalise the
                         output BEV map to [0,1].  Set False to preserve absolute
                         magnitudes for cross-condition comparison.

    Returns:
        bev_unc: [B, H_bev, W_bev].  In [0,1] when normalize_output=True.
    """
    B, N_views, C, H, W = images.shape
    device = images.device

    grid_size = int(2 * bev_range / bev_resolution)  # e.g. 256
    H_bev = W_bev = grid_size

    # ─── Patch quality for all views ───
    if patch_quality is None:
        patch_quality = compute_patch_quality(images, patch_size=patch_size)
    # [B, N_views, N_patches], N_patches = (H // patch_size) * (W // patch_size)
    patch_unc = 1.0 - patch_quality  # [B, N_views, N_patches]

    # ─── Patch centres in pixel space ───
    centers_uv = _build_patch_centers(H, W, patch_size).to(device)  # [N_p, 2]
    N_p = centers_uv.shape[0]

    bev_out = torch.zeros(B, H_bev, W_bev, device=device)
    weight_out = torch.zeros(B, H_bev, W_bev, device=device)

    # ─── Splat each view onto BEV grid ───
    for vi in range(N_views):
        K = cam_intrinsics[vi]          # [3, 3]
        cam2ego = cam2egos[vi]          # [4, 4]

        xy_bev = _ipm_project(centers_uv, K, cam2ego, bev_range)  # [N_p, 2]
        valid_mask = ~torch.isnan(xy_bev[:, 0])  # [N_p]

        if valid_mask.sum() == 0:
            continue

        # Convert metres → grid indices (origin at centre)
        # x: left-right, y: forward-back in ego frame
        # BEV grid: row 0 = +y (forward top), col 0 = -x (left)
        xy_valid = xy_bev[valid_mask]  # [N_v, 2]
        col = ((xy_valid[:, 0] + bev_range) / bev_resolution).long()  # x → col
        row = ((bev_range - xy_valid[:, 1]) / bev_resolution).long()  # y → row (flipped)
        in_grid = (col >= 0) & (col < W_bev) & (row >= 0) & (row < H_bev)

        col = col[in_grid]
        row = row[in_grid]
        valid_idx = torch.where(valid_mask)[0][in_grid]  # original patch indices

        # Simple Gaussian splat (σ=1.5 bev_pixels ≈ 0.6 m)
        rad = int(sigma * 3)
        for b in range(B):
            unc_b = patch_unc[b, vi]  # [N_p]
            for i in range(len(col)):
                r0, c0 = row[i].item(), col[i].item()
                v = unc_b[valid_idx[i]].item()
                r_lo = max(0, r0 - rad)
                r_hi = min(H_bev, r0 + rad + 1)
                c_lo = max(0, c0 - rad)
                c_hi = min(W_bev, c0 + rad + 1)
                rs = torch.arange(r_lo, r_hi, device=device)
                cs = torch.arange(c_lo, c_hi, device=device)
                gr, gc = torch.meshgrid(rs, cs, indexing="ij")
                g = torch.exp(-((gr - r0) ** 2 + (gc - c0) ** 2) / (2 * sigma ** 2))
                bev_out[b, r_lo:r_hi, c_lo:c_hi] += v * g
                weight_out[b, r_lo:r_hi, c_lo:c_hi] += g

    # Weighted average of uncertainty (unweighted areas stay at 0)
    bev_unc = bev_out / (weight_out + 1e-8)
    # Zero out areas with no camera coverage (weight ≈ 0)
    bev_unc = bev_unc * (weight_out > 1e-6).float()

    if normalize_output:
        # Per-sample min-max to [0, 1] over COVERED pixels only
        # (uncovered pixels are 0; we normalise the non-zero range)
        covered = weight_out > 1e-6
        for b in range(B):
            vals = bev_unc[b][covered[b]]
            if vals.numel() > 0:
                mn = vals.min()
                mx = vals.max()
                if mx > mn:
                    bev_unc[b] = torch.where(
                        covered[b],
                        (bev_unc[b] - mn) / (mx - mn + 1e-8),
                        torch.zeros_like(bev_unc[b]),
                    )

    return bev_unc  # [B, H_bev, W_bev]


def make_b2d_calibration(img_h: int = 900, img_w: int = 1600) -> tuple:
    """Return hard-coded B2D camera calibrations (intrinsics, cam2egos).

    Based on the camera positions in team_code/orion_b2d_agent.py.
    Camera convention: x=right, y=down, z=forward (CARLA/OpenCV).

    Returns:
        intrinsics: list of 6 [3, 3] tensors.
        cam2egos:   list of 6 [4, 4] tensors.
        cam_names:  list of 6 camera name strings.
    """
    import math

    cam_names = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]

    # (tx, ty, tz_ego, yaw_deg, fov_deg) — yaw in ego frame (CARLA: x=forward, y=right)
    # B2D agent.py positions (ego x=forward, y=right, z=up):
    cam_specs = [
        # tx,    ty,    tz,   yaw_deg, fov_deg
        (0.80,  0.00,  1.60,    0.0,  70.0),  # FRONT
        (0.27, -0.55,  1.60,  -55.0,  70.0),  # FRONT_LEFT
        (0.27,  0.55,  1.60,   55.0,  70.0),  # FRONT_RIGHT
        (-2.0,  0.00,  1.60,  180.0, 110.0),  # BACK
        (-0.32, -0.55, 1.60, -110.0,  70.0),  # BACK_LEFT
        (-0.32,  0.55, 1.60,  110.0,  70.0),  # BACK_RIGHT
    ]

    intrinsics = []
    cam2egos = []

    for tx, ty, tz, yaw_deg, fov_deg in cam_specs:
        # ─── Intrinsics ───
        f = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        K = torch.tensor([
            [f,   0.0, img_w / 2.0],
            [0.0, f,   img_h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        intrinsics.append(K)

        # ─── cam2ego ───
        # Ego frame: x=forward, y=right, z=up (CARLA UE4)
        # Camera frame: x=right, y=down, z=forward (CARLA camera / OpenCV)
        # We build R_ego2cam first, then invert for cam2ego.
        #
        # Step 1: rotate ego axes so camera z points at yaw_deg:
        #   camera_z in ego = (cos(yaw), sin(yaw), 0)
        #   camera_x in ego = (-sin(yaw), cos(yaw), 0)
        #   camera_y in ego = (0, 0, -1)  [y=down → z_ego=-1]
        yaw = math.radians(yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # Rows of R_ego2cam: camera axes expressed in ego frame
        # camera x (right) in ego
        cx_in_ego = torch.tensor([-sy, cy, 0.0])
        # camera y (down) in ego: down = -z_ego
        cy_in_ego = torch.tensor([0.0, 0.0, -1.0])
        # camera z (forward) in ego
        cz_in_ego = torch.tensor([cy, sy, 0.0])

        # R_cam2ego: columns = camera axes in ego frame
        R_cam2ego = torch.stack([cx_in_ego, cy_in_ego, cz_in_ego], dim=1)  # [3, 3]

        t_cam_in_ego = torch.tensor([tx, ty, tz], dtype=torch.float32)

        cam2ego_mat = torch.eye(4, dtype=torch.float32)
        cam2ego_mat[:3, :3] = R_cam2ego
        cam2ego_mat[:3, 3] = t_cam_in_ego
        cam2egos.append(cam2ego_mat)

    return intrinsics, cam2egos, cam_names


# ── Attention Hook Helper ───────────────────────────────────────────


def register_attn_hook(model) -> dict:
    """Register a forward hook to capture QT-Former cross-attention weights.

    Requires ``flash_attn=False`` in config (Flash Attention doesn't return
    attention weights).

    Args:
        model: the ORION detector model.

    Returns:
        storage: dict with key 'attn_weights' populated after each forward.
    """
    storage: dict = {}

    # Target: last decoder layer, 3rd sub-layer (cross-attention to image tokens)
    target = (
        model.pts_bbox_head.transformer
        .query_decoder._layers[-1]
        .transformer_layers[2]
    )

    def _hook(module, inputs, outputs):
        # MultiHeadAttentionwDropout returns (output, attn_weights) when
        # attn_weights are available
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            storage["attn_weights"] = outputs[1]

    target.register_forward_hook(_hook)
    return storage


# ── Smoke Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== BEV Uncertainty Smoke Test ===\n")

    B, N_views, N_patches, N_q = 2, 6, 256, 600
    N_modes, N_steps = 20, 6

    # ① Patch quality
    images = torch.rand(B, N_views, 3, 640, 640)
    pq = compute_patch_quality(images, patch_size=16)
    assert pq.shape == (B, N_views, 1600), f"patch_quality shape: {pq.shape}"
    assert pq.min() >= 0 and pq.max() <= 1, "patch_quality out of [0,1]"
    print(f"[PASS] compute_patch_quality: {pq.shape}, range [{pq.min():.3f}, {pq.max():.3f}]")

    # ② BEV uncertainty (use mock N_patches=256 for speed)
    pq_mock = torch.rand(B, N_views, N_patches)
    attn_w = torch.softmax(torch.randn(B, N_q, N_views * N_patches), dim=-1)
    bev = compute_bev_uncertainty(attn_w, pq_mock)
    assert bev.shape == (B, N_q), f"bev_unc shape: {bev.shape}"
    print(f"[PASS] compute_bev_uncertainty: {bev.shape}, range [{bev.min():.3f}, {bev.max():.3f}]")

    # ③ Trajectory cost
    ref_xy = torch.randn(N_q, 2) * 30  # random BEV positions in metres
    waypoints = torch.randn(B, N_modes, N_steps, 2) * 20
    tc = compute_trajectory_cost(bev, ref_xy, waypoints, k=5)
    assert tc.shape == (B, N_modes), f"traj_cost shape: {tc.shape}"
    print(f"[PASS] compute_trajectory_cost: {tc.shape}")

    # ④ Adjust scores
    poses_cls = torch.softmax(torch.randn(B, N_modes), dim=-1)
    score = torch.tensor([[0.8], [0.2]])
    adj = adjust_mode_scores(poses_cls, tc, score, lambda_val=0.5)
    assert adj.shape == (B, N_modes), f"adjusted shape: {adj.shape}"
    # score=0 identity check
    zero_score = torch.zeros(B, 1)
    adj_zero = adjust_mode_scores(poses_cls, tc, zero_score)
    torch.testing.assert_close(adj_zero, poses_cls)
    print(f"[PASS] adjust_mode_scores: identity at score=0 verified")

    # ⑤ Render heatmap
    img = render_bev_heatmap(bev[0], ref_xy, waypoints[0])
    assert img.ndim == 3 and img.shape[2] == 3, f"heatmap shape: {img.shape}"
    print(f"[PASS] render_bev_heatmap: {img.shape}")

    print("\n=== All smoke tests PASSED ===")
