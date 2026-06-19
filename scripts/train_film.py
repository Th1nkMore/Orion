"""
FiLM L1 fine-tuning script.

Trains only the FiLM modulation layers (film_gamma, film_beta) in the
PETRTemporalTransformer while keeping ALL other ORION parameters frozen.

Uses teacher-forcing through the LLM (not generate()) to maintain gradient
flow from trajectory loss through FiLM:
    trajectory_loss → VAE → ego_feature → LLM(teacher forcing) →
    vlm_memory → QT-Former(+FiLM) → FiLM layers

Usage:
    python scripts/train_film.py \
        --config adzoo/orion/configs/orion_stage3_infer.py \
        --checkpoint ckpts/Orion.pth \
        --epochs 3 \
        --lr 1e-3 \
        --out checkpoints/film/best.pt
"""
import argparse
import copy
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv.utils import Config, load_checkpoint, set_random_seed, ProgressBar
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader
from uq_estimator.grounding import grounding_loss
from uq_estimator.training import low_uq_consistency_loss


def parse_args():
    p = argparse.ArgumentParser(description='FiLM L1 fine-tuning')
    p.add_argument('--config', default='adzoo/orion/configs/orion_stage3_infer.py')
    p.add_argument('--checkpoint', default='ckpts/Orion.pth')
    p.add_argument('--ann-file', default=None,
                   help='override annotation file (default: val set from config)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grad-accum', type=int, default=4,
                   help='gradient accumulation steps (effective batch = grad_accum)')
    p.add_argument('--max-samples', type=int, default=None,
                   help='limit training samples (for testing)')
    p.add_argument('--lambda-col', type=float, default=1.0,
                   help='weight for collision margin loss (0 to disable)')
    p.add_argument('--col-margin', type=float, default=4.0,
                   help='safety margin in meters for collision loss')
    p.add_argument('--lambda-film-reg', type=float, default=0.0,
                   help='weight for low-UQ FiLM amplitude regularization')
    p.add_argument('--lambda-progress', type=float, default=0.0,
                   help='weight for under-progress penalty against GT trajectory')
    p.add_argument('--lambda-comfort', type=float, default=0.0,
                   help='weight for trajectory smoothness penalty (acc + jerk)')
    p.add_argument('--out', default='checkpoints/film/best.pt')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ── Custom FP16 wrapper (from test.py) ────────────────────────────��─────
custom_fp16 = dict(map_head=False, pts_bbox_head=False)

def custom_wrap_fp16_model(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        if module_name in model._modules:
            model._modules[module_name].fp16_enabled = v


def gt_collision_margin_loss(ego_fut_preds, gt_attr_labels, gt_bboxes_3d,
                             uq_score=None, margin=2.0):
    """Differentiable collision margin loss using GT agent trajectories.

    Args:
        ego_fut_preds: [B, 20, 6, 2] predicted trajectory offsets (differentiable)
        gt_attr_labels: [N, 34] GT agent attributes (dims 0-11 = future offsets)
        gt_bboxes_3d: LiDARInstance3DBoxes [N, 9] (x,y,z,w,l,h,yaw,vx,vy)
        uq_score: [B, 1] uncertainty score (detached), or None
        margin: safety distance in meters

    Returns:
        Scalar loss tensor
    """
    device = ego_fut_preds.device

    # No agents → no collision risk
    if gt_attr_labels is None or gt_attr_labels.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    # gt_bboxes_3d may be a list (from dataloader); unwrap to get the boxes object
    bboxes = gt_bboxes_3d
    while isinstance(bboxes, list):
        if len(bboxes) == 0:
            return torch.tensor(0.0, device=device)
        bboxes = bboxes[0]

    # Agent current position from bboxes
    agent_xy = bboxes.tensor[:, :2].to(device)  # [N, 2]

    # Agent future trajectory offsets from gt_attr_labels dims 0-11 → [N, 6, 2]
    if gt_attr_labels.dim() == 1:
        gt_attr_labels = gt_attr_labels.unsqueeze(0)
    N = gt_attr_labels.shape[0]
    n_cols = gt_attr_labels.shape[1] if gt_attr_labels.dim() > 1 else 0
    if n_cols < 12:
        return torch.tensor(0.0, device=device)
    fut_data = gt_attr_labels[:, :12]  # [N, 12]
    agent_fut_offsets = fut_data.reshape(N, 6, 2).to(device)

    # Cumulative sum to get absolute future positions relative to current pos
    agent_abs = agent_xy[:, None, :] + agent_fut_offsets.cumsum(dim=1)  # [N, 6, 2]

    # Ego best-mode cumulative trajectory: [B, 6, 2]
    ego_cum = ego_fut_preds[:, 0].cumsum(dim=1)

    # Pairwise distance: ego [B, 6, 1, 2] vs agent [1, 1, N, 2] → [B, 6, N]
    dist = torch.norm(
        ego_cum[:, :, None, :] - agent_abs[None, :, :, :].permute(0, 2, 1, 3),
        dim=-1)

    # Min distance to any agent at each timestep: [B, 6]
    min_dist = dist.min(dim=2)[0]

    # Hinge loss: penalize when closer than margin
    violation = torch.relu(margin - min_dist)  # [B, 6]

    # UQ-weighted: higher uncertainty → stronger collision penalty
    if uq_score is not None:
        violation = violation * uq_score  # [B, 1] broadcasts to [B, 6]

    return violation.mean()


def freeze_all_except_film(model):
    """Freeze all parameters except FiLM layers in the transformer."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze FiLM layers
    film_params = []
    for name, param in model.named_parameters():
        if 'film_gamma' in name or 'film_beta' in name:
            param.requires_grad = True
            film_params.append((name, param))

    n_trainable = sum(p.numel() for _, p in film_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'[FiLM] Trainable: {n_trainable:,} / {n_total:,} params')
    for name, p in film_params:
        print(f'  {name}: {p.shape}')
    return [p for _, p in film_params]


def compute_film_reg_loss(orion, uncertainty_score):
    """Regularize FiLM to stay near identity on low-UQ samples."""
    losses = []
    sample_weight = None
    if uncertainty_score is not None:
        sample_weight = (1.0 - uncertainty_score.detach()).view(-1, 1)

    def _weighted_identity(gamma_layer, beta_layer):
        gamma_w = gamma_layer.weight
        gamma_b = gamma_layer.bias - 1.0
        beta_w = beta_layer.weight
        beta_b = beta_layer.bias
        base = gamma_w.pow(2).mean() + gamma_b.pow(2).mean()
        base = base + beta_w.pow(2).mean() + beta_b.pow(2).mean()
        if sample_weight is None:
            return base
        return sample_weight.mean() * base

    transformer = orion.pts_bbox_head.transformer
    if hasattr(transformer, 'film_gamma') and hasattr(transformer, 'film_beta'):
        losses.append(_weighted_identity(transformer.film_gamma, transformer.film_beta))
    if hasattr(orion, 'film_gamma_l2') and hasattr(orion, 'film_beta_l2'):
        losses.append(_weighted_identity(orion.film_gamma_l2, orion.film_beta_l2))

    if not losses:
        return torch.tensor(0.0, device='cuda')
    return sum(losses)


def compute_progress_loss(ego_fut_preds, ego_fut_trajs, ego_fut_masks):
    """Penalize predictions that lag behind GT forward progress."""
    pred_abs = ego_fut_preds[:, 0].cumsum(dim=1)  # [B, 6, 2]
    gt_abs = ego_fut_trajs[:, 0].cumsum(dim=1)    # [B, 6, 2]
    valid = ego_fut_masks.float()                 # [B, 6]
    pred_forward = pred_abs[..., 1]
    gt_forward = gt_abs[..., 1]
    under_progress = torch.relu(gt_forward - pred_forward)
    denom = valid.sum().clamp_min(1.0)
    return (under_progress * valid).sum() / denom


def compute_comfort_loss(ego_fut_preds, ego_fut_masks):
    """Smooth trajectory in offset space via acceleration and jerk penalties."""
    pred_abs = ego_fut_preds[:, 0].cumsum(dim=1)  # [B, 6, 2]
    vel = pred_abs[:, 1:] - pred_abs[:, :-1]      # [B, 5, 2]
    acc = vel[:, 1:] - vel[:, :-1]                # [B, 4, 2]
    jerk = acc[:, 1:] - acc[:, :-1]               # [B, 3, 2]

    mask = ego_fut_masks.float()
    acc_mask = (mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).unsqueeze(-1)
    jerk_mask = (mask[:, 3:] * mask[:, 2:-1] * mask[:, 1:-2] * mask[:, :-3]).unsqueeze(-1)

    acc_loss = (acc.pow(2) * acc_mask).sum() / acc_mask.sum().clamp_min(1.0)
    jerk_loss = (jerk.pow(2) * jerk_mask).sum() / jerk_mask.sum().clamp_min(1.0)
    return acc_loss + jerk_loss


def forward_film_training(model, data, lambda_col=0.0, col_margin=2.0,
                          lambda_film_reg=0.0, lambda_progress=0.0,
                          lambda_comfort=0.0, lambda_uq_consistency=0.0,
                          lambda_plan=1.0, lambda_vae=0.1, lambda_vlm=0.01,
                          lambda_ground=0.0, uq_mode="correct",
                          shuffled_uq=None, grounding_only=False,
                          counterfactual_grounding=False,
                          token_input="score_direction"):
    """Custom forward for FiLM training with test-format data.

    Replicates ORION's inference path through QT-Former + FiLM, then
    teacher-forces through LLM and computes planning loss.

    Args:
        lambda_col: weight for collision margin loss (0 to disable)
        col_margin: safety margin in meters

    Returns dict with 'loss' tensor and 'log_vars' dict.
    """
    orion = model.module if hasattr(model, 'module') else model
    B = 1

    # ─── Step 1: unpack test-format data ───
    img_metas = data['img_metas']
    SPECIAL_KEYS = {'img', 'input_ids', 'gt_bboxes_3d', 'gt_attr_labels', 'vlm_labels', 'img_metas'}
    for key in list(data.keys()):
        if key in SPECIAL_KEYS:
            continue
        data[key] = data[key][0][0].unsqueeze(0).cuda()

    data['img'] = data['img'][0].cuda()

    # input_ids: unwrap nested lists to tensor
    raw_ids = data['input_ids']
    while isinstance(raw_ids, list) and len(raw_ids) > 0:
        raw_ids = raw_ids[0]
    # vlm_labels: use input_ids as labels (test format has string labels)
    input_ids_1d = raw_ids  # [N]
    vlm_labels_1d = raw_ids.clone()

    data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]

    # Unpack gt_attr_labels for collision loss: nested list → [N, 34] tensor
    if 'gt_attr_labels' in data:
        attr = data['gt_attr_labels']
        while isinstance(attr, (list, tuple)):
            if len(attr) == 0:
                attr = None
                break
            attr = attr[0]
        if attr is not None and torch.is_tensor(attr):
            data['gt_attr_labels'] = attr.cuda()
        else:
            data.pop('gt_attr_labels', None)

    # Skip samples without valid future trajectory
    ego_fut_masks = data['ego_fut_masks']  # [1, 1, 1, 6] after unpacking
    if not grounding_only and ego_fut_masks.sum() == 0:
        return None

    # ─── Step 2: extract features ───
    data['img_feats'] = orion.extract_feat(data['img'])
    if data['img'].dim() == 4:
        data['img'] = data['img'].unsqueeze(0)

    img_metas_list = img_metas[0]
    data.pop('img_metas', None)

    # ─── Step 3: position embedding ───
    location = orion.prepare_location(img_metas_list, **data)
    pos_embed = orion.position_embeding(data, location, img_metas_list)

    # ─── Step 4: detection + UQ + FiLM L1 ───
    outs_bbox, det_query, uncertainty_emb, uncertainty_score = orion.pts_bbox_head(img_metas_list, pos_embed, **data)
    target_uncertainty_score = uncertainty_score.detach()
    uq_output = getattr(orion.pts_bbox_head, 'uq_output', None)
    active_embedding = getattr(uq_output, 'active_embedding', None)
    if token_input == "score_only":
        active_embedding = torch.zeros_like(active_embedding)
    elif token_input != "score_direction":
        raise ValueError(f"Unknown token_input mode: {token_input}")
    vision_embeded_obj = det_query.clone()

    # ─── Step 5: map head ───
    if orion.with_map_head:
        outs_lane, map_query = orion.map_head(img_metas_list, pos_embed, **data)
        vision_embeded_map = map_query.clone()

    # ─── Step 6: LLM teacher forcing ───
    if not (orion.with_lm_head and orion.use_gen_token):
        return None
    vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
    baseline_vision_embeded = vision_embeded
    if uq_mode == "correct":
        vision_embeded = orion._append_uq_tokens(
            vision_embeded,
            uncertainty_emb,
            uncertainty_score,
            active_embedding=active_embedding,
        )
    elif uq_mode == "zero":
        vision_embeded = orion._append_uq_tokens(
            vision_embeded,
            torch.zeros_like(uncertainty_emb),
            torch.zeros_like(uncertainty_score),
            active_embedding=torch.zeros_like(active_embedding),
        )
    elif uq_mode == "shuffled":
        if shuffled_uq is None:
            raise ValueError("shuffled mode requires shuffled_uq")
        shuffled_active, shuffled_score = shuffled_uq
        if token_input == "score_only":
            shuffled_active = torch.zeros_like(shuffled_active)
        vision_embeded = orion._append_uq_tokens(
            vision_embeded,
            uncertainty_emb,
            shuffled_score,
            active_embedding=shuffled_active,
        )
    elif uq_mode != "none":
        raise ValueError(f"Unknown UQ token mode: {uq_mode}")

    if orion.tokenizer is None:
        return None
    pad_id = orion.tokenizer.pad_token_id or 0
    input_ids = input_ids_1d.unsqueeze(0).cuda()
    vlm_labels = vlm_labels_1d.unsqueeze(0).cuda()
    input_ids = input_ids[:, :orion.tokenizer.model_max_length]
    vlm_labels = vlm_labels[:, :orion.tokenizer.model_max_length]
    vlm_attn_mask = input_ids.ne(pad_id)

    vlm_output, ego_feature = orion.lm_head(
        input_ids=input_ids, attention_mask=vlm_attn_mask,
        labels=vlm_labels, images=vision_embeded,
        use_cache=False, return_ego_feature=True)
    # vlm_output may be a CausalLMOutputWithPast or a tuple
    if hasattr(vlm_output, 'loss'):
        vlm_loss = vlm_output.loss  # HuggingFace output object
    elif isinstance(vlm_output, (tuple, list)):
        vlm_loss = vlm_output[0]
    else:
        vlm_loss = vlm_output

    consistency_loss_val = torch.tensor(0.0, device=ego_feature.device)
    if lambda_uq_consistency > 0:
        with torch.no_grad():
            _, baseline_ego_feature = orion.lm_head(
                input_ids=input_ids,
                attention_mask=vlm_attn_mask,
                labels=None,
                images=baseline_vision_embeded,
                use_cache=False,
                return_ego_feature=True,
            )
        consistency_loss_val = low_uq_consistency_loss(
            ego_feature,
            baseline_ego_feature,
            uncertainty_score,
        )

    predicted_grounding_score = orion.uq_grounding_head(ego_feature)
    grounding_loss_val = grounding_loss(
        predicted_grounding_score,
        target_uncertainty_score,
    )
    counterfactual_loss_val = torch.tensor(0.0, device=ego_feature.device)
    if counterfactual_grounding:
        if shuffled_uq is None:
            raise ValueError(
                "counterfactual grounding requires shuffled_uq"
            )
        shuffled_active, shuffled_score = shuffled_uq
        if token_input == "score_only":
            shuffled_active = torch.zeros_like(shuffled_active)
        counterfactual_vision = orion._append_uq_tokens(
            baseline_vision_embeded,
            uncertainty_emb,
            shuffled_score,
            active_embedding=shuffled_active,
        )
        _, counterfactual_feature = orion.lm_head(
            input_ids=input_ids,
            attention_mask=vlm_attn_mask,
            labels=None,
            images=counterfactual_vision,
            use_cache=False,
            return_ego_feature=True,
        )
        counterfactual_prediction = orion.uq_grounding_head(
            counterfactual_feature
        )
        counterfactual_loss_val = grounding_loss(
            counterfactual_prediction,
            shuffled_score,
        )
        grounding_loss_val = 0.5 * (
            grounding_loss_val + counterfactual_loss_val
        )

    if grounding_only:
        total = lambda_vlm * vlm_loss + lambda_ground * grounding_loss_val
        total = total + lambda_uq_consistency * consistency_loss_val
        return {
            'loss': total,
            'predicted_score': predicted_grounding_score.detach(),
            'target_score': target_uncertainty_score.detach(),
            'log_vars': {
                'vlm': float(vlm_loss.detach()),
                'uq_ground': float(grounding_loss_val.detach()),
                'uq_ground_counterfactual': float(
                    counterfactual_loss_val.detach()
                ),
                'uq_consistency': float(consistency_loss_val.detach()),
                'weighted_vlm': float((lambda_vlm * vlm_loss).detach()),
                'weighted_ground': float(
                    (lambda_ground * grounding_loss_val).detach()
                ),
                'weighted_consistency': float(
                    (lambda_uq_consistency * consistency_loss_val).detach()
                ),
                'total': float(total.detach()),
                'target_score': float(target_uncertainty_score.mean()),
                'predicted_score': float(predicted_grounding_score.mean()),
            },
        }

    # Handle mixed QA training
    if orion.mix_qa_training:
        dummy_ego_feature = orion.lm_head.get_model().embed_tokens(
            torch.tensor([[orion.lm_head.config.waypoint_token_idx] for _ in range(B)]).cuda())
        dummy_ego_feature = dummy_ego_feature.squeeze(1)
        valid_input_mask = (input_ids == orion.lm_head.config.waypoint_token_idx).sum(dim=-1).to(torch.bool)
        dummy_ego_feature[valid_input_mask] = ego_feature
        ego_feature = dummy_ego_feature
        data['ego_fut_masks'][:, 0, 0] *= valid_input_mask.unsqueeze(-1)

    current_states = ego_feature.unsqueeze(1)

    # [UQ] Score-Gated FiLM L2: modulate before VAE
    if hasattr(orion, 'use_uncertainty_l2') and orion.use_uncertainty_l2 and uncertainty_emb is not None:
        gamma_raw_l2 = orion.film_gamma_l2(uncertainty_emb)
        beta_raw_l2 = orion.film_beta_l2(uncertainty_emb)
        if uncertainty_score is not None:
            s = uncertainty_score.unsqueeze(-1)  # [B, 1, 1]
            gamma_l2 = 1.0 + s * (gamma_raw_l2.unsqueeze(1) - 1.0)
            beta_l2 = s * beta_raw_l2.unsqueeze(1)
        else:
            gamma_l2 = gamma_raw_l2.unsqueeze(1)
            beta_l2 = beta_raw_l2.unsqueeze(1)
        current_states = gamma_l2 * current_states + beta_l2

    # ─── Step 7: VAE → trajectory ───
    if not orion.use_diff_decoder and not orion.use_mlp_decoder:
        ego_fut_trajs = data['ego_fut_trajs']
        distribution_comp = {}
        noise = None
        orion.fut_ts = 6

        future_distribution_inputs = ego_fut_trajs.reshape(B, ego_fut_trajs.shape[1], -1)
        if orion.PROBABILISTIC:
            sample, output_distribution = orion.distribution_forward(
                current_states, future_distribution_inputs, noise)
            distribution_comp = {**distribution_comp, **output_distribution}

        hidden_states = ego_feature.unsqueeze(1)
        states_hs, future_states_hs = orion.future_states_predict(
            B, sample, hidden_states, current_states)

        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
        ego_fut_trajs_list = []
        for i in range(orion.fut_ts):
            outputs_ego_trajs = orion.ego_fut_decoder(
                ego_query_hs[i]).reshape(B, orion.ego_fut_mode, 2)
            ego_fut_trajs_list.append(outputs_ego_trajs)

        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)

        # ─── Step 8: planning loss (reg only) ───
        saved_col = orion.use_col_loss
        orion.use_col_loss = False
        loss_planning = orion.loss_planning(
            ego_fut_preds, ego_fut_trajs[:, 0],
            data['ego_fut_masks'][:, 0, 0], data['ego_fut_cmd'][:, 0, 0])
        orion.use_col_loss = saved_col

        loss_vae = orion.loss_vae_gen(distribution_comp, data['ego_fut_masks'][:, 0, 0])
        loss_vae = torch.nan_to_num(loss_vae)

        # Combine losses
        plan_loss = loss_planning.get('loss_plan_reg', torch.tensor(0.0, device='cuda'))
        vlm_loss_val = vlm_loss if torch.is_tensor(vlm_loss) else torch.tensor(0.0, device='cuda')
        total = (
            lambda_plan * plan_loss
            + lambda_vae * loss_vae
            + lambda_vlm * vlm_loss_val
        )
        total = total + lambda_uq_consistency * consistency_loss_val
        total = total + lambda_ground * grounding_loss_val

        # Collision margin loss (Plan C)
        col_loss_val = torch.tensor(0.0, device='cuda')
        if lambda_col > 0 and 'gt_attr_labels' in data:
            # Get UQ score (detached — no backprop through UQ estimator)
            uq_score = None
            if hasattr(orion.pts_bbox_head, 'uq_output') and orion.pts_bbox_head.uq_output is not None:
                uq_score = orion.pts_bbox_head.uq_output.score.detach()

            col_loss_val = gt_collision_margin_loss(
                ego_fut_preds,
                data['gt_attr_labels'],
                data['gt_bboxes_3d'],
                uq_score=uq_score,
                margin=col_margin)
            total = total + lambda_col * col_loss_val

        film_reg_val = torch.tensor(0.0, device='cuda')
        if lambda_film_reg > 0:
            film_reg_val = compute_film_reg_loss(orion, uncertainty_score)
            total = total + lambda_film_reg * film_reg_val

        progress_loss_val = torch.tensor(0.0, device='cuda')
        if lambda_progress > 0:
            progress_loss_val = compute_progress_loss(
                ego_fut_preds,
                ego_fut_trajs,
                data['ego_fut_masks'][:, 0, 0],
            )
            total = total + lambda_progress * progress_loss_val

        comfort_loss_val = torch.tensor(0.0, device='cuda')
        if lambda_comfort > 0:
            comfort_loss_val = compute_comfort_loss(
                ego_fut_preds,
                data['ego_fut_masks'][:, 0, 0],
            )
            total = total + lambda_comfort * comfort_loss_val

        log_vars = {
            'plan_reg': plan_loss.item(),
            'vae': loss_vae.item(),
            'vlm': vlm_loss_val.item() if torch.is_tensor(vlm_loss_val) else 0,
            'col': col_loss_val.item(),
            'film_reg': film_reg_val.item(),
            'progress': progress_loss_val.item(),
            'comfort': comfort_loss_val.item(),
            'uq_consistency': consistency_loss_val.item(),
            'uq_ground': grounding_loss_val.item(),
            'weighted_plan': (lambda_plan * plan_loss).item(),
            'weighted_vae': (lambda_vae * loss_vae).item(),
            'weighted_vlm': (lambda_vlm * vlm_loss_val).item(),
            'weighted_consistency': (
                lambda_uq_consistency * consistency_loss_val
            ).item(),
            'weighted_ground': (lambda_ground * grounding_loss_val).item(),
            'total': total.item(),
        }
        ego_command = data['ego_fut_cmd'][:, 0, 0].to(
            dtype=ego_fut_preds.dtype
        )
        active_ego_fut_preds = (
            ego_fut_preds * ego_command[..., None, None]
        ).sum(dim=1)
        return {
            'loss': total,
            'log_vars': log_vars,
            'predicted_score': predicted_grounding_score.detach(),
            'target_score': target_uncertainty_score.detach(),
            'planning_prediction': active_ego_fut_preds.detach(),
            'planning_target': ego_fut_trajs[:, 0].detach(),
            'planning_mask': data['ego_fut_masks'][:, 0, 0].detach(),
        }

    return None


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)

    # ─── Config: use inference config with test pipeline ───
    # Test pipeline provides ego_fut_trajs when using the full val set
    # (individual frames in the small val set may lack sequential neighbors)
    cfg = Config.fromfile(args.config)
    cfg.model.pts_bbox_head.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = 'checkpoints/density_uq/best.pt'

    # Training mode selection via env vars (used by run_ablation.sh)
    l2_only = os.environ.get('USE_FILM_L2_ONLY', '0') == '1'
    l1l2 = os.environ.get('USE_FILM_L1L2', '0') == '1'

    if l2_only:
        # Group C: L2 only — disable FiLM L1, enable L2
        cfg.model.pts_bbox_head.transformer.use_uncertainty = False
        cfg.model.use_uncertainty_l2 = True
        print('[FiLM] Mode: L2 only (QT-Former FiLM disabled, VAE FiLM enabled)')
    elif l1l2:
        # Group D: L1 + L2
        cfg.model.pts_bbox_head.transformer.use_uncertainty = True
        cfg.model.use_uncertainty_l2 = True
        print('[FiLM] Mode: L1 + L2 (both QT-Former and VAE FiLM enabled)')
    else:
        # Group B: L1 only (default)
        cfg.model.pts_bbox_head.transformer.use_uncertainty = True
        cfg.model.use_uncertainty_l2 = False
        print('[FiLM] Mode: L1 only (QT-Former FiLM enabled)')

    # Use full val set (has sequential frames for ego_fut_trajs)
    ann_file = args.ann_file or 'data/infos/b2d_infos_val.pkl'
    cfg.data.test.ann_file = ann_file

    # ─── Build dataset and dataloader ───
    dataset = build_dataset(cfg.data.test)
    if args.max_samples and args.max_samples < len(dataset):
        # Subsample dataset — keep extra margin for sequential access (sample_interval=5)
        keep = min(args.max_samples + 50, len(dataset))
        dataset.data_infos = dataset.data_infos[:keep]

    # Ensure flag matches current dataset length (required by GroupSampler)
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=True,  # shuffle for training
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f'Training dataset: {len(dataset)} samples, dataloader: {len(data_loader)} batches')

    # ─── Build model ───
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)

    # Load ORION checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print(f'Loaded ORION checkpoint from {args.checkpoint}')

    # Reload UQ checkpoint
    pts_cfg = cfg.model.get('pts_bbox_head', {})
    if pts_cfg.get('use_uncertainty') and pts_cfg.get('uq_checkpoint'):
        uq_ckpt_path = pts_cfg['uq_checkpoint']
        if os.path.exists(uq_ckpt_path):
            uq_ckpt = torch.load(uq_ckpt_path, map_location='cpu', weights_only=False)
            from uq_estimator.density import get_uq_state_dict
            model.pts_bbox_head.uq_estimator.load_state_dict(
                get_uq_state_dict(uq_ckpt), strict=False)
            print(f'[UQ] Reloaded UQEstimator from {uq_ckpt_path}')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    # ─── Freeze all except FiLM ───
    film_params = freeze_all_except_film(model)

    model.cuda()
    model.train()  # train mode needed for RNN backward (GRU in VAE)

    # Disable denoising training in detection head (requires training-only data fields)
    model.pts_bbox_head.with_dn = False

    # Disable gradient checkpointing in LLM — it blocks gradient flow from
    # vision_embeded through the LLM to the planning loss
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'gradient_checkpointing_disable'):
        model.lm_head.gradient_checkpointing_disable()
        print('[FiLM] Disabled gradient checkpointing in LLM for gradient flow')

    # Keep batch norm and dropout in eval mode for frozen layers
    for name, module in model.named_modules():
        if 'film_gamma' not in name and 'film_beta' not in name:
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)):
                module.eval()

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(film_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(data_loader), eta_min=args.lr * 0.01)

    # ─── Training loop ───
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_loss = float('inf')

    print(f'\nStarting FiLM training: {args.epochs} epochs, lr={args.lr}')
    print(f'Gradient accumulation: {args.grad_accum} steps')

    for epoch in range(args.epochs):
        epoch_losses = []
        optimizer.zero_grad()
        prog_bar = ProgressBar(len(data_loader))

        for step, data in enumerate(data_loader):
            try:
                outputs = forward_film_training(
                    model, data,
                    lambda_col=args.lambda_col,
                    col_margin=args.col_margin,
                    lambda_film_reg=args.lambda_film_reg,
                    lambda_progress=args.lambda_progress,
                    lambda_comfort=args.lambda_comfort)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f'\n[WARN] Step {step} failed: {e}\n{tb}')
                prog_bar.update()
                continue

            if outputs is None or 'loss' not in outputs:
                prog_bar.update()
                continue

            total_loss = outputs['loss']

            if total_loss.requires_grad:
                (total_loss / args.grad_accum).backward()
                epoch_losses.append(total_loss.item())
                if step % 200 == 0:
                    log_vars = outputs.get('log_vars', {})
                    print(f'\n  [Step {step}] '
                          f'{" ".join(f"{k}={v:.4f}" for k,v in log_vars.items())}')

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(film_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            prog_bar.update()

        # Epoch summary
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            print(f'\nEpoch {epoch+1}/{args.epochs}: loss={mean_loss:.4f} '
                  f'(lr={scheduler.get_last_lr()[0]:.2e})')

            # Save best
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_dict = {'epoch': epoch + 1, 'loss': mean_loss}
                # Save FiLM L1 weights if present
                transformer = model.pts_bbox_head.transformer
                if hasattr(transformer, 'film_gamma'):
                    save_dict['film_gamma_weight'] = transformer.film_gamma.weight.data.cpu()
                    save_dict['film_gamma_bias'] = transformer.film_gamma.bias.data.cpu()
                    save_dict['film_beta_weight'] = transformer.film_beta.weight.data.cpu()
                    save_dict['film_beta_bias'] = transformer.film_beta.bias.data.cpu()
                # Save FiLM L2 weights if present
                if hasattr(model, 'film_gamma_l2'):
                    save_dict['film_gamma_l2_weight'] = model.film_gamma_l2.weight.data.cpu()
                    save_dict['film_gamma_l2_bias'] = model.film_gamma_l2.bias.data.cpu()
                    save_dict['film_beta_l2_weight'] = model.film_beta_l2.weight.data.cpu()
                    save_dict['film_beta_l2_bias'] = model.film_beta_l2.bias.data.cpu()
                torch.save(save_dict, args.out)
                print(f'  Saved best checkpoint (loss={mean_loss:.4f})')
        else:
            print(f'\nEpoch {epoch+1}/{args.epochs}: no valid losses computed')

    # Print final FiLM statistics
    print(f'\nFinal FiLM statistics:')
    transformer = model.pts_bbox_head.transformer
    if hasattr(transformer, 'film_gamma'):
        for name, param in [('gamma_w', transformer.film_gamma.weight.data),
                            ('gamma_b', transformer.film_gamma.bias.data),
                            ('beta_w', transformer.film_beta.weight.data),
                            ('beta_b', transformer.film_beta.bias.data)]:
            print(f'  L1 {name}: mean={param.mean():.6f}, std={param.std():.6f}')
    if hasattr(model, 'film_gamma_l2'):
        for name, param in [('gamma_l2_w', model.film_gamma_l2.weight.data),
                            ('gamma_l2_b', model.film_gamma_l2.bias.data),
                            ('beta_l2_w', model.film_beta_l2.weight.data),
                            ('beta_l2_b', model.film_beta_l2.bias.data)]:
            print(f'  L2 {name}: mean={param.mean():.6f}, std={param.std():.6f}')
    print(f'\nTraining complete. Best loss: {best_loss:.4f}')
    print(f'Checkpoint saved to: {args.out}')


if __name__ == '__main__':
    main()
