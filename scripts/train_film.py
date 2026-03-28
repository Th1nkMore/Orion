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


def forward_film_training(model, data):
    """Custom forward for FiLM training with test-format data.

    Replicates ORION's inference path through QT-Former + FiLM, then
    teacher-forces through LLM and computes planning loss.

    Returns dict with 'loss' tensor and 'log_vars' dict.
    """
    orion = model.module if hasattr(model, 'module') else model
    B = 1

    # ─── Step 1: unpack test-format data ───
    img_metas = data['img_metas']
    SPECIAL_KEYS = {'img', 'input_ids', 'gt_bboxes_3d', 'vlm_labels', 'img_metas'}
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

    # Skip samples without valid future trajectory
    ego_fut_masks = data['ego_fut_masks']  # [1, 1, 1, 6] after unpacking
    if ego_fut_masks.sum() == 0:
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
    outs_bbox, det_query, uncertainty_emb = orion.pts_bbox_head(img_metas_list, pos_embed, **data)
    vision_embeded_obj = det_query.clone()

    # ─── Step 5: map head ───
    if orion.with_map_head:
        outs_lane, map_query = orion.map_head(img_metas_list, pos_embed, **data)
        vision_embeded_map = map_query.clone()

    # ─── Step 6: LLM teacher forcing ───
    if not (orion.with_lm_head and orion.use_gen_token):
        return None
    vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)

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

    # [UQ] FiLM L2: modulate before VAE
    if hasattr(orion, 'use_uncertainty_l2') and orion.use_uncertainty_l2 and uncertainty_emb is not None:
        gamma_l2 = orion.film_gamma_l2(uncertainty_emb)
        beta_l2 = orion.film_beta_l2(uncertainty_emb)
        current_states = gamma_l2.unsqueeze(1) * current_states + beta_l2.unsqueeze(1)

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
        total = plan_loss + 0.1 * loss_vae + 0.01 * vlm_loss_val

        log_vars = {
            'plan_reg': plan_loss.item(),
            'vae': loss_vae.item(),
            'vlm': vlm_loss_val.item() if torch.is_tensor(vlm_loss_val) else 0,
            'total': total.item(),
        }
        return {'loss': total, 'log_vars': log_vars}

    return None


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)

    # ─── Config: use inference config with test pipeline ───
    # Test pipeline provides ego_fut_trajs when using the full val set
    # (individual frames in the small val set may lack sequential neighbors)
    cfg = Config.fromfile(args.config)
    cfg.model.pts_bbox_head.use_uncertainty = True
    cfg.model.pts_bbox_head.uq_checkpoint = 'checkpoints/uq/best.pt'

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
            model.pts_bbox_head.uq_estimator.load_state_dict(
                uq_ckpt['model_state_dict'], strict=False)
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
                outputs = forward_film_training(model, data)
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
                save_dict = {
                    'epoch': epoch + 1,
                    'film_gamma_weight': model.pts_bbox_head.transformer.film_gamma.weight.data.cpu(),
                    'film_gamma_bias': model.pts_bbox_head.transformer.film_gamma.bias.data.cpu(),
                    'film_beta_weight': model.pts_bbox_head.transformer.film_beta.weight.data.cpu(),
                    'film_beta_bias': model.pts_bbox_head.transformer.film_beta.bias.data.cpu(),
                    'loss': mean_loss,
                }
                # [UQ] Also save FiLM L2 weights if present
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
    gamma_w = model.pts_bbox_head.transformer.film_gamma.weight.data
    gamma_b = model.pts_bbox_head.transformer.film_gamma.bias.data
    beta_w = model.pts_bbox_head.transformer.film_beta.weight.data
    beta_b = model.pts_bbox_head.transformer.film_beta.bias.data
    print(f'\nFinal FiLM statistics:')
    print(f'  gamma weight: mean={gamma_w.mean():.6f}, std={gamma_w.std():.6f}')
    print(f'  gamma bias:   mean={gamma_b.mean():.6f}, std={gamma_b.std():.6f} (init=1.0)')
    print(f'  beta weight:  mean={beta_w.mean():.6f}, std={beta_w.std():.6f}')
    print(f'  beta bias:    mean={beta_b.mean():.6f}, std={beta_b.std():.6f} (init=0.0)')
    print(f'\nTraining complete. Best loss: {best_loss:.4f}')
    print(f'Checkpoint saved to: {args.out}')


if __name__ == '__main__':
    main()
