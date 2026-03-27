"""
Extract EVAViT patch tokens from ORION model for UQ Estimator training.

Usage:
    python scripts/extract_orion_features.py \
        --checkpoint ckpts/Orion.pth \
        --output_dir data/features \
        --ann_file data/infos/b2d_infos_val.pkl \
        --batch_size 4 \
        --num_workers 4
"""

import argparse
import os
import torch
from tqdm import tqdm

from mmcv.utils import Config
from mmcv.models import build_model
from mmcv.datasets import build_dataset, build_dataloader
from mmcv.models.backbones.eva_vit import EVAViT


# Weather patterns for scene classification
NORMAL_WEATHERS = {'ClearNoon', 'ClearSunset'}
ADVERSE_WEATHERS = {
    'HardRain', 'WetNoon', 'WetSunset',
    'SoftRainNoon', 'SoftRainSunset', 'SoftRainNight',
    'MidRainNoon', 'MidRainSunset', 'MidRainNight',
    'HardRainNoon', 'HardRainSunset', 'HardRainNight',
}


def classify_scene(filename: str) -> str:
    """Classify scene as normal/adverse based on weather pattern in filename."""
    for w in ADVERSE_WEATHERS:
        if w in filename:
            return 'adverse'
    for w in NORMAL_WEATHERS:
        if w in filename:
            return 'normal'
    return 'unknown'


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ORION EVAViT features')
    parser.add_argument('--config', default='adzoo/orion/configs/orion_stage3_infer.py', help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--output_dir', default='data/features', help='output dir')
    parser.add_argument('--ann_file', default=None, help='annotation file (overrides config)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)
    return parser.parse_args()


def extract_features(backbone, data_loader, output_dir):
    """Extract features and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    backbone.eval()

    saved_count = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Extracting features'):
            # img: [B, N_views, C, H, W] in a list [DC]
            img_dc = data['img'][0]
            img = img_dc.data.cuda()  # [B, N_views, C, H, W]

            # Reshape for backbone: [B*N_views, C, H, W]
            B, N_views, C, H, W = img.shape
            img_flat = img.flatten(0, 1)  # [B*N_views, C, H, W]

            # Extract features through EVAViT backbone
            img_feats = backbone(img_flat)  # returns list

            # img_feats is a list: [tensor of shape [B*N, D, H_feat, W_feat]]
            img_feats = img_feats[0]

            BN, D, H_feat, W_feat = img_feats.shape
            img_feats = img_feats.reshape(B, N_views, D, H_feat, W_feat)

            # Reshape to patch token format: [B, N_views, N_patches, D]
            N_patches = H_feat * W_feat
            patch_tokens = img_feats.permute(0, 1, 3, 4, 2).reshape(B, N_views, N_patches, D)

            # Get scene info from img_metas
            img_metas_list = data['img_metas'][0]  # list of B dicts

            # Process each sample in batch
            for i in range(B):
                meta = img_metas_list[i]
                filename_list = meta.get('filename', [])
                if filename_list and len(filename_list) > 0:
                    filename = filename_list[0]
                    scene_id = os.path.splitext(os.path.basename(filename))[0]
                else:
                    scene_id = f'sample_{saved_count:06d}'

                scene_type = classify_scene(scene_id)

                # Save: tokens in fp16, no image
                feat_data = {
                    'tokens': patch_tokens[i].cpu().half(),  # [N_views, N_patches, D] fp16
                    'scene_type': scene_type,
                }
                out_path = os.path.join(output_dir, f'{scene_id}.pt')
                torch.save(feat_data, out_path)
                saved_count += 1

    return saved_count


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file

    print(f"Config loaded: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Ann file: {cfg.data.test.ann_file}")

    # Build dataset
    dataset = build_dataset(cfg.data.test)
    print(f"Dataset size: {len(dataset)}")

    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
    )

    # Build only the img_backbone (EVAViT) from config
    backbone_cfg = cfg.model.img_backbone.copy()
    # Remove keys not accepted by EVAViT
    backbone_cfg.pop('type', None)
    backbone_cfg.pop('pretrained', None)
    backbone = EVAViT(**backbone_cfg)
    backbone = backbone.cuda()
    backbone.eval()
    print("EVAViT backbone built and loaded")

    # Load only img_backbone weights from full Orion checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Filter to only backbone keys and strip the 'img_backbone.' prefix
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('img_backbone.'):
            new_k = k[len('img_backbone.'):]  # strip prefix
            backbone_state_dict[new_k] = v

    missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
    print(f"Loaded backbone weights: {len(backbone_state_dict)} keys")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys (first 5): {unexpected[:5]}")

    # Extract features
    saved = extract_features(backbone, data_loader, args.output_dir)
    print(f"Extracted {saved} samples to {args.output_dir}")


if __name__ == '__main__':
    main()
