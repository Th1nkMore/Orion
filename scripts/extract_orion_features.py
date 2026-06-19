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


# B2D uses CARLA weather preset IDs (WeatherN in route folder names).
# Keep this split aligned with eval_openloop.py and the report protocol:
# Weather 0-3 are clear/cloudy daytime conditions; all other presets are
# treated as adverse, including clear night.
NORMAL_WEATHER_IDS = {0, 1, 2, 3}
ADVERSE_WEATHER_IDS = set(range(27)) - NORMAL_WEATHER_IDS


def classify_scene(route_folder: str) -> str:
    """
    Classify scene as normal/adverse from B2D route folder name.
    Folder format: '{Scenario}_Town{N}_Route{M}_Weather{K}'
    WeatherK is mapped to CARLA preset IDs.
    """
    import re
    m = re.search(r'Weather(\d+)', route_folder)
    if m:
        wid = int(m.group(1))
        if wid in NORMAL_WEATHER_IDS:
            return 'normal'
        if wid in ADVERSE_WEATHER_IDS:
            return 'adverse'
    return 'unknown'


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ORION EVAViT features')
    parser.add_argument('--config', default='adzoo/orion/configs/orion_stage3_infer.py', help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--output_dir', default='data/features', help='output dir')
    parser.add_argument('--ann_file', default=None, help='annotation file (overrides config)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def get_scene_id_and_type(meta: dict, global_idx: int) -> tuple:
    """
    Extract a globally unique scene_id and scene_type from img_metas.

    Path structure (absolute): .../v1/{route_folder}/camera/rgb_{cam}/{frame}.jpg
    → parts[-4] = route_folder, parts[-1] = frame.jpg

    scene_id  = '{route_folder}__{frame_basename}'
    scene_type derived from route_folder (contains weather tag)
    Falls back to f'{global_idx:06d}' if path is unavailable.
    """
    filename_list = meta.get('filename', [])
    if not filename_list:
        return f'{global_idx:06d}', 'unknown'

    full_path = filename_list[0].replace('\\', '/')
    parts = full_path.split('/')

    # Structure: .../v1/{route_folder}/camera/rgb_front/{frame}.jpg
    # parts[-1]=frame.jpg  parts[-2]=rgb_front  parts[-3]=camera  parts[-4]=route_folder
    if len(parts) >= 4:
        route_folder = parts[-4]
        frame_base = os.path.splitext(parts[-1])[0]
    else:
        route_folder = 'unknown_route'
        frame_base = f'{global_idx:06d}'

    scene_id = f'{route_folder}__{frame_base}'
    scene_type = classify_scene(route_folder)   # weather tag is in route folder name
    return scene_id, scene_type


def extract_features(backbone, data_loader, output_dir):
    """Extract features and save to disk. Skips already-extracted samples."""
    os.makedirs(output_dir, exist_ok=True)
    backbone.eval()

    saved_count = 0
    skipped_count = 0
    global_idx = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Extracting features'):
            # img: [B, N_views, C, H, W] in a list [DC]
            img_dc = data['img'][0]
            img_metas_list = data['img_metas'][0]  # list of B dicts
            B_actual = img_dc.data.shape[0]

            # Pre-compute unique scene_ids and output paths for this batch
            scene_infos = [
                get_scene_id_and_type(img_metas_list[i], global_idx + i)
                for i in range(B_actual)
            ]
            out_paths = [os.path.join(output_dir, f'{sid}.pt') for sid, _ in scene_infos]

            # Skip entire batch if all files already exist
            if all(os.path.exists(p) for p in out_paths):
                skipped_count += B_actual
                global_idx += B_actual
                continue

            img = img_dc.data.cuda()  # [B, N_views, C, H, W]

            # Reshape for backbone: [B*N_views, C, H, W]
            B, N_views, C, H, W = img.shape
            img_flat = img.flatten(0, 1)  # [B*N_views, C, H, W]

            # Extract features through EVAViT backbone
            img_feats = backbone(img_flat)  # returns list of tensors
            img_feats = img_feats[0]        # [B*N_views, D, H_feat, W_feat]

            BN, D, H_feat, W_feat = img_feats.shape
            img_feats = img_feats.reshape(B, N_views, D, H_feat, W_feat)

            # Reshape to patch token format: [B, N_views, N_patches, D]
            N_patches = H_feat * W_feat
            patch_tokens = img_feats.permute(0, 1, 3, 4, 2).reshape(B, N_views, N_patches, D)

            # Save each sample
            for i in range(B):
                out_path = out_paths[i]
                if os.path.exists(out_path):
                    skipped_count += 1
                    continue
                _, scene_type = scene_infos[i]
                feat_data = {
                    'tokens': patch_tokens[i].cpu().half(),  # [N_views, N_patches, D] fp16
                    'scene_type': scene_type,
                }
                torch.save(feat_data, out_path)
                saved_count += 1

            global_idx += B_actual

    return saved_count, skipped_count


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

    # Report already-done files
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith('.pt')]) if os.path.exists(args.output_dir) else 0
    print(f"Already extracted: {existing} / {len(dataset)} files — will skip")

    # Extract features
    saved, skipped = extract_features(backbone, data_loader, args.output_dir)
    print(f"Done: {saved} new + {skipped} skipped = {saved + skipped} total → {args.output_dir}")


if __name__ == '__main__':
    main()
