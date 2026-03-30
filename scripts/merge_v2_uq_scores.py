"""Merge v2 UQEstimator scores into existing eval_openloop results.

Replaces uq_score in each record, recomputes aggregate stats, AUROC,
Spearman correlations, and saves as a new .pt file.

Usage:
    python scripts/merge_v2_uq_scores.py \
        --input results/eval_openloop_full.pt \
        --output results/eval_openloop_v2.pt \
        --uq-checkpoint checkpoints/uq/best.pt \
        --feature-dir data/features \
        --stat-cache data/stat_cache.pt
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uq_estimator.model import UQEstimator
from uq_estimator.dataset import compute_stat_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Existing eval .pt file")
    parser.add_argument("--output", required=True, help="Output .pt file with v2 scores")
    parser.add_argument("--uq-checkpoint", default="checkpoints/uq/best.pt")
    parser.add_argument("--feature-dir", default="data/features")
    parser.add_argument("--stat-cache", default="data/stat_cache.pt")
    parser.add_argument("--config", default="configs/uq_train.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-patches-subsample", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(args.uq_checkpoint, map_location="cpu", weights_only=False)
    model = UQEstimator(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"Loaded UQEstimator from epoch {ckpt['epoch']}")

    # Load stat_cache
    stat_cache = {}
    if args.stat_cache and Path(args.stat_cache).is_file():
        stat_cache = torch.load(args.stat_cache, weights_only=True, map_location="cpu")
        print(f"Loaded stat_cache: {len(stat_cache)} entries")

    # Load eval results
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    records = data["records"]
    print(f"Loaded {len(records)} records from {args.input}")

    # Build record → feature filename mapping
    feature_dir = Path(args.feature_dir)

    def record_to_fname(rec):
        folder = rec["folder"]
        scenario = folder.split("/")[-1] if "/" in folder else folder
        return f"{scenario}__{rec['frame_idx']:05d}.pt"

    # Batch compute v2 UQ scores
    fnames = [record_to_fname(r) for r in records]
    fname_to_score = {}

    t0 = time.time()
    batch_tokens, batch_stats, batch_fnames = [], [], []

    for i, fname in enumerate(fnames):
        fp = feature_dir / fname
        if not fp.exists():
            continue

        d = torch.load(str(fp), map_location="cpu", weights_only=True)
        tokens = d["tokens"].float()

        if tokens.shape[1] > args.n_patches_subsample and args.n_patches_subsample > 0:
            rng = torch.Generator()
            rng.manual_seed(42)
            perm = torch.randperm(tokens.shape[1], generator=rng)[:args.n_patches_subsample]
            tokens = tokens[:, perm, :]

        stat = stat_cache[fname].float() if fname in stat_cache else compute_stat_features(tokens.unsqueeze(0)).squeeze(0)
        batch_tokens.append(tokens)
        batch_stats.append(stat)
        batch_fnames.append(fname)

        if len(batch_tokens) >= args.batch_size or i == len(fnames) - 1:
            if batch_tokens:
                t_batch = torch.stack(batch_tokens).to(device)
                s_batch = torch.stack(batch_stats).to(device)
                with torch.no_grad():
                    out = model(t_batch, s_batch)
                    scores = out.score.squeeze(-1).cpu().tolist()
                for fn, sc in zip(batch_fnames, scores):
                    fname_to_score[fn] = sc
                batch_tokens, batch_stats, batch_fnames = [], [], []

        if (i + 1) % 640 == 0 or i == len(fnames) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(fnames) - i - 1)
            print(f"  [{i+1}/{len(fnames)}] elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"scores_computed={len(fname_to_score)}")

    print(f"\nComputed {len(fname_to_score)} v2 UQ scores in {time.time()-t0:.0f}s")

    # Replace UQ scores in records
    replaced = 0
    for rec in records:
        fn = record_to_fname(rec)
        if fn in fname_to_score:
            rec["uq_score"] = fname_to_score[fn]
            replaced += 1
    print(f"Replaced {replaced}/{len(records)} UQ scores")

    # Recompute aggregate stats
    normal_records = [r for r in records if not r["is_adverse"]]
    adverse_records = [r for r in records if r["is_adverse"]]

    def agg_stats(recs, label):
        if not recs:
            return {}
        scores = np.array([r["uq_score"] for r in recs])
        l2_3s = np.array([r["plan_L2_3s"] for r in recs])
        col_3s = np.array([r["plan_obj_col_3s"] for r in recs])
        return {
            "n": len(recs),
            "uq_mean": float(scores.mean()),
            "uq_std": float(scores.std()),
            "uq_median": float(np.median(scores)),
            "l2_3s_mean": float(l2_3s.mean()),
            "col_3s_mean": float(col_3s.mean()),
        }

    data["stats"] = {
        "all": agg_stats(records, "all"),
        "normal": agg_stats(normal_records, "normal"),
        "adverse": agg_stats(adverse_records, "adverse"),
    }

    # Per-weather stats
    weather_groups = defaultdict(list)
    for r in records:
        weather_groups[r["weather_name"]].append(r)

    weather_stats = {}
    for wname, recs in sorted(weather_groups.items()):
        scores = np.array([r["uq_score"] for r in recs])
        l2_3s = np.array([r["plan_L2_3s"] for r in recs])
        col_3s = np.array([r["plan_obj_col_3s"] for r in recs])
        weather_stats[wname] = {
            "n": len(recs),
            "uq_mean": float(scores.mean()),
            "uq_std": float(scores.std()),
            "l2_3s_mean": float(l2_3s.mean()),
            "col_3s_mean": float(col_3s.mean()),
            "is_adverse": recs[0]["is_adverse"],
        }
    data["stats"]["per_weather"] = weather_stats

    # Recompute correlations
    all_uq = np.array([r["uq_score"] for r in records])
    all_l2 = np.array([r["plan_L2_3s"] for r in records])
    all_col = np.array([r["plan_obj_col_3s"] for r in records])
    all_adverse = np.array([r["is_adverse"] for r in records]).astype(float)

    sp_l2, sp_l2_p = spearmanr(all_uq, all_l2)
    sp_col, sp_col_p = spearmanr(all_uq, all_col)
    pe_l2, pe_l2_p = pearsonr(all_uq, all_l2)
    auroc = roc_auc_score(all_adverse, all_uq)

    data["correlation"] = {
        "spearman_uq_vs_L2_3s": {"rho": float(sp_l2), "p": float(sp_l2_p)},
        "spearman_uq_vs_col_3s": {"rho": float(sp_col), "p": float(sp_col_p)},
        "pearson_uq_vs_L2_3s": {"r": float(pe_l2), "p": float(pe_l2_p)},
        "auroc_adverse": float(auroc),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"v2 UQ Score Summary")
    print(f"{'='*60}")
    for group in ["normal", "adverse", "all"]:
        s = data["stats"][group]
        print(f"  [{group:8s}] n={s['n']:5d}  uq_mean={s['uq_mean']:.4f}  "
              f"uq_std={s['uq_std']:.4f}  uq_median={s['uq_median']:.4f}")

    print(f"\n  AUROC (adverse detection): {auroc:.4f}")
    print(f"  Spearman(UQ, L2@3s):  rho={sp_l2:.4f}  p={sp_l2_p:.2e}")
    print(f"  Spearman(UQ, Col@3s): rho={sp_col:.4f}  p={sp_col_p:.2e}")
    print(f"  Pearson(UQ, L2@3s):   r={pe_l2:.4f}  p={pe_l2_p:.2e}")

    gap = data["stats"]["adverse"]["uq_mean"] - data["stats"]["normal"]["uq_mean"]
    print(f"  Normal/Adverse gap: {gap:.4f}")

    print(f"\n  Per-weather UQ scores:")
    for wname, ws in sorted(weather_stats.items(), key=lambda x: x[1]["uq_mean"]):
        tag = "ADV" if ws["is_adverse"] else "NOR"
        print(f"    [{tag}] {wname:25s}  n={ws['n']:4d}  "
              f"uq={ws['uq_mean']:.4f}  l2@3s={ws['l2_3s_mean']:.2f}m  "
              f"col@3s={ws['col_3s_mean']*100:.2f}%")

    # Save
    data["uq_version"] = "v2"
    data["uq_checkpoint"] = args.uq_checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(output_path))
    print(f"\nSaved to {output_path}")

    # Also save JSON summary
    summary = {
        "n_samples": len(records),
        "auroc": auroc,
        "uq_version": "v2",
        "stats": data["stats"],
        "correlation": data["correlation"],
    }
    # Convert non-serializable types
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
