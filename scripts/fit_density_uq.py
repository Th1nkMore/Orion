"""Fit and validate a normal-feature density model for EVAViT."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from uq_estimator.density import DensityUQEstimator


def split_routes(
    routes: list[str],
    scene_types: list[str],
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    """Assign complete routes to train/calibration/test, stratified by scene type."""
    route_type: dict[str, str] = {}
    for route, scene_type in zip(routes, scene_types):
        previous = route_type.setdefault(route, scene_type)
        if previous != scene_type:
            raise ValueError(f"Route {route} has mixed scene types")

    grouped: dict[str, list[str]] = defaultdict(list)
    for route, scene_type in route_type.items():
        grouped[scene_type].append(route)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    split_names = ("train", "calibration", "test")
    for group_routes in grouped.values():
        group_routes.sort()
        rng.shuffle(group_routes)
        count = len(group_routes)
        train_end = round(count * ratios[0])
        cal_end = train_end + round(count * ratios[1])
        for index, route in enumerate(group_routes):
            split_index = 0 if index < train_end else 1 if index < cal_end else 2
            assignment[route] = split_names[split_index]
    return assignment


def bootstrap_route_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    routes: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    unique_routes = np.unique(routes)
    route_indices = {route: np.flatnonzero(routes == route) for route in unique_routes}
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_routes, size=len(unique_routes), replace=True)
        indices = np.concatenate([route_indices[route] for route in sampled])
        if np.unique(labels[indices]).size < 2:
            continue
        aucs.append(float(roc_auc_score(labels[indices], scores[indices])))
    if not aucs:
        return float("nan"), float("nan")
    return tuple(np.percentile(aucs, [2.5, 97.5]).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/density_uq.yaml")
    parser.add_argument("--descriptor-cache", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    cache_path = args.descriptor_cache or data_cfg["descriptor_cache"]
    output_dir = Path(args.output_dir or data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    descriptors = cache["descriptors"].float().numpy()
    routes = np.asarray(cache["routes"])
    weather_ids = cache["weather_ids"].numpy().astype(np.int64)
    scene_types = np.asarray(cache["scene_types"])
    labels = (scene_types == "adverse").astype(np.int64)

    ratios = tuple(data_cfg["split_ratios"])
    assignment = split_routes(
        cache["routes"], cache["scene_types"], int(data_cfg["seed"]), ratios
    )
    splits = np.asarray([assignment[route] for route in routes])
    normal = labels == 0
    train_mask = (splits == "train") & normal
    calibration_mask = (splits == "calibration") & normal
    test_mask = splits == "test"
    if train_mask.sum() <= int(model_cfg["n_components"]):
        raise ValueError("Not enough normal training samples for requested PCA size")
    if calibration_mask.sum() == 0:
        raise ValueError("Calibration split contains no normal samples")
    if np.unique(labels[test_mask]).size < 2:
        raise ValueError("Test split must contain both normal and adverse samples")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(descriptors[train_mask])
    pca = IncrementalPCA(
        n_components=int(model_cfg["n_components"]),
        batch_size=int(model_cfg["pca_batch_size"]),
    )
    train_latent = pca.fit_transform(train_scaled)

    covariance = LedoitWolf(
        assume_centered=False,
        store_precision=True,
    ).fit(train_latent)
    precision = covariance.precision_.astype(np.float64)
    whitening = np.linalg.cholesky(precision)

    def transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = scaler.transform(values)
        latent = pca.transform(scaled)
        residual = latent - covariance.location_
        whitened = residual @ whitening
        distances = np.linalg.norm(whitened, axis=1)
        embeddings = whitened / np.maximum(distances[:, None], 1e-8)
        return embeddings.astype(np.float32), distances.astype(np.float32)

    _, calibration_distances = transform(descriptors[calibration_mask])
    calibration_sorted = np.sort(calibration_distances)
    test_embeddings, test_distances = transform(descriptors[test_mask])
    test_scores = np.searchsorted(
        calibration_sorted, test_distances, side="right"
    ) / len(calibration_sorted)
    test_labels = labels[test_mask]
    test_routes = routes[test_mask]
    test_weather = weather_ids[test_mask]

    auc = float(roc_auc_score(test_labels, test_scores))
    auprc = float(average_precision_score(test_labels, test_scores))
    ci_low, ci_high = bootstrap_route_auc(
        test_labels,
        test_scores,
        test_routes,
        int(data_cfg["seed"]),
        int(eval_cfg["bootstrap_iterations"]),
    )

    model_state = {
        "descriptor_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "descriptor_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "pca_components": torch.from_numpy(pca.components_.astype(np.float32)),
        "latent_mean": torch.from_numpy(covariance.location_.astype(np.float32)),
        # Runtime uses residual @ whitening.T, matching residual @ L above.
        "whitening": torch.from_numpy(whitening.T.astype(np.float32)),
        "calibration_distances": torch.from_numpy(calibration_sorted),
    }
    output_dim = int(model_cfg.get("output_dim", model_cfg["n_components"]))
    active_dim = int(model_cfg["n_components"])
    if output_dim < active_dim:
        raise ValueError("output_dim must be greater than or equal to n_components")
    output_projection = torch.zeros(output_dim, active_dim)
    output_projection[:active_dim, :active_dim] = torch.eye(active_dim)
    model_state["output_projection"] = output_projection
    checkpoint = {
        "model_state": model_state,
        "config": config,
        "metrics": {
            "auroc": auc,
            "auprc": auprc,
            "auroc_route_bootstrap_95_ci": [ci_low, ci_high],
        },
        "split_assignment": assignment,
    }
    checkpoint_path = output_dir / "density_uq.pt"
    torch.save(checkpoint, checkpoint_path)

    runtime = DensityUQEstimator.from_checkpoint(checkpoint)
    with torch.no_grad():
        runtime_embedding, runtime_score, runtime_distance = runtime.encode_descriptor(
            torch.from_numpy(descriptors[test_mask])
        )
    np.testing.assert_allclose(
        runtime_embedding[:, :active_dim].numpy(),
        test_embeddings,
        rtol=2e-4,
        atol=2e-4,
    )
    if output_dim > active_dim:
        np.testing.assert_allclose(
            runtime_embedding[:, active_dim:].numpy(), 0.0, atol=1e-7
        )
    np.testing.assert_allclose(
        runtime_distance.numpy(), test_distances, rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        runtime_score.squeeze(-1).numpy(), test_scores, atol=1 / len(calibration_sorted)
    )

    weather_rows = []
    for weather_id in sorted(np.unique(test_weather)):
        mask = test_weather == weather_id
        values = test_scores[mask]
        weather_rows.append(
            {
                "weather_id": int(weather_id),
                "scene_type": "normal" if weather_id in {0, 1, 2, 3} else "adverse",
                "count": int(mask.sum()),
                "score_mean": float(values.mean()),
                "score_p50": float(np.quantile(values, 0.5)),
                "score_p90": float(np.quantile(values, 0.9)),
            }
        )
    with open(output_dir / "weather_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=weather_rows[0].keys())
        writer.writeheader()
        writer.writerows(weather_rows)

    metrics = {
        "samples": int(len(descriptors)),
        "routes": int(len(set(routes))),
        "normal_train_samples": int(train_mask.sum()),
        "normal_calibration_samples": int(calibration_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "test_normal_samples": int(((test_labels == 0)).sum()),
        "test_adverse_samples": int(((test_labels == 1)).sum()),
        "auroc": auc,
        "auprc": auprc,
        "auroc_route_bootstrap_95_ci": [ci_low, ci_high],
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "active_embedding_dim": active_dim,
        "output_embedding_dim": output_dim,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plt.figure(figsize=(7, 4.5))
    plt.hist(
        test_scores[test_labels == 0], bins=30, alpha=0.65, density=True, label="normal"
    )
    plt.hist(
        test_scores[test_labels == 1], bins=30, alpha=0.65, density=True, label="adverse"
    )
    plt.xlabel("Calibrated density uncertainty")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    for label, name in ((0, "normal"), (1, "adverse")):
        mask = test_labels == label
        plt.scatter(
            test_embeddings[mask, 0],
            test_embeddings[mask, 1],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("Whitened residual dimension 1")
    plt.ylabel("Whitened residual dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_projection.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved density UQ checkpoint and validation report to {output_dir}")


if __name__ == "__main__":
    main()
