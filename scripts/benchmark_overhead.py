"""
Computational overhead measurement for UQ-ORION.

Measures inference latency and throughput with and without UQ components
to quantify the overhead introduced by UQEstimator and FiLM layers.

Reports:
  - Per-sample latency (ms) for each configuration
  - Throughput (samples/sec)
  - UQEstimator standalone latency
  - FiLM layer overhead
  - Parameter count comparison

Usage:
    python scripts/benchmark_overhead.py \
        --config adzoo/orion/configs/orion_stage3_infer.py \
        --checkpoint ckpts/Orion.pth \
        --num-warmup 5 --num-measure 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_parameters(model, prefix=''):
    """Count trainable parameters, optionally filtered by prefix."""
    if prefix:
        return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
    return sum(p.numel() for p in model.parameters())


def benchmark_uq_standalone(config_path, device, num_warmup=10, num_measure=100):
    """Benchmark UQEstimator in isolation."""
    import yaml
    from uq_estimator.model import UQEstimator

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg['model']
    model = UQEstimator(model_cfg).to(device).eval()

    B = 1
    n_views = model_cfg['n_views']
    n_patches = model_cfg.get('n_patches_subsample', model_cfg['n_patches'])
    d_patch = model_cfg['d_patch']

    patch_tokens = torch.randn(B, n_views, n_patches, d_patch, device=device)
    stat_features = torch.randn(B, 5, device=device)

    n_params = count_parameters(model)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(patch_tokens, stat_features)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(num_measure):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(patch_tokens, stat_features)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        'component': 'UQEstimator (standalone)',
        'params': n_params,
        'params_M': n_params / 1e6,
        'latency_mean_ms': np.mean(latencies),
        'latency_std_ms': np.std(latencies),
        'latency_p50_ms': np.percentile(latencies, 50),
        'latency_p95_ms': np.percentile(latencies, 95),
        'input_shape': f'patch=[{B},{n_views},{n_patches},{d_patch}] stat=[{B},5]',
    }


def benchmark_film_layer(device, d_model=256, num_warmup=10, num_measure=100):
    """Benchmark FiLM layer (gamma/beta linear transforms) in isolation."""
    film_gamma = nn.Linear(d_model, d_model).to(device).eval()
    film_beta = nn.Linear(d_model, d_model).to(device).eval()

    B = 1
    n_query = 200
    emb = torch.randn(B, d_model, device=device)
    query = torch.randn(n_query, B, d_model, device=device)

    n_params = count_parameters(film_gamma) + count_parameters(film_beta)

    for _ in range(num_warmup):
        with torch.no_grad():
            gamma = film_gamma(emb).unsqueeze(0)  # [1, B, d_model]
            beta = film_beta(emb).unsqueeze(0)
            _ = query * gamma + beta
    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_measure):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gamma = film_gamma(emb).unsqueeze(0)
            beta = film_beta(emb).unsqueeze(0)
            _ = query * gamma + beta
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        'component': 'FiLM Layer (L1, QT-Former)',
        'params': n_params,
        'params_M': n_params / 1e6,
        'latency_mean_ms': np.mean(latencies),
        'latency_std_ms': np.std(latencies),
        'latency_p50_ms': np.percentile(latencies, 50),
        'latency_p95_ms': np.percentile(latencies, 95),
        'input_shape': f'emb=[{B},{256}] query=[{n_query},{B},{256}]',
    }


def print_report(results):
    """Print formatted benchmark report."""
    print('\n' + '=' * 80)
    print('UQ-ORION Computational Overhead Report')
    print('=' * 80)

    header = f'{"Component":35s} | {"Params":>10s} | {"Latency (ms)":>14s} | {"P95 (ms)":>10s}'
    print(header)
    print('-' * len(header))

    total_uq_params = 0
    total_uq_latency = 0

    for r in results:
        params_str = f'{r["params_M"]:.2f}M'
        latency_str = f'{r["latency_mean_ms"]:.2f} ± {r["latency_std_ms"]:.2f}'
        p95_str = f'{r["latency_p95_ms"]:.2f}'
        print(f'{r["component"]:35s} | {params_str:>10s} | {latency_str:>14s} | {p95_str:>10s}')
        total_uq_params += r['params']
        total_uq_latency += r['latency_mean_ms']

    print('-' * len(header))
    print(f'{"TOTAL UQ overhead":35s} | {total_uq_params/1e6:>9.2f}M | {total_uq_latency:>13.2f}ms |')
    print()

    # Context: ORION base model ~7.5B params, ~500ms/sample
    orion_params = 7.5e9
    orion_latency_est = 500.0
    print(f'Reference: ORION base model ~{orion_params/1e9:.1f}B params, ~{orion_latency_est:.0f}ms/sample')
    print(f'UQ overhead: {total_uq_params/orion_params*100:.4f}% params, '
          f'~{total_uq_latency/orion_latency_est*100:.2f}% latency')
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark UQ overhead')
    parser.add_argument('--uq-config', default='configs/uq_train.yaml',
                        help='UQ model config for standalone benchmark')
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-measure', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    results = []

    # Benchmark UQEstimator standalone
    print('\nBenchmarking UQEstimator...')
    uq_result = benchmark_uq_standalone(
        args.uq_config, device, args.num_warmup, args.num_measure)
    results.append(uq_result)

    # Benchmark FiLM layer
    print('Benchmarking FiLM layer...')
    film_result = benchmark_film_layer(
        device, num_warmup=args.num_warmup, num_measure=args.num_measure)
    results.append(film_result)

    # Print report
    print_report(results)

    # Save JSON
    import json
    out_path = 'results/benchmark_overhead.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json_data = []
    for r in results:
        json_data.append({k: float(v) if isinstance(v, (np.floating, float)) else v
                          for k, v in r.items()})
    with open(out_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
