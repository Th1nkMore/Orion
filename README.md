# UQ-ORION

Uncertainty-aware extension for the [ORION](https://github.com/xiaomi-mlab/Orion) end-to-end autonomous driving framework, improving safety in adverse weather and low-visibility scenarios through uncertainty quantification.

## Project Structure

```
uq-orion/
├── uq_estimator/              # UQ extension module (all new code lives here)
│   ├── __init__.py            # Public API: UQEstimator, UQOutput, CombinedUQLoss, UQFeatureDataset
│   ├── model.py               # UQEstimator model (2.24M params)
│   ├── losses.py              # Regression + ranking + calibration losses
│   └── dataset.py             # Feature dataset with stat-feature computation
├── scripts/
│   ├── generate_labels.py     # Compute uncertainty pseudo-labels from features
│   ├── train_uq.py            # Full training script with warmup, cosine LR, checkpointing
│   ├── validate_uq.py         # Generate validation report with plots
│   └── e2e_mock_test.py       # End-to-end pipeline verification (no real data needed)
├── configs/
│   └── uq_train.yaml          # Model, training, data, and logging configuration
├── tests/
│   ├── test_uq_model.py       # Model shape, range, and parameter count tests
│   ├── test_generate_labels.py # Label generation tests with scene separation
│   ├── test_training.py       # Training loop smoke, resume, and loss tests
│   └── fixtures.py            # Mock feature generators (normal / adverse / random)
├── adzoo/                     # ORION original code (do not modify)
├── team_code/                 # ORION original code
├── mmcv/                      # ORION dependency
├── requirements.txt           # ORION original dependencies
├── requirements_uq.txt        # UQ project dependencies (managed by uv)
├── CLAUDE.md                  # Development context and coding conventions
└── .gitignore
```

## Requirements

- Python >= 3.10
- CUDA >= 11.8 (recommended for training; CPU is fully supported)
- [uv](https://docs.astral.sh/uv/) (package management)

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/uq-orion.git
cd uq-orion
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements_uq.txt
```

### Verify Installation (no data required)

```bash
PYTHONPATH=. python scripts/e2e_mock_test.py
```

This runs the full pipeline with synthetic data and should print `Phase 1 end-to-end verification PASSED`.

### Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

### Training Pipeline (with real data)

**Step 1: Extract features** (run on a machine with large GPU memory)

Feature extraction uses the ORION Vision Encoder. See `adzoo/` for ORION inference scripts. Each scene produces a `.pt` file with format described below.

**Step 2: Generate pseudo-labels**

```bash
PYTHONPATH=. python scripts/generate_labels.py \
    --feature_dir /path/to/features \
    --output_file ./data/labels/uq_labels.pt \
    --n_workers 8
```

**Step 3: Train UQ Estimator**

```bash
PYTHONPATH=. python scripts/train_uq.py \
    --config configs/uq_train.yaml \
    --data_dir /path/to/features \
    --label_file ./data/labels/uq_labels.pt
```

**Step 4: Validate**

```bash
PYTHONPATH=. python scripts/validate_uq.py \
    --checkpoint ./checkpoints/uq/best.pt \
    --feature_dir /path/to/features \
    --label_file ./data/labels/uq_labels.pt
```

## Data Format

**Feature file** (`.pt`, one per scene):
```python
{
    "tokens": torch.Tensor,  # [N_views, N_patches, D]  — D=1152 (Qwen2-VL output)
    "image":  torch.Tensor,  # [N_views, 3, H, W]       — multi-view camera images
}
```

**Label file** (`.pt`, single file):
```python
{
    "sample_0000.pt": 0.32,  # filename → uncertainty score in [0, 1]
    "sample_0001.pt": 0.78,
    ...
}
```

## Relationship to ORION

This project is a research extension built on top of [ORION](https://github.com/xiaomi-mlab/Orion) (ICCV 2025). The original ORION pipeline is:

```
Vision Encoder → QT-Former → VLM → planning token → VAE → trajectory
```

UQ-ORION adds a parallel UQ Estimator branch that:
1. Consumes the same Vision Encoder patch tokens
2. Outputs an uncertainty embedding and scalar score
3. Injects the embedding into QT-Former via an uncertainty token
4. Modulates VAE output based on uncertainty level

**Current status (Phase 1):** The UQ Estimator module, training pipeline, and evaluation tools are implemented. No modifications to ORION original code have been made yet. Integration with QT-Former and VAE (Phase 2) is pending.

## Development Conventions

- All new code goes in `uq_estimator/`; ORION files in `adzoo/` are read-only unless explicitly requested
- Every function must include shape annotations: `# [B, N, D]`
- No hard-coded dimensions in model code — read from config
- Commits modifying ORION original files must start with `[UQ]`
- All tests use mock data and do not depend on real datasets
- Environment: use the project `.venv/` managed by uv; never install into base conda

## Citation

If you use this work, please also cite the original ORION paper:

```bibtex
@inproceedings{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Haoyu Fu and Diankun Zhang and Zongchuang Zhao and Jianfeng Cui and Dingkang Liang and Chong Zhang and Dingyuan Zhang and Hongwei Xie and Bing Wang and Xiang Bai},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
