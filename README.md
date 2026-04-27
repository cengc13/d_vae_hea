## Data-efficient and interpretable inverse materials design using a disentangled variational autoencoder
Cheng Zeng, Zulqarnain Khan, Nathan Post. Accepted at [AI & Materials](https://www.elspub.com/papers/j/1838650894880686080.html) as an Editor's Choice.

## Overview

This repository implements a **Semi-Supervised Variational Autoencoder (SSVAE)** for high-entropy alloy (HEA) phase prediction and inverse design. The model learns a disentangled 2D latent space from alloy composition and thermodynamic features, enabling:

- Phase classification (single-phase vs. multi-phase) from engineered features
- Reconstruction and generation of alloy compositions via the latent space
- Iterative inverse design: find compositions likely to form a target phase
- Interpretability via SHAP feature attribution

The model achieves **87.7% test accuracy** using 864 labelled and 296 unlabelled training samples.

## Installation

### Conda (recommended)

```bash
# GPU (PyTorch cu128 wheel for CUDA 12.8)
mamba env create -f environment.yml
mamba activate d-vae-hea

# CPU-only — edit environment.yml first:
#   replace
#     --extra-index-url https://download.pytorch.org/whl/cu128
#   with
#     --extra-index-url https://download.pytorch.org/whl/cpu
mamba env create -f environment.yml
mamba activate d-vae-hea
```

### Package compatibility notes

| Package | Constraint | Reason |
|---------|-----------|--------|
| `python` | 3.11 | Safe target for the refreshed stack; `pyro-ppl` 1.9.1 officially supports up to Python 3.11 |
| `numba` | 0.61.2 | Supports `numpy` 2.2 and keeps SHAP runtime compatibility |
| `numpy` | 2.2.x | Works with `numba` 0.61.2 and current `pandas` / `scipy` releases |
| `pyro-ppl` | 1.9.1 | Requires `torch` ≥ 2.0; still installed via `pip` |
| `torch` | 2.7.0 + CUDA 12.8 | Official `cu128` wheels are provided by PyTorch via `pip` |

If you are on a different CUDA runtime, replace the PyTorch wheel index in `environment.yml`
with the matching one from the official PyTorch install matrix, for example `cu126` or `cpu`.

## Project structure

```
d_vae_hea/
├── ssvae/                  # Core model package
│   ├── model.py            #   SSVAE class (encoder_y, encoder_z, decoder)
│   ├── dataset.py          #   HEAFeatureDataset, HEAFeatureDatasetUnlabelled
│   └── trainer.py          #   run_inference_for_epoch, evaluate_model, get_accuracy
├── utils/
│   ├── custom_mlp.py       # Flexible MLP builder used by SSVAE
│   └── featurization.py    # Alloy formula parsing and engineered feature calculation
├── train.py                # Train the SSVAE from scratch
├── evaluate.py             # Accuracy, ROC curves, latent space plots
├── shap_analysis.py        # SHAP feature importance for the classifier
├── reconstruct.py          # Encode→decode error analysis for test alloys
├── interpolate.py          # Latent space scanning and iterative inverse design
├── data/
│   ├── HEA_top30_comps.csv             # Composition vectors (30 elements)
│   ├── HEA_feature_engineered.csv      # Thermodynamic engineered features
│   ├── labelled_hea.pk                 # Labelled training split (864 samples)
│   ├── unlabelled_hea.pk               # Unlabelled training split (296 samples)
│   ├── validation_hea.pk               # Validation split (75 samples)
│   ├── test_hea.pk                     # Test split (138 samples)
│   ├── look_up_dict.pkl                # Element property look-up table
│   ├── mixing_enthalpy_dict.pkl        # Pairwise mixing enthalpy table
│   ├── hyper-parameter-tuning.json     # Scikit-learn MLP hyperparameter search results
│   └── test_data_reconstruction_analysis.csv  # Reconstruction error results
├── models/
│   └── ssvae.model         # Trained checkpoint (test accuracy 0.877)
├── figures/                # Output figures
└── notebooks/              # Original Jupyter notebooks (kept for reference)
    ├── 1_SSVAE_Model_training.ipynb
    ├── 2_SSVAE_model_inference.ipynb
    ├── 3_SSVAE_SHAP_analysis.ipynb
    ├── 4_SSVAE_Alloy_reconstruction.ipynb
    └── 5_SSVAE_Interpolation_study.ipynb
```

## Quick start

All scripts run from the project root and default to the pre-trained checkpoint at `models/ssvae.model`.

### Evaluate the trained model
```bash
python evaluate.py
```
Prints train/val/test accuracy and saves ROC curves and latent space scatter plots to `figures/`.

### Reconstruct test alloys
```bash
python reconstruct.py
```
Encodes each test alloy to the latent space and decodes it back, reporting composition / phase / latent MAE and saving distribution figures.

### SHAP feature importance
```bash
python shap_analysis.py
```
Runs SHAP KernelExplainer on the test set and saves a summary beeswarm plot plus per-sample waterfall plots.

### Latent space scanning and inverse design
```bash
python interpolate.py
python interpolate.py --alloy AlCoCrFeNi --target-prob 0.7
python interpolate.py --z1-range -1 1 --z2-range -1 1 --n-points 7
```

### Train from scratch
```bash
python train.py --epochs 20000 --lr 1e-4 --hidden-dims 100 100 --aux-loss-multiplier 10
```
Data splits are loaded from `data/*.pk` if present, otherwise created from the raw CSVs. Checkpoints are saved to `models/`.

## Model architecture

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `encoder_y` | 8 engineered features | single-phase probability | Phase classifier |
| `encoder_z` | 30-dim composition + phase label | 2D latent (μ, σ) | Composition encoder |
| `decoder`   | 2D latent + phase label | 30-dim composition probs | Composition generator |

Training uses Pyro's SVI with a Trace-ELBO objective plus an auxiliary supervised classification loss (Kingma et al., 2014).

## Data

| File | Description |
|------|-------------|
| `HEA_top30_comps.csv` | Composition fractions for the 30 most common HEA elements |
| `HEA_feature_engineered.csv` | Bulk modulus, molar volume, melting temperature, VEC, atomic size difference, electronegativity difference, mixing entropy, mixing enthalpy |
| `look_up_dict.pkl` | Per-element property table for feature calculation |
| `mixing_enthalpy_dict.pkl` | Pairwise Miedema mixing enthalpies |
| `hyper-parameter-tuning.json` | Grid search results for scikit-learn MLP baseline; key format: `[alpha]-[hls]-[lr]` |

## Citation

```
Zeng C, Khan Z, Post N. Data-efficient and interpretable inverse materials design
using a disentangled variational autoencoder. AI Mater. 2025;1(1):0002.
doi:10.55092/aimat20250002.
```
