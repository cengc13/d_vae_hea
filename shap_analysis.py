#!/usr/bin/env python
"""SHAP feature importance analysis for the trained SSVAE classifier.

Usage:
    python shap_analysis.py
    python shap_analysis.py --model-path models/ssvae_best_val.model
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import pyro

from ssvae.model import SSVAE


FEATURE_NAMES = [
    "Bulk Modulus",
    "Molar Volume",
    "Melting Temperature",
    "Val. Electron Conc.",
    "Atomic Size Diff.",
    "Pauling Electronegativity Diff.",
    "Mixing Entropy",
    "Mixing Enthalpy",
]


def parse_args():
    p = argparse.ArgumentParser(description="SHAP analysis of trained SSVAE classifier")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-path", default="models/ssvae.model")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[100, 100])
    p.add_argument("--aux-loss-multiplier", type=float, default=50.0)
    return p.parse_args()


def normalize_shap_output(shap_values, expected_value):
    """Handle SHAP's old list output and newer ndarray output consistently."""
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[0])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            if values.shape[2] != 1:
                raise ValueError(
                    f"Unsupported SHAP output shape {values.shape}; expected a single output classifier."
                )
            values = values[:, :, 0]

    base_value = np.asarray(expected_value)
    if base_value.ndim > 0:
        base_value = float(base_value.reshape(-1)[0])
    else:
        base_value = float(base_value)

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP values shape {values.shape}; expected (n_samples, n_features)."
        )

    return values, base_value


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)

    labelled_hea = pickle.load(open(data_dir / "labelled_hea.pk", "rb"))
    test_hea = pickle.load(open(data_dir / "test_hea.pk", "rb"))

    pyro.clear_param_store()
    model = SSVAE(
        output_size=1,
        input_size=30,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_dims,
        use_cuda=False,
        aux_loss_multiplier=args.aux_loss_multiplier,
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    train_engg = np.array(labelled_hea.loc[:, "k":"delta_h_mix"].values, np.float32)
    test_engg = np.array(test_hea.loc[:, "k":"delta_h_mix"].values, np.float32)
    test_labels = test_hea["Class"].values

    print(f"Background samples: {len(train_engg)} (training set)")
    print("Building SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model.predict_proba, train_engg)

    print(f"Computing SHAP values for {len(test_engg)} test samples...")
    shap_values = explainer.shap_values(test_engg)
    shap_values, expected_value = normalize_shap_output(
        shap_values, explainer.expected_value
    )
    print(f"SHAP values shape: {shap_values.shape}")

    # Summary beeswarm plot
    shap.summary_plot(shap_values, test_engg, FEATURE_NAMES, show=False)
    plt.title("Multiple Phase  <--|-->  Single Phase")
    plt.tight_layout()
    plt.savefig(figures_dir / "SHAP_summary.pdf", dpi=300)
    plt.savefig(figures_dir / "SHAP_summary.png", dpi=300)
    plt.close()
    print(f"SHAP summary -> {figures_dir}/SHAP_summary.{{pdf,png}}")

    # Waterfall plots for each test sample
    waterfall_dir = figures_dir / "shap_waterfall"
    waterfall_dir.mkdir(exist_ok=True)
    for i in range(len(shap_values)):
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=expected_value,
            data=test_engg[i],
            feature_names=FEATURE_NAMES,
        )
        shap.plots.waterfall(exp, show=False)
        label = test_labels[i]
        title = "True Label: Single Phase" if label == 1 else "True Label: Multiple Phase"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(waterfall_dir / f"waterfall_{i:03d}.png", dpi=150)
        plt.close()

    print(f"Waterfall plots -> {waterfall_dir}/")


if __name__ == "__main__":
    main()
