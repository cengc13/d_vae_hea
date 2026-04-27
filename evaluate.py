#!/usr/bin/env python
"""Evaluate trained SSVAE: accuracy, ROC curves, and latent space plots.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/ssvae_best_val.model
"""

import argparse
import pickle
import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import auc, roc_curve

import pyro

from ssvae.dataset import HEAFeatureDataset
from ssvae.model import SSVAE
from ssvae.trainer import get_accuracy
from utils.featurization import top30, calculate_compositions, calculate_engineered_features


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained SSVAE")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-path", default="models/ssvae.model")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[100, 100])
    p.add_argument("--aux-loss-multiplier", type=float, default=50.0)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


def load_model(model_path, hidden_dims, z_dim, aux_loss_multiplier, cuda=False):
    pyro.clear_param_store()
    model = SSVAE(
        output_size=1,
        input_size=30,
        z_dim=z_dim,
        hidden_layers=hidden_dims,
        use_cuda=cuda,
        aux_loss_multiplier=aux_loss_multiplier,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def encode_dataframe(df, model):
    """Add z1, z2, Predicted_Class columns to df using the trained encoder."""
    data = torch.tensor(df.loc[:, "Fe":"Sc"].values * 100, dtype=torch.float32)
    labels = torch.tensor(df["Class"].values, dtype=torch.float32).unsqueeze(-1)
    engg = torch.tensor(df.loc[:, "k":"delta_h_mix"].values, dtype=torch.float32)

    with torch.no_grad():
        z_loc, _ = model.encoder_z([data, labels])
        preds = model.classifier(engg)

    df = df.copy()
    df["Predicted_Class"] = preds.numpy().squeeze().astype(int).astype(str)
    df["z1"] = z_loc[:, 0].numpy()
    df["z2"] = z_loc[:, 1].numpy()
    return df, data, labels, engg


def count_elements(alloy):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.*\d*?(?=\D|$))")
    parts = [(x, float(y)) if y else (x, 1) for x, y in pattern.findall(alloy)]
    return len(parts)


def plot_roc_curves(splits, model, figures_dir):
    """Plot ROC curves for train, validation, and test splits."""
    fig, ax = plt.subplots()
    colors = {"Training": "g", "Validation": "b", "Test": "r"}
    styles = {"Training": "--o", "Validation": "-.*", "Test": "-.*"}

    for name, (labels, engg) in splits.items():
        with torch.no_grad():
            probs = model.encoder_y(engg).numpy()
        fpr, tpr, _ = roc_curve(labels.numpy(), probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, styles[name], label=f"{name} AUC = {roc_auc:.2f}", color=colors[name])

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=12)
    fig.tight_layout()
    out = Path(figures_dir)
    fig.savefig(out / "ROC_curves.pdf")
    fig.savefig(out / "ROC_curves.png")
    plt.close(fig)
    print(f"ROC curves -> {figures_dir}/ROC_curves.{{pdf,png}}")


def plot_latent_space(all_hea, figures_dir):
    """Scatter plot of latent coordinates colored by phase label."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_hea["z1"], all_hea["z2"], c=all_hea["Class"], cmap="viridis", s=30)
    legend_elements = [
        Line2D([0], [0], color="w", marker="o", markerfacecolor="#fde725", markersize=8, label="Single phase"),
        Line2D([0], [0], color="w", marker="o", markerfacecolor="#440154", markersize=8, label="Multiple phase"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=12)
    leg.get_frame().set_edgecolor("grey")
    leg.get_frame().set_linewidth(0.2)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.tight_layout()
    out = Path(figures_dir)
    fig.savefig(out / "latent_space.pdf")
    fig.savefig(out / "latent_space.png")
    plt.close(fig)
    print(f"Latent space -> {figures_dir}/latent_space.{{pdf,png}}")


def plot_alloy_type_distribution(model, figures_dir):
    """Show how noble, magnetic, and refractory alloy classes cluster in latent space."""
    noble = ["Ag", "Cu", "Pd", "Au", "Pt", "Zn"]
    magnetic = ["Fe", "Co", "Ni", "Ru", "Rh", "Ir"]
    refractory = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Ta", "W", "Hf"]

    z1_vals, z2_vals, colors = [], [], []
    color_map = [(noble, "g"), (magnetic, "b"), (refractory, "r")]

    for elements, color in color_map:
        for combo in combinations(elements, 4):
            formula = "".join(combo)
            ftr = calculate_engineered_features(formula)
            y = model.encoder_y(torch.tensor(ftr, dtype=torch.float32))
            comp, _, __ = calculate_compositions(formula)
            z = model.encoder_z([torch.tensor(comp, dtype=torch.float32), y])[0]
            z1_vals.append(z[0].item())
            z2_vals.append(z[1].item())
            colors.append(color)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z1_vals, z2_vals, c=colors)
    legend_elements = [
        Line2D([0], [0], color="w", marker="o", markerfacecolor="r", markersize=8, label="Refractory"),
        Line2D([0], [0], color="w", marker="o", markerfacecolor="b", markersize=8, label="Magnetic"),
        Line2D([0], [0], color="w", marker="o", markerfacecolor="g", markersize=8, label="Noble"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.tight_layout()
    out = Path(figures_dir)
    fig.savefig(out / "alloy_type_distribution.pdf")
    fig.savefig(out / "alloy_type_distribution.png")
    plt.close(fig)
    print(f"Alloy type distribution -> {figures_dir}/alloy_type_distribution.{{pdf,png}}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)

    labelled_hea = pickle.load(open(data_dir / "labelled_hea.pk", "rb"))
    unlabelled_hea = pickle.load(open(data_dir / "unlabelled_hea.pk", "rb"))
    test_hea = pickle.load(open(data_dir / "test_hea.pk", "rb"))
    validation_hea = pickle.load(open(data_dir / "validation_hea.pk", "rb"))

    model = load_model(args.model_path, args.hidden_dims, args.z_dim, args.aux_loss_multiplier, args.cuda)

    ds_sup = HEAFeatureDataset(labelled_hea)
    ds_unsup = HEAFeatureDataset(unlabelled_hea)
    ds_val = HEAFeatureDataset(validation_hea)
    ds_test = HEAFeatureDataset(test_hea)
    loader_sup = torch.utils.data.DataLoader(ds_sup, batch_size=32, shuffle=False)
    loader_unsup = torch.utils.data.DataLoader(ds_unsup, batch_size=32, shuffle=False)
    loader_val = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False)

    # Accuracy report
    train_acc, _, _ = get_accuracy(loader_sup, model.classifier, cuda=args.cuda)
    val_acc, _, _ = get_accuracy(loader_val, model.classifier, cuda=args.cuda)
    test_acc, _, _ = get_accuracy(loader_test, model.classifier, cuda=args.cuda)
    unsup_acc, _, _ = get_accuracy(loader_unsup, model.classifier, cuda=args.cuda)
    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Unsup accuracy:      {unsup_acc:.4f}")

    # Encode all splits to get latent coordinates
    labelled_hea, _, labelled_labels, labelled_engg = encode_dataframe(labelled_hea, model)
    test_hea, _, test_labels, test_engg = encode_dataframe(test_hea, model)
    validation_hea, _, val_labels, val_engg = encode_dataframe(validation_hea, model)

    # ROC curves
    plot_roc_curves(
        {
            "Training": (labelled_labels.squeeze(), labelled_engg),
            "Validation": (val_labels.squeeze(), val_engg),
            "Test": (test_labels.squeeze(), test_engg),
        },
        model,
        figures_dir,
    )

    # Latent space
    all_hea = pd.concat([labelled_hea, validation_hea, test_hea])
    all_hea["Class"] = all_hea["Class"].astype(int)
    all_hea["num_ele"] = all_hea["Alloys"].apply(count_elements)
    plot_latent_space(all_hea, figures_dir)

    # Alloy type clustering
    plot_alloy_type_distribution(model, figures_dir)


if __name__ == "__main__":
    main()
