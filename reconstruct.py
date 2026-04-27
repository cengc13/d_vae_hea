#!/usr/bin/env python
"""Alloy reconstruction analysis: encode to latent space, decode back, measure error.

Usage:
    python reconstruct.py
    python reconstruct.py --model-path models/ssvae_best_val.model
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import pyro

from ssvae.model import SSVAE
from utils.featurization import top30, calculate_compositions, calculate_engineered_features


def parse_args():
    p = argparse.ArgumentParser(description="Alloy reconstruction analysis")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-path", default="models/ssvae.model")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--output-csv", default="data/test_data_reconstruction_analysis.csv")
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[100, 100])
    p.add_argument("--aux-loss-multiplier", type=float, default=50.0)
    return p.parse_args()


def count_elements(alloy):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.*\d*?(?=\D|$))")
    parts = [(x, float(y)) if y else (x, 1) for x, y in pattern.findall(alloy)]
    return len(parts)


def get_latent(alloy, model):
    ftr = calculate_engineered_features(alloy)
    y = model.encoder_y(torch.tensor(ftr, dtype=torch.float32))
    comp, _, __ = calculate_compositions(alloy)
    z = model.encoder_z([torch.tensor(comp, dtype=torch.float32), y])[0]
    return z.detach().numpy().round(3), y.item()


def reconstruct_alloy(alloy, model, test_hea):
    """Encode alloy through the trained SSVAE and decode back to a composition."""
    engg = torch.tensor(test_hea[test_hea["Alloys"] == alloy].loc[:, "k":"delta_h_mix"].values, dtype=torch.float32)
    z = torch.tensor(test_hea[test_hea["Alloys"] == alloy].loc[:, "z1":"z2"].values, dtype=torch.float32)

    y = model.encoder_y(engg)
    _, __, old_alloy = calculate_compositions(alloy)
    comp_orig, _, __ = calculate_compositions(old_alloy)
    comp_orig = np.array(comp_orig)

    with torch.no_grad():
        inv_comps = model.decoder([z, y])
    inv_alloy_comps = (torch.round(inv_comps, decimals=2) * 100).numpy()[0]
    new_alloy = "".join([str(e) + str(int(c)) for e, c in zip(top30, inv_alloy_comps) if c > 0])

    new_comp, _, __ = calculate_compositions(new_alloy)
    new_comp = np.array(new_comp)
    new_z, new_y = get_latent(new_alloy, model)

    z_arr = z.numpy().round(3)[0]
    y_val = round(y.item(), 2)
    new_y = round(new_y, 2)

    return old_alloy, new_alloy, comp_orig, new_comp, y_val, new_y, z_arr, new_z


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)

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

    # Encode test set to populate z1, z2 columns
    test_data = torch.tensor(test_hea.loc[:, "Fe":"Sc"].values * 100, dtype=torch.float32)
    test_labels = torch.tensor(test_hea["Class"].values, dtype=torch.float32).unsqueeze(-1)
    test_engg = torch.tensor(test_hea.loc[:, "k":"delta_h_mix"].values, dtype=torch.float32)

    with torch.no_grad():
        z_loc, _ = model.encoder_z([test_data, test_labels])
        label_pred = model.classifier(test_engg)

    test_hea = test_hea.copy()
    test_hea["Predicted_Class"] = label_pred.numpy().squeeze().astype(int).astype(str)
    test_hea["z1"] = z_loc[:, 0].numpy()
    test_hea["z2"] = z_loc[:, 1].numpy()
    test_hea["num_ele"] = test_hea["Alloys"].apply(count_elements)

    # Reconstruct all test alloys
    new_cols = [
        "normalized_alloy", "reconstructed_alloy",
        "old_comp_fv", "reconstructed_comp_fv",
        "y", "reconstructed_y", "z", "reconstructed_z",
    ]
    test_hea[new_cols] = test_hea.apply(
        lambda row: reconstruct_alloy(row.Alloys, model, test_hea),
        axis="columns",
        result_type="expand",
    )

    test_hea["comp_diff"] = np.abs(test_hea["old_comp_fv"] - test_hea["reconstructed_comp_fv"])
    test_hea["y_diff"] = np.abs(test_hea["y"] - test_hea["reconstructed_y"])
    test_hea["z_diff"] = np.abs(test_hea["z"] - test_hea["reconstructed_z"])
    test_hea["comp_diff_mae"] = test_hea["comp_diff"].apply(np.mean)
    test_hea["z_diff_mae"] = test_hea["z_diff"].apply(np.mean)

    print(f"Mean composition MAE (%): {np.mean(test_hea.comp_diff_mae):.4f}")
    print(f"Mean phase-prob MAE:      {np.mean(test_hea.y_diff):.4f}")
    print(f"Mean latent MAE:          {np.mean(test_hea.z_diff_mae):.4f}")

    # Distribution plots grouped by number of elements
    for col, xlabel, fname in [
        ("comp_diff_mae", "Composition error (%)", "Test_composition_MAE"),
        ("y_diff",        "Single-phase prob. error (-)", "Test_y_MAE"),
        ("z_diff_mae",    "Latent variable error (-)", "Test_z_MAE"),
    ]:
        sns.displot(test_hea, x=col, hue="num_ele", stat="probability", palette="tab10", multiple="stack")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.savefig(figures_dir / f"{fname}.pdf")
        plt.savefig(figures_dir / f"{fname}.png")
        plt.close()

    print(f"Figures saved to {figures_dir}/")

    cols_to_save = ["Alloys"] + new_cols + ["comp_diff", "y_diff", "z_diff", "comp_diff_mae", "z_diff_mae"]
    test_hea.to_csv(args.output_csv, columns=cols_to_save, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
