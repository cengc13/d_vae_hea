#!/usr/bin/env python
"""Latent space scanning and iterative inverse design of HEA compositions.

Usage:
    python interpolate.py                          # scan default region + example inversion
    python interpolate.py --alloy AlCoCrFeNi       # invert a specific alloy
    python interpolate.py --z1-range -1 1 --z2-range -1 1 --n-points 7
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch
import pyro

from ssvae.model import SSVAE
from utils.featurization import top30, calculate_compositions, calculate_engineered_features


def parse_args():
    p = argparse.ArgumentParser(description="Latent space interpolation and inverse design")
    p.add_argument("--model-path", default="models/ssvae.model")
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[100, 100])
    p.add_argument("--aux-loss-multiplier", type=float, default=50.0)
    p.add_argument("--z1-range", type=float, nargs=2, default=[-0.1, 0.1], metavar=("Z1_MIN", "Z1_MAX"))
    p.add_argument("--z2-range", type=float, nargs=2, default=[-0.5, -0.3], metavar=("Z2_MIN", "Z2_MAX"))
    p.add_argument("--n-points", type=int, default=5)
    p.add_argument("--phase-label", type=float, default=1.0, help="Phase label for decoding (0=multi, 1=single)")
    p.add_argument("--alloy", default="Al1.4Co0.9Cr1.4Cu0.5Fe0.9Ni1", help="Alloy for inverse design example")
    p.add_argument("--target-prob", type=float, default=0.6, help="Target single-phase probability for inversion")
    p.add_argument("--max-iter", type=int, default=10)
    return p.parse_args()


def load_model(model_path, hidden_dims, z_dim, aux_loss_multiplier):
    pyro.clear_param_store()
    model = SSVAE(
        output_size=1,
        input_size=30,
        z_dim=z_dim,
        hidden_layers=hidden_dims,
        use_cuda=False,
        aux_loss_multiplier=aux_loss_multiplier,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def generate_alloy(z, phase_label, model):
    """Decode a latent coordinate to an alloy composition string."""
    z_t = torch.tensor([z], dtype=torch.float32)
    y_t = torch.tensor([[phase_label]], dtype=torch.float32)
    with torch.no_grad():
        comps = model.decoder([z_t, y_t])
    comps_pct = (torch.round(comps, decimals=2) * 100).numpy().squeeze()
    return "".join([str(e) + str(int(c)) for e, c in zip(top30, comps_pct) if c > 0])


def get_phase_probability(alloy, model):
    """Predict single-phase probability from engineered features."""
    ftr = calculate_engineered_features(alloy)
    with torch.no_grad():
        y = model.encoder_y(torch.tensor(ftr, dtype=torch.float32))
    return y.item()


def scan_latent_space(z1_range, z2_range, n_points, phase_label, model):
    """Generate alloys at a grid of latent coordinates and print predictions."""
    z1_vals = np.linspace(*z1_range, num=n_points)
    z2_vals = np.linspace(*z2_range, num=n_points)

    print(f"\n{'z':>20}  {'alloy (y=label)':>35}  p(single)  {'alloy (y=1-label)':>35}  p(single)")
    print("-" * 120)

    results = []
    for z1, z2 in product(z1_vals, z2_vals):
        z = [z1, z2]
        alloy = generate_alloy(z, phase_label, model)
        inv_alloy = generate_alloy(z, 1 - phase_label, model)
        y_fwd = get_phase_probability(alloy, model)
        y_inv = get_phase_probability(inv_alloy, model)
        print(f"[{z1:5.2f}, {z2:5.2f}]  {alloy:>35}  {y_fwd:.3f}      {inv_alloy:>35}  {y_inv:.3f}")
        results.append({"z": z, "alloy": alloy, "phase_prob": y_fwd,
                        "inv_alloy": inv_alloy, "inv_phase_prob": y_inv})

    return results


def iterative_inverse_design(start_alloy, model, target_prob=0.6, max_iter=10):
    """Iteratively decode with inverted phase label to find single-phase compositions.

    Starting from a multi-phase alloy, each iteration encodes the current alloy,
    then decodes with the complement phase label to push toward the single-phase region.
    """
    _, __, alloy = calculate_compositions(start_alloy)
    print(f"\nStarting alloy: {alloy}")

    ftr = calculate_engineered_features(alloy)
    with torch.no_grad():
        y = model.encoder_y(torch.tensor(ftr, dtype=torch.float32))

    for step in range(max_iter):
        p = y.item()
        print(f"  Step {step}: p(single)={p:.3f}  alloy={alloy}")
        if p >= target_prob:
            break

        comp, _, __ = calculate_compositions(alloy)
        with torch.no_grad():
            z = model.encoder_z([torch.tensor(comp, dtype=torch.float32), y])[0]
            inv_comps = model.decoder([z.unsqueeze(0), 1 - y])
        inv_comps_pct = (torch.round(inv_comps, decimals=2) * 100).numpy().squeeze()
        alloy = "".join([str(e) + str(int(c)) for e, c in zip(top30, inv_comps_pct) if c > 0])

        ftr = calculate_engineered_features(alloy)
        with torch.no_grad():
            y = model.encoder_y(torch.tensor(ftr, dtype=torch.float32))

    comp, _, __ = calculate_compositions(alloy)
    with torch.no_grad():
        z = model.encoder_z([torch.tensor(comp, dtype=torch.float32), y])[0]

    return alloy, y.item(), z.detach().numpy().round(3)


def main():
    args = parse_args()
    model = load_model(args.model_path, args.hidden_dims, args.z_dim, args.aux_loss_multiplier)

    print("=== Latent Space Scan ===")
    scan_latent_space(
        z1_range=args.z1_range,
        z2_range=args.z2_range,
        n_points=args.n_points,
        phase_label=args.phase_label,
        model=model,
    )

    print("\n=== Iterative Inverse Design ===")
    result_alloy, result_prob, result_z = iterative_inverse_design(
        args.alloy, model, target_prob=args.target_prob, max_iter=args.max_iter
    )
    print(f"\nFinal alloy:              {result_alloy}")
    print(f"Single-phase probability: {result_prob:.4f}")
    print(f"Latent coordinates:       {result_z}")


if __name__ == "__main__":
    main()
