#!/usr/bin/env python
"""Train the SSVAE model for HEA phase prediction.

Usage:
    python train.py                          # defaults
    python train.py --epochs 5000 --lr 1e-4
    python train.py --cuda --epochs 20000
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ReduceLROnPlateau
from torch.optim import Adam

from ssvae.dataset import HEAFeatureDataset
from ssvae.model import SSVAE
from ssvae.trainer import evaluate_model, get_accuracy, run_inference_for_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train SSVAE for HEA phase prediction")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[100, 100])
    p.add_argument("--aux-loss-multiplier", type=float, default=10.0)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


def load_splits(data_dir):
    """Load pre-saved train/val/test splits, or create and save them from raw CSVs."""
    data_dir = Path(data_dir)
    split_files = ["labelled_hea.pk", "unlabelled_hea.pk", "test_hea.pk", "validation_hea.pk"]

    if all((data_dir / f).exists() for f in split_files):
        labelled = pickle.load(open(data_dir / "labelled_hea.pk", "rb"))
        unlabelled = pickle.load(open(data_dir / "unlabelled_hea.pk", "rb"))
        test = pickle.load(open(data_dir / "test_hea.pk", "rb"))
        val = pickle.load(open(data_dir / "validation_hea.pk", "rb"))
        return labelled, unlabelled, test, val

    print("Split files not found — creating splits from raw CSVs...")
    top30 = pd.read_csv(data_dir / "HEA_top30_comps.csv", comment="#")
    engineered = pd.read_csv(data_dir / "HEA_feature_engineered.csv", comment="#")
    top30 = top30.drop_duplicates(subset="Alloys", keep="first")
    engineered = engineered.drop_duplicates(subset="Alloys", keep="first")
    merged = pd.merge(top30, engineered.drop(columns="Class"), on="Alloys", how="inner")

    labelled, test = train_test_split(merged, test_size=0.1, random_state=42)
    labelled, unlabelled = train_test_split(labelled, test_size=0.3, random_state=42)
    unlabelled, val = train_test_split(unlabelled, test_size=0.2, random_state=42)

    pickle.dump(labelled, open(data_dir / "labelled_hea.pk", "wb"))
    pickle.dump(unlabelled, open(data_dir / "unlabelled_hea.pk", "wb"))
    pickle.dump(test, open(data_dir / "test_hea.pk", "wb"))
    pickle.dump(val, open(data_dir / "validation_hea.pk", "wb"))
    return labelled, unlabelled, test, val


def main():
    args = parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)

    labelled, unlabelled, test, val = load_splits(args.data_dir)

    ds_sup = HEAFeatureDataset(labelled)
    ds_unsup = HEAFeatureDataset(unlabelled)
    ds_val = HEAFeatureDataset(val)
    ds_test = HEAFeatureDataset(test)

    loader_sup = torch.utils.data.DataLoader(ds_sup, batch_size=args.batch_size, shuffle=True)
    loader_unsup = torch.utils.data.DataLoader(ds_unsup, batch_size=args.batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    data_loaders = {"sup": loader_sup, "unsup": loader_unsup, "val": loader_val, "test": loader_test}

    print(f"Labelled: {len(ds_sup)} | Unlabelled: {len(ds_unsup)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    pyro.clear_param_store()
    model = SSVAE(
        output_size=1,
        input_size=30,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_dims,
        use_cuda=use_cuda,
        aux_loss_multiplier=args.aux_loss_multiplier,
    )

    adam_params = {"lr": args.lr, "betas": (0.9, 0.999)}
    scheduler = ReduceLROnPlateau({
        "optimizer": Adam,
        "optim_args": adam_params,
        "mode": "min",
        "factor": 0.5,
        "patience": 200,
        "verbose": True,
    })
    loss_basic = SVI(model.model, model.guide, scheduler, loss=Trace_ELBO())
    loss_aux = SVI(model.model_classify, model.guide_classify, scheduler, loss=Trace_ELBO())
    losses = [loss_basic, loss_aux]

    best_val_acc = 0.0
    best_train_acc = 0.0

    print(f"Training for {args.epochs} epochs...")
    try:
        for epoch in range(args.epochs):
            model.train()
            sup_losses, unsup_losses = run_inference_for_epoch(data_loaders, losses, cuda=use_cuda)

            val_losses = evaluate_model(loader_val, model, losses, cuda=use_cuda)
            model.train()

            torch.nn.utils.clip_grad_norm_(model.encoder_z.parameters(), max_norm=1.0)

            train_acc, _, _ = get_accuracy(loader_sup, model.classifier, cuda=use_cuda)
            val_acc, _, _ = get_accuracy(loader_val, model.classifier, cuda=use_cuda)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                torch.save(model.state_dict(), model_dir / "ssvae_best_train.model")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_dir / "ssvae_best_val.model")

            avg_sup = sup_losses[0] / len(ds_sup)
            print(
                f"Epoch {epoch:5d} | elbo: {avg_sup:.4f} | "
                f"train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}"
            )

            scheduler.step(sup_losses[0] / len(ds_sup))

            if args.save_interval > 0 and epoch > 0 and epoch % args.save_interval == 0:
                torch.save(model.state_dict(), model_dir / f"ssvae_epoch{epoch}.model")

    finally:
        ts = datetime.now().strftime("%m%d%Y_%H%M%S")
        torch.save(model.state_dict(), model_dir / f"ssvae_{ts}.model")
        torch.save(model.state_dict(), model_dir / "ssvae.model")
        print("Checkpoint saved.")

    model.eval()
    test_acc, _, _ = get_accuracy(loader_test, model.classifier, cuda=use_cuda)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
