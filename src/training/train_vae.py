# =============================================================================
# src/training/train_vae.py
# Task 2: Train β-VAE for multi-genre music generation
# Algorithm 2 from the project spec
# =============================================================================

import os, sys, json, time
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (DEVICE, NUM_EPOCHS_VAE, LEARNING_RATE, BATCH_SIZE,
                    OUTPUT_PLOTS, N_PITCHES, SEQ_LEN,
                    VAE_HIDDEN_DIM, VAE_LATENT_DIM, VAE_NUM_LAYERS,
                    VAE_DROPOUT, VAE_BETA)
from models.vae import MusicVAE
from preprocessing.piano_roll import load_splits, get_dataloaders

ROOT     = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, device, beta):
    model.train()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    for x, _ in loader:
        x = x.to(device)
        optimiser.zero_grad()
        x_hat, mu, log_var = model(x)
        losses = model.loss(x, x_hat, mu, log_var, beta=beta)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        for k in totals:
            totals[k] += losses[k].item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, beta):
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    for x, _ in loader:
        x = x.to(device)
        x_hat, mu, log_var = model(x)
        losses = model.loss(x, x_hat, mu, log_var, beta=beta)
        for k in totals:
            totals[k] += losses[k].item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_vae] Device: {device}")

    splits = load_splits()
    train_dl, val_dl, _ = get_dataloaders(splits, BATCH_SIZE)

    model = MusicVAE(
        input_dim=N_PITCHES, hidden_dim=VAE_HIDDEN_DIM,
        latent_dim=VAE_LATENT_DIM, seq_len=SEQ_LEN,
        num_layers=VAE_NUM_LAYERS, dropout=VAE_DROPOUT,
        beta=VAE_BETA
    ).to(device)
    print(f"[train_vae] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=NUM_EPOCHS_VAE)

    history = {"train_total": [], "train_recon": [], "train_kl": [],
               "val_total":   [], "val_recon":   [], "val_kl":   []}
    best_val = float("inf")

    # KL annealing: linearly ramp beta from 0 → VAE_BETA over first 20 epochs
    warmup_epochs = 20

    for epoch in range(1, NUM_EPOCHS_VAE + 1):
        t0   = time.time()
        beta = min(VAE_BETA, VAE_BETA * epoch / warmup_epochs)

        tr  = train_one_epoch(model, train_dl, optimiser, device, beta)
        val = evaluate(model, val_dl, device, beta)
        scheduler.step()

        for k in ["total", "recon", "kl"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(val[k])

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS_VAE}  β={beta:.2f}  "
              f"train [tot={tr['total']:.4f} rec={tr['recon']:.4f} kl={tr['kl']:.4f}]  "
              f"val [tot={val['total']:.4f}]  ({time.time()-t0:.1f}s)")

        if val["total"] < best_val:
            best_val = val["total"]
            torch.save(model.state_dict(), CKPT_DIR / "vae_best.pt")

    torch.save(model.state_dict(), CKPT_DIR / "vae_final.pt")
    with open(CKPT_DIR / "vae_history.json", "w") as f:
        json.dump(history, f)

    # ---- Plots (deliverable) ----
    epochs = range(1, NUM_EPOCHS_VAE + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
            axes,
            ["total", "recon", "kl"],
            ["Total Loss", "Reconstruction Loss", "KL Divergence"]):
        ax.plot(history[f"train_{key}"], label="Train")
        ax.plot(history[f"val_{key}"],   label="Val")
        ax.set_title(f"Task 2 VAE — {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, "task2_vae_loss.png"), dpi=150)
    plt.close()
    print(f"[train_vae] Saved. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()