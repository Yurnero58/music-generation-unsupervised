# =============================================================================
# src/training/train_ae.py
# Task 1: Train LSTM Autoencoder
# Algorithm 1 from the project spec
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
from config import (DEVICE, NUM_EPOCHS_AE, LEARNING_RATE, BATCH_SIZE,
                    OUTPUT_PLOTS, DATA_SPLIT, N_PITCHES, SEQ_LEN,
                    AE_HIDDEN_DIM, AE_LATENT_DIM, AE_NUM_LAYERS, AE_DROPOUT)
from models.autoencoder import LSTMAutoencoder
from preprocessing.piano_roll import load_splits, get_dataloaders

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimiser.zero_grad()
        x_hat, z = model(x)
        loss = LSTMAutoencoder.loss(x, x_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        total_loss += LSTMAutoencoder.loss(x, x_hat).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_ae] Device: {device}")

    # Data
    splits     = load_splits()
    train_dl, val_dl, _ = get_dataloaders(splits, BATCH_SIZE)

    # Model
    model = LSTMAutoencoder(
        input_dim=N_PITCHES, hidden_dim=AE_HIDDEN_DIM,
        latent_dim=AE_LATENT_DIM, seq_len=SEQ_LEN,
        num_layers=AE_NUM_LAYERS, dropout=AE_DROPOUT
    ).to(device)
    print(f"[train_ae] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=5,
        factor=0.5
        )

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS_AE + 1):
        t0 = time.time()
        tr_loss  = train_one_epoch(model, train_dl, optimiser, device)
        val_loss = evaluate(model, val_dl, device)
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]['lr']
        print(f"[LR] {current_lr}")
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS_AE}  "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       CKPT_DIR / "ae_best.pt")

    # Save final
    torch.save(model.state_dict(), CKPT_DIR / "ae_final.pt")

    # Save loss history
    history = {"train": train_losses, "val": val_losses}
    with open(CKPT_DIR / "ae_history.json", "w") as f:
        json.dump(history, f)

    # Plot reconstruction loss curve (required deliverable)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Task 1 — LSTM Autoencoder: Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, "task1_ae_loss.png"), dpi=150)
    plt.close()
    print(f"[train_ae] Loss curve saved. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()