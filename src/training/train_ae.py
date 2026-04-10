# =============================================================================
# src/training/train_ae.py
# =============================================================================

import os, sys, json, time
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEVICE, NUM_EPOCHS_AE, LEARNING_RATE, BATCH_SIZE, OUTPUT_PLOTS
from models.autoencoder import LSTMAutoencoder
from preprocessing.piano_roll import load_splits, get_dataloaders


ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    total = 0

    for x, _ in loader:
        x = x.to(device)

        optim.zero_grad()

        x_hat, _ = model(x)

        loss = model.loss(x, x_hat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()

        total += loss.item()

    return total / len(loader)


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0

    for x, _ in loader:
        x = x.to(device)

        x_hat, _ = model(x)

        loss = model.loss(x, x_hat)

        total += loss.item()

    return total / len(loader)


# ---------------------------------------------------------------------------
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("[train_ae] Device:", device)

    splits = load_splits()
    train_dl, val_dl, _ = get_dataloaders(splits, BATCH_SIZE)

    model = LSTMAutoencoder().to(device)

    print("[train_ae] Params:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []

    best = float("inf")

    for epoch in range(NUM_EPOCHS_AE):

        t0 = time.time()

        tr = train_one_epoch(model, train_dl, optimizer, device)
        va = evaluate(model, val_dl, device)

        train_losses.append(tr)
        val_losses.append(va)

        print(f"Epoch {epoch+1} | train={tr:.5f} | val={va:.5f} | time={time.time()-t0:.1f}s")

        if va < best:
            best = va
            torch.save(model.state_dict(), CKPT_DIR / "ae_best.pt")

    torch.save(model.state_dict(), CKPT_DIR / "ae_final.pt")

    # Plot
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("LSTM Autoencoder Loss")
    plt.savefig(os.path.join(OUTPUT_PLOTS, "ae_loss.png"))