# =============================================================================
# src/generation/sample_latent.py
# Task 2 deliverable: Latent space interpolation + 2D visualisation (t-SNE)
# =============================================================================

import os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (DEVICE, OUTPUT_PLOTS, OUTPUT_MIDI,
                    N_PITCHES, SEQ_LEN, VAE_LATENT_DIM,
                    VAE_HIDDEN_DIM, VAE_NUM_LAYERS, VAE_DROPOUT, VAE_BETA)
from models.vae import MusicVAE
from preprocessing.piano_roll import load_splits, PianoRollDataset
from generation.midi_export import pianoroll_to_midi, save_midi

ROOT     = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"


@torch.no_grad()
def latent_interpolation(n_steps: int = 8, save: bool = True):
    """Generate interpolated samples between two random latent codes."""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model  = MusicVAE(N_PITCHES, VAE_HIDDEN_DIM, VAE_LATENT_DIM, SEQ_LEN,
                      VAE_NUM_LAYERS, VAE_DROPOUT, VAE_BETA).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "vae_best.pt", map_location=device))
    model.eval()

    z1 = torch.randn(1, VAE_LATENT_DIM, device=device)
    z2 = torch.randn(1, VAE_LATENT_DIM, device=device)

    alphas = torch.linspace(0, 1, n_steps, device=device)
    rolls  = []
    for i, a in enumerate(alphas):
        z_interp = (1 - a) * z1 + a * z2
        x_hat    = model.decoder(z_interp).squeeze(0).cpu().numpy()
        roll_bin = (x_hat > 0.5).astype(np.float32)
        rolls.append(roll_bin)
        if save:
            pm   = pianoroll_to_midi(roll_bin)
            path = os.path.join(OUTPUT_MIDI, f"task2_interp_step{i+1:02d}.mid")
            save_midi(pm, path)
            print(f"[interp] {path}")

    # Piano-roll grid plot
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 3))
    for ax, roll, a in zip(axes, rolls, alphas.cpu().numpy()):
        ax.imshow(roll.T, aspect="auto", origin="lower",
                  cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"α={a:.2f}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Task 2 VAE — Latent Interpolation")
    plt.tight_layout()
    out = os.path.join(OUTPUT_PLOTS, "task2_latent_interpolation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[interp] Plot → {out}")
    return rolls


@torch.no_grad()
def tsne_latent_space(n_samples: int = 500):
    """t-SNE visualisation of latent codes coloured by genre."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[tsne] scikit-learn not installed, skipping t-SNE plot.")
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model  = MusicVAE(N_PITCHES, VAE_HIDDEN_DIM, VAE_LATENT_DIM, SEQ_LEN,
                      VAE_NUM_LAYERS, VAE_DROPOUT, VAE_BETA).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "vae_best.pt", map_location=device))
    model.eval()

    splits = load_splits()
    X = torch.tensor(splits["X_val"][:n_samples], dtype=torch.float32)
    y = splits["y_val"][:n_samples]

    mu_list = []
    bs = 64
    for i in range(0, len(X), bs):
        xb = X[i:i+bs].to(device)
        mu, _ = model.encoder(xb)
        mu_list.append(mu.cpu().numpy())
    mus = np.concatenate(mu_list, axis=0)

    print("[tsne] Running t-SNE …")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(mus)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                          c=y, cmap="tab10", alpha=0.6, s=10)
    plt.colorbar(scatter, label="Genre ID")
    plt.title("Task 2 VAE — t-SNE of Latent Space")
    plt.tight_layout()
    out = os.path.join(OUTPUT_PLOTS, "task2_tsne_latent.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[tsne] Plot → {out}")


if __name__ == "__main__":
    latent_interpolation()
    tsne_latent_space()