# =============================================================================
# src/generation/generate_music.py
# Generate MIDI samples for all 4 tasks from trained checkpoints
# =============================================================================

import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (DEVICE, OUTPUT_MIDI,
                    N_SAMPLES_AE, N_SAMPLES_VAE, N_SAMPLES_TR, N_SAMPLES_RL,
                    N_PITCHES, SEQ_LEN, AE_LATENT_DIM, VAE_LATENT_DIM,
                    TR_VOCAB_SIZE, TR_D_MODEL, TR_NHEAD, TR_NUM_LAYERS,
                    TR_DIM_FF, TR_MAX_SEQ_LEN,
                    AE_HIDDEN_DIM, AE_NUM_LAYERS, AE_DROPOUT,
                    VAE_HIDDEN_DIM, VAE_NUM_LAYERS, VAE_DROPOUT, VAE_BETA,
                    TEMPERATURE, TOP_K)
from models.autoencoder  import LSTMAutoencoder
from models.vae          import MusicVAE
from models.transformer  import MusicTransformer
from generation.midi_export import pianoroll_to_midi, save_midi
from preprocessing.tokenizer import tokens_to_pianoroll

ROOT     = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
TR_SEQ   = 128


# ---------------------------------------------------------------------------
def load_ae(device):
    model = LSTMAutoencoder(N_PITCHES, AE_HIDDEN_DIM, AE_LATENT_DIM,
                            SEQ_LEN, AE_NUM_LAYERS, AE_DROPOUT).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "ae_best.pt", map_location=device))
    model.eval()
    return model


def load_vae(device):
    model = MusicVAE(N_PITCHES, VAE_HIDDEN_DIM, VAE_LATENT_DIM, SEQ_LEN,
                     VAE_NUM_LAYERS, VAE_DROPOUT, VAE_BETA).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "vae_best.pt", map_location=device))
    model.eval()
    return model


def load_transformer(device, rl: bool = False):
    ckpt = "rl_best.pt" if rl else "transformer_best.pt"
    model = MusicTransformer(TR_VOCAB_SIZE, TR_D_MODEL, TR_NHEAD,
                             TR_NUM_LAYERS, TR_DIM_FF, 0.0, TR_SEQ).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / ckpt, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_ae_samples(n: int = N_SAMPLES_AE, device=None, tag="task1"):
    model = load_ae(device)
    rolls = []
    for i in range(n):
        z     = torch.randn(1, AE_LATENT_DIM, device=device)
        x_hat = model.decode(z).squeeze(0).cpu().numpy()   # (T, n_p)
        x_bin = (x_hat > 0.5).astype(np.float32)
        rolls.append(x_bin)
        pm    = pianoroll_to_midi(x_bin)
        path  = os.path.join(OUTPUT_MIDI, f"{tag}_sample_{i+1:02d}.mid")
        save_midi(pm, path)
        print(f"[gen] {path}")
    return rolls


@torch.no_grad()
def generate_vae_samples(n: int = N_SAMPLES_VAE, device=None, tag="task2"):
    model = load_vae(device)
    rolls = []
    for i in range(n):
        x_hat = model.sample(1, device=str(device)).squeeze(0).cpu().numpy()
        x_bin = (x_hat > 0.5).astype(np.float32)
        rolls.append(x_bin)
        pm    = pianoroll_to_midi(x_bin)
        path  = os.path.join(OUTPUT_MIDI, f"{tag}_sample_{i+1:02d}.mid")
        save_midi(pm, path)
        print(f"[gen] {path}")
    return rolls


@torch.no_grad()
def generate_transformer_samples(n: int = N_SAMPLES_TR, device=None,
                                  tag="task3", rl=False):
    model = load_transformer(device, rl=rl)
    rolls = []
    for i in range(n):
        tokens = model.generate(max_new=TR_SEQ, temperature=TEMPERATURE,
                                top_k=TOP_K, device=str(device))
        roll   = tokens_to_pianoroll(tokens, SEQ_LEN)
        rolls.append(roll)
        pm     = pianoroll_to_midi(roll)
        path   = os.path.join(OUTPUT_MIDI, f"{tag}_sample_{i+1:02d}.mid")
        save_midi(pm, path)
        print(f"[gen] {path}")
    return rolls


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1","2","3","4","all"], default="all")
    args   = parser.parse_args()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_MIDI, exist_ok=True)

    if args.task in ("1", "all"):
        print("\n=== Task 1: LSTM Autoencoder Samples ===")
        generate_ae_samples(N_SAMPLES_AE, device, "task1_ae")

    if args.task in ("2", "all"):
        print("\n=== Task 2: VAE Samples ===")
        generate_vae_samples(N_SAMPLES_VAE, device, "task2_vae")

    if args.task in ("3", "all"):
        print("\n=== Task 3: Transformer Samples ===")
        generate_transformer_samples(N_SAMPLES_TR, device, "task3_tr", rl=False)

    if args.task in ("4", "all"):
        print("\n=== Task 4: RLHF Samples ===")
        generate_transformer_samples(N_SAMPLES_RL, device, "task4_rl", rl=True)

    print("\n[gen] All samples saved to", OUTPUT_MIDI)


if __name__ == "__main__":
    main()