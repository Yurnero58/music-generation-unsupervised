# =============================================================================
# src/models/autoencoder.py
# Task 1: LSTM Autoencoder for single-genre music generation
# z = f_phi(X)   X_hat = g_theta(z)   L = ||X - X_hat||^2
# =============================================================================

import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (N_PITCHES, SEQ_LEN,
                    AE_HIDDEN_DIM, AE_LATENT_DIM, AE_NUM_LAYERS, AE_DROPOUT)


class LSTMEncoder(nn.Module):
    """Encode piano-roll sequence → fixed-size latent vector z."""

    def __init__(self, input_dim: int = N_PITCHES,
                 hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM,
                 num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=False)
        self.fc   = nn.Linear(hidden_dim, latent_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_p)
        _, (h_n, _) = self.lstm(x)       # h_n: (num_layers, B, H)
        h_last = h_n[-1]                  # (B, H) — top layer final hidden
        z = self.fc(self.drop(h_last))    # (B, latent_dim)
        return z


class LSTMDecoder(nn.Module):
    """Decode latent z → reconstructed piano-roll sequence."""

    def __init__(self, latent_dim: int = AE_LATENT_DIM,
                 hidden_dim: int = AE_HIDDEN_DIM,
                 output_dim: int = N_PITCHES,
                 seq_len: int = SEQ_LEN,
                 num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_in  = nn.Linear(latent_dim, hidden_dim)
        self.lstm   = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.size(0)
        h0 = torch.tanh(self.fc_in(z))              # (B, H)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)   # (L, B, H)
        c0 = torch.zeros_like(h0)

        # Repeat z as input at each time step
        inp = z.unsqueeze(1).repeat(1, self.seq_len, 1)       # (B, T, latent)
        inp = self.drop(inp)
        # Project to hidden size via the LSTM input at each step
        inp_proj = torch.tanh(self.fc_in(inp))                 # (B, T, H)

        out, _ = self.lstm(inp_proj, (h0, c0))                # (B, T, H)
        x_hat  = torch.sigmoid(self.fc_out(out))              # (B, T, n_p)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder (Task 1)."""

    def __init__(self, input_dim: int = N_PITCHES,
                 hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM,
                 seq_len: int = SEQ_LEN,
                 num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim,
                                   num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim,
                                   seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)       # (B, latent_dim)
        x_hat = self.decoder(z)       # (B, T, n_p)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss: L_AE = sum_t ||x_t - x_hat_t||^2"""
        return nn.functional.mse_loss(x_hat, x, reduction="mean")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, n_p = 8, SEQ_LEN, N_PITCHES
    model = LSTMAutoencoder()
    x     = torch.rand(B, T, n_p)
    x_hat, z = model(x)
    loss  = LSTMAutoencoder.loss(x, x_hat)
    print(f"x:     {x.shape}")
    print(f"z:     {z.shape}")
    print(f"x_hat: {x_hat.shape}")
    print(f"loss:  {loss.item():.4f}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")