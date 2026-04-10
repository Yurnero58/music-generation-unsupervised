# =============================================================================
# src/models/vae.py
# Task 2: Variational Autoencoder for multi-genre music generation
# q_phi(z|X) = N(mu(X), sigma(X))
# L_VAE = L_recon + β * D_KL(q_phi(z|X) || p(z))
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (N_PITCHES, SEQ_LEN,
                    VAE_HIDDEN_DIM, VAE_LATENT_DIM, VAE_NUM_LAYERS,
                    VAE_DROPOUT, VAE_BETA)


class VAEEncoder(nn.Module):
    """Bidirectional LSTM encoder → (mu, log_var)."""

    def __init__(self, input_dim: int = N_PITCHES,
                 hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM,
                 num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop    = nn.Dropout(dropout)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, n_p)
        out, _ = self.lstm(x)                    # (B, T, hidden_dim)
        # Mean-pool over time for a compact sequence representation
        h = out.mean(dim=1)                       # (B, hidden_dim)
        h = self.drop(h)
        mu      = self.fc_mu(h)                   # (B, latent_dim)
        log_var = self.fc_logv(h)                 # (B, latent_dim)
        return mu, log_var


class VAEDecoder(nn.Module):
    """LSTM decoder conditioned on sampled z → reconstructed sequence."""

    def __init__(self, latent_dim: int = VAE_LATENT_DIM,
                 hidden_dim: int = VAE_HIDDEN_DIM,
                 output_dim: int = N_PITCHES,
                 seq_len: int = SEQ_LEN,
                 num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_init = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm    = nn.LSTM(latent_dim, hidden_dim, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out  = nn.Linear(hidden_dim, output_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.size(0)
        # Build initial hidden state from z
        h = torch.tanh(self.fc_init(z))                        # (B, H*L)
        h = h.view(self.num_layers, B, self.hidden_dim)        # (L, B, H)
        c = torch.zeros_like(h)

        inp  = z.unsqueeze(1).repeat(1, self.seq_len, 1)       # (B, T, latent)
        inp  = self.drop(inp)
        out, _ = self.lstm(inp, (h, c))                        # (B, T, H)
        x_hat  = torch.sigmoid(self.fc_out(out))               # (B, T, n_p)
        return x_hat


class MusicVAE(nn.Module):
    """β-VAE for multi-genre music generation (Task 2)."""

    def __init__(self, input_dim: int = N_PITCHES,
                 hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM,
                 seq_len: int = SEQ_LEN,
                 num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT,
                 beta: float = VAE_BETA):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim,
                                  num_layers, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim,
                                  seq_len, num_layers, dropout)

    # ------------------------------------------------------------------
    def reparameterise(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps,  eps ~ N(0, I)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    # ------------------------------------------------------------------
    @staticmethod
    def recon_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x, reduction="mean")

    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """D_KL(q(z|X) || p(z)) closed form for Gaussian."""
        # = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        return -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = None) -> dict:
        beta = beta if beta is not None else self.beta
        l_recon = self.recon_loss(x, x_hat)
        l_kl    = self.kl_loss(mu, log_var)
        l_total = l_recon + beta * l_kl
        return {"total": l_total, "recon": l_recon, "kl": l_kl}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample z ~ N(0,I) and decode."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    steps: int = 8) -> torch.Tensor:
        """Latent interpolation between two piano-rolls."""
        mu1, _ = self.encoder(x1.unsqueeze(0))
        mu2, _ = self.encoder(x2.unsqueeze(0))
        alphas  = torch.linspace(0, 1, steps, device=mu1.device)
        zs      = torch.stack([(1 - a) * mu1 + a * mu2 for a in alphas]).squeeze(1)
        return self.decoder(zs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, n_p = 8, SEQ_LEN, N_PITCHES
    model = MusicVAE()
    x     = torch.rand(B, T, n_p)
    x_hat, mu, log_var = model(x)
    losses = model.loss(x, x_hat, mu, log_var)
    print(f"x:       {x.shape}")
    print(f"x_hat:   {x_hat.shape}")
    print(f"mu:      {mu.shape}")
    print(f"Losses → total: {losses['total'].item():.4f}  "
          f"recon: {losses['recon'].item():.4f}  "
          f"kl: {losses['kl'].item():.4f}")
    samples = model.sample(4)
    print(f"Samples: {samples.shape}")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")