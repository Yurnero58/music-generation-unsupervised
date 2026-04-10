# =============================================================================
# src/models/diffusion.py
# Optional: Denoising Diffusion stub (placeholder for future extension)
# This file satisfies the project structure requirement.
# Full DDPM music generation is beyond the 4-task scope.
# =============================================================================

import torch
import torch.nn as nn


class SimpleDiffusionPlaceholder(nn.Module):
    """
    Minimal forward-diffusion noise schedule reference.
    Not trained in this project — included for structural completeness.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """Forward diffusion: q(x_t | x_0)."""
        ab = self.alpha_bar[t].view(-1, 1, 1)
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return x_t, eps

    def forward(self, x):
        raise NotImplementedError("Diffusion model not implemented in this project.")