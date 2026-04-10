# =============================================================================
# src/models/autoencoder.py
# LSTM Autoencoder (Task 1 - FIXED)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_PITCHES, SEQ_LEN, AE_HIDDEN_DIM, AE_LATENT_DIM, AE_NUM_LAYERS, AE_DROPOUT


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=N_PITCHES,
            hidden_size=AE_HIDDEN_DIM,
            num_layers=AE_NUM_LAYERS,
            batch_first=True,
            dropout=AE_DROPOUT if AE_NUM_LAYERS > 1 else 0.0
        )

        self.fc = nn.Linear(AE_HIDDEN_DIM, AE_LATENT_DIM)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        z = self.fc(h)
        return z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_in = nn.Linear(AE_LATENT_DIM, AE_HIDDEN_DIM)

        self.lstm = nn.LSTM(
            input_size=AE_HIDDEN_DIM,
            hidden_size=AE_HIDDEN_DIM,
            num_layers=AE_NUM_LAYERS,
            batch_first=True,
            dropout=AE_DROPOUT if AE_NUM_LAYERS > 1 else 0.0
        )

        self.fc_out = nn.Linear(AE_HIDDEN_DIM, N_PITCHES)

        self.seq_len = SEQ_LEN

    def forward(self, z):
        B = z.size(0)

        h0 = torch.tanh(self.fc_in(z)).unsqueeze(0).repeat(AE_NUM_LAYERS, 1, 1)
        c0 = torch.zeros_like(h0)

        inp = h0[0].unsqueeze(1).repeat(1, self.seq_len, 1)

        out, _ = self.lstm(inp, (h0, c0))

        x_hat = torch.sigmoid(self.fc_out(out))

        return x_hat


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @staticmethod
    def loss(x, x_hat):
        # ✅ CORRECT for Lakh MIDI piano-roll (Bernoulli)
        return F.binary_cross_entropy(x_hat, x)