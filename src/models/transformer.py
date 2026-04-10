# =============================================================================
# src/models/transformer.py
# Task 3: Causal Transformer decoder for long-horizon music generation
# p(X) = prod_t p(x_t | x_{<t})
# L_TR = -sum_t log p_theta(x_t | x_{<t})
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (TR_VOCAB_SIZE, TR_D_MODEL, TR_NHEAD, TR_NUM_LAYERS,
                    TR_DIM_FF, TR_DROPOUT, TR_MAX_SEQ_LEN,
                    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
                    TOP_K, TEMPERATURE)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


# ---------------------------------------------------------------------------
# Music Transformer (decoder-only, causal)
# ---------------------------------------------------------------------------
class MusicTransformer(nn.Module):
    """
    GPT-style decoder-only Transformer for autoregressive music generation.
    h_t = Emb(x_t) + Emb_pos(t)
    p(x_t | x_{<t}) = softmax(W h_t)
    """

    def __init__(self, vocab_size: int = TR_VOCAB_SIZE,
                 d_model: int = TR_D_MODEL,
                 nhead: int = TR_NHEAD,
                 num_layers: int = TR_NUM_LAYERS,
                 dim_feedforward: int = TR_DIM_FF,
                 dropout: float = TR_DROPOUT,
                 max_seq_len: int = TR_MAX_SEQ_LEN):
        super().__init__()
        self.d_model  = d_model
        self.vocab    = vocab_size

        self.embed    = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc  = PositionalEncoding(d_model, dropout, max_seq_len + 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True)          # Pre-LN for stability
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out   = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
        nn.init.normal_(self.fc_out.weight, std=0.02)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x == PAD_TOKEN)   # (B, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) token ids
        Returns logits: (B, T, vocab_size)
        """
        B, T = x.shape
        causal = self._causal_mask(T, x.device)
        key_pad = self._pad_mask(x)

        h = self.embed(x) * math.sqrt(self.d_model)   # (B, T, d_model)
        h = self.pos_enc(h)

        # Decoder-only: use h as both memory and target (GPT-style trick)
        out = self.transformer(
            tgt=h, memory=h,
            tgt_mask=causal,
            memory_mask=causal,
            tgt_key_padding_mask=key_pad,
            memory_key_padding_mask=key_pad)           # (B, T, d_model)

        logits = self.fc_out(out)                      # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy autoregressive loss.
        L_TR = -sum_t log p_theta(x_t | x_{<t})
        logits:  (B, T, V)
        targets: (B, T)
        """
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=PAD_TOKEN)

    @staticmethod
    def perplexity(loss_val: float) -> float:
        """Perplexity = exp(L_TR)"""
        return math.exp(loss_val)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: list[int] = None,
                 max_new: int = 256,
                 temperature: float = TEMPERATURE,
                 top_k: int = TOP_K,
                 device: str = "cpu") -> list[int]:
        """Autoregressive generation with top-k sampling."""
        self.eval()
        if prompt is None:
            prompt = [BOS_TOKEN]
        seq = torch.tensor([prompt], dtype=torch.long, device=device)

        for _ in range(max_new):
            # Clip to max_seq_len context
            inp  = seq[:, -TR_MAX_SEQ_LEN:]
            logits = self(inp)                        # (1, T, V)
            next_logits = logits[0, -1] / temperature  # (V,)

            # Top-k filtering
            if top_k > 0:
                vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < vals[-1]] = -float("inf")

            probs   = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)     # (1,)
            seq = torch.cat([seq, next_id.unsqueeze(0)], dim=1)

            if next_id.item() == EOS_TOKEN:
                break

        return seq[0].tolist()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model  = MusicTransformer()
    B, T   = 4, 128
    x      = torch.randint(4, TR_VOCAB_SIZE, (B, T))
    logits = model(x)
    loss   = MusicTransformer.loss(logits[:, :-1], x[:, 1:])
    ppl    = MusicTransformer.perplexity(loss.item())
    print(f"logits: {logits.shape}")
    print(f"loss:   {loss.item():.4f}")
    print(f"perplexity: {ppl:.2f}")
    tokens = model.generate(max_new=64)
    print(f"generated: {len(tokens)} tokens")
    print(f"params:  {sum(p.numel() for p in model.parameters()):,}")