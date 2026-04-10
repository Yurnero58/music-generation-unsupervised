# =============================================================================
# src/training/train_transformer.py
# Task 3: Train causal Transformer for autoregressive music generation
# Algorithm 3 from the project spec
# =============================================================================

import os, sys, json, time, math
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (DEVICE, NUM_EPOCHS_TR, LEARNING_RATE, BATCH_SIZE,
                    OUTPUT_PLOTS, TR_VOCAB_SIZE, TR_D_MODEL, TR_NHEAD,
                    TR_NUM_LAYERS, TR_DIM_FF, TR_DROPOUT, TR_MAX_SEQ_LEN)
from models.transformer import MusicTransformer
from preprocessing.piano_roll import load_splits
from preprocessing.tokenizer import get_transformer_dataloaders

ROOT     = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Sequence length for the Transformer (shorter than full to keep memory ok on Colab)
TR_SEQ = 128


# ---------------------------------------------------------------------------
# Warmup + cosine LR scheduler
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimiser, warmup_steps, total_steps, base_lr, last_epoch=-1):
        self.warmup  = warmup_steps
        self.total   = total_steps
        self.base_lr = base_lr
        super().__init__(optimiser, last_epoch)

    def get_lr(self):
        s = self.last_epoch + 1
        if s < self.warmup:
            scale = s / max(1, self.warmup)
        else:
            progress = (s - self.warmup) / max(1, self.total - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.base_lr * scale for _ in self.base_lrs]


# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, scheduler, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(x)
        loss   = MusicTransformer.loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += MusicTransformer.loss(logits, y).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_tr] Device: {device}")

    splits   = load_splits()
    train_dl, val_dl, _ = get_transformer_dataloaders(
        splits, max_len=TR_SEQ + 1, batch_size=BATCH_SIZE)

    model = MusicTransformer(
        vocab_size=TR_VOCAB_SIZE, d_model=TR_D_MODEL,
        nhead=TR_NHEAD, num_layers=TR_NUM_LAYERS,
        dim_feedforward=TR_DIM_FF, dropout=TR_DROPOUT,
        max_seq_len=TR_SEQ
    ).to(device)
    print(f"[train_tr] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    total_steps  = NUM_EPOCHS_TR * len(train_dl)
    warmup_steps = total_steps // 10
    optimiser    = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                               weight_decay=1e-2)
    scheduler    = WarmupCosineScheduler(optimiser, warmup_steps,
                                         total_steps, LEARNING_RATE)

    history      = {"train": [], "val": [], "train_ppl": [], "val_ppl": []}
    best_val     = float("inf")

    for epoch in range(1, NUM_EPOCHS_TR + 1):
        t0       = time.time()
        tr_loss  = train_one_epoch(model, train_dl, optimiser, scheduler, device)
        val_loss = evaluate(model, val_dl, device)

        tr_ppl  = MusicTransformer.perplexity(tr_loss)
        val_ppl = MusicTransformer.perplexity(val_loss)

        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        history["train_ppl"].append(tr_ppl)
        history["val_ppl"].append(val_ppl)

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS_TR}  "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f}  "
              f"val_ppl={val_ppl:.2f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "transformer_best.pt")

    torch.save(model.state_dict(), CKPT_DIR / "transformer_final.pt")
    with open(CKPT_DIR / "transformer_history.json", "w") as f:
        json.dump(history, f)

    # ---- Plots ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train"], label="Train Loss")
    ax1.plot(history["val"],   label="Val Loss")
    ax1.set_title("Task 3 — Transformer: Cross-Entropy Loss")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(history["train_ppl"], label="Train PPL")
    ax2.plot(history["val_ppl"],   label="Val PPL")
    ax2.set_title("Task 3 — Transformer: Perplexity")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, "task3_transformer_loss.png"), dpi=150)
    plt.close()
    print(f"[train_tr] Done. Best val PPL: {MusicTransformer.perplexity(best_val):.2f}")


if __name__ == "__main__":
    main()