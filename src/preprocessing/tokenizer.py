# =============================================================================
# src/preprocessing/tokenizer.py
# Event-based MIDI tokenizer for Task 3 (Transformer)
# Converts piano-roll windows → integer token sequences
# =============================================================================

import numpy as np
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (N_PITCHES, VELOCITY_BINS, PITCH_RANGE,
                    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN,
                    TR_VOCAB_SIZE, SEQ_LEN)

# Token layout:
#   0           = PAD
#   1           = BOS
#   2           = EOS
#   3           = MASK
#   4 … 4+N_P-1            = NOTE_ON  (pitch offset from PITCH_RANGE[0])
#   4+N_P … 4+N_P+V_BINS-1 = VELOCITY bucket tokens  (unused in simple mode)

NOTE_OFFSET = 4
VEL_OFFSET  = NOTE_OFFSET + N_PITCHES


def pianoroll_to_tokens(roll: np.ndarray) -> list[int]:
    """
    Convert a piano-roll window (T, n_pitches) → list of token ints.
    Strategy: for each time step, emit NOTE_ON tokens for active pitches,
    or a TIME_STEP token if silent.
    """
    tokens = [BOS_TOKEN]
    T, n_p = roll.shape
    for t in range(T):
        active = np.where(roll[t] > 0)[0]
        if len(active) == 0:
            continue
        for p in active:
            tok = NOTE_OFFSET + int(p)
            tokens.append(tok)
    tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_pianoroll(tokens: list[int], seq_len: int = SEQ_LEN) -> np.ndarray:
    """Reverse: token list → binary piano-roll (seq_len, n_pitches)."""
    roll = np.zeros((seq_len, N_PITCHES), dtype=np.float32)
    step = 0
    for tok in tokens:
        if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN):
            if tok == EOS_TOKEN:
                break
            continue
        if NOTE_OFFSET <= tok < NOTE_OFFSET + N_PITCHES:
            pitch = tok - NOTE_OFFSET
            if step < seq_len:
                roll[step, pitch] = 1.0
                step += 1
    return roll


class EventTokenDataset(torch.utils.data.Dataset):
    """Dataset for Transformer: returns padded token sequences."""

    def __init__(self, X: np.ndarray, max_len: int = 256):
        self.max_len = max_len
        self.sequences = []
        for i in range(len(X)):
            toks = pianoroll_to_tokens(X[i])[:max_len]
            # pad to max_len
            toks += [PAD_TOKEN] * (max_len - len(toks))
            self.sequences.append(toks)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        x   = seq[:-1]
        y   = seq[1:]
        return x, y


def get_transformer_dataloaders(splits: dict, max_len: int = 256,
                                batch_size: int = 64, num_workers: int = 2):
    from torch.utils.data import DataLoader
    train_ds = EventTokenDataset(splits["X_train"], max_len)
    val_ds   = EventTokenDataset(splits["X_val"],   max_len)
    test_ds  = EventTokenDataset(splits["X_test"],  max_len)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, drop_last=True)
    return (DataLoader(train_ds, shuffle=True,  **kwargs),
            DataLoader(val_ds,   shuffle=False, **kwargs),
            DataLoader(test_ds,  shuffle=False, **kwargs))


if __name__ == "__main__":
    # Smoke test
    dummy = np.random.randint(0, 2, (64, N_PITCHES)).astype(np.float32)
    toks  = pianoroll_to_tokens(dummy)
    back  = tokens_to_pianoroll(toks)
    print("tokens:", len(toks), "| roll shape:", back.shape)
    print("Vocab size check:", TR_VOCAB_SIZE)