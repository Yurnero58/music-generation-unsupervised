# =============================================================================
# src/preprocessing/piano_roll.py
# Piano-roll windowing, normalisation, and PyTorch Dataset wrappers
# =============================================================================

import os, pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (SEQ_LEN, N_PITCHES, BATCH_SIZE,
                    TEST_SPLIT, VAL_SPLIT, DATA_SPLIT, DATA_PROCESSED)


# ---------------------------------------------------------------------------
# 1. Windowing
# ---------------------------------------------------------------------------
def roll_to_windows(roll: np.ndarray, seq_len: int = SEQ_LEN,
                    stride: int = None) -> np.ndarray:
    """Slice a piano-roll (T, n_p) into overlapping windows."""
    if stride is None:
        stride = seq_len // 2
    T, n_p = roll.shape
    if T < seq_len:
        return np.zeros((0, seq_len, n_p), dtype=np.float32)
    starts = range(0, T - seq_len + 1, stride)
    windows = np.stack([roll[s:s + seq_len] for s in starts])   # (N, L, n_p)
    return windows.astype(np.float32)


def build_windows(rolls_by_genre: dict,
                  seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert all genre rolls to windows.
    Returns:
        X : (N, seq_len, n_pitches)  float32
        y : (N,)                     int   genre label
    """
    all_X, all_y = [], []
    genre_list = sorted(rolls_by_genre.keys())
    genre2id   = {g: i for i, g in enumerate(genre_list)}

    for genre, rolls in rolls_by_genre.items():
        gid = genre2id[genre]
        for roll in rolls:
            wins = roll_to_windows(roll, seq_len)
            if wins.shape[0] == 0:
                continue
            all_X.append(wins)
            all_y.append(np.full(wins.shape[0], gid, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"[piano_roll] Total windows: {X.shape[0]}  shape: {X.shape}")
    return X, y, genre2id


# ---------------------------------------------------------------------------
# 2. Train / Val / Test split
# ---------------------------------------------------------------------------
def split_and_save(X: np.ndarray, y: np.ndarray,
                   val_frac: float = VAL_SPLIT,
                   test_frac: float = TEST_SPLIT,
                   dest: str = DATA_SPLIT):
    N = len(X)
    idx = np.random.permutation(N)
    n_test = int(N * test_frac)
    n_val  = int(N * val_frac)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    splits = {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val":   X[val_idx],   "y_val":   y[val_idx],
        "X_test":  X[test_idx],  "y_test":  y[test_idx],
    }
    for name, arr in splits.items():
        np.save(os.path.join(dest, f"{name}.npy"), arr)
    print(f"[piano_roll] Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    return splits


def load_splits(dest: str = DATA_SPLIT) -> dict:
    keys = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    return {k: np.load(os.path.join(dest, f"{k}.npy")) for k in keys}


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------------------------------------
class PianoRollDataset(Dataset):
    """Piano-roll window dataset. Returns (x, label) tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)      # (N, L, n_p)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None \
                 else torch.zeros(len(X), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(splits: dict,
                    batch_size: int = BATCH_SIZE,
                    num_workers: int = 2) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = PianoRollDataset(splits["X_train"], splits["y_train"])
    val_ds   = PianoRollDataset(splits["X_val"],   splits["y_val"])
    test_ds  = PianoRollDataset(splits["X_test"],  splits["y_test"])

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, drop_last=True)
    train_dl = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kwargs)
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from midi_parser import load_parsed
    rolls = load_parsed()
    X, y, g2id = build_windows(rolls)
    split_and_save(X, y)
    print("[piano_roll] Done.", g2id)