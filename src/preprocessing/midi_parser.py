# =============================================================================
# src/preprocessing/midi_parser.py
# Lakh MIDI Dataset parser (Groove removed completely)
# Outputs: { "lakh": [piano_rolls] }
# =============================================================================

import os
import pickle
from pathlib import Path
import numpy as np
import pretty_midi

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, DATA_PROCESSED, PITCH_RANGE


# ---------------------------------------------------------------------------
# 1. Dataset path (NO download — Lakh is manual)
# ---------------------------------------------------------------------------
def download_lakh(dest: str = DATA_RAW) -> str:
    """
    Lakh MIDI is too large for auto-download.
    Put dataset manually in:
        data/raw_midi/lakh/
    """
    path = os.path.join(dest, "lakh")
    os.makedirs(path, exist_ok=True)

    print(f"[parser] Using Lakh dataset at: {path}")
    return path


# ---------------------------------------------------------------------------
# 2. MIDI → Piano Roll
# ---------------------------------------------------------------------------
def midi_to_pianoroll(midi_path: str,
                      pitch_lo: int = PITCH_RANGE[0],
                      pitch_hi: int = PITCH_RANGE[1]) -> np.ndarray:
    """
    Convert MIDI file → binary piano roll (T, n_pitches)
    """

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[parser] Skip {midi_path}: {e}")
        return None

    # Fixed resolution (stable for deep learning)
    roll = pm.get_piano_roll(fs=16)

    # Binary representation
    roll = (roll > 0).astype(np.float32)

    # Crop pitch range
    roll = roll[pitch_lo:pitch_hi, :]

    return roll.T  # (T, n_pitches)


# ---------------------------------------------------------------------------
# 3. Load full Lakh dataset
# ---------------------------------------------------------------------------
def load_lakh_dataset(lakh_root: str) -> dict:
    """
    Traverse dataset and return:
        {"lakh": [piano_rolls]}
    """

    rolls = []

    midi_files = list(Path(lakh_root).rglob("*.mid"))
    print(f"[parser] Found {len(midi_files)} MIDI files")

    for path in midi_files:
        roll = midi_to_pianoroll(str(path))

        if roll is None:
            continue

        # filter too-short sequences
        if roll.shape[0] < 32:
            continue

        rolls.append(roll)

    print(f"[parser] Valid sequences: {len(rolls)}")

    return {"lakh": rolls}


# ---------------------------------------------------------------------------
# 4. Save / Load processed dataset
# ---------------------------------------------------------------------------
def save_parsed(rolls: dict, dest: str = DATA_PROCESSED):
    os.makedirs(dest, exist_ok=True)

    path = os.path.join(dest, "lakh_rolls.pkl")

    with open(path, "wb") as f:
        pickle.dump(rolls, f)

    print(f"[parser] Saved → {path}")


def load_parsed(dest: str = DATA_PROCESSED) -> dict:
    path = os.path.join(dest, "lakh_rolls.pkl")

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = download_lakh()
    rolls = load_lakh_dataset(root)
    save_parsed(rolls)
    print("[parser] Done.")