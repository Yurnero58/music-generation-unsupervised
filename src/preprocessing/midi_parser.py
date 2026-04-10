# =============================================================================
# src/preprocessing/midi_parser.py
# Download + parse Groove MIDI dataset into raw note arrays
# =============================================================================

import os, zipfile, urllib.request, csv, pickle
from pathlib import Path
import numpy as np
import pretty_midi

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (DATA_RAW, DATA_PROCESSED, GROOVE_URL,
                    PITCH_RANGE, STEPS_PER_BAR, GENRES)


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_groove(dest: str = DATA_RAW) -> str:
    """Download Groove MIDI zip if not already present."""
    zip_path = os.path.join(dest, "groove.zip")
    extract_path = os.path.join(dest, "groove")
    if os.path.isdir(extract_path):
        print(f"[parser] Groove already extracted at {extract_path}")
        return extract_path

    print(f"[parser] Downloading Groove MIDI (~1 GB) …")
    urllib.request.urlretrieve(GROOVE_URL, zip_path,
        reporthook=lambda b, bs, t: print(
            f"\r  {min(b*bs, t)/1e6:.1f}/{t/1e6:.1f} MB", end="", flush=True))
    print()
    print("[parser] Extracting …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    os.remove(zip_path)
    return extract_path


# ---------------------------------------------------------------------------
# 2. Parse a single MIDI file → piano-roll matrix
# ---------------------------------------------------------------------------
def midi_to_pianoroll(midi_path: str,
                      steps_per_bar: int = STEPS_PER_BAR,
                      pitch_lo: int = PITCH_RANGE[0],
                      pitch_hi: int = PITCH_RANGE[1]) -> np.ndarray:
    """
    Convert one MIDI file to a binary piano-roll.
    Returns shape (T, n_pitches) where T = number of 16th-note steps.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[parser] Skip {midi_path}: {e}")
        return None

    # Resolution: 1 step = 1 sixteenth note at the file's tempo
    # pretty_midi.get_piano_roll uses fs (frames per second)
    # We compute fs from the median tempo.
    tempos, _ = pm.get_tempo_change_times()
    bpm = float(np.median(pm.get_tempo_change_times()[0])) if len(tempos) > 0 else 120.0
    # 1 bar = 4 beats; steps_per_bar steps → steps_per_bar/(4 beats) steps/beat
    steps_per_beat = steps_per_bar / 4
    fs = steps_per_beat * (bpm / 60.0)          # steps per second

    roll = pm.get_piano_roll(fs=fs)              # (128, T)
    roll = (roll > 0).astype(np.float32)         # binarise
    roll = roll[pitch_lo:pitch_hi, :]            # crop pitch range → (n_p, T)
    return roll.T                                # (T, n_p)


# ---------------------------------------------------------------------------
# 3. Walk the dataset and collect all rolls
# ---------------------------------------------------------------------------
def load_groove_dataset(groove_root: str) -> dict:
    """
    Walk Groove folder structure and return dict:
        { genre_tag: [pianoroll_array, ...] }
    Uses the metadata CSV if present, otherwise uses folder name heuristics.
    """
    rolls_by_genre: dict = {g: [] for g in GENRES}
    rolls_by_genre["other"] = []

    # Groove ships a metadata CSV
    meta_csv = os.path.join(groove_root, "info.csv")
    file_genre_map: dict = {}
    if os.path.isfile(meta_csv):
        with open(meta_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("midi_filename", "")
                style = row.get("style", "other").lower().split("/")[0]
                file_genre_map[fname] = style

    midi_files = sorted(Path(groove_root).rglob("*.mid"))
    print(f"[parser] Found {len(midi_files)} MIDI files")

    for mid_path in midi_files:
        rel = str(mid_path.relative_to(groove_root))
        genre = file_genre_map.get(rel, None)
        if genre is None:
            # Fallback: match folder name
            parts = rel.lower().split(os.sep)
            genre = next((g for g in GENRES if any(g in p for p in parts)), "other")

        roll = midi_to_pianoroll(str(mid_path))
        if roll is None or roll.shape[0] < 32:
            continue

        bucket = genre if genre in rolls_by_genre else "other"
        rolls_by_genre[bucket].append(roll)

    for g, lst in rolls_by_genre.items():
        print(f"  {g:12s}: {len(lst)} files")
    return rolls_by_genre


# ---------------------------------------------------------------------------
# 4. Save / load
# ---------------------------------------------------------------------------
def save_parsed(rolls_by_genre: dict, dest: str = DATA_PROCESSED):
    path = os.path.join(dest, "rolls_by_genre.pkl")
    with open(path, "wb") as f:
        pickle.dump(rolls_by_genre, f)
    print(f"[parser] Saved parsed data → {path}")


def load_parsed(dest: str = DATA_PROCESSED) -> dict:
    path = os.path.join(dest, "rolls_by_genre.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    groove_root = download_groove()
    rolls = load_groove_dataset(groove_root)
    save_parsed(rolls)
    print("[parser] Done.")