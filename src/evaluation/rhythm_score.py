# =============================================================================
# src/evaluation/rhythm_score.py
# Rhythm analysis — diversity, syncopation, density plots
# =============================================================================

import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_PLOTS, N_PITCHES, STEPS_PER_BAR
from evaluation.metrics import rhythm_diversity, repetition_ratio


def note_density_profile(roll: np.ndarray) -> np.ndarray:
    """Notes-per-step profile: (T,) float."""
    return roll.sum(axis=1)


def syncopation_score(roll: np.ndarray,
                      steps_per_bar: int = STEPS_PER_BAR) -> float:
    """
    Proxy syncopation: fraction of note onsets on off-beats.
    Beat positions: 0, 4, 8, 12 (for 16 steps/bar).
    """
    T, n_p = roll.shape
    beat_mask = np.zeros(T, dtype=bool)
    beat_steps = [0, 4, 8, 12]   # quarter-note positions in 16-step bar
    for t in range(T):
        if (t % steps_per_bar) in beat_steps:
            beat_mask[t] = True

    total_onsets = 0
    off_beat_onsets = 0
    for t in range(1, T):
        onsets = (roll[t] > 0) & (roll[t-1] == 0)   # new note starts
        n_on = onsets.sum()
        total_onsets += n_on
        if not beat_mask[t]:
            off_beat_onsets += n_on
    return float(off_beat_onsets / max(1, total_onsets))


def plot_rhythm_comparison(model_rolls: dict, save_path: str = None):
    """
    model_rolls: {model_name: list_of_rolls}
    Bar chart comparing rhythm diversity + syncopation across models.
    """
    labels, r_divs, syncs, rep_rats = [], [], [], []
    for name, rolls in model_rolls.items():
        labels.append(name)
        rd_vals, sy_vals, rr_vals = [], [], []
        for roll in rolls:
            if roll.shape[0] == 0:
                continue
            rd_vals.append(rhythm_diversity(roll))
            sy_vals.append(syncopation_score(roll))
            rr_vals.append(repetition_ratio(roll))
        r_divs.append(np.mean(rd_vals) if rd_vals else 0)
        syncs.append(np.mean(sy_vals)  if sy_vals  else 0)
        rep_rats.append(np.mean(rr_vals) if rr_vals else 0)

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, r_divs,   width=w, label="Rhythm Diversity")
    ax.bar(x,     syncs,    width=w, label="Syncopation Score")
    ax.bar(x + w, rep_rats, width=w, label="Repetition Ratio (lower=better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Rhythm Metrics Comparison Across Models")
    ax.legend()
    plt.tight_layout()

    out = save_path or os.path.join(OUTPUT_PLOTS, "rhythm_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[rhythm_score] Saved → {out}")