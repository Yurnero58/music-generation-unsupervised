# =============================================================================
# src/evaluation/pitch_histogram.py
# Standalone pitch histogram analysis — thin wrapper around metrics.py
# =============================================================================

import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_PLOTS, N_PITCHES
from evaluation.metrics import pitch_histogram, pitch_histogram_similarity


def compare_all_models(model_rolls: dict, save_path: str = None):
    """
    model_rolls: {model_name: list_of_rolls}
    Plots chroma histograms + pairwise similarity heatmap.
    """
    labels = list(model_rolls.keys())
    hists  = {}
    for name, rolls in model_rolls.items():
        combined = np.concatenate(
            [r for r in rolls if r.shape[0] > 0], axis=0) \
            if rolls else np.zeros((1, N_PITCHES))
        hists[name] = pitch_histogram(combined)

    n = len(labels)
    sim_matrix = np.zeros((n, n))
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            sim_matrix[i, j] = np.sum(np.abs(hists[la] - hists[lb]))

    chroma = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram bars
    for name, hist in hists.items():
        axes[0].plot(chroma, hist, marker="o", label=name)
    axes[0].set_title("Pitch Class Histograms")
    axes[0].set_xlabel("Pitch Class")
    axes[0].set_ylabel("Normalised Frequency")
    axes[0].legend(fontsize=8)

    # Similarity heatmap
    im = axes[1].imshow(sim_matrix, cmap="RdYlGn_r", vmin=0)
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_yticks(range(n)); axes[1].set_yticklabels(labels)
    axes[1].set_title("Pitch Histogram Distance H(p,q)")
    plt.colorbar(im, ax=axes[1])
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{sim_matrix[i,j]:.2f}",
                         ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out = save_path or os.path.join(OUTPUT_PLOTS, "pitch_histogram_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[pitch_histogram] Saved → {out}")
    return sim_matrix