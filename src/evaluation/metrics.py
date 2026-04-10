# =============================================================================
# src/evaluation/metrics.py
# All quantitative metrics from the project spec + comparison table generation
# =============================================================================

import os, sys, json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_PLOTS, N_PITCHES


# ---------------------------------------------------------------------------
# 1. Pitch Histogram Similarity
#    H(p, q) = sum_{i=1}^{12} |p_i - q_i|
# ---------------------------------------------------------------------------
def pitch_histogram(roll: np.ndarray) -> np.ndarray:
    """
    Compute 12-bin chroma histogram from piano-roll (T, n_pitches).
    Each pitch bin is reduced mod 12 for chroma.
    """
    hist = np.zeros(12, dtype=np.float32)
    T, n_p = roll.shape
    for p in range(n_p):
        chroma = p % 12
        hist[chroma] += roll[:, p].sum()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pitch_histogram_similarity(roll_a: np.ndarray,
                                roll_b: np.ndarray) -> float:
    """H(p, q) = sum |p_i - q_i|  (lower = more similar)"""
    p = pitch_histogram(roll_a)
    q = pitch_histogram(roll_b)
    return float(np.sum(np.abs(p - q)))


def pitch_histogram_entropy(roll: np.ndarray) -> float:
    """Entropy of pitch histogram (higher = more diverse)."""
    hist = pitch_histogram(roll) + 1e-9
    return float(-np.sum(hist * np.log(hist)))


# ---------------------------------------------------------------------------
# 2. Rhythm Diversity Score
#    D_rhythm = #unique_durations / #total_notes
# ---------------------------------------------------------------------------
def rhythm_diversity(roll: np.ndarray) -> float:
    """
    Estimate rhythm diversity from a piano-roll.
    'Duration' = consecutive active steps for the same pitch.
    """
    T, n_p = roll.shape
    durations = []
    for p in range(n_p):
        col = roll[:, p]
        in_note = False
        dur = 0
        for t in range(T):
            if col[t] > 0:
                in_note = True
                dur += 1
            else:
                if in_note:
                    durations.append(dur)
                    dur = 0
                    in_note = False
        if in_note and dur > 0:
            durations.append(dur)

    if len(durations) == 0:
        return 0.0
    return len(set(durations)) / len(durations)


# ---------------------------------------------------------------------------
# 3. Repetition Ratio
#    R = #repeated_patterns / #total_patterns
# ---------------------------------------------------------------------------
def repetition_ratio(roll: np.ndarray, pattern_len: int = 4) -> float:
    """
    Slide a window of `pattern_len` time steps over the roll.
    R = #repeated patterns / #total patterns
    """
    T, n_p = roll.shape
    if T < pattern_len * 2:
        return 0.0

    patterns = []
    for t in range(T - pattern_len + 1):
        pat = tuple(roll[t:t + pattern_len].flatten().astype(int))
        patterns.append(pat)

    counts = Counter(patterns)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / max(1, len(counts))


# ---------------------------------------------------------------------------
# 4. Human Listening Score  (collects from JSON survey file)
# ---------------------------------------------------------------------------
def load_human_scores(survey_path: str) -> dict:
    """Load human scores from a JSON dict {sample_id: score 1-5}."""
    with open(survey_path) as f:
        data = json.load(f)
    scores = [float(v) for v in data.values()]
    return {
        "mean":   float(np.mean(scores)),
        "std":    float(np.std(scores)),
        "n":      len(scores),
        "raw":    data,
    }


# ---------------------------------------------------------------------------
# 5. Batch evaluation of a list of generated piano-rolls
# ---------------------------------------------------------------------------
def evaluate_batch(rolls: list[np.ndarray],
                   reference_rolls: list[np.ndarray] = None,
                   label: str = "model") -> dict:
    """
    Compute all metrics for a list of piano-roll arrays.
    Returns a summary dict.
    """
    pitch_entropies, r_divs, r_ratios, ph_sims = [], [], [], []

    ref_hist = None
    if reference_rolls:
        # Pool all reference rolls into one big histogram
        all_ref = np.concatenate([r for r in reference_rolls if r.shape[0] > 0], axis=0)
        ref_hist = pitch_histogram(all_ref)

    for roll in rolls:
        if roll.shape[0] == 0:
            continue
        pitch_entropies.append(pitch_histogram_entropy(roll))
        r_divs.append(rhythm_diversity(roll))
        r_ratios.append(repetition_ratio(roll))
        if ref_hist is not None:
            gen_hist = pitch_histogram(roll)
            ph_sims.append(float(np.sum(np.abs(gen_hist - ref_hist))))

    summary = {
        "label":              label,
        "n_samples":          len(rolls),
        "pitch_entropy_mean": float(np.mean(pitch_entropies)) if pitch_entropies else 0,
        "rhythm_diversity":   float(np.mean(r_divs))          if r_divs          else 0,
        "repetition_ratio":   float(np.mean(r_ratios))        if r_ratios        else 0,
    }
    if ph_sims:
        summary["pitch_histogram_sim"] = float(np.mean(ph_sims))
    return summary


# ---------------------------------------------------------------------------
# 6. Comparison table (baselines + all tasks)
# ---------------------------------------------------------------------------
def print_comparison_table(results: list[dict]):
    """Print a comparison table like Table 3 in the spec."""
    header = f"{'Model':<30} {'PitchEntropy':>13} {'RhythmDiv':>10} {'RepRatio':>9} {'HumanScore':>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        human = f"{r['human_score']:.2f}" if "human_score" in r else "  N/A"
        print(f"{r['label']:<30} "
              f"{r.get('pitch_entropy_mean', 0):>13.4f} "
              f"{r.get('rhythm_diversity',   0):>10.4f} "
              f"{r.get('repetition_ratio',   0):>9.4f} "
              f"{human:>11}")
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# 7. Plot pitch histograms side by side
# ---------------------------------------------------------------------------
def plot_pitch_histograms(rolls_dict: dict, save_path: str = None):
    """
    rolls_dict: {label: [roll, ...]}
    """
    n = len(rolls_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    chroma_labels = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    for ax, (label, rolls) in zip(axes, rolls_dict.items()):
        all_roll = np.concatenate([r for r in rolls if r.shape[0] > 0], axis=0)
        hist = pitch_histogram(all_roll)
        ax.bar(chroma_labels, hist, color="steelblue")
        ax.set_title(label)
        ax.set_xlabel("Pitch Class")
    axes[0].set_ylabel("Normalised Frequency")
    plt.suptitle("Pitch Histogram Comparison")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    roll = (np.random.rand(64, N_PITCHES) > 0.85).astype(np.float32)
    print("Pitch entropy:   ", pitch_histogram_entropy(roll))
    print("Rhythm diversity:", rhythm_diversity(roll))
    print("Repetition ratio:", repetition_ratio(roll))
    res = evaluate_batch([roll], label="test")
    print_comparison_table([res])