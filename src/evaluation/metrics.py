import os
import sys
import glob
import numpy as np

# Ensure Python can find the adjacent modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pitch_histogram import get_pitch_histogram, calculate_histogram_similarity
from rhythm_score import calculate_rhythm_diversity, calculate_repetition_ratio

def evaluate_folder(generated_dir, target_histogram=None):
    """
    Runs quantitative metrics on a directory of MIDI files.
    """
    midi_files = glob.glob(os.path.join(generated_dir, "*.mid"))
    if not midi_files:
        print(f"No MIDI files found in {generated_dir}")
        return

    print(f"Evaluating {len(midi_files)} files in {os.path.basename(generated_dir)}...\n")

    similarities = []
    diversities = []
    repetitions = []

    for file in midi_files:
        # 1. Pitch Histogram
        gen_hist = get_pitch_histogram(file)
        if target_histogram is not None:
            sim = calculate_histogram_similarity(gen_hist, target_histogram)
            similarities.append(sim)

        # 2. Rhythm Diversity
        div = calculate_rhythm_diversity(file)
        diversities.append(div)

        # 3. Repetition Ratio
        rep = calculate_repetition_ratio(file)
        repetitions.append(rep)

    print("-" * 40)
    print("QUANTITATIVE EVALUATION RESULTS")
    print("-" * 40)
    
    if target_histogram is not None:
        print(f"Avg Pitch Histogram Distance (H): {np.mean(similarities):.4f}")
    
    print(f"Avg Rhythm Diversity Score:       {np.mean(diversities):.4f}")
    print(f"Avg Repetition Ratio:             {np.mean(repetitions):.4f}")
    print("-" * 40)

if __name__ == "__main__":
    # Point this to whichever folder you want to test (Task 1, 2, 3, or 4)
    target_dir = '/content/music-generation-unsupervised/outputs/generated_midis'
    
    # Optional: If you want to calculate H(p,q), define a 'perfect' target distribution.
    # E.g., A uniform distribution across all 12 notes:
    dummy_target_hist = np.ones(12) / 12.0 
    
    evaluate_folder(target_dir, target_histogram=dummy_target_hist)