import pretty_midi
import numpy as np

def get_pitch_histogram(midi_path):
    """
    Extracts a normalized 12-bin pitch class histogram from a MIDI file.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return np.zeros(12)

    histogram = np.zeros(12)
    total_notes = 0
    
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_class = note.pitch % 12
                histogram[pitch_class] += 1
                total_notes += 1
                
    if total_notes == 0:
        return np.zeros(12)
        
    # Normalize to create a probability distribution (sum = 1.0)
    return histogram / total_notes

def calculate_histogram_similarity(hist_p, hist_q):
    """
    Calculates H(p, q) = sum(|p_i - q_i|)
    Lower is better (0.0 means identical distributions).
    """
    return np.sum(np.abs(hist_p - hist_q))