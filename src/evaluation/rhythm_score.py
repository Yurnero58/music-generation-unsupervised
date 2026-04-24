import pretty_midi
import numpy as np

def calculate_rhythm_diversity(midi_path, decimals=2):
    """
    Calculates D_rhythm = (#unique_durations) / (#total_notes)
    Durations are rounded to avoid microscopic floating-point noise.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return 0.0

    durations = []
    
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                duration = round(note.end - note.start, decimals)
                durations.append(duration)
                
    total_notes = len(durations)
    if total_notes == 0:
        return 0.0
        
    unique_durations = len(set(durations))
    
    return unique_durations / total_notes

def calculate_repetition_ratio(midi_path, n_gram_length=4):
    """
    Calculates R = (#repeated_patterns) / (#total_patterns)
    Uses sequences of 'n' consecutive pitches as a pattern.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return 0.0

    pitches = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitches.append(note.pitch)
                
    if len(pitches) < n_gram_length:
        return 0.0
        
    # Extract overlapping n-grams
    patterns = [tuple(pitches[i:i+n_gram_length]) for i in range(len(pitches) - n_gram_length + 1)]
    total_patterns = len(patterns)
    
    if total_patterns == 0:
        return 0.0
        
    unique_patterns = set(patterns)
    repeated_patterns = total_patterns - len(unique_patterns)
    
    return repeated_patterns / total_patterns