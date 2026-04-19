import pretty_midi
import numpy as np
import glob
import os

def calculate_metrics(midi_folder):
    # Ensure the path is absolute for Windows
    abs_path = os.path.abspath(midi_folder)
    files = glob.glob(os.path.join(abs_path, "*.mid"))
    
    if not files:
        print(f"Warning: No MIDI files found in {abs_path}")
        return None, None

    pitch_varieties = []
    note_densities = []

    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(f)
            # Pitch Variety: Number of unique MIDI notes
            pitches = [note.pitch for inst in pm.instruments for note in inst.notes]
            if pitches:
                pitch_varieties.append(len(set(pitches)))
                
                # Note Density: Notes per second
                duration = pm.get_end_time()
                if duration > 0:
                    note_densities.append(len(pitches) / duration)
        except Exception as e:
            continue
            
    if not pitch_varieties:
        return None, None
        
    return np.mean(pitch_varieties), np.mean(note_densities)

if __name__ == "__main__":
    # Update these to the exact folders on your I: drive
    t1_path = "outputs/generated_midis"
    t2_path = "outputs/task2"

    t1_pitch, t1_dense = calculate_metrics(t1_path)
    t2_pitch, t2_dense = calculate_metrics(t2_path)

    print("\n--- Evaluation Results ---")
    if t1_pitch is not None:
        print(f"Task 1 (Classical)   | Pitch Variety: {t1_pitch:.2f} | Note Density: {t1_dense:.2f}")
    else:
        print("Task 1: No data found.")

    if t2_pitch is not None:
        print(f"Task 2 (Multi-Genre) | Pitch Variety: {t2_pitch:.2f} | Note Density: {t2_dense:.2f}")
    else:
        print("Task 2: No data found.")