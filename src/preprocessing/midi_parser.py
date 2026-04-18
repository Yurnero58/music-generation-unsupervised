import os
import glob
import numpy as np
import pretty_midi

def parse_maestro_to_npy(raw_path, output_file, fs=4, window_size=64):
    all_sequences = []
    midi_files = glob.glob(os.path.join(raw_path, '**/*.mid*'), recursive=True)
    
    print(f"Found {len(midi_files)} files. Starting preprocessing with silence filtering...")

    filtered_out_count = 0

    for file_path in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            piano_roll = midi_data.get_piano_roll(fs=fs)
            
            # Clip to 88 keys and normalize
            piano_roll = piano_roll[21:109, :] / 127.0
            piano_roll = piano_roll.T # Shape: (Time, 88)
            
            for i in range(0, piano_roll.shape[0] - window_size, window_size):
                segment = piano_roll[i : i + window_size, :]
                
                if segment.shape[0] == window_size:
                    # SILENCE FILTER: Keep only segments with at least 15 active note events
                    if np.sum(segment > 0) >= 15:
                        all_sequences.append(segment)
                    else:
                        filtered_out_count += 1
                        
        except Exception as e:
            continue

    data_array = np.array(all_sequences)
    np.save(output_file, data_array)
    print(f"Success! Saved {data_array.shape[0]} active music sequences.")
    print(f"Filtered out {filtered_out_count} silent/empty sequences.")

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    parse_maestro_to_npy('data/raw_midi/', 'data/processed/classical_piano.npy')