import os
import glob
import numpy as np
import pretty_midi

def preprocess_lmd_multi_genre(raw_base_path, output_file, window_size=64, fs=4):
    processed_data = []
    # Targeted path for Task 2 data
    clean_midi_path = os.path.join(raw_base_path, 'clean_midi')
    
    if not os.path.exists(clean_midi_path):
        print(f"Error: {clean_midi_path} not found!")
        return

    # Subdirectories in clean_midi are organized by artist/genre
    subdirs = [d for d in os.listdir(clean_midi_path) if os.path.isdir(os.path.join(clean_midi_path, d))]
    print(f"Found {len(subdirs)} artist folders in clean_midi. Processing...")

    # Balancing the dataset: Take a limited number of segments from many artists
    for subdir in subdirs[:400]: 
        subdir_path = os.path.join(clean_midi_path, subdir)
        files = glob.glob(os.path.join(subdir_path, "**/*.mid"), recursive=True)
        
        for file in files[:2]: # 2 files per artist to maximize diversity
            try:
                midi = pretty_midi.PrettyMIDI(file)
                # Filter to 88 piano keys and normalize
                roll = midi.get_piano_roll(fs=fs)[21:109, :] / 127.0
                roll = roll.T 
                
                for i in range(0, roll.shape[0] - window_size, window_size):
                    chunk = roll[i:i+window_size, :]
                    # Ensure the segment has enough musical density
                    if chunk.shape[0] == window_size and np.sum(chunk > 0) > 15:
                        processed_data.append(chunk)
            except:
                continue
                
    np.save(output_file, np.array(processed_data))
    print(f"Task 2 Dataset Ready: {len(processed_data)} segments saved to {output_file}")

if __name__ == "__main__":
    preprocess_lmd_multi_genre('data/raw_midi', 'data/processed/multi_genre_lmd.npy')