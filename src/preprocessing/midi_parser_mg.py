import os
import glob
import numpy as np
import pretty_midi

def preprocess_lmd_multi_genre(root_dir, output_file, window_size=64, fs=4):
    processed_data = []
    
    # Using the absolute path from your screenshot
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found!")
        return

    # subdirs in lakh_clean are organized by artist/genre
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Found {len(subdirs)} folders. Processing for Task 2...")

    for subdir in subdirs[:400]: 
        subdir_path = os.path.join(root_dir, subdir)
        files = glob.glob(os.path.join(subdir_path, "**/*.mid"), recursive=True)
        
        for file in files[:2]: 
            try:
                midi = pretty_midi.PrettyMIDI(file)
                # Task 2 Requirement: Multi-genre generation
                roll = midi.get_piano_roll(fs=fs)[21:109, :] / 127.0
                roll = roll.T 
                
                for i in range(0, roll.shape[0] - window_size, window_size):
                    chunk = roll[i:i+window_size, :]
                    if chunk.shape[0] == window_size and np.sum(chunk > 0) > 15:
                        processed_data.append(chunk)
            except:
                continue
                
    if len(processed_data) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, np.array(processed_data))
        print(f"Task 2 Dataset Ready: {len(processed_data)} segments saved to {output_file}")
    else:
        print("Error: Path found, but no MIDI files were discovered. Check subfolder contents.")

if __name__ == "__main__":
    # ABSOLUTE PATH based on your image_a44d5e.png
    data_path = '/content/music-generation-unsupervised/data/raw_midi/lakh_clean'
    preprocess_lmd_multi_genre(data_path, 'data/processed/multi_genre_lmd.npy')