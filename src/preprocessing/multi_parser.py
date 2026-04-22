import pretty_midi
import numpy as np
import glob
import os

def parse_multi_instrument_midi(midi_path, fs=10, seq_len=512):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except:
        return None

    # We will track 4 families: 0=Drums, 1=Piano/Keys, 2=Bass/Guitar, 3=Strings/Synth
    # Shape for each: (88 keys, Time)
    tracks = [np.zeros((88, seq_len)) for _ in range(4)]
    
    for inst in pm.instruments:
        # Get piano roll and trim to standard 88 keys (MIDI notes 21-108)
        roll = inst.get_piano_roll(fs=fs)[21:109, :]
        
        # Pad or trim to our fixed sequence length
        if roll.shape[1] > seq_len:
            roll = roll[:, :seq_len]
        else:
            pad = np.zeros((88, seq_len - roll.shape[1]))
            roll = np.hstack([roll, pad])
            
        roll = (roll > 0).astype(float)
        
        # Categorize the instrument and merge it into the correct track
        if inst.is_drum:
            tracks[0] = np.maximum(tracks[0], roll)
        elif 0 <= inst.program <= 7: # MIDI Programs 0-7 are Pianos/Keys
            tracks[1] = np.maximum(tracks[1], roll)
        elif 24 <= inst.program <= 39: # MIDI Programs 24-39 are Guitars and Basses
            tracks[2] = np.maximum(tracks[2], roll)
        else: # Everything else (Strings, Brass, Synth, Pads)
            tracks[3] = np.maximum(tracks[3], roll)
            
    # Stack them vertically. New shape: (352, Time)
    combined_matrix = np.vstack(tracks)
    # Transpose to (Time, 352) to match standard model inputs
    return combined_matrix.T

def build_multi_dataset(midi_folder, output_path, max_files=800):
    files = glob.glob(f"{midi_folder}/**/*.mid", recursive=True)
    dataset = []
    
    print(f"Found {len(files)} files. Processing the first {max_files} for Multi-Track...")
    
    for i, f in enumerate(files[:max_files]):
        matrix = parse_multi_instrument_midi(f)
        if matrix is not None and np.sum(matrix) > 10: # Ensure it's not empty
            dataset.append(matrix)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{max_files} files...")
            
    if dataset:
        final_data = np.array(dataset)
        np.save(output_path, final_data)
        print(f"Success! Multi-instrument dataset saved to {output_path} | Shape: {final_data.shape}")
    else:
        print("Error: No valid data extracted.")

if __name__ == "__main__":
    # UPDATE THIS PATH to where your Lakh MIDI files are located
    midi_folder = 'data/raw/lmd_aligned' 
    output_path = 'data/processed/multi_track_lmd.npy'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    build_multi_dataset(midi_folder, output_path)