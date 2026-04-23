import os
import glob
import pickle
import numpy as np
import pretty_midi
# Import the tokenizer we created earlier
from tokenizer import MusicTokenizer

def quantize_velocity(velocity, num_bins=32):
    """Maps a 0-127 MIDI velocity to a 0-(num_bins-1) bin."""
    return int((velocity / 128.0) * num_bins)

def extract_events_from_midi(midi_path, max_time_shift=100):
    """Converts a MIDI file into a string of event tokens."""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        return [] # Skip corrupted files

    events = []
    # Extract every note from every instrument (ignore drums for now)
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                # Note On event
                events.append({'time': note.start, 'type': 'on', 'pitch': note.pitch, 'velocity': note.velocity})
                # Note Off event
                events.append({'time': note.end, 'type': 'off', 'pitch': note.pitch, 'velocity': 0})

    # Sort all events chronologically
    events.sort(key=lambda x: x['time'])

    string_tokens = []
    last_time = 0.0

    for event in events:
        # 1. Handle Time Shifts
        delta_time = event['time'] - last_time
        if delta_time > 0.01: # 10ms minimum resolution
            # Convert seconds to a quantized time shift token (e.g., 10ms steps)
            shift_bins = min(int(delta_time * 100), max_time_shift)
            if shift_bins > 0:
                string_tokens.append(f"TIME_SHIFT_{shift_bins}")
            last_time = event['time']

        # 2. Handle Notes
        if event['type'] == 'on':
            quantized_vel = quantize_velocity(event['velocity'])
            string_tokens.append(f"VELOCITY_{quantized_vel}")
            string_tokens.append(f"NOTE_ON_{event['pitch']}")
        elif event['type'] == 'off':
            string_tokens.append(f"NOTE_OFF_{event['pitch']}")

    return string_tokens

def build_dataset(root_dir, output_file, vocab_file):
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found!")
        return

    # Initialize Tokenizer
    tokenizer = MusicTokenizer()
    
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Found {len(subdirs)} artist folders. Extracting sequences...")

    all_encoded_tokens = []

    # Using the same subset limits you used in the VAE parser
    for subdir in subdirs[:400]: 
        subdir_path = os.path.join(root_dir, subdir)
        files = glob.glob(os.path.join(subdir_path, "**/*.mid"), recursive=True)
        
        for file in files[:2]: 
            # 1. Get string tokens: ["TIME_SHIFT_10", "VELOCITY_16", "NOTE_ON_60", ...]
            string_tokens = extract_events_from_midi(file)
            
            if len(string_tokens) > 50: # Only keep sequences with actual content
                # 2. Convert to integers: [45, 12, 164, ...]
                encoded_ids = tokenizer.encode(string_tokens)
                all_encoded_tokens.extend(encoded_ids)

    if len(all_encoded_tokens) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the dataset as a flat numpy array of integers
        np.save(output_file, np.array(all_encoded_tokens, dtype=np.uint16))
        
        # Save the vocabulary so the Generator knows how to decode it later
        tokenizer.save(vocab_file)
        
        print(f"Task 3 Dataset Ready: {len(all_encoded_tokens)} total tokens saved to {output_file}")
    else:
        print("Error: No valid MIDI sequences found.")

if __name__ == "__main__":
    # Adjust paths based on your Colab/Local setup
    data_path = '/content/music-generation-unsupervised/data/raw_midi/lakh_clean'
    output_path = '/content/music-generation-unsupervised/data/processed/transformer_tokens.npy'
    vocab_path = '/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl'
    
    build_dataset(data_path, output_path, vocab_path)