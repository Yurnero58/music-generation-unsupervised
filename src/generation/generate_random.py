import os
import sys
import numpy as np
import pretty_midi

# Adjust imports based on your exact file structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.tokenizer import MusicTokenizer

def tokens_to_midi(string_tokens, output_path):
    """Converts a list of string events back into a playable MIDI file."""
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    current_time = 0.0
    current_velocity = 64 # Default velocity
    active_notes = {}
    
    for token in string_tokens:
        if token.startswith("TIME_SHIFT_"):
            shift_bins = int(token.split("_")[2])
            current_time += shift_bins / 100.0 # Revert 10ms bins back to seconds
        
        elif token.startswith("VELOCITY_"):
            bin_val = int(token.split("_")[1])
            current_velocity = int((bin_val / 32.0) * 128) # Revert bins back to 0-127
            
        elif token.startswith("NOTE_ON_"):
            pitch = int(token.split("_")[2])
            # Store when the note started and how loud it was
            active_notes[pitch] = (current_time, current_velocity)
            
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[2])
            if pitch in active_notes:
                start_time, vel = active_notes.pop(pitch)
                # Create the note and add it to the instrument
                note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start_time, end=current_time)
                piano.notes.append(note)
                
    midi.instruments.append(piano)
    midi.write(output_path)

def generate_random_baseline(num_compositions=10, max_length=1500):
    print("Generating Naive Random Baseline...")

    # 1. Load Tokenizer to get the exact vocab size
    tokenizer = MusicTokenizer()
    vocab_path = '/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl'
    tokenizer.load(vocab_path)
    
    output_dir = '/content/music-generation-unsupervised/outputs/baseline_random/'
    os.makedirs(output_dir, exist_ok=True)
    
    sos_token_id = tokenizer.token_to_id[tokenizer.sos_token]

    for i in range(num_compositions):
        # 2. Pick random tokens between 1 and vocab_size
        # (We skip 0 if it is reserved for padding)
        random_ids = np.random.randint(1, tokenizer.vocab_size, size=(max_length,))
        
        # Force the first token to be the Start-Of-Sequence token 
        random_ids[0] = sos_token_id
        
        # 3. Decode IDs to String Events
        comp_ids = random_ids.tolist()
        string_events = tokenizer.decode(comp_ids)
        
        # 4. Save as MIDI
        out_mid_file = os.path.join(output_dir, f'random_{i+1}.mid')
        
        try:
            tokens_to_midi(string_events, out_mid_file)
            
            # Save the raw tokens too
            out_npy_file = os.path.join(output_dir, f'random_{i+1}.npy')
            np.save(out_npy_file, np.array(comp_ids, dtype=np.int64))
            
            print(f"Saved: {out_mid_file} AND {out_npy_file}")
            
        except Exception as e:
            # Random tokens often create impossible MIDI logic (like turning off a note that never started)
            print(f"Track {i+1} failed to parse into MIDI (Expected for random noise): {e}")

if __name__ == "__main__":
    generate_random_baseline()