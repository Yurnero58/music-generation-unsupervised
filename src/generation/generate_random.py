import os
import sys
import numpy as np
import pretty_midi

# 1. Setup pathing to find the tokenizer class
# This adds the root 'music-generation-unsupervised' directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

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
            try:
                shift_bins = int(token.split("_")[2])
                current_time += shift_bins / 100.0 # Revert 10ms bins back to seconds
            except: continue
        
        elif token.startswith("VELOCITY_"):
            try:
                bin_val = int(token.split("_")[1])
                current_velocity = int((bin_val / 32.0) * 128)
            except: continue
            
        elif token.startswith("NOTE_ON_"):
            try:
                pitch = int(token.split("_")[2])
                active_notes[pitch] = (current_time, current_velocity)
            except: continue
            
        elif token.startswith("NOTE_OFF_"):
            try:
                pitch = int(token.split("_")[2])
                if pitch in active_notes:
                    start_time, vel = active_notes.pop(pitch)
                    note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start_time, end=current_time)
                    piano.notes.append(note)
            except: continue
                
    midi.instruments.append(piano)
    midi.write(output_path)

def generate_random_baseline(num_compositions=10, max_length=1500):
    print("--- Generating Naive Random Baseline ---")

    # 2. Dynamic Vocabulary Pathing
    vocab_path = os.path.join(BASE_DIR, 'data', 'processed', 'tokenizer_vocab.pkl')
    
    if not os.path.exists(vocab_path):
        print(f"CRITICAL ERROR: Vocabulary not found at {vocab_path}")
        print("Please ensure you have run your Task 1 preprocessing first.")
        return

    print(f"Loading vocabulary from: {vocab_path}")
    tokenizer = MusicTokenizer()
    tokenizer.load(vocab_path)
    
    # 3. Setup Output Directory
    output_dir = os.path.join(BASE_DIR, 'outputs', 'baseline_random')
    os.makedirs(output_dir, exist_ok=True)
    
    sos_token_id = tokenizer.token_to_id[tokenizer.sos_token]

    for i in range(num_compositions):
        # 4. Pick random tokens across the entire vocabulary range
        # Use vocab_size-1 because randint is inclusive
        random_ids = np.random.randint(0, tokenizer.vocab_size, size=(max_length,))
        
        # Enforce the SOS token at the start
        random_ids[0] = sos_token_id
        
        # 5. Decode and Save
        comp_ids = random_ids.tolist()
        string_events = tokenizer.decode(comp_ids)
        
        out_mid_file = os.path.join(output_dir, f'random_{i+1}.mid')
        out_npy_file = os.path.join(output_dir, f'random_{i+1}.npy')
        
        try:
            tokens_to_midi(string_events, out_mid_file)
            np.save(out_npy_file, np.array(comp_ids, dtype=np.int64))
            print(f"Successfully generated: {out_mid_file}")
            
        except Exception as e:
            # We expect random tokens to fail sometimes (invalid MIDI logic)
            print(f"Track {i+1} failed to parse (standard for noise): {e}")

if __name__ == "__main__":
    generate_random_baseline()