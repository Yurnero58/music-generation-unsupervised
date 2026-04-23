import os
import sys
import numpy as np
import pretty_midi
import pickle
from collections import defaultdict

# Setup pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.preprocessing.tokenizer import MusicTokenizer

def tokens_to_midi(string_tokens, output_path):
    """Reusing your project's MIDI conversion logic"""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    current_time, current_velocity, active_notes = 0.0, 64, {}
    
    for token in string_tokens:
        if token.startswith("TIME_SHIFT_"):
            current_time += int(token.split("_")[2]) / 100.0
        elif token.startswith("VELOCITY_"):
            current_velocity = int((int(token.split("_")[1]) / 32.0) * 128)
        elif token.startswith("NOTE_ON_"):
            pitch = int(token.split("_")[2])
            active_notes[pitch] = (current_time, current_velocity)
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[2])
            if pitch in active_notes:
                st, vel = active_notes.pop(pitch)
                piano.notes.append(pretty_midi.Note(vel, pitch, st, current_time))
    midi.instruments.append(piano)
    midi.write(output_path)

def train_markov_chain(token_data_path):
    """Builds a transition matrix: {current_token: [list_of_next_tokens]}"""
    print("Training Markov Chain from dataset...")
    transitions = defaultdict(list)
    
    # Load your processed .npy file from Task 1
    data = np.load(token_data_path)
    
    for i in range(len(data) - 1):
        current_t = data[i]
        next_t = data[i+1]
        transitions[current_t].append(next_t)
    
    return transitions

def generate_markov_baseline(num_compositions=10, max_length=1500):
    # Paths
    vocab_path = os.path.join(BASE_DIR, 'data', 'processed', 'tokenizer_vocab.pkl')
    token_data_path = os.path.join(BASE_DIR, 'data', 'processed', 'transformer_tokens.npy')
    output_dir = os.path.join(BASE_DIR, 'outputs', 'baseline_markov')
    os.makedirs(output_dir, exist_ok=True)

    # Load Tokenizer
    tokenizer = MusicTokenizer()
    tokenizer.load(vocab_path)
    sos_token_id = tokenizer.token_to_id[tokenizer.sos_token]

    # Train the chain
    chain = train_markov_chain(token_data_path)

    print(f"--- Generating {num_compositions} Markov Compositions ---")
    for i in range(num_compositions):
        current_token = sos_token_id
        gen_ids = [current_token]

        for _ in range(max_length - 1):
            if current_token in chain:
                # Pick the next token based on learned probabilities
                next_token = np.random.choice(chain[current_token])
            else:
                # Fallback to random if we hit a dead end
                next_token = np.random.randint(0, tokenizer.vocab_size)
            
            gen_ids.append(next_token)
            current_token = next_token

        # Save outputs
        string_events = tokenizer.decode(gen_ids)
        out_mid = os.path.join(output_dir, f'markov_{i+1}.mid')
        out_npy = os.path.join(output_dir, f'markov_{i+1}.npy')
        
        tokens_to_midi(string_events, out_mid)
        np.save(out_npy, np.array(gen_ids, dtype=np.int64))
        print(f"Generated: {out_mid}")

if __name__ == "__main__":
    generate_markov_baseline()