import os
import sys
import torch
import torch.nn.functional as F
import pretty_midi
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import MusicTokenizer

def generate_tokens(model, start_token_id, max_length=1024, temperature=0.9, device="cuda"):
    model.eval()
    sequence = torch.tensor([[start_token_id]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(sequence)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat([sequence, next_token], dim=1)
                
    return sequence[0].tolist()

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

def generate_10_compositions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tokenizer
    tokenizer = MusicTokenizer()
    vocab_path = '/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl'
    tokenizer.load(vocab_path)
    
    # Load Model
    model = MusicTransformer(vocab_size=tokenizer.vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
    weights_path = '/content/music-generation-unsupervised/src/models/transformer_rlhf_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    output_dir = '/content/music-generation-unsupervised/outputs/generated_midis'
    os.makedirs(output_dir, exist_ok=True)
    
    sos_token_id = tokenizer.token_to_id[tokenizer.sos_token]
    
    print("Generating 10 long-sequence compositions for Human Feedback...")
    for i in range(10):
        # 1. Autoregressive Generation (This is the array of integers)
        comp_ids = generate_tokens(model, sos_token_id, max_length=1500, temperature=0.95, device=device)
        
        # 2. Decode IDs to String Events
        string_events = tokenizer.decode(comp_ids)
        
        # 3. Save as a MIDI file (For humans to listen to)
        out_mid_file = os.path.join(output_dir, f'Task-4 midi file {i+1}.mid')
        tokens_to_midi(string_events, out_mid_file)
        
        # 4. NEW: Save the raw tokens as a Numpy array (For the Reward Model to read)
        out_npy_file = os.path.join(output_dir, f'Task-4 npy file {i+1}.npy')
        np.save(out_npy_file, np.array(comp_ids, dtype=np.int64))
        
        print(f"Saved: {out_mid_file} AND {out_npy_file}")



if __name__ == "__main__":
    generate_10_compositions()