import sys
import os
import torch
import numpy as np
import pretty_midi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder

def matrix_to_midi(matrix, output_path, fs=4):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    step_duration = 1.0 / fs
    
    for pitch_idx in range(matrix.shape[1]):
        pitch = pitch_idx + 21
        note_events = matrix[:, pitch_idx] == 1.0
        
        start_time = None
        for t, is_playing in enumerate(note_events):
            current_time = t * step_duration
            if is_playing and start_time is None:
                start_time = current_time
            elif not is_playing and start_time is not None:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=current_time)
                instrument.notes.append(note)
                start_time = None
                
        if start_time is not None:
            end_time = matrix.shape[0] * step_duration
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
            instrument.notes.append(note)
            
    pm.instruments.append(instrument)
    pm.write(output_path)

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder().to(device)
    
    # Load your successfully trained weights
    model.load_state_dict(torch.load('src/models/ae_weights.pt', map_location=device))
    model.eval()
    
    seed_data = np.load('data/processed/classical_piano.npy', mmap_mode='r')
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    with torch.no_grad():
        for i in range(5):
            # 1. Get a random seed
            idx = np.random.randint(0, len(seed_data))
            seed = torch.from_numpy(np.copy(seed_data[idx])).float().unsqueeze(0).to(device)
            
            # Encode seed to get the context (z)
            _, (h_n, _) = model.encoder(seed)
            z = model.fc_latent(h_n[-1])
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            seq_len = 128 # 32 seconds of music
            
            # FIX 1: Start the loop with the LAST step of the actual music, not pure silence
            decoder_input = seed[:, -1:, :] 
            
            generated_matrix = np.zeros((seq_len, 88))
            
            for t in range(seq_len):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                pred = torch.sigmoid(model.fc_out(out))
                
                # Get raw probabilities for this specific 1/4 second
                prob_array = pred.squeeze(0).squeeze(0).cpu().numpy()
                
                # FIX 2: Dynamic Sampling to prevent infinite silence loops
                step_max = np.max(prob_array)
                binary_step = np.zeros(88)
                
                if step_max > 0.01:
                    # Normalize and pick notes that are highly confident relative to the max
                    normalized = prob_array / step_max
                    binary_step[normalized > 0.8] = 1.0
                    
                    # Cap polyphony to 4 notes max to prevent cluster-chord noise
                    if np.sum(binary_step) > 4:
                        top_4 = np.argsort(prob_array)[-4:]
                        binary_step = np.zeros(88)
                        binary_step[top_4] = 1.0
                
                generated_matrix[t, :] = binary_step
                
                # Feed exactly what we just played into the next step
                decoder_input = torch.tensor(binary_step).float().unsqueeze(0).unsqueeze(0).to(device)
            
            out_file = f'outputs/generated_midis/task1_proper_{i+1}.mid'
            matrix_to_midi(generated_matrix, out_file)
            print(f"Generated: {out_file}")

if __name__ == "__main__":
    generate()