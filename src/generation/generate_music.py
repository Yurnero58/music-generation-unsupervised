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
        note_events = matrix[:, pitch_idx] == 1.0 # Strict binary check
        
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
    model.load_state_dict(torch.load('src/models/ae_weights.pt', map_location=device))
    model.eval()
    
    seed_data = np.load('data/processed/classical_piano.npy', mmap_mode='r')
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    with torch.no_grad():
        for i in range(5):
            idx = np.random.randint(0, len(seed_data))
            seed = torch.from_numpy(np.copy(seed_data[idx])).float().unsqueeze(0).to(device)
            
            # Encode seed
            _, (h_n, _) = model.encoder(seed)
            z = model.fc_latent(h_n[-1])
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            # Start generating! 
            # 128 steps = 32 seconds of continuous music
            seq_len = 128 
            decoder_input = torch.zeros(1, 1, 88).to(device)
            generated_matrix = np.zeros((seq_len, 88))
            
            for t in range(seq_len):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                pred = torch.sigmoid(model.fc_out(out))
                
                # Convert probability to hard binary note (0.0 or 1.0)
                prob_array = pred.squeeze(0).cpu().numpy()
                binary_step = (prob_array > 0.5).astype(float)
                
                generated_matrix[t, :] = binary_step[0]
                
                # Feed the strictly played notes into the next time step
                decoder_input = torch.tensor(binary_step).float().unsqueeze(0).to(device)
            
            out_file = f'outputs/generated_midis/task1_proper_{i+1}.mid'
            matrix_to_midi(generated_matrix, out_file)
            print(f"Generated: {out_file}")

if __name__ == "__main__":
    generate()