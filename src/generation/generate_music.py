import sys
import os
import torch
import numpy as np
import pretty_midi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder

def tensor_to_instrument(tensor, instrument, start_offset, fs=4, max_notes=4):
    step_duration = 1.0 / fs 
    
    # 1. TOP-K MASKING: Enforce human constraints (max 4 notes per time step)
    binary_mask = np.zeros_like(tensor)
    for t in range(tensor.shape[0]):
        # Get the indices of the highest probabilities at this exact millisecond
        top_indices = np.argsort(tensor[t])[-max_notes:]
        
        # Only activate them if they are above a bare minimum noise floor
        for idx in top_indices:
            if tensor[t, idx] > 0.02: 
                binary_mask[t, idx] = 1.0

    # 2. NOTE WRITING: Use the clean mask to write MIDI events
    for pitch_idx in range(binary_mask.shape[1]):
        pitch = pitch_idx + 21 
        notes = binary_mask[:, pitch_idx] > 0.5 # Mask is already 1.0 or 0.0
        
        start_time = None
        for i, val in enumerate(notes):
            current_time = start_offset + (i * step_duration)
            if val and start_time is None:
                start_time = current_time
            elif not val and start_time is not None:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=current_time)
                instrument.notes.append(note)
                start_time = None
        
        if start_time is not None:
            end_time = start_offset + (binary_mask.shape[0] * step_duration)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
            instrument.notes.append(note)

def generate_samples(num_samples=5, total_seconds=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMAutoencoder(input_dim=88, hidden_dim=256, latent_dim=128, num_layers=2).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), '../models/ae_weights.pt')
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Load the real dataset to use as seeds
    data_path = 'data/processed/classical_piano.npy'
    if not os.path.exists(data_path):
        print("Data not found. Run preprocessing first.")
        return
        
    seed_data = np.load(data_path, mmap_mode='r')
    
    fs = 4 
    steps_per_segment = 64
    duration_per_segment = steps_per_segment / fs 
    num_segments = int(np.ceil(total_seconds / duration_per_segment))

    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    with torch.no_grad():
        for s in range(num_samples):
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            
            for seg in range(num_segments):
                # 1. Grab a random real segment
                idx = np.random.randint(0, len(seed_data))
                seed_tensor = torch.from_numpy(np.copy(seed_data[idx])).float().unsqueeze(0).to(device)
                
                # 2. Encode it to get a GUARANTEED VALID latent vector z
                _, (h_n, _) = model.encoder(seed_tensor)
                last_hidden = h_n[-1]
                z = model.fc_hidden(last_hidden) 
                
                # 3. Decode the valid z
                h_d = model.fc_latent(z) 
                h_0 = h_d.unsqueeze(0).repeat(model.num_layers, 1, 1) 
                c_0 = torch.zeros_like(h_0)
                
                decoder_input = h_d.unsqueeze(1).repeat(1, steps_per_segment, 1)
                
                out, _ = model.decoder(decoder_input, (h_0, c_0))
                reconstruction = torch.sigmoid(model.output_layer(out)).squeeze(0).cpu().numpy()
                
                tensor_to_instrument(reconstruction, instrument, start_offset=seg*duration_per_segment, fs=fs)
            
            pm.instruments.append(instrument)
            output_file = f'outputs/generated_midis/task1_final_{s+1}.mid'
            pm.write(output_file)
            print(f"Generated {total_seconds}s file based on real seed: {output_file}")

if __name__ == "__main__":
    generate_samples()