import sys
import os
import torch
import numpy as np
import pretty_midi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder

def tensor_to_instrument(tensor, instrument, start_offset, fs=4):
    step_duration = 1.0 / fs 
    
    # DYNAMIC THRESHOLD: Adapts to the conservative outputs of MSE
    max_val = np.max(tensor)
    threshold = max_val * 0.7 if max_val > 0.05 else 0.02 

    for pitch_idx in range(tensor.shape[1]):
        pitch = pitch_idx + 21 
        notes = tensor[:, pitch_idx] > threshold 
        
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
            end_time = start_offset + (tensor.shape[0] * step_duration)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
            instrument.notes.append(note)

def generate_samples(num_samples=5, total_seconds=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder().to(device)
    weights_path = os.path.join(os.path.dirname(__file__), '../models/ae_weights.pt')
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
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
                z = torch.randn(1, 128).to(device) 
                
                h_d = torch.relu(model.fc_latent(z)).unsqueeze(0)
                decoder_input = h_d.repeat(1, steps_per_segment, 1).view(1, steps_per_segment, -1)
                
                out, _ = model.decoder(decoder_input, (h_d, torch.zeros_like(h_d)))
                reconstruction = torch.sigmoid(model.output_layer(out)).squeeze(0).cpu().numpy()
                
                tensor_to_instrument(reconstruction, instrument, start_offset=seg*duration_per_segment, fs=fs)
            
            pm.instruments.append(instrument)
            output_file = f'outputs/generated_midis/task1_final_{s+1}.mid'
            pm.write(output_file)
            print(f"Generated {total_seconds}s file: {output_file}")

if __name__ == "__main__":
    generate_samples()