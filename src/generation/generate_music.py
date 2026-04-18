import sys
import os
import torch
import numpy as np
import pretty_midi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder

def tensor_to_midi_smooth(tensor, output_file, fs=4, max_polyphony=3):
    """
    State-tracking MIDI converter. Prevents cluster chords and stuttering.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
    step_duration = 1.0 / fs
    
    # Tracks currently playing notes: {pitch: start_time}
    active_notes = {} 
    
    for t in range(tensor.shape[0]):
        current_time = t * step_duration
        step_probs = tensor[t]
        
        # 1. Find the top 3 strongest signals at this exact millisecond
        top_indices = np.argsort(step_probs)[-max_polyphony:]
        step_max = np.max(step_probs)
        
        # 2. Filter out background noise (must be at least 50% as strong as the loudest note)
        if step_max < 0.01:
            active_pitches_this_step = set()
        else:
            threshold = step_max * 0.5
            valid_indices = [idx for idx in top_indices if step_probs[idx] >= threshold]
            active_pitches_this_step = set([idx + 21 for idx in valid_indices])
            
        # 3. Turn OFF notes that are no longer playing
        for pitch in list(active_notes.keys()):
            if pitch not in active_pitches_this_step:
                start_time = active_notes.pop(pitch)
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=current_time)
                instrument.notes.append(note)
                
        # 4. Turn ON new notes
        for pitch in active_pitches_this_step:
            if pitch not in active_notes:
                active_notes[pitch] = current_time
                
    # 5. Clean up any notes still being held at the very end of the file
    end_time = tensor.shape[0] * step_duration
    for pitch, start_time in active_notes.items():
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)
        
    pm.instruments.append(instrument)
    pm.write(output_file)

def generate_final_deliverables():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_dim=88, hidden_dim=256, latent_dim=128, num_layers=2).to(device)
    weights_path = 'src/models/ae_weights.pt'
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    data_path = 'data/processed/classical_piano.npy'
    seed_data = np.load(data_path, mmap_mode='r')
    
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    print("Generating 5 clean, strictly-constrained MIDI samples...")
    with torch.no_grad():
        for s in range(5):
            # Grab a random real seed from your dataset
            idx = np.random.randint(0, len(seed_data))
            seed_tensor = torch.from_numpy(np.copy(seed_data[idx])).float().unsqueeze(0).to(device)
            
            # Encode it to get a valid latent vector
            _, (h_n, _) = model.encoder(seed_tensor)
            z = model.fc_hidden(h_n[-1])
            
            h_d = model.fc_latent(z)
            h_0 = h_d.unsqueeze(0).repeat(model.num_layers, 1, 1)
            c_0 = torch.zeros_like(h_0)
            
            # Decode into a 128-step (~32 seconds) sequence
            decoder_input = h_d.unsqueeze(1).repeat(1, 128, 1)
            out, _ = model.decoder(decoder_input, (h_0, c_0))
            
            reconstruction = torch.sigmoid(model.output_layer(out)).squeeze(0).cpu().numpy()
            
            output_file = f'outputs/generated_midis/task1_{s+1}.mid'
            tensor_to_midi_smooth(reconstruction, output_file, fs=4, max_polyphony=3)
            print(f"Generated: {output_file}")

if __name__ == "__main__":
    generate_final_deliverables()