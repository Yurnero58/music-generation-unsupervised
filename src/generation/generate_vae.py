import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_soothing_reconstruction(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    
    weights_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    seed_data = np.load(data_path, mmap_mode='r')

    output_dir = '/content/music-generation-unsupervised/outputs/task2_soothing'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} SLOW & SOOTHING reconstructions...")

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(seed_data))
            seed_matrix = seed_data[idx]
            
            # Find the music, skip the silence
            active_steps = np.sum(seed_matrix, axis=1)
            first_note = np.argmax(active_steps > 0) if np.max(active_steps) > 0 else 0
            trimmed_seed = seed_matrix[first_note : first_note + 160] # Slightly longer
            
            seed_tensor = torch.from_numpy(np.copy(trimmed_seed)).float().unsqueeze(0).to(device)

            _, (h_n, _) = model.encoder(seed_tensor)
            h_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            z = model.fc_mu(h_cat) 

            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []
            
            # Rhythmic Cooldown to prevent the "terererere" drone
            cooldown = 0 

            for t in range(trimmed_seed.shape[0]):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # If we just played a note, force silence for a moment to create "breath"
                if cooldown > 0:
                    sample = torch.zeros_like(probs)
                    cooldown -= 1
                else:
                    # Only play notes the model is VERY sure about
                    sample = (probs > 0.45).float()
                    if sample.sum() > 0:
                        cooldown = 2 # Wait 2 frames before next note
                
                generated_seq.append(sample)
                decoder_input = seed_tensor[:, t:t+1, :] if t < trimmed_seed.shape[0] else sample
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            out_file = os.path.join(output_dir, f"soothing_artist_{i+1}.mid")
            
            # THE FIX: Slow the FS to 4 for a calm, human-like tempo
            multi_matrix_to_midi(matrix > 0.5, out_file, fs=4)
            print(f"Saved soothing track {i+1} (Original index: {idx})")

if __name__ == "__main__":
    generate_soothing_reconstruction()