import os
import sys
import torch
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_reconstructions(num_samples=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    model.load_state_dict(torch.load('/content/music-generation-unsupervised/src/models/vae_weights.pt', map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2'
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}")
        return
        
    seed_data = np.load(data_path, mmap_mode='r')
    print(f"Loaded dataset. Generating {num_samples} WILDLY different reconstructions...")

    with torch.no_grad():
        for i in range(num_samples):
            # 1. TEMPO MUTATION: Randomize the speed of the song
            # Lower fs = very slow/sludgy. Higher fs = very fast/frantic.
            current_fs = random.choice([4, 6, 9, 14, 18, 22])
            
            # 2. RHYTHM MUTATION: Randomize the threshold gate
            # High threshold = sparse notes. Low threshold = dense walls of sound.
            current_threshold = random.uniform(0.05, 0.30)
            
            idx = np.random.randint(0, len(seed_data))
            seed_matrix = seed_data[idx]
            
            seed_tensor = torch.from_numpy(np.copy(seed_matrix)).float().unsqueeze(0).to(device)
            seq_len = seed_tensor.shape[1]
            
            _, (h_n, _) = model.encoder(seed_tensor)
            h_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            
            mu = model.fc_mu(h_cat)
            logvar = model.fc_logvar(h_cat)
            z = model.reparameterize(mu, logvar)
            
            # 3. LATENT MUTATION: Smash the latent space with heavy noise to break the "average" curse
            z = z + (torch.randn_like(z) * 1.8)
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for t in range(seq_len):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                prob_array = probs.squeeze().cpu().numpy()
                binary_step = np.zeros(352)
                
                # Apply the randomized rhythm gate
                binary_step[prob_array > current_threshold] = 1.0 
                
                sample = torch.tensor(binary_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                generated_seq.append(sample)
                
                # Teacher forcing
                decoder_input = seed_tensor[:, t:t+1, :]
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            
            # Name includes the tempo so you know it worked
            recon_path = os.path.join(output_dir, f"mutated_track_{i+1}_Tempo{current_fs}.mid")
            
            # Inject the random tempo directly into the MIDI creator
            multi_matrix_to_midi(matrix > 0.5, recon_path, fs=current_fs)
            
            print(f"Saved: Track {i+1} | Speed: {current_fs} FPS | Rhythm Gate: {current_threshold:.2f}")

if __name__ == "__main__":
    generate_reconstructions()