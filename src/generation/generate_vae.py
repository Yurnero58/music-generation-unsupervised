import os
import sys
import torch
import numpy as np

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
    
    # Load the actual dataset to pick real songs
    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}")
        return
        
    seed_data = np.load(data_path, mmap_mode='r')
    print(f"Loaded dataset. Generating {num_samples} Guided Reconstructions...")

    with torch.no_grad():
        for i in range(num_samples):
            # 1. Pick a random real song from the dataset
            idx = np.random.randint(0, len(seed_data))
            seed_matrix = seed_data[idx]
            
            # Save the ORIGINAL song so you can compare how well the VAE learned it
            original_path = os.path.join(output_dir, f"original_song_{i+1}.mid")
            multi_matrix_to_midi(seed_matrix > 0.5, original_path)
            
            # 2. Encode the real song to get its specific latent signature
            seed_tensor = torch.from_numpy(np.copy(seed_matrix)).float().unsqueeze(0).to(device)
            seq_len = seed_tensor.shape[1]
            
            _, (h_n, _) = model.encoder(seed_tensor)
            h_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            
            mu = model.fc_mu(h_cat)
            logvar = model.fc_logvar(h_cat)
            z = model.reparameterize(mu, logvar)
            
            # 3. Decode from that specific artist's signature
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for t in range(seq_len):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # Clean thresholding for sharper notes
                prob_array = probs.squeeze().cpu().numpy()
                binary_step = np.zeros(352)
                step_max = np.max(prob_array)
                
                if step_max > 0.15:
                    normalized = prob_array / step_max
                    binary_step[normalized > 0.6] = 1.0
                
                sample = torch.tensor(binary_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                generated_seq.append(sample)
                
                # Autoregressive feed-forward
                decoder_input = sample
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            recon_path = os.path.join(output_dir, f"reconstructed_song_{i+1}.mid")
            
            multi_matrix_to_midi(matrix > 0.5, recon_path)
            print(f"Saved: Original vs. Reconstruction Pair {i+1}")

if __name__ == "__main__":
    generate_reconstructions()