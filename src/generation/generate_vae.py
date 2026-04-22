import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_multi_artist_reconstruction(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    
    weights_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load the actual data so we can pick real, different songs
    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    seed_data = np.load(data_path, mmap_mode='r')

    output_dir = '/content/music-generation-unsupervised/outputs/task2_final'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reconstructing {num_samples} DIFFERENT songs from the dataset...")

    with torch.no_grad():
        for i in range(num_samples):
            # 1. Pick a random, UNIQUE song index from the dataset
            idx = np.random.randint(0, len(seed_data))
            seed_matrix = seed_data[idx]
            
            # 2. Trim silence to ensure the music starts immediately
            active_steps = np.sum(seed_matrix, axis=1)
            first_note = np.argmax(active_steps > 0) if np.max(active_steps) > 0 else 0
            trimmed_seed = seed_matrix[first_note : first_note + 128]
            
            # Padding if the song is too short
            if trimmed_seed.shape[0] < 128:
                pad = np.zeros((128 - trimmed_seed.shape[0], 352))
                trimmed_seed = np.vstack([trimmed_seed, pad])
            
            seed_tensor = torch.from_numpy(np.copy(trimmed_seed)).float().unsqueeze(0).to(device)

            # 3. Encode THIS specific song into the latent space
            _, (h_n, _) = model.encoder(seed_tensor)
            h_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            z = model.fc_mu(h_cat) # Use the exact Mean for perfect artist match

            # 4. Decode the signature of THIS artist
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for t in range(128):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # Use Greedy Decoding for the most accurate reconstruction
                sample = (probs > 0.4).float()
                generated_seq.append(sample)
                
                # Guided Reconstruction: feed the real note back in to keep it on track
                decoder_input = seed_tensor[:, t:t+1, :]
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            out_file = os.path.join(output_dir, f"reconstructed_artist_{i+1}.mid")
            
            multi_matrix_to_midi(matrix > 0.5, out_file, fs=8)
            print(f"Saved reconstruction for song index {idx}")

if __name__ == "__main__":
    generate_multi_artist_reconstruction()