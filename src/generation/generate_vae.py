import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_pure_reconstruction(num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your trained model
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    weights_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    
    if not os.path.exists(weights_path):
        print("Error: Model weights not found. Train the VAE first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2'
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    seed_data = np.load(data_path, mmap_mode='r')
    
    print("Initiating STRICT GUIDED Reconstruction...")

    with torch.no_grad():
        for i in range(num_samples):
            # 1. Grab a random song and trim silence
            idx = np.random.randint(0, len(seed_data))
            seed_matrix = seed_data[idx]
            
            active_steps = np.sum(seed_matrix, axis=1)
            if np.max(active_steps) == 0: continue 
            first_note_idx = np.argmax(active_steps > 0)
            
            trimmed_seed = seed_matrix[first_note_idx : first_note_idx + 128]
            if trimmed_seed.shape[0] < 128:
                pad = np.zeros((128 - trimmed_seed.shape[0], 352))
                trimmed_seed = np.vstack([trimmed_seed, pad])
                
            # Save the EXACT input we are asking the model to memorize
            original_path = os.path.join(output_dir, f"pure_original_{i+1}.mid")
            multi_matrix_to_midi(trimmed_seed > 0.5, original_path, fs=10)
            
            seed_tensor = torch.from_numpy(np.copy(trimmed_seed)).float().unsqueeze(0).to(device)
            
            # 2. Encode to Latent Space
            _, (h_n, _) = model.encoder(seed_tensor)
            h_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            
            mu = model.fc_mu(h_cat)
            
            # Zero Latent Noise. Force the exact mean.
            z = mu 
            
            # 3. Decode Autoregressively
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for t in range(128):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # THE FIX #1: Forgiving Greedy Threshold. 
                # Ensure the notes fire even if the model is slightly hesitant.
                sample = (probs > 0.25).float()
                generated_seq.append(sample)
                
                # THE FIX #2: Guided Reconstruction (Teacher Forcing)
                # Feed the REAL note from the original song into the next step.
                # This guarantees the LSTM cannot derail into garbage noise.
                decoder_input = seed_tensor[:, t:t+1, :]
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            recon_path = os.path.join(output_dir, f"pure_reconstruction_{i+1}.mid")
            
            multi_matrix_to_midi(matrix > 0.5, recon_path, fs=10)
            print(f"Saved: Original vs. Pure Reconstruction Pair {i+1}")

if __name__ == "__main__":
    generate_pure_reconstruction()