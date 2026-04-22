import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_soothing_vae(num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    weights_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    
    if not os.path.exists(weights_path):
        print("Error: Weights missing. Run Task 2 training first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2_soothing'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Soothing Multi-Track Compositions...")

    with torch.no_grad():
        for i in range(num_samples):
            # THE SOOTHING FIX: Sample from a very narrow latent space (low variance)
            # This avoids the chaotic "edges" of the model's knowledge
            z = (torch.randn(1, 256) * 0.3).to(device)
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for t in range(256): # Longer sequence for a slow song
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # THE SOOTHING FIX: Sparse Gating
                # Only play notes the model is EXTREMELY sure about (>0.7).
                # This prevents the noisy "drone" effect.
                sample = (probs > 0.7).float()
                generated_seq.append(sample)
                
                # Autoregressive feedback
                decoder_input = sample
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            out_file = os.path.join(output_dir, f"soothing_vae_{i+1}.mid")
            
            # THE SOOTHING FIX: Slow Tempo (fs=4)
            # This spreads the notes out, making it sound like a calm ambient track.
            multi_matrix_to_midi(matrix > 0.5, out_file, fs=4)
            print(f"Generated: {out_file} at 4 FPS")

if __name__ == "__main__":
    generate_soothing_vae()