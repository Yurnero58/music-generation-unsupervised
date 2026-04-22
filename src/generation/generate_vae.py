import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_audible_soothing(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    
    weights_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2_final'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Task 2 Final (Anti-Silence Optimized)...")
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from the safe, melodic center
            z = (torch.randn(1, 256) * 0.4).to(device)
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for t in range(256):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # Stochastic Sampling: Now works because probs are ~0.97
                sample = torch.bernoulli(probs.clamp(0, 1))
                
                # Clean up chaotic noise (Max 3 notes at once for extra soothing)
                if sample.sum() > 3:
                    mask = (torch.rand_like(sample) > 0.8).float()
                    sample = sample * mask
                
                generated_seq.append(sample)
                decoder_input = sample
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            out_file = os.path.join(output_dir, f"task2_soothing_{i+1}.mid")
            
            # Slow, clear tempo
            multi_matrix_to_midi(matrix > 0.5, out_file, fs=6)
            print(f"Generated: {out_file}")

if __name__ == "__main__":
    generate_audible_soothing()