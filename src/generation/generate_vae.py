import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_task2_samples(num_samples=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    model.load_state_dict(torch.load('/content/music-generation-unsupervised/src/models/vae_weights.pt', map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} Multi-Track VAE samples with increasing variance...")

    with torch.no_grad():
        for i in range(num_samples):
            # THE FIX: Latent Space Temperature Scaling
            # We increase the temperature for each sample to push it into different "genre" clusters
            # Sample 0 gets T=1.0 (Average). Sample 7 gets T=2.5 (Extreme Edge).
            temperature = 1.0 + (i * 0.2) 
            
            # Multiply randn by temperature to stretch the sampling distribution
            z = (torch.randn(1, 256) * temperature).to(device)
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for _ in range(128):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                # We also add a tiny bit of noise to the probabilities to break repetitive loops
                probs = torch.clamp(probs + (torch.randn_like(probs) * 0.05), 0.0, 1.0)
                
                sample = torch.bernoulli(probs) 
                generated_seq.append(sample)
                decoder_input = sample
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            
            # Name the file with its temperature so you can hear the difference
            midi_path = os.path.join(output_dir, f"diverse_multitrack_T{temperature:.1f}.mid")
            
            multi_matrix_to_midi(matrix > 0.5, midi_path)
            print(f"Saved: {midi_path} | Latent Temperature: {temperature:.1f}")

if __name__ == "__main__":
    generate_task2_samples()