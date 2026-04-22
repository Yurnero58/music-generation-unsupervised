import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from generation.generate_music import multi_matrix_to_midi

def generate_task2_samples(num_samples=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 352 input, 1024 hidden to match the new Multi-Track VAE
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    model.load_state_dict(torch.load('/content/music-generation-unsupervised/src/models/vae_weights.pt', map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} Multi-Track VAE samples...")

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 256).to(device)
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            # Initialize with 352 dimensions
            decoder_input = torch.zeros(1, 1, 352).to(device)
            generated_seq = []

            for _ in range(128):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                probs = torch.sigmoid(model.fc_out(out))
                
                sample = torch.bernoulli(probs) 
                generated_seq.append(sample)
                decoder_input = sample
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            midi_path = os.path.join(output_dir, f"diverse_multitrack_{i+1}.mid")
            
            # Save using the 352-key multi-track function
            multi_matrix_to_midi(matrix > 0.5, midi_path)
            print(f"Saved: {midi_path}")

if __name__ == "__main__":
    generate_task2_samples()