import os
import sys
import torch
import numpy as np

# Set path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import MusicVAE
from generation.generate_music import matrix_to_midi

def generate_task2_samples(num_samples=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CRITICAL: Match the new 512 hidden / 256 latent dimensions
    model = MusicVAE(input_dim=88, hidden_dim=512, latent_dim=256).to(device)
    
    weights_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found. Please run training first.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task2'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} samples from the 512/256 architecture...")

    with torch.no_grad():
        for i in range(num_samples):
            # Sample z ~ N(0, I) with size 256
            z = torch.randn(1, 256).to(device) 
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            decoder_input = torch.zeros(1, 1, 88).to(device)
            generated_seq = []

            for _ in range(128):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                pred = torch.sigmoid(model.fc_out(out))
                generated_seq.append(pred)
                decoder_input = pred
            
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            midi_path = os.path.join(output_dir, f"diverse_sample_{i+1}.mid")
            
            # Lowered threshold to ensure notes are captured
            matrix_to_midi(matrix > 0.15, midi_path) 
            print(f"Saved: {midi_path}")

if __name__ == "__main__":
    generate_task2_samples()