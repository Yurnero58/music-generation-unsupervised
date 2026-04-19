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
    model = MusicVAE().to(device)
    
    # Load the trained weights
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/vae_weights.pt'))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../outputs/task2'))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} diverse samples from the latent space...")

    with torch.no_grad():
        for i in range(num_samples):
            # Sample z ~ N(0, I)
            z = torch.randn(1, 128).to(device)
            
            # Initial decoder state from latent vector
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            # Start token (zero frame)
            decoder_input = torch.zeros(1, 1, 88).to(device)
            generated_seq = []

            for _ in range(128): # Generate ~32 seconds of music
                out, (h, c) = model.decoder(decoder_input, (h, c))
                pred = torch.sigmoid(model.fc_out(out))
                generated_seq.append(pred)
                decoder_input = pred
            
            # Convert to binary piano roll
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            midi_path = os.path.join(output_dir, f"diverse_sample_{i+1}.mid")
            matrix_to_midi(matrix > 0.3, midi_path) # Thresholding for better note clarity
            print(f"Saved: {midi_path}")

if __name__ == "__main__":
    generate_task2_samples()