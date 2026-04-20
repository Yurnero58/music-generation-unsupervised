import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer import MusicTransformer
from generation.generate_music import matrix_to_midi

def generate_task3_long_sequences(num_samples=10, seq_len=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer().to(device)
    weights_path = '/content/music-generation-unsupervised/src/models/transformer_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task3'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} long-horizon compositions...")

    with torch.no_grad():
        for i in range(num_samples):
            # Start with a silent frame
            generated_seq = [torch.zeros(1, 1, 88).to(device)]
            
            for t in range(seq_len):
                input_tensor = torch.cat(generated_seq, dim=1)
                # Use causal mask to maintain autoregressive integrity
                mask = model.generate_mask(input_tensor.size(1)).to(device)
                probs = model(input_tensor, mask)[:, -1:, :]
                
                # Sampling with temperature to ensure coherence over time
                sample = torch.bernoulli(probs)
                generated_seq.append(sample)
                
                # Keep sliding window to manage memory if seq gets too long
                if len(generated_seq) > 256:
                    generated_seq.pop(0)

            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            midi_path = os.path.join(output_dir, f"long_composition_{i+1}.mid")
            matrix_to_midi(matrix > 0.5, midi_path)
            print(f"Saved: {midi_path}")

if __name__ == "__main__":
    generate_task3_long_sequences()