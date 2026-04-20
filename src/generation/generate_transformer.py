import os
import sys
import torch
import numpy as np

# Ensure PYTHONPATH is correct for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer import MusicTransformer
from generation.generate_music import matrix_to_midi

def generate_task3_long_sequences(num_samples=10, seq_len=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with Task 3 dimensions
    model = MusicTransformer(input_dim=88, d_model=256, nhead=8).to(device)
    
    weights_path = '/content/music-generation-unsupervised/src/models/transformer_weights.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Transformer weights not found at {weights_path}. Train the model first!")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task3'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} long-horizon compositions (Length: {seq_len} steps)...")

    with torch.no_grad():
        for i in range(num_samples):
            # Start with a silent frame as a seed
            generated_seq = [torch.zeros(1, 1, 88).to(device)]
            
            for t in range(seq_len):
                # Concatenate previous tokens for context
                input_tensor = torch.cat(generated_seq, dim=1)
                
                # Causal mask ensures we only look at the past
                mask = model.generate_mask(input_tensor.size(1)).to(device)
                
                # Predict next token distribution p(xt | x<t)
                full_output = model(input_tensor, mask)
                probs = full_output[:, -1:, :] # Get only the last prediction
                
                # Use Temperature Sampling to prevent repetitive 'loops'
                # Temperature < 1.0 makes it conservative; > 1.0 makes it more random
                temp = 0.9
                probs = torch.pow(probs, 1.0 / temp)
                
                # Stochastic sampling to ensure variety
                sample = torch.bernoulli(probs)
                generated_seq.append(sample)
                
                # Sliding context window (Transformer memory limit)
                if len(generated_seq) > 512:
                    generated_seq.pop(0)

            # Convert the list of tensors into a single matrix
            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            
            midi_path = os.path.join(output_dir, f"transformer_long_{i+1}.mid")
            # Apply a confidence threshold for the final MIDI file
            matrix_to_midi(matrix > 0.5, midi_path)
            print(f"Generated and saved: {midi_path}")

if __name__ == "__main__":
    generate_task3_long_sequences()