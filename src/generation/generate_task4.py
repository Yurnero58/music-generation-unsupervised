import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer import MusicTransformer
from generation.generate_music import matrix_to_midi

def generate_rlhf_samples(num_samples=10, seq_len=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(input_dim=88, d_model=256, nhead=8).to(device)
    
    # LOAD TASK 4 WEIGHTS
    weights_path = '/content/music-generation-unsupervised/src/models/rlhf_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    output_dir = '/content/music-generation-unsupervised/outputs/task4'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} Task 4 (RLHF) compositions...")

    with torch.no_grad():
        for i in range(num_samples):
            generated_seq = [torch.zeros(1, 1, 88).to(device)]
            for t in range(seq_len):
                input_tensor = torch.cat(generated_seq, dim=1)
                mask = model.generate_mask(input_tensor.size(1)).to(device)
                
                full_output = model(input_tensor, mask)
                probs = full_output[:, -1:, :]
                
                # Temperature 0.8 for cleaner rhythm
                temp = 0.8
                probs = torch.pow(probs, 1.0 / temp)
                
                sample = torch.bernoulli(probs)
                generated_seq.append(sample)
                
                if len(generated_seq) > 512:
                    generated_seq.pop(0)

            matrix = torch.cat(generated_seq, dim=1).squeeze(0).cpu().numpy()
            midi_path = os.path.join(output_dir, f"rlhf_tuned_{i+1}.mid")
            matrix_to_midi(matrix > 0.5, midi_path)
            print(f"Saved: {midi_path}")

if __name__ == "__main__":
    generate_rlhf_samples()