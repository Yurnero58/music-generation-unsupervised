import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder
from generation.generate_music import matrix_to_midi

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder().to(device)
    
    model.load_state_dict(torch.load('/content/music-generation-unsupervised/src/models/ae_weights.pt', map_location=device))
    model.eval()
    
    seed_data = np.load('/content/music-generation-unsupervised/data/processed/classical_piano.npy', mmap_mode='r')
    output_dir = '/content/music-generation-unsupervised/outputs/generated_midis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initiating Guided Reconstruction Protocol (Task 1)...")
    with torch.no_grad():
        for i in range(5):
            idx = np.random.randint(0, len(seed_data))
            seed = torch.from_numpy(np.copy(seed_data[idx])).float().unsqueeze(0).to(device)
            seq_len = seed.shape[1]
            
            _, (h_n, _) = model.encoder(seed)
            z = model.fc_latent(h_n[-1])
            
            h = model.fc_dec_init(z).unsqueeze(0)
            c = torch.zeros_like(h)
            
            generated_matrix = np.zeros((seq_len, 88))
            decoder_input = torch.zeros(1, 1, 88).to(device)
            
            for t in range(seq_len):
                out, (h, c) = model.decoder(decoder_input, (h, c))
                pred = torch.sigmoid(model.fc_out(out))
                
                prob_array = pred.squeeze().cpu().numpy()
                binary_step = np.zeros(88)
                
                step_max = np.max(prob_array)
                if step_max > 0.1:
                    normalized = prob_array / step_max
                    binary_step[normalized > 0.6] = 1.0
                
                generated_matrix[t, :] = binary_step
                decoder_input = seed[:, t:t+1, :]
            
            out_file = os.path.join(output_dir, f'Task-1 midi file {i+1}.mid')
            matrix_to_midi(generated_matrix, out_file)
            print(f"Generated: {out_file}")

if __name__ == "__main__":
    generate()