import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import sys

# Ensure PYTHONPATH is correct for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer import MusicTransformer
from preprocessing.piano_roll import get_loader

def train():
    # Use CUDA_LAUNCH_BLOCKING for cleaner error reporting if it crashes again
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicTransformer(input_dim=88, d_model=256, nhead=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss() 

    data_path = '/content/music-generation-unsupervised/data/processed/multi_genre_lmd.npy'
    train_loader = get_loader(data_path, batch_size=32)

    print(f"Starting Task 3 (Transformer) Training on {device}...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Causal mask for autoregressive probability p(xt | x<t)
            mask = model.generate_mask(batch.size(1)).to(device)
            output = model(batch, mask)
            
            # FIX: Clamp target batch to [0, 1] to prevent CUDA Assert error
            target = torch.clamp(batch[:, 1:, :], 0.0, 1.0)
            input_seq = output[:, :-1, :]
            
            loss = criterion(input_seq, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        # Deliverable: Perplexity = exp(LTR)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1}/30 | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    save_path = '/content/music-generation-unsupervised/src/models/transformer_weights.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

if __name__ == "__main__":
    train()