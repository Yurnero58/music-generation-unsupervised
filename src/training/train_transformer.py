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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Using Binary Cross Entropy for multi-label note prediction
    criterion = nn.BCELoss() 

    # CORRECTED PATH: Using your existing Lakh processed data
    data_path = '/content/music-generation-unsupervised/data/processed/multi_genre_lmd.npy'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run your Task 2 parser first!")
        return

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
            
            # Autoregressive loss: predict next token
            loss = criterion(output[:, :-1, :], batch[:, 1:, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        # Deliverable: Perplexity = exp(1/T * LTR)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1}/30 | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    # Save weights specifically for the Transformer
    save_path = '/content/music-generation-unsupervised/src/models/transformer_weights.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Transformer Weights Saved to {save_path}")

if __name__ == "__main__":
    train()