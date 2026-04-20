import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from models.transformer import MusicTransformer
from preprocessing.piano_roll import get_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss() # For multi-label note prediction

    # Using MAESTRO for long-horizon coherence
    data_path = 'data/processed/maestro_v3.npy'
    train_loader = get_loader(data_path, batch_size=32)

    print(f"Starting Task 3 (Transformer) Training on {device}...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            # Autoregressive setup: x is input, y is shifted by one
            optimizer.zero_grad()
            mask = model.generate_mask(batch.size(1)).to(device)
            output = model(batch, mask)
            
            loss = criterion(output[:, :-1, :], batch[:, 1:, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        # Calculate Perplexity deliverable
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    torch.save(model.state_dict(), 'src/models/transformer_weights.pt')

if __name__ == "__main__":
    train()