import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn

# Ensure PYTHONPATH is correct for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import MusicVAE
from preprocessing.piano_roll import get_loader

def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    # BCE requires target strictly in [0, 1]
    x_clamped = torch.clamp(x, 0.0, 1.0)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x_clamped, reduction='sum')
    
    # KL Divergence helps organize the multi-genre latent space
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return (recon_loss + beta * kl_loss) / x.size(0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for stability
    
    # Absolute path for Colab
    data_path = '/content/music-generation-unsupervised/data/processed/multi_genre_lmd.npy'
    train_loader = get_loader(data_path, batch_size=64)
    
    epochs = 40
    print(f"Restarting Task 2 Training (Rescue Mode) on {device}...")
    
    for epoch in range(epochs):
        model.train()
        # CYCLICAL ANNEALING: beta resets every 10 epochs to kick the model out of collapse
        beta = (epoch % 10) / 1000.0 if epoch < 30 else 0.01 
        
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f} | Beta: {beta:.4f}")

    save_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train()