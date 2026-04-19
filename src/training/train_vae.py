import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import MusicVAE
from preprocessing.piano_roll import get_loader

def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    # Clamp target to [0, 1] for BCE stability
    x_clamped = torch.clamp(x, 0.0, 1.0)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x_clamped, reduction='sum')
    
    # KL-Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_loss) / x.size(0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicVAE(input_dim=88, hidden_dim=512, latent_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    data_path = '/content/music-generation-unsupervised/data/processed/multi_genre_lmd.npy'
    train_loader = get_loader(data_path, batch_size=64)
    
    print(f"Starting Rescue Training on {device}...")
    for epoch in range(25):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/25 | Avg Loss: {total_loss/len(train_loader):.4f}")

    save_path = '/content/music-generation-unsupervised/src/models/vae_weights.pt'
    torch.save(model.state_dict(), save_path)
    print("Training Complete.")

if __name__ == "__main__":
    train()