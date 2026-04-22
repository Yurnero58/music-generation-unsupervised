import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from preprocessing.piano_roll import get_loader

def vae_loss(recon_x, x, mu, logvar, beta=0.0):
    x_clamped = torch.clamp(x, 0.0, 1.0)
    # Binary Cross Entropy with KL penalty
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x_clamped, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_loss) / x.size(0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIX: Initialize with 352 input dimensions and 1024 hidden dimensions
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # FIX: Point to the new multi-track numpy file
    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Did the multi_parser finish?")
        return
        
    train_loader = get_loader(data_path, batch_size=64)
    
    print(f"Starting Task 2: Multi-Instrument VAE Training on {device}...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        # Beta stays at 0 for half the training to ensure dense reconstruction
        beta = 0.0 if epoch < 15 else 0.001 
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/30 | Loss: {total_loss/len(train_loader):.4f} | Beta: {beta}")

    torch.save(model.state_dict(), '/content/music-generation-unsupervised/src/models/vae_weights.pt')
    print("Training Complete. Multi-Track VAE Weights Saved.")

if __name__ == "__main__":
    train()