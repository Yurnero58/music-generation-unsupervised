import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import MusicVAE
from preprocessing.piano_roll import get_loader

def weighted_vae_loss(recon_x, x, mu, logvar, beta=0.0):
    # THE FIX: The matrix is 98% zeros. 
    # Standard BCE lets the model cheat by predicting all zeros.
    bce_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='none')
    
    # We apply a 20x penalty multiplier specifically where the true note is ON (1.0)
    # This forces the model to actually care about predicting notes.
    weight_matrix = torch.where(x == 1.0, 20.0, 1.0)
    weighted_bce = torch.sum(bce_loss * weight_matrix) / x.size(0)
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return weighted_bce + (beta * kl_loss)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicVAE(input_dim=352, hidden_dim=1024, latent_dim=256).to(device)
    # Slightly lower learning rate to handle the aggressive new loss multiplier
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    
    data_path = '/content/music-generation-unsupervised/data/processed/multi_track_lmd.npy'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Did the multi_parser finish?")
        return
        
    train_loader = get_loader(data_path, batch_size=32)
    
    print(f"Initiating ANTI-SILENCE Training on {device}...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        beta = 0.0 if epoch < 10 else 0.001 
        
        # THE FIX: Slowly remove Teacher Forcing so the model learns to walk on its own
        tf_ratio = max(0.2, 0.8 - (epoch * 0.02))
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(batch, teacher_forcing_ratio=tf_ratio)
            loss = weighted_vae_loss(recon, batch, mu, logvar, beta=beta)
            
            loss.backward()
            # Gradient clipping to prevent the weighted loss from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # DIAGNOSTIC CHECK: 
        # We print the highest probability in the batch. 
        # We want this to climb past 0.5! If it stays at 0.05, the model is failing.
        max_prob = recon.max().item()
        print(f"Epoch {epoch+1}/30 | Loss: {total_loss/len(train_loader):.2f} | TF: {tf_ratio:.2f} | Max Prob: {max_prob:.3f}")

    torch.save(model.state_dict(), '/content/music-generation-unsupervised/src/models/vae_weights.pt')
    print("Training Complete. Anti-Silence Weights Saved.")

if __name__ == "__main__":
    train()