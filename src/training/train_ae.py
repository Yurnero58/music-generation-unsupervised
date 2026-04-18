import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder
from preprocessing.piano_roll import get_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LSTMAutoencoder(input_dim=88, hidden_dim=256, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Restored to strictly follow Task 1 mathematical formulation
    criterion = nn.MSELoss() 
    
    npy_path = 'data/processed/classical_piano.npy'
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found. Run midi_parser.py first.")
        return

    train_loader = get_loader(npy_path, batch_size=128, shuffle=True)
    epochs = 20
    loss_history = []

    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.6f}")

    # Deliverables
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='MSE Reconstruction Loss')
    plt.title("Task 1: LSTM Autoencoder Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig('outputs/plots/ae_loss_curve.png')
    
    os.makedirs('src/models', exist_ok=True)
    torch.save(model.state_dict(), 'src/models/ae_weights.pt')
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train()