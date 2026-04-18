import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import LSTMAutoencoder
from preprocessing.piano_roll import get_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Clean Slate Training on: {device}")
    
    model = LSTMAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Strict adherence to task requirements
    
    npy_path = 'data/processed/classical_piano.npy'
    train_loader = get_loader(npy_path, batch_size=128, shuffle=True)
    
    epochs = 50
    loss_history = []

    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.6f}")

    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('src/models', exist_ok=True)
    
    plt.plot(loss_history)
    plt.title("MSE Reconstruction Loss")
    plt.savefig('outputs/plots/ae_loss_curve.png')
    torch.save(model.state_dict(), 'src/models/ae_weights.pt')
    print("Training complete. Weights secured.")

if __name__ == "__main__":
    train()