import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Ensure python can find your modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import MusicTokenizer

class TokenizedMusicDataset(Dataset):
    def __init__(self, token_data, seq_len=256):
        self.data = token_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # x is sequence t, y is sequence t+1
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # 1. Load the Tokenizer to get the exact vocab size
    tokenizer = MusicTokenizer()
    vocab_path = '/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl'
    tokenizer.load(vocab_path)
    vocab_size = tokenizer.vocab_size

    # 2. Load the Multi-Genre Token Array
    data_path = '/content/music-generation-unsupervised/data/processed/transformer_tokens.npy'
    print(f"Loading dataset from {data_path}...")
    token_data = np.load(data_path).astype(np.int64) # Convert uint16 to standard int for PyTorch
    
    # 3. Initialize Dataset and Dataloader
    seq_len = 256 # Context window for the transformer
    dataset = TokenizedMusicDataset(token_data, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 4. Initialize Model
    model = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size)
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    # Save weights
    os.makedirs('/content/music-generation-unsupervised/src/models/', exist_ok=True)
    torch.save(model.state_dict(), '/content/music-generation-unsupervised/src/models/transformer_weights.pt')
    print("Training complete. Deliverable 2 (Perplexity Report) data generated.")

if __name__ == "__main__":
    train_transformer()