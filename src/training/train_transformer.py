import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # <--- Added for a progress bar!

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import MusicTokenizer

class TokenizedMusicDataset(Dataset):
    def __init__(self, token_data, seq_len=256):
        self.data = token_data
        self.seq_len = seq_len
        # Chunk the data instead of sliding by 1
        self.num_samples = (len(self.data) - 1) // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Jump forward by seq_len chunks
        start_idx = idx * self.seq_len
        
        x = self.data[start_idx : start_idx + self.seq_len]
        y = self.data[start_idx + 1 : start_idx + self.seq_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    tokenizer = MusicTokenizer()
    vocab_path = '/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl'
    tokenizer.load(vocab_path)
    vocab_size = tokenizer.vocab_size

    # UPDATED PATH: Pointing to the new train split
    data_path = '/content/music-generation-unsupervised/data/train_test_split/transformer_tokens_train.npy'
    print(f"Loading dataset from {data_path}...")
    token_data = np.load(data_path).astype(np.int64)
    
    seq_len = 256
    dataset = TokenizedMusicDataset(token_data, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Total sequences: {len(dataset)} | Total batches per epoch: {len(dataloader)}")

    model = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Wrapped the dataloader in tqdm for a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update the progress bar with the current loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)
        print(f"End of Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}\n")

    os.makedirs('/content/music-generation-unsupervised/src/models/', exist_ok=True)
    torch.save(model.state_dict(), '/content/music-generation-unsupervised/src/models/transformer_weights.pt')
    print("Training complete.")

if __name__ == "__main__":
    train_transformer()