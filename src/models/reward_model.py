import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- 1. The Reward Model Architecture (Unchanged) ---
class MusicRewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super(MusicRewardModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        final_state = hidden[-1] 
        val = self.relu(self.fc1(final_state))
        score = self.fc_out(val)
        return score.squeeze(-1)


# --- 2. NEW: Real Human Feedback Dataset Loader ---
class HumanFeedbackDataset(Dataset):
    def __init__(self, csv_file, npy_dir, max_len=1500, pad_token_id=0):
        self.data_frame = pd.read_csv(csv_file)
        self.npy_dir = npy_dir
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # Swap .mid extension for .npy to get the token array
        npy_filename = row['file_name'].replace('.mid', '.npy')
        npy_path = os.path.join(self.npy_dir, npy_filename)

        # Load the raw tokens generated in Task 3
        token_array = np.load(npy_path)

        # Pad or truncate the sequence so PyTorch can batch them
        if len(token_array) > self.max_len:
            token_array = token_array[:self.max_len]
        else:
            pad_len = self.max_len - len(token_array)
            token_array = np.pad(token_array, (0, pad_len), 'constant', constant_values=self.pad_token_id)

        # The human score (1 to 10)
        score = row['score']

        return torch.tensor(token_array, dtype=torch.long), torch.tensor(score, dtype=torch.float32)


# --- 3. Updated Training Loop ---
def train_reward_model(vocab_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Reward Model on {device}...")

    # Paths to your feedback data
    csv_path = '/content/music-generation-unsupervised/data/processed/human_feedback.csv'
    npy_dir = '/content/music-generation-unsupervised/outputs/transformer/'
    
    # Initialize Dataset and DataLoader
    # Batch size is small (2) because you only have ~10 survey responses
    dataset = HumanFeedbackDataset(csv_file=csv_path, npy_dir=npy_dir, max_len=1500)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    rm = MusicRewardModel(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(rm.parameters(), lr=0.001)
    criterion = nn.MSELoss() 

    # We run more epochs (e.g., 20) because the dataset is very small
    epochs = 20
    for epoch in range(epochs):
        rm.train()
        total_loss = 0
        
        for sequences, human_scores in dataloader:
            sequences, human_scores = sequences.to(device), human_scores.to(device)
            
            optimizer.zero_grad()
            predicted_scores = rm(sequences)
            
            loss = criterion(predicted_scores, human_scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"RM Epoch {epoch+1}/{epochs} | Avg MSE Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs('/content/music-generation-unsupervised/src/models/', exist_ok=True)
    torch.save(rm.state_dict(), '/content/music-generation-unsupervised/src/models/reward_model.pt')
    print("Reward Model Saved and ready for RLHF.")

if __name__ == "__main__":
    # Ensure this matches the vocab_size from your Tokenizer (around 392-400)
    train_reward_model(vocab_size=400)