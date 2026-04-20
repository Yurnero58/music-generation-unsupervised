import torch
import torch.optim as optim
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer import MusicTransformer
from models.reward_model import MusicRewardModel

def train_rlhf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your Task 3 Pretrained Model
    model = MusicTransformer().to(device)
    weights_path = '/content/music-generation-unsupervised/src/models/transformer_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Very low LR for fine-tuning
    reward_model = MusicRewardModel()
    
    print("Starting Task 4 (RLHF) Preference Tuning...")
    
    for iteration in range(50): # K steps
        model.train()
        
        # 1. Generate a sample: Xgen ~ p(X)
        # Using a shorter sequence for faster RL iterations
        seed = torch.zeros(1, 1, 88).to(device)
        log_probs = []
        current_seq = seed
        
        for _ in range(64):
            mask = model.generate_mask(current_seq.size(1)).to(device)
            probs = model(current_seq, mask)[:, -1:, :]
            
            # Action sampling for Policy Gradient
            m = torch.distributions.Bernoulli(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action).sum())
            current_seq = torch.cat([current_seq, action], dim=1)
        
        # 2. Collect Reward: r = HumanScore(Xgen)
        reward = reward_model.get_reward(current_seq.squeeze(0))
        
        # 3. Policy Gradient Update
        # J(theta) = E[r * log p(X)]
        loss = -torch.stack(log_probs).mean() * reward
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration} | Reward: {reward.item():.4f}")

    torch.save(model.state_dict(), '/content/music-generation-unsupervised/src/models/rlhf_weights.pt')
    print("RLHF Tuning Complete.")

if __name__ == "__main__":
    train_rlhf()