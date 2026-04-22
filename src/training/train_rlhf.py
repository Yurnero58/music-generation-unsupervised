import torch
import torch.optim as optim
import os
import sys

# Ensure PYTHONPATH is correct for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer import MusicTransformer
from models.reward_model import MusicRewardModel

def train_rlhf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1: Initialize policy parameters theta
    model = MusicTransformer(input_dim=88, d_model=256, nhead=8).to(device)
    
    weights_path = '/content/music-generation-unsupervised/src/models/transformer_weights.pt'
    if not os.path.exists(weights_path):
        print("Error: Task 3 weights not found. You must complete Task 3 first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Very low learning rate to prevent catastrophic forgetting of Task 3 musical knowledge
    optimizer = optim.Adam(model.parameters(), lr=5e-6) 
    reward_model = MusicRewardModel()
    
    print(f"Starting Task 4 (RLHF) Optimization on {device}...")
    
    epochs = 150 # K iterations
    seq_len = 64 # Short sequences for faster RL iterations
    
    for iteration in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 3: Generate music sample: Xgen ~ p(X)
        generated_seq = [torch.zeros(1, 1, 88).to(device)]
        log_probs = []
        
        for t in range(seq_len):
            input_tensor = torch.cat(generated_seq, dim=1)
            mask = model.generate_mask(input_tensor.size(1)).to(device)
            
            # Get probabilities for next note
            probs = model(input_tensor, mask)[:, -1:, :]
            
            # Sample action and calculate log probability for REINFORCE
            m = torch.distributions.Bernoulli(probs)
            action = m.sample()
            
            # Sum log probs across the 88 keys
            log_prob = m.log_prob(action).sum(dim=-1) 
            log_probs.append(log_prob)
            
            generated_seq.append(action)
            
        # Compile sequence for scoring
        full_sequence = torch.cat(generated_seq, dim=1).squeeze(0)
        
        # 4: Collect human preference score: r = HumanScore(Xgen)
        reward = reward_model.get_reward(full_sequence).to(device)
        
        # 5 & 6: Compute expected reward objective and Policy Gradient
        # J(theta) = E[r * log p(X)] --> Loss = -Reward * sum(log_probs)
        loss = -reward * torch.stack(log_probs).sum()
        
        # 7: Update generator parameters
        loss.backward()
        
        # Gradient clipping to prevent exploding updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{epochs} | Reward: {reward.item():.4f} | Loss: {loss.item():.4f}")

    # 9: Output RLHF-tuned music generation model
    save_path = '/content/music-generation-unsupervised/src/models/rlhf_weights.pt'
    torch.save(model.state_dict(), save_path)
    print(f"RLHF Tuning Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train_rlhf()