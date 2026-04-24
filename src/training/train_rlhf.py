import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import MusicTokenizer
from src.models.reward_model import MusicRewardModel

def generate_sequence_for_rl(model, start_token_id, max_length=200, device="cuda"):
    """Generates a sequence without gradients to save memory during sampling."""
    model.eval() # Switches to evaluation mode (turns off Dropout)
    sequence = torch.tensor([[start_token_id]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(sequence)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat([sequence, next_token], dim=1)
            
    return sequence

def rl_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting RLHF on {device}...")

    # 1. Load Tokenizer (Stays in processed, does not get split)
    tokenizer = MusicTokenizer()
    tokenizer.load('/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl')
    vocab_size = tokenizer.vocab_size
    sos_id = tokenizer.token_to_id[tokenizer.sos_token]

    # 2. Load Pretrained Generator (Task 3)
    generator = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
    generator.load_state_dict(torch.load('/content/music-generation-unsupervised/src/models/transformer_weights.pt', map_location=device))
    
    # 3. Load Trained Reward Model
    reward_model = MusicRewardModel(vocab_size=vocab_size).to(device)
    reward_model.load_state_dict(torch.load('/content/music-generation-unsupervised/src/models/reward_model.pt', map_location=device))
    reward_model.eval() # RM is frozen during this step

    # Algorithm Step 1: Initialize Policy parameters
    optimizer = optim.Adam(generator.parameters(), lr=1e-5) 
    
    baseline_reward = 5.0 # A running average to stabilize gradients

    # Algorithm Step 2: for iteration = 1 to K
    K_iterations = 500
    
    pbar = tqdm(range(K_iterations), desc="RL Fine-tuning")
    for iteration in pbar:
        # Algorithm Step 3: Generate music samples X_gen ~ p_theta(X)
        seq = generate_sequence_for_rl(generator, sos_id, max_length=256, device=device)
        
        # Algorithm Step 4: Collect feedback score r(X_gen)
        with torch.no_grad():
            reward = reward_model(seq).item()
            
        # Update baseline (exponential moving average)
        baseline_reward = 0.9 * baseline_reward + 0.1 * reward
        advantage = reward - baseline_reward # Center the reward to reduce variance

        # Switch generator back to training mode so Dropout is active for the backward pass!
        generator.train() 
        
        # Algorithm Step 6: Policy Gradient Update
        optimizer.zero_grad()
        logits = generator(seq)
        
        # Shift logits and targets for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = seq[:, 1:].contiguous()
        
        # Calculate log probabilities of the exact sequence we generated
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
        
        # Objective: J(θ) = E[ r * log p(X) ]. Minimize the negative to maximize.
        loss = -1 * advantage * target_log_probs.sum()
        
        # Algorithm Step 7: Update generator parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer.step()
        
        pbar.set_postfix({"Reward": f"{reward:.2f}", "Advantage": f"{advantage:.2f}"})

    # Algorithm Step 9: Output RLHF-tuned model
    os.makedirs('/content/music-generation-unsupervised/src/models/', exist_ok=True)
    torch.save(generator.state_dict(), '/content/music-generation-unsupervised/src/models/transformer_rlhf_weights.pt')
    print("\nRLHF Tuning Complete.")

if __name__ == "__main__":
    rl_finetune()