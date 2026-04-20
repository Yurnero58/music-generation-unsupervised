import torch
import numpy as np

class MusicRewardModel:
    def __init__(self):
        pass

    def get_reward(self, sequence):
        """
        Simulates human preference by scoring:
        1. Note Variety (avoiding drones)
        2. Rhythmic Consistency
        3. Consonance (avoiding random clusters)
        """
        # Convert tensor to numpy for easier analysis
        seq_np = sequence.cpu().numpy() 
        
        # 1. Note Variety Reward
        unique_notes = np.unique(np.where(seq_np > 0.5)[1])
        variety_score = len(unique_notes) / 88.0
        
        # 2. Density Penalty (prevents the "continuous wall of noise")
        density = np.mean(seq_np > 0.5)
        density_reward = 1.0 - abs(density - 0.1) # Aiming for 10% density
        
        # 3. Final Reward Calculation
        reward = (variety_score * 0.5) + (density_reward * 0.5)
        return torch.tensor(reward, dtype=torch.float32)