import torch
import numpy as np

class MusicRewardModel:
    def __init__(self):
        pass

    def get_reward(self, sequence):
        """
        Simulates human preference by scoring the sequence.
        sequence shape: (seq_len, 88)
        """
        # Convert to numpy for easier heuristic calculation
        seq_np = sequence.cpu().numpy()
        
        # 1. Density Penalty (Fixes the "wall of noise" problem)
        # We want the model to learn silence. Target density is around 5-10% notes ON.
        density = np.mean(seq_np > 0.5)
        # Sharp penalty if density strays far from 5%
        density_reward = 1.0 - (abs(density - 0.05) * 5.0) 
        
        # 2. Rhythmic Variance (Fixes the "continuous note" drone problem)
        # Calculates how often a note changes state (ON to OFF, or OFF to ON)
        note_changes = np.sum(np.abs(np.diff(seq_np, axis=0)))
        max_possible_changes = seq_np.shape[0] * seq_np.shape[1]
        rhythm_reward = (note_changes / max_possible_changes) * 10.0
        
        # 3. Pitch Variety
        unique_notes = len(np.unique(np.where(seq_np > 0.5)[1]))
        variety_reward = unique_notes / 88.0
        
        # Total Reward (Bounded roughly between -1 and 2)
        total_reward = density_reward + rhythm_reward + (variety_reward * 0.5)
        
        # Return as a float tensor for the policy gradient
        return torch.tensor(total_reward, dtype=torch.float32)