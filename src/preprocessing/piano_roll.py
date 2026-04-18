import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MusicDataset(Dataset):
    """
    Custom Dataset for loading large piano-roll files without crashing RAM.
    """
    def __init__(self, npy_file):
        # mmap_mode='r' keeps the file on disk and loads slices on demand.
        self.data = np.load(npy_file, mmap_mode='r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract sequence and explicitly copy to memory.
        # This prevents PyTorch warnings about non-writable memory-mapped arrays.
        sequence = np.copy(self.data[idx])
        sample = torch.from_numpy(sequence).float()
        return sample

def get_loader(npy_file, batch_size=128, shuffle=True, num_workers=0):
    """
    Returns a DataLoader optimized for GPU training as per Algorithm 1 requirements.
    """
    dataset = MusicDataset(npy_file)
    
    # Check if a GPU is available to safely enable pin_memory
    use_pin_memory = torch.cuda.is_available()
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory, # Accelerates CPU to GPU data transfers
        drop_last=True             # Prevents batch-size mismatch in LSTM hidden states
    )