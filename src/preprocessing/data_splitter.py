import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save():
    # 1. Setup Directories
    processed_dir = 'data/processed'
    output_dir = 'data/train_test_split'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Define the datasets to split
    datasets = {
        'classical_piano.npy': 'random',       # Task 1
        'multi_genre_lmd.npy': 'random',       # Task 2
        'transformer_tokens.npy': 'sequential' # Task 3 & 4
    }
    
    for filename, split_type in datasets.items():
        filepath = os.path.join(processed_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename} - File not found.")
            continue
            
        print(f"Loading {filename}...")
        data = np.load(filepath)
        
        if split_type == 'random':
            # For Tasks 1 & 2: Randomly shuffle the isolated piano-roll chunks
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            
        elif split_type == 'sequential':
            # For Task 3: Split sequentially to maintain temporal token order
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            
        # 3. Save the splits
        name, ext = os.path.splitext(filename)
        train_path = os.path.join(output_dir, f"{name}_train{ext}")
        test_path = os.path.join(output_dir, f"{name}_test{ext}")
        
        np.save(train_path, train_data)
        np.save(test_path, test_data)
        
        print(f"Success: {filename} split into Train ({len(train_data)}) and Test ({len(test_data)})")

if __name__ == "__main__":
    split_and_save()