import os
import sys
import numpy as np

# Adjust imports based on your exact file structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.tokenizer import MusicTokenizer

# Assuming you have a function to convert string events back to MIDI
from src.preprocessing.midi_parser_mg import tokens_to_midi 

def generate_random_baseline(num_compositions=10, max_length=1500):
    print("Generating Naive Random Baseline...")

    # 1. Load Tokenizer to get the exact vocab size
    tokenizer = MusicTokenizer()
    vocab_path = '/content/music-generation-unsupervised/data/processed/tokenizer_vocab.pkl'
    tokenizer.load(vocab_path)
    
    output_dir = '/content/music-generation-unsupervised/outputs/baseline_random/'
    os.makedirs(output_dir, exist_ok=True)
    
    sos_token_id = tokenizer.token_to_id[tokenizer.sos_token]

    for i in range(num_compositions):
        # 2. Pick random tokens between 1 and vocab_size
        # (We skip 0 if it is reserved for padding)
        random_ids = np.random.randint(1, tokenizer.vocab_size, size=(max_length,))
        
        # Force the first token to be the Start-Of-Sequence token 
        random_ids[0] = sos_token_id
        
        # 3. Decode IDs to String Events
        comp_ids = random_ids.tolist()
        string_events = tokenizer.decode(comp_ids)
        
        # 4. Save as MIDI
        out_mid_file = os.path.join(output_dir, f'random_{i+1}.mid')
        
        try:
            tokens_to_midi(string_events, out_mid_file)
            
            # Save the raw tokens too, so you can test them against your Reward Model!
            out_npy_file = os.path.join(output_dir, f'random_{i+1}.npy')
            np.save(out_npy_file, np.array(comp_ids, dtype=np.int64))
            
            print(f"Saved: {out_mid_file}")
            
        except Exception as e:
            # Random tokens often create impossible MIDI logic (like turning off a note that never started)
            print(f"Track {i+1} failed to parse into MIDI (Expected for random noise): {e}")

if __name__ == "__main__":
    generate_random_baseline()