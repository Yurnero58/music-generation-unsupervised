import os
import pickle
import numpy as np

class MusicTokenizer:
    """
    A tokenizer for converting MIDI events or piano roll sequences into 
    discrete integer tokens for Transformer-based music generation.
    """
    def __init__(self, num_velocities=32, max_time_shift=100):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"  # Start of Sequence
        self.eos_token = "<EOS>"  # End of Sequence
        self.unk_token = "<UNK>"  # Unknown token

        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # 1. Pitch Tokens (Standard MIDI 0-127, though piano is usually 21-108)
        self.vocab.extend([f"NOTE_ON_{i}" for i in range(128)])
        self.vocab.extend([f"NOTE_OFF_{i}" for i in range(128)])
        
        # 2. Time Shift Tokens (Quantized time steps between notes)
        self.vocab.extend([f"TIME_SHIFT_{i}" for i in range(1, max_time_shift + 1)])
        
        # 3. Velocity Tokens (Loudness, quantized into bins)
        self.vocab.extend([f"VELOCITY_{i}" for i in range(num_velocities)])

        # Create mapping dictionaries
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        """Returns the total number of tokens in the vocabulary."""
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    def encode(self, token_sequence):
        """
        Converts a list of string tokens into a list of integer IDs.
        Adds <SOS> at the beginning and <EOS> at the end.
        """
        encoded = [self.token_to_id[self.sos_token]]
        for token in token_sequence:
            encoded.append(self.token_to_id.get(token, self.token_to_id[self.unk_token]))
        encoded.append(self.token_to_id[self.eos_token])
        return encoded

    def decode(self, id_sequence, remove_special_tokens=True):
        """
        Converts a list of integer IDs back into a list of string tokens.
        """
        decoded = []
        for idx in id_sequence:
            # Handle potential tensor inputs
            if hasattr(idx, 'item'):
                idx = idx.item()
                
            token = self.id_to_token.get(idx, self.unk_token)
            
            if remove_special_tokens and token in [self.pad_token, self.sos_token, self.eos_token]:
                continue
                
            decoded.append(token)
        return decoded

    def save(self, filepath):
        """Saves the vocabulary mapping to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.token_to_id, f)
        print(f"Tokenizer vocabulary saved to {filepath}")

    def load(self, filepath):
        """Loads a vocabulary mapping from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
            
        with open(filepath, 'rb') as f:
            self.token_to_id = pickle.load(f)
            self.id_to_token = {i: t for t, i in self.token_to_id.items()}
            self.vocab = list(self.token_to_id.keys())
        print("Tokenizer vocabulary loaded successfully.")

# --- Helper function to test the tokenizer if run directly ---
if __name__ == "__main__":
    tokenizer = MusicTokenizer()
    print(f"Initialized Tokenizer with Vocab Size: {tokenizer.vocab_size}")
    
    # Mock data to simulate extracting events from a MIDI file
    sample_events = ["NOTE_ON_60", "VELOCITY_16", "TIME_SHIFT_10", "NOTE_OFF_60"]
    
    encoded_ids = tokenizer.encode(sample_events)
    print(f"\nOriginal Events: {sample_events}")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded Events: {tokenizer.decode(encoded_ids)}")