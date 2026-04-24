# config.py

# Task 3 & 4: Transformer Model Configuration
TRANSFORMER_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "seq_len": 256,
    "dropout": 0.1
}

# Task 1 & 2: Autoencoder / VAE Configuration
AE_VAE_CONFIG = {
    "input_dim": 88,
    "hidden_dim": 512,  # Used 512 for VAE, 256 for AE
    "latent_dim": 256   # Used 256 for VAE, 128 for AE
}

# Training Configuration
TRAINING_CONFIG = {
    "transformer_lr": 0.0005,
    "vae_lr": 0.001,
    "rlhf_lr": 1e-5,
    "batch_size": 64,  # Used across most tasks
    "num_epochs_transformer": 20,
    "num_epochs_vae": 30
}

# Data Configuration
DATA_CONFIG = {
    "raw_dir": "data/raw_midi",
    "processed_dir": "data/processed",
    "split_dir": "data/train_test_split",
    "vocab_path": "data/processed/tokenizer_vocab.pkl"
}