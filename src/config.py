# =============================================================================
# config.py — Central configuration for all tasks
# CSE425/EEE474 Neural Networks — Multi-Genre Music Generation
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(ROOT_DIR, "data", "raw_midi")
DATA_PROCESSED  = os.path.join(ROOT_DIR, "data", "processed")
DATA_SPLIT      = os.path.join(ROOT_DIR, "data", "train_test_split")
OUTPUT_MIDI     = os.path.join(ROOT_DIR, "outputs", "generated_midis")
OUTPUT_PLOTS    = os.path.join(ROOT_DIR, "outputs", "plots")
OUTPUT_SURVEY   = os.path.join(ROOT_DIR, "outputs", "survey_results")

for _p in [DATA_RAW, DATA_PROCESSED, DATA_SPLIT,
           OUTPUT_MIDI, OUTPUT_PLOTS, OUTPUT_SURVEY]:
    os.makedirs(_p, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset — Groove MIDI
# ---------------------------------------------------------------------------
GROOVE_URL      = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
DATASET_NAME    = "groove"
GENRES          = ["funk", "rock", "jazz", "hiphop", "latin"]   # Groove sub-styles

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
STEPS_PER_BAR   = 16          # time resolution
SEQ_LEN         = 64          # tokens per training window
PITCH_RANGE     = (36, 84)    # MIDI notes kept (kick → hi-hat range)
N_PITCHES       = PITCH_RANGE[1] - PITCH_RANGE[0]   # 48
VELOCITY_BINS   = 32
TEST_SPLIT      = 0.10
VAL_SPLIT       = 0.10

# ---------------------------------------------------------------------------
# Shared training
# ---------------------------------------------------------------------------
BATCH_SIZE      = 64
NUM_EPOCHS_AE   = 50
NUM_EPOCHS_VAE  = 60
NUM_EPOCHS_TR   = 40
LEARNING_RATE   = 1e-3
DEVICE          = "cuda"       # Colab GPU

# ---------------------------------------------------------------------------
# Task 1 — LSTM Autoencoder
# ---------------------------------------------------------------------------
AE_HIDDEN_DIM   = 256
AE_LATENT_DIM   = 64
AE_NUM_LAYERS   = 2
AE_DROPOUT      = 0.3

# ---------------------------------------------------------------------------
# Task 2 — VAE
# ---------------------------------------------------------------------------
VAE_HIDDEN_DIM  = 512
VAE_LATENT_DIM  = 128
VAE_NUM_LAYERS  = 2
VAE_BETA        = 4.0          # β-VAE weight for KL term
VAE_DROPOUT     = 0.3

# ---------------------------------------------------------------------------
# Task 3 — Transformer
# ---------------------------------------------------------------------------
TR_VOCAB_SIZE   = N_PITCHES + VELOCITY_BINS + 4   # notes + vel + special tokens
TR_D_MODEL      = 256
TR_NHEAD        = 8
TR_NUM_LAYERS   = 6
TR_DIM_FF       = 1024
TR_DROPOUT      = 0.1
TR_MAX_SEQ_LEN  = 512
TR_GEN_LEN      = 512          # tokens to generate

# Special token IDs
PAD_TOKEN       = 0
BOS_TOKEN       = 1
EOS_TOKEN       = 2
MASK_TOKEN      = 3

# ---------------------------------------------------------------------------
# Task 4 — RLHF
# ---------------------------------------------------------------------------
RL_STEPS        = 200
RL_LR           = 1e-4
RL_SAMPLES_PER_STEP = 8
REWARD_WEIGHTS  = {            # for automatic reward proxy
    "pitch_diversity": 0.3,
    "rhythm_diversity": 0.3,
    "no_repetition":   0.2,
    "note_density":    0.2,
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
N_SAMPLES_AE    = 5
N_SAMPLES_VAE   = 8
N_SAMPLES_TR    = 10
N_SAMPLES_RL    = 10
TEMPERATURE     = 1.0
TOP_K           = 40