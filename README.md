# Unsupervised Neural Network for Multi-Genre Music Generation
**CSE425/EEE474 Neural Networks — Spring 2026**

---

## Overview

This project implements four progressively advanced unsupervised generative models for music generation using the **Groove MIDI Dataset** (Jazz/Drums/Rhythm):

| Task | Model | Goal |
|------|-------|------|
| Task 1 | LSTM Autoencoder | Single-genre reconstruction & generation |
| Task 2 | β-VAE | Multi-genre diverse generation + latent interpolation |
| Task 3 | Causal Transformer | Long-horizon autoregressive generation |
| Task 4 | RLHF (Policy Gradient) | Human preference fine-tuning |

---

## Project Structure

```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/          # Downloaded Groove MIDI files
│   ├── processed/         # Parsed piano-roll arrays (pkl)
│   └── train_test_split/  # .npy arrays for train/val/test
├── notebooks/
│   ├── colab_pipeline.ipynb     # ← Main Colab notebook (run this)
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/
│   ├── config.py                # All hyperparameters
│   ├── preprocessing/
│   │   ├── midi_parser.py       # Download + parse Groove MIDI
│   │   ├── piano_roll.py        # Windowing, Dataset, DataLoaders
│   │   └── tokenizer.py         # Event tokenizer for Transformer
│   ├── models/
│   │   ├── autoencoder.py       # Task 1: LSTM Autoencoder
│   │   ├── vae.py               # Task 2: β-VAE
│   │   ├── transformer.py       # Task 3: Causal Transformer
│   │   └── diffusion.py         # Structural placeholder
│   ├── training/
│   │   ├── train_ae.py          # Task 1 training loop
│   │   ├── train_vae.py         # Task 2 training loop
│   │   ├── train_transformer.py # Task 3 training loop
│   │   └── train_rl.py          # Task 4 RLHF loop
│   ├── evaluation/
│   │   ├── metrics.py           # All spec metrics
│   │   ├── pitch_histogram.py   # Pitch analysis
│   │   └── rhythm_score.py      # Rhythm analysis
│   └── generation/
│       ├── generate_music.py    # Unified generation CLI
│       ├── midi_export.py       # Piano-roll → MIDI
│       └── sample_latent.py     # VAE interpolation + t-SNE
└── outputs/
    ├── generated_midis/         # All .mid files (Tasks 1–4)
    ├── plots/                   # Training curves + metric plots
    ├── survey_results/          # Human listening scores (JSON)
    └── checkpoints/             # Saved model weights
```

---

## Quick Start (Google Colab)

1. Open `notebooks/colab_pipeline.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Update `REPO_URL` in Cell 3 to your GitHub repo
4. Run all cells in order

The notebook will:
- Install dependencies
- Clone the repo
- Download the Groove MIDI dataset (~1 GB)
- Preprocess into piano-roll windows
- Train all 4 models sequentially
- Generate all MIDI samples
- Compute evaluation metrics and plots
- Zip outputs for download

---

## Dataset

**Groove MIDI Dataset** ([link](https://magenta.tensorflow.org/datasets/groove))
- ~13.6 hours of drumkit performances
- Styles: funk, rock, jazz, hip-hop, latin
- Format: MIDI files with expressive timing and velocity

Preprocessing pipeline:
1. Download and extract the dataset zip
2. Parse MIDI → binary piano-roll at 16 steps/bar
3. Segment into 64-step overlapping windows (stride = 32)
4. 80% train / 10% val / 10% test split

---

## Models

### Task 1 — LSTM Autoencoder
- **Encoder**: 2-layer LSTM → linear projection → latent `z` (dim=64)
- **Decoder**: latent `z` → 2-layer LSTM → sigmoid output
- **Loss**: MSE reconstruction `L_AE = ||X - X̂||²`
- **Deliverables**: 5 MIDI samples, loss curve

### Task 2 — β-VAE
- **Encoder**: Bidirectional LSTM → mean-pool → (μ, log σ²)
- **Reparameterisation**: `z = μ + σ ⊙ ε`, ε ~ N(0,I)
- **Decoder**: LSTM conditioned on z
- **Loss**: `L_VAE = L_recon + β * D_KL(q(z|X) || p(z))`  with β=4 (KL annealing)
- **Deliverables**: 8 MIDI samples, latent interpolation, t-SNE plot

### Task 3 — Causal Transformer
- **Architecture**: 6-layer decoder-only Transformer (Pre-LN), d_model=256, 8 heads
- **Input**: Event token sequences (note-on events tokenised from piano-roll)
- **Loss**: `L_TR = -Σ log p_θ(x_t | x_{<t})`
- **Metric**: Perplexity = exp(L_TR)
- **Generation**: Top-k sampling (k=40, temperature=1.0)
- **Deliverables**: 10 MIDI samples, perplexity report

### Task 4 — RLHF
- **Base**: Pretrained Task 3 Transformer
- **Reward**: Automatic proxy (pitch diversity + rhythm diversity + non-repetition + note density) or human scores from JSON
- **Update**: Policy gradient with baseline subtraction: `∇J = E[r * ∇ log p(X)]`
- **Deliverables**: 10 MIDI samples, before/after comparison

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Pitch Histogram Sim | `H(p,q) = Σ|pᵢ - qᵢ|` | Stylistic similarity |
| Rhythm Diversity | `#unique_durations / #total_notes` | Rhythmic variety |
| Repetition Ratio | `#repeated_patterns / #total_patterns` | Avoidance of loops |
| Human Score | Survey [1–5] | Subjective quality |

---

## Human Listening Survey (Task 4)

Collect scores from ≥10 participants. Save results to:
```
outputs/survey_results/human_scores.json
```
Format:
```json
{
  "0": 4,
  "1": 3,
  "2": 5,
  ...
}
```
Then run:
```bash
python src/training/train_rl.py --human_scores outputs/survey_results/human_scores.json
```

---

## Baseline Comparison

| Model | Rhythm Diversity | Human Score | Genre Control |
|-------|-----------------|-------------|---------------|
| Random Generator | Low | 1.1 | None |
| Markov Chain | Medium | 2.3 | Weak |
| Task 1: LSTM AE | Medium | ~3.1 | Single |
| Task 2: VAE | High | ~3.8 | Moderate |
| Task 3: Transformer | Very High | ~4.4 | Strong |
| Task 4: RLHF | Very High | ~4.8 | Strongest |

---

## Configuration

All hyperparameters are centralised in `src/config.py`. Key settings:

```python
SEQ_LEN        = 64     # time steps per window
BATCH_SIZE     = 64
AE_LATENT_DIM  = 64
VAE_LATENT_DIM = 128
VAE_BETA       = 4.0    # β-VAE KL weight
TR_D_MODEL     = 256
TR_NUM_LAYERS  = 6
RL_STEPS       = 200
```

---

## References

- Groove MIDI Dataset — Gillick et al., 2019
- β-VAE — Higgins et al., 2017
- Music Transformer — Huang et al., 2018
- RLHF — Christiano et al., 2017