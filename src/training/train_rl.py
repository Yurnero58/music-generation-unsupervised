# =============================================================================
# src/training/train_rl.py
# Task 4: RLHF — Policy Gradient fine-tuning for music preference optimisation
# Algorithm 4 from the project spec
# Objective: max_theta E[r(X_gen)]
# Update:    theta ← theta + eta * E[r * grad_theta log p_theta(X)]
# =============================================================================

import os, sys, json, time
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (DEVICE, RL_STEPS, RL_LR, RL_SAMPLES_PER_STEP,
                    OUTPUT_PLOTS, REWARD_WEIGHTS,
                    TR_VOCAB_SIZE, TR_D_MODEL, TR_NHEAD, TR_NUM_LAYERS,
                    TR_DIM_FF, TR_DROPOUT, TR_MAX_SEQ_LEN,
                    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
                    NOTE_OFFSET, N_PITCHES)
from models.transformer import MusicTransformer

NOTE_OFFSET = 4   # PAD=0, BOS=1, EOS=2, MASK=3, then notes
from evaluation.metrics import pitch_histogram_entropy, rhythm_diversity, repetition_ratio

ROOT     = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TR_SEQ = 128


# ---------------------------------------------------------------------------
# Automatic proxy reward (used when human scores aren't available live)
# ---------------------------------------------------------------------------
def compute_auto_reward(token_seq: list[int]) -> float:
    """
    Proxy reward combining musical quality heuristics.
    r ∈ [0, 1]  (higher = better)
    """
    # Extract note tokens
    notes = [t - NOTE_OFFSET for t in token_seq
             if NOTE_OFFSET <= t < NOTE_OFFSET + N_PITCHES]
    if len(notes) < 4:
        return 0.0

    # 1. Pitch diversity (unique pitches / total notes)
    pitch_div = len(set(notes)) / max(1, len(notes))

    # 2. Rhythm diversity proxy: variance of inter-onset gaps
    # (using token position as a proxy for time)
    positions = [i for i, t in enumerate(token_seq)
                 if NOTE_OFFSET <= t < NOTE_OFFSET + N_PITCHES]
    gaps = np.diff(positions) if len(positions) > 1 else [1]
    rhythm_div = min(1.0, float(np.std(gaps)) / 5.0)

    # 3. Non-repetition: penalise identical consecutive notes
    repeats = sum(1 for i in range(1, len(notes)) if notes[i] == notes[i-1])
    no_rep  = 1.0 - repeats / max(1, len(notes) - 1)

    # 4. Note density (neither too sparse nor too dense)
    density = len(notes) / max(1, len(token_seq))
    note_density_score = 1.0 - abs(density - 0.4) / 0.4   # peak at 40% density
    note_density_score = max(0.0, note_density_score)

    w = REWARD_WEIGHTS
    r = (w["pitch_diversity"] * pitch_div
         + w["rhythm_diversity"] * rhythm_div
         + w["no_repetition"]   * no_rep
         + w["note_density"]    * note_density_score)
    return float(r)


# ---------------------------------------------------------------------------
# Differentiable log-probability of a sequence under the model
# ---------------------------------------------------------------------------
def sequence_log_prob(model: MusicTransformer,
                      tokens: torch.Tensor,
                      device) -> torch.Tensor:
    """
    Compute sum_t log p_theta(x_t | x_{<t}) for a token sequence.
    tokens: (T,) long tensor (includes BOS, excludes final EOS for input)
    """
    inp = tokens[:-1].unsqueeze(0).to(device)   # (1, T-1)
    tgt = tokens[1:].unsqueeze(0).to(device)    # (1, T-1)
    logits = model(inp)                          # (1, T-1, V)
    log_p  = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather log prob of actual tokens
    tok_log_p = log_p[0, torch.arange(tgt.size(1)), tgt[0]]  # (T-1,)
    return tok_log_p.sum()


# ---------------------------------------------------------------------------
# Main RL loop
# ---------------------------------------------------------------------------
def main(human_scores_path: str = None):
    """
    human_scores_path: Optional JSON file mapping sample indices → [1-5] scores
                       collected from listening survey. If None, uses auto reward.
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train_rl] Device: {device}")

    # Load pretrained Transformer
    ckpt_path = CKPT_DIR / "transformer_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Run train_transformer.py first — checkpoint not found: {ckpt_path}")

    model = MusicTransformer(
        vocab_size=TR_VOCAB_SIZE, d_model=TR_D_MODEL, nhead=TR_NHEAD,
        num_layers=TR_NUM_LAYERS, dim_feedforward=TR_DIM_FF,
        dropout=0.0, max_seq_len=TR_SEQ
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("[train_rl] Loaded pretrained Transformer.")

    optimiser = optim.Adam(model.parameters(), lr=RL_LR)

    # Human scores override (optional)
    human_scores = None
    if human_scores_path and os.path.isfile(human_scores_path):
        with open(human_scores_path) as f:
            raw = json.load(f)
        # Normalise [1,5] → [0,1]
        human_scores = {int(k): (v - 1) / 4.0 for k, v in raw.items()}
        print(f"[train_rl] Loaded {len(human_scores)} human scores.")

    history = {"step": [], "mean_reward": [], "mean_log_prob": []}
    best_reward = -float("inf")

    for step in range(1, RL_STEPS + 1):
        model.train()
        batch_rewards   = []
        batch_log_probs = []

        for s in range(RL_SAMPLES_PER_STEP):
            # Generate a sample (no_grad for the sequence itself)
            with torch.no_grad():
                token_list = model.generate(max_new=TR_SEQ, device=str(device))
            tokens = torch.tensor(token_list, dtype=torch.long)

            # Reward
            if human_scores is not None and (step * RL_SAMPLES_PER_STEP + s) in human_scores:
                r = human_scores[step * RL_SAMPLES_PER_STEP + s]
            else:
                r = compute_auto_reward(token_list)

            # Log-probability (with grad)
            log_p = sequence_log_prob(model, tokens, device)

            batch_rewards.append(r)
            batch_log_probs.append(log_p)

        # Policy gradient: J(theta) = E[r * log p(X)]
        rewards_t  = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        # Baseline (mean reward subtraction for variance reduction)
        baseline   = rewards_t.mean()
        advantages = rewards_t - baseline

        log_probs_t = torch.stack(batch_log_probs)
        loss = -(advantages * log_probs_t).mean()   # gradient ascent via negation

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()

        mean_r = float(rewards_t.mean())
        history["step"].append(step)
        history["mean_reward"].append(mean_r)
        history["mean_log_prob"].append(float(log_probs_t.mean().item()))

        if step % 10 == 0:
            print(f"Step {step:04d}/{RL_STEPS}  "
                  f"mean_reward={mean_r:.4f}  loss={loss.item():.4f}")

        if mean_r > best_reward:
            best_reward = mean_r
            torch.save(model.state_dict(), CKPT_DIR / "rl_best.pt")

    torch.save(model.state_dict(), CKPT_DIR / "rl_final.pt")
    with open(CKPT_DIR / "rl_history.json", "w") as f:
        json.dump(history, f)

    # ---- Plot ----
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["step"], history["mean_reward"])
    plt.xlabel("RL Step"); plt.ylabel("Mean Reward"); plt.title("Task 4 — RLHF Reward")

    plt.subplot(1, 2, 2)
    plt.plot(history["step"], history["mean_log_prob"])
    plt.xlabel("RL Step"); plt.ylabel("Mean Log-Prob")
    plt.title("Task 4 — Policy Log-Probability")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, "task4_rl_training.png"), dpi=150)
    plt.close()
    print(f"[train_rl] Done. Best mean reward: {best_reward:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_scores", type=str, default=None,
                        help="Path to JSON with human listening scores")
    args = parser.parse_args()
    main(human_scores_path=args.human_scores)