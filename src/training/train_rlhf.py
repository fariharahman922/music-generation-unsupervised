import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Project root setup
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.transformer import MusicTransformer
from src.generation.midi_export import save_piano_roll_as_midi
from src.preprocessing.tokenizer import midi_to_token_ids


# =========================================================
# Paths
# =========================================================
VOCAB_PATH = PROJECT_ROOT / "data" / "train_test_split" / "lmd_tokenized" / "token_vocab.json"
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "outputs" / "transformer_best.pt"

SURVEY_DIR = PROJECT_ROOT / "outputs" / "survey_results"
BEFORE_RL_DIR = SURVEY_DIR / "task4_before_rl"
AFTER_RL_DIR = SURVEY_DIR / "task4_after_rl"

REWARD_CSV = SURVEY_DIR / "task4_reward_scores.csv"
AFTER_SURVEY_TEMPLATE = SURVEY_DIR / "task4_after_rl_survey_template.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

RLHF_MODEL_PATH = OUTPUT_DIR / "task4_rlhf_best.pt"
RLHF_HISTORY_PATH = OUTPUT_DIR / "task4_rlhf_history.json"
RLHF_PLOT_PATH = PLOTS_DIR / "task4_rlhf_training_curve.png"
RLHF_SUMMARY_PATH = OUTPUT_DIR / "task4_rlhf_summary.json"


# =========================================================
# Token / MIDI settings
# =========================================================
LOWEST_PITCH = 21
HIGHEST_PITCH = 108
N_PITCHES = HIGHEST_PITCH - LOWEST_PITCH + 1

MAX_SEQ_LEN = 256
MAX_NEW_TOKENS = 1024
NUM_AFTER_RL_SAMPLES = 10

TEMPERATURE = 1.0
TOP_K = 50

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

BATCH_SIZE = 4
RL_STEPS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

USE_MIXED_PRECISION = True
SEED = 42


# =========================================================
# Reproducibility / device utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch_for_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        return device
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        return device


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    AFTER_RL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Vocab / model
# =========================================================
def load_vocab():
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Could not find vocab file: {VOCAB_PATH}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    token_to_id = vocab_data["token_to_id"]
    id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}

    pad_id = vocab_data["special_tokens"]["PAD_ID"]
    bos_id = vocab_data["special_tokens"]["BOS_ID"]
    eos_id = vocab_data["special_tokens"]["EOS_ID"]

    vocab_size = len(token_to_id)

    print("Vocab size:", vocab_size)
    print("PAD_ID:", pad_id, "| BOS_ID:", bos_id, "| EOS_ID:", eos_id)

    return token_to_id, id_to_token, pad_id, bos_id, eos_id, vocab_size


def build_model(vocab_size, device):
    model = MusicTransformer(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_genres=None
    ).to(device)

    if not PRETRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find pretrained Transformer model: {PRETRAINED_MODEL_PATH}")

    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded pretrained Transformer from:", PRETRAINED_MODEL_PATH)

    return model


# =========================================================
# Reward dataset: first human-feedback batch
# =========================================================
def load_reward_scores():
    if not REWARD_CSV.exists():
        raise FileNotFoundError(f"Could not find reward CSV: {REWARD_CSV}")

    reward_df = pd.read_csv(REWARD_CSV)

    required_cols = {"sample_id", "mean_human_score"}
    if not required_cols.issubset(set(reward_df.columns)):
        raise ValueError(
            f"Reward CSV must contain columns {required_cols}, "
            f"but found {list(reward_df.columns)}"
        )

    print("Loaded reward scores from:", REWARD_CSV)
    return reward_df


def pad_or_truncate(token_ids, seq_len, pad_id):
    token_ids = token_ids[:seq_len]
    if len(token_ids) < seq_len:
        token_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
    return token_ids


def build_first_feedback_batch_dataset(pad_id):
    """
    This uses the FIRST rated batch:
      before_rl_01 ... before_rl_10
    These were generated from the pretrained Transformer and rated by humans.
    """
    reward_df = load_reward_scores()

    sequences = []
    rewards = []
    sample_ids = []

    for _, row in reward_df.iterrows():
        sample_id = row["sample_id"]
        reward = float(row["mean_human_score"])

        midi_path = BEFORE_RL_DIR / f"{sample_id}.mid"
        if not midi_path.exists():
            print(f"Skipping missing rated MIDI: {midi_path}")
            continue

        try:
            token_ids = midi_to_token_ids(str(midi_path))
            token_ids = pad_or_truncate(token_ids, MAX_SEQ_LEN, pad_id)

            sequences.append(token_ids)
            rewards.append(reward)
            sample_ids.append(sample_id)

        except Exception as e:
            print(f"Skipping {midi_path} due to tokenization error: {e}")

    if len(sequences) == 0:
        raise ValueError("No valid RLHF training sequences could be built.")

    X = np.array(sequences, dtype=np.int64)
    y = np.array(rewards, dtype=np.float32)

    reward_mean = y.mean()
    reward_std = y.std() if y.std() > 0 else 1.0
    y_norm = (y - reward_mean) / reward_std

    print("Built first RL feedback dataset.")
    print("Num rated sequences:", len(X))
    print("Raw reward mean:", reward_mean)
    print("Raw reward std:", reward_std)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_norm.astype(np.float32))
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader, sample_ids, y, y_norm, reward_mean, reward_std


# =========================================================
# RL objective
# =========================================================
def compute_sequence_logprob(logits, target_tokens, pad_id):
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)

    valid_mask = (target_tokens != pad_id).float()
    seq_logprob = (target_log_probs * valid_mask).sum(dim=1)

    return seq_logprob


def policy_gradient_loss(model, tokens, rewards, pad_id):
    """
    Practical first-round policy-gradient style update.

    The rated before-RL samples were generated from the current
    pretrained policy. We now increase the likelihood of high-reward
    sequences and decrease the likelihood of low-reward ones.

    Loss = - E[ reward * log p_theta(X) ]
    """
    input_tokens = tokens[:, :-1]
    target_tokens = tokens[:, 1:]

    logits = model(input_tokens)
    seq_logprob = compute_sequence_logprob(logits, target_tokens, pad_id)

    loss = -(rewards * seq_logprob).mean()

    avg_seq_logprob = seq_logprob.mean().item()
    avg_reward = rewards.mean().item()

    return loss, avg_seq_logprob, avg_reward


# =========================================================
# RL training loop
# =========================================================
def train_one_step(model, dataloader, optimizer, device, scaler, use_amp, pad_id):
    model.train()

    total_loss_sum = 0.0
    total_seq_logprob_sum = 0.0
    total_reward_sum = 0.0
    total_batches = 0

    for batch in dataloader:
        tokens = batch[0].to(device)
        rewards = batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss, avg_seq_logprob, avg_reward = policy_gradient_loss(
                model, tokens, rewards, pad_id
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss_sum += loss.item()
        total_seq_logprob_sum += avg_seq_logprob
        total_reward_sum += avg_reward
        total_batches += 1

    step_loss = total_loss_sum / total_batches
    step_seq_logprob = total_seq_logprob_sum / total_batches
    step_reward = total_reward_sum / total_batches

    return step_loss, step_seq_logprob, step_reward


def plot_training_curve(history, save_path):
    steps = range(1, len(history["rlhf_loss"]) + 1)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(steps, history["rlhf_loss"], label="RLHF Loss")
    plt.xlabel("RL Step")
    plt.ylabel("Loss")
    plt.title("Task 4 RLHF First-Round Update")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(steps, history["avg_seq_logprob"], label="Avg Sequence LogProb")
    plt.plot(steps, history["avg_reward"], label="Avg Normalized Reward")
    plt.xlabel("RL Step")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Generation of AFTER-RL survey batch
# =========================================================
def token_ids_to_piano_roll(token_ids, id_to_token):
    active_notes = set()
    frames = []

    def current_frame():
        frame = np.zeros(N_PITCHES, dtype=np.uint8)
        for pitch in active_notes:
            if LOWEST_PITCH <= pitch <= HIGHEST_PITCH:
                frame[pitch - LOWEST_PITCH] = 1
        return frame

    for token_id in token_ids:
        token_name = id_to_token.get(int(token_id), None)
        if token_name is None:
            continue

        if token_name in {"PAD", "BOS"}:
            continue

        if token_name == "EOS":
            break

        if token_name.startswith("NOTE_ON_"):
            pitch = int(token_name.split("_")[-1])
            active_notes.add(pitch)

        elif token_name.startswith("NOTE_OFF_"):
            pitch = int(token_name.split("_")[-1])
            active_notes.discard(pitch)

        elif token_name.startswith("TIME_SHIFT_"):
            shift = int(token_name.split("_")[-1])
            for _ in range(shift):
                frames.append(current_frame())

    if len(frames) == 0:
        frames.append(np.zeros(N_PITCHES, dtype=np.uint8))

    return np.stack(frames, axis=0)


@torch.no_grad()
def generate_after_rl_samples(model, device, bos_id, eos_id, id_to_token, num_samples=10):
    model.eval()
    sample_ids = []

    for i in range(num_samples):
        sample_name = f"after_rl_{i + 1:02d}"
        sample_ids.append(sample_name)

        start_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        generated = model.generate(
            start_tokens=start_tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            genre_ids=None,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            eos_token_id=eos_id
        )

        token_ids = generated.squeeze(0).cpu().numpy().tolist()
        piano_roll = token_ids_to_piano_roll(token_ids, id_to_token)

        output_path = AFTER_RL_DIR / f"{sample_name}.mid"
        save_piano_roll_as_midi(piano_roll, str(output_path), tempo=120)
        print(f"Saved after-RL MIDI: {output_path}")

    return sample_ids


def create_after_rl_survey_template(sample_ids, num_participants=10):
    import csv

    rows = []
    for p in range(1, num_participants + 1):
        participant_id = f"P{p:02d}"
        row = {"participant_id": participant_id}
        for sample_id in sample_ids:
            row[sample_id] = ""
        rows.append(row)

    with open(AFTER_SURVEY_TEMPLATE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["participant_id"] + sample_ids
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved after-RL survey template:", AFTER_SURVEY_TEMPLATE)


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    setup_torch_for_gpu()
    ensure_dirs()

    device = get_device()
    use_amp = (device.type == "cuda" and USE_MIXED_PRECISION)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    token_to_id, id_to_token, pad_id, bos_id, eos_id, vocab_size = load_vocab()
    model = build_model(vocab_size, device)

    rlhf_loader, sample_ids, raw_rewards, norm_rewards, reward_mean, reward_std = build_first_feedback_batch_dataset(pad_id)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = {
        "rlhf_loss": [],
        "avg_seq_logprob": [],
        "avg_reward": []
    }

    best_loss = float("inf")

    print("\nStarting Task 4 RLHF first-round update...\n")
    print("Mixed precision enabled:", use_amp)

    for step in range(RL_STEPS):
        step_loss, step_seq_logprob, step_reward = train_one_step(
            model=model,
            dataloader=rlhf_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            pad_id=pad_id
        )

        history["rlhf_loss"].append(step_loss)
        history["avg_seq_logprob"].append(step_seq_logprob)
        history["avg_reward"].append(step_reward)

        print(
            f"RL Step [{step + 1}/{RL_STEPS}] "
            f"Loss: {step_loss:.6f} | "
            f"Avg Seq LogProb: {step_seq_logprob:.6f} | "
            f"Avg Norm Reward: {step_reward:.6f}"
        )

        if step_loss < best_loss:
            best_loss = step_loss
            torch.save(model.state_dict(), RLHF_MODEL_PATH)
            print(f"Best RLHF model saved to: {RLHF_MODEL_PATH}")

    with open(RLHF_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_curve(history, RLHF_PLOT_PATH)

    summary = {
        "task": "Task 4 RLHF - First RL Update Round",
        "pretrained_model": str(PRETRAINED_MODEL_PATH),
        "reward_csv": str(REWARD_CSV),
        "num_rated_sequences": len(sample_ids),
        "before_rl_mean_human_score": float(raw_rewards.mean()),
        "before_rl_std_human_score": float(raw_rewards.std(ddof=0)),
        "reward_normalization_mean": float(reward_mean),
        "reward_normalization_std": float(reward_std),
        "best_rlhf_loss": float(best_loss),
        "next_required_step": "Run second listening survey on after-RL outputs"
    }

    with open(RLHF_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nReloading best RLHF model for after-RL generation...\n")
    model.load_state_dict(torch.load(RLHF_MODEL_PATH, map_location=device))
    model.to(device)

    after_sample_ids = generate_after_rl_samples(
        model=model,
        device=device,
        bos_id=bos_id,
        eos_id=eos_id,
        id_to_token=id_to_token,
        num_samples=NUM_AFTER_RL_SAMPLES
    )

    create_after_rl_survey_template(
        sample_ids=after_sample_ids,
        num_participants=10
    )

    print("\nDone.")
    print("Saved:")
    print("-", RLHF_MODEL_PATH)
    print("-", RLHF_HISTORY_PATH)
    print("-", RLHF_PLOT_PATH)
    print("-", RLHF_SUMMARY_PATH)
    print("-", AFTER_SURVEY_TEMPLATE)
    for sample_id in after_sample_ids:
        print("-", AFTER_RL_DIR / f"{sample_id}.mid")


if __name__ == "__main__":
    main()