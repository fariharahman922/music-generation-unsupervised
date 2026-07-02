import os
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Project root import setup
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.transformer import MusicTransformer
from src.generation.midi_export import save_piano_roll_as_midi


# =========================================================
# Paths
# =========================================================
MANIFEST_PATH = PROJECT_ROOT / "data" / "train_test_split" / "lmd_tokenized" / "token_shard_manifest.json"
VOCAB_PATH = PROJECT_ROOT / "data" / "train_test_split" / "lmd_tokenized" / "token_vocab.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
GENERATED_DIR = OUTPUT_DIR / "generated_midis"

BEST_MODEL_PATH = OUTPUT_DIR / "transformer_best.pt"
HISTORY_PATH = OUTPUT_DIR / "transformer_history.json"
LOSS_PLOT_PATH = PLOTS_DIR / "transformer_training_curves.png"
PERPLEXITY_REPORT_PATH = OUTPUT_DIR / "transformer_perplexity_report.json"


# =========================================================
# Token / MIDI settings
# =========================================================
LOWEST_PITCH = 21
HIGHEST_PITCH = 108
N_PITCHES = HIGHEST_PITCH - LOWEST_PITCH + 1

MAX_SEQ_LEN = 256
MAX_NEW_TOKENS = 1024  
NUM_GENERATED_SAMPLES = 10


# =========================================================
# Training settings
# =========================================================
USE_MIXED_PRECISION = True
NUM_WORKERS = 0   

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

TEMPERATURE = 1.0
TOP_K = 50

SEED = 42

MAX_TRAIN_SHARDS = None
MAX_VAL_SHARDS = None


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
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load token metadata
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


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Could not find token shard manifest: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_shards = [item["path"] for item in manifest["train"]]
    val_shards = [item["path"] for item in manifest["validation"]]

    if MAX_TRAIN_SHARDS is not None:
        train_shards = train_shards[:MAX_TRAIN_SHARDS]
    if MAX_VAL_SHARDS is not None:
        val_shards = val_shards[:MAX_VAL_SHARDS]

    print("Train shards:", len(train_shards))
    print("Validation shards:", len(val_shards))

    return train_shards, val_shards


def load_shard_as_loader(shard_path, batch_size, shuffle, pin_memory):
    X = np.load(shard_path).astype(np.int64)
    dataset = TensorDataset(torch.from_numpy(X))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0)
    )
    return loader, X.shape[0]


# =========================================================
# Loss / metrics
# =========================================================
def autoregressive_loss(logits, targets, pad_id):
    """
    Cross-entropy over next-token prediction.
    This corresponds to the negative log-likelihood form in Task 3.
    """
    vocab_size = logits.size(-1)

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="mean"
    )
    return loss


def compute_perplexity(avg_nll):
    """
    PDF metric:
        Perplexity = exp( L_TR / T )
    With token-level mean NLL, this becomes exp(avg_nll).
    """
    return float(math.exp(avg_nll))


# =========================================================
# Training / validation
# =========================================================
def train_one_epoch(model, shard_paths, optimizer, device, scaler, use_amp, pin_memory, pad_id):
    model.train()

    total_nll_sum = 0.0
    total_pred_tokens = 0

    shuffled_shards = shard_paths[:]
    random.shuffle(shuffled_shards)

    for shard_idx, shard_path in enumerate(shuffled_shards, start=1):
        loader, _ = load_shard_as_loader(
            shard_path=shard_path,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=pin_memory
        )

        for batch in loader:
            tokens = batch[0].to(device, non_blocking=pin_memory)  # (B, T)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_tokens)
                loss = autoregressive_loss(logits, target_tokens, pad_id=pad_id)

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

            valid_tokens = (target_tokens != pad_id).sum().item()
            total_nll_sum += loss.item() * valid_tokens
            total_pred_tokens += valid_tokens

        if shard_idx % 10 == 0 or shard_idx == len(shuffled_shards):
            print(f"  Processed train shard {shard_idx}/{len(shuffled_shards)}")

        del loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_nll = total_nll_sum / total_pred_tokens
    perplexity = compute_perplexity(avg_nll)

    return avg_nll, perplexity


def validate_one_epoch(model, shard_paths, device, use_amp, pin_memory, pad_id):
    model.eval()

    total_nll_sum = 0.0
    total_pred_tokens = 0

    with torch.no_grad():
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            loader, _ = load_shard_as_loader(
                shard_path=shard_path,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=pin_memory
            )

            for batch in loader:
                tokens = batch[0].to(device, non_blocking=pin_memory)

                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(input_tokens)
                    loss = autoregressive_loss(logits, target_tokens, pad_id=pad_id)

                valid_tokens = (target_tokens != pad_id).sum().item()
                total_nll_sum += loss.item() * valid_tokens
                total_pred_tokens += valid_tokens

            if shard_idx % 10 == 0 or shard_idx == len(shard_paths):
                print(f"  Processed validation shard {shard_idx}/{len(shard_paths)}")

            del loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_nll = total_nll_sum / total_pred_tokens
    perplexity = compute_perplexity(avg_nll)

    return avg_nll, perplexity


# =========================================================
# Plotting / saving
# =========================================================
def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Transformer Training Curves")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_perplexity"], label="Train Perplexity")
    plt.plot(epochs, history["val_perplexity"], label="Val Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_history(history, best_val_loss, best_val_perplexity):
    payload = dict(history)
    payload["best_val_loss"] = best_val_loss
    payload["best_val_perplexity"] = best_val_perplexity

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_perplexity_report(best_val_loss, best_val_perplexity):
    report = {
        "task": "Task 3 Transformer-Based Music Generator",
        "best_validation_loss": best_val_loss,
        "best_validation_perplexity": best_val_perplexity,
        "formula_note": "Perplexity = exp(mean token-level negative log-likelihood)"
    }

    with open(PERPLEXITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# =========================================================
# Token decoding for MIDI generation
# =========================================================
def token_ids_to_piano_roll(token_ids, id_to_token):
    """
    Decode generated token ids back into a binary piano-roll.

    We reconstruct time by consuming TIME_SHIFT tokens and keep track
    of active notes between shifts.
    """
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
            if pitch in active_notes:
                active_notes.remove(pitch)

        elif token_name.startswith("TIME_SHIFT_"):
            shift = int(token_name.split("_")[-1])
            for _ in range(shift):
                frames.append(current_frame())

    if len(frames) == 0:
        frames.append(np.zeros(N_PITCHES, dtype=np.uint8))

    piano_roll = np.stack(frames, axis=0)
    return piano_roll


@torch.no_grad()
def generate_long_compositions(model, device, bos_id, eos_id, id_to_token, num_samples=10):
    model.eval()

    for i in range(num_samples):
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

        output_path = GENERATED_DIR / f"task3_transformer_sample_{i+1}.mid"
        save_piano_roll_as_midi(piano_roll, str(output_path), tempo=120)
        print(f"Saved generated MIDI: {output_path}")


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    setup_torch_for_gpu()
    ensure_dirs()

    device = get_device()
    pin_memory = (device.type == "cuda")
    use_amp = (device.type == "cuda" and USE_MIXED_PRECISION)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    token_to_id, id_to_token, pad_id, bos_id, eos_id, vocab_size = load_vocab()
    train_shards, val_shards = load_manifest()

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = {
        "train_loss": [],
        "train_perplexity": [],
        "val_loss": [],
        "val_perplexity": []
    }

    best_val_loss = float("inf")
    best_val_perplexity = float("inf")

    print("\nStarting Transformer training...\n")
    print("Mixed precision enabled:", use_amp)
    print("Pin memory enabled:", pin_memory)

    for epoch in range(EPOCHS):
        train_loss, train_ppl = train_one_epoch(
            model=model,
            shard_paths=train_shards,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            pin_memory=pin_memory,
            pad_id=pad_id
        )

        val_loss, val_ppl = validate_one_epoch(
            model=model,
            shard_paths=val_shards,
            device=device,
            use_amp=use_amp,
            pin_memory=pin_memory,
            pad_id=pad_id
        )

        history["train_loss"].append(train_loss)
        history["train_perplexity"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Perplexity: {train_ppl:.4f} || "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Perplexity: {val_ppl:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_perplexity = val_ppl
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved to: {BEST_MODEL_PATH}")

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)
    print("Best validation perplexity:", best_val_perplexity)

    save_history(history, best_val_loss, best_val_perplexity)
    print("Training history saved to:", HISTORY_PATH)

    plot_training_curves(history, LOSS_PLOT_PATH)
    print("Training curves saved to:", LOSS_PLOT_PATH)

    save_perplexity_report(best_val_loss, best_val_perplexity)
    print("Perplexity report saved to:", PERPLEXITY_REPORT_PATH)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

    print("\nGenerating Task 3 long compositions...\n")
    generate_long_compositions(
        model=model,
        device=device,
        bos_id=bos_id,
        eos_id=eos_id,
        id_to_token=id_to_token,
        num_samples=NUM_GENERATED_SAMPLES
    )

    print("\nDone.")
    print("Outputs saved:")
    print("-", BEST_MODEL_PATH)
    print("-", HISTORY_PATH)
    print("-", LOSS_PLOT_PATH)
    print("-", PERPLEXITY_REPORT_PATH)
    for i in range(NUM_GENERATED_SAMPLES):
        print("-", GENERATED_DIR / f"task3_transformer_sample_{i+1}.mid")


if __name__ == "__main__":
    main()