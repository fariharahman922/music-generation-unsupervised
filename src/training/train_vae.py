import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Make project root importable
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.vae import LSTMVAE
from src.generation.midi_export import save_piano_roll_as_midi


# =========================================================
# Configuration
# =========================================================
MANIFEST_PATH = PROJECT_ROOT / "data" / "train_test_split" / "lmd_matched" / "lmd_shard_manifest.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
GENERATED_DIR = OUTPUT_DIR / "generated_midis"

BEST_MODEL_PATH = OUTPUT_DIR / "vae_best.pt"
HISTORY_PATH = OUTPUT_DIR / "vae_history.json"
LOSS_PLOT_PATH = PLOTS_DIR / "vae_loss_curve.png"

USE_MIXED_PRECISION = True
NUM_WORKERS = 0  

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
BETA = 0.001

INPUT_DIM = 88
HIDDEN_DIM = 256
LATENT_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

SEQ_LEN = 128
THRESHOLD = 0.5

NUM_GENERATED_SAMPLES = 8
NUM_INTERPOLATION_STEPS = 5

SEED = 42

MAX_TRAIN_SHARDS = None
MAX_VAL_SHARDS = None


# =========================================================
# Utilities
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


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Shard manifest not found: {MANIFEST_PATH}")

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
    X = np.load(shard_path).astype(np.float32)
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


def vae_loss_function(x_hat, x, mu, logvar, beta=0.001):
    """
    Task 2 objective from the PDF:
    L_VAE = L_recon + beta * D_KL
    """
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_one_epoch(model, shard_paths, optimizer, device, beta, scaler, use_amp, pin_memory):
    model.train()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0

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
            x = batch[0].to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                x_hat, mu, logvar = model(x)
                total_loss, recon_loss, kl_loss = vae_loss_function(
                    x_hat, x, mu, logvar, beta=beta
                )

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_loss_sum += total_loss.item() * batch_size
            recon_loss_sum += recon_loss.item() * batch_size
            kl_loss_sum += kl_loss.item() * batch_size
            total_samples += batch_size

        if shard_idx % 10 == 0 or shard_idx == len(shuffled_shards):
            print(f"  Processed train shard {shard_idx}/{len(shuffled_shards)}")

        del loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    epoch_total = total_loss_sum / total_samples
    epoch_recon = recon_loss_sum / total_samples
    epoch_kl = kl_loss_sum / total_samples

    return epoch_total, epoch_recon, epoch_kl


def validate_one_epoch(model, shard_paths, device, beta, use_amp, pin_memory):
    model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            loader, _ = load_shard_as_loader(
                shard_path=shard_path,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=pin_memory
            )

            for batch in loader:
                x = batch[0].to(device, non_blocking=pin_memory)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    x_hat, mu, logvar = model(x)
                    total_loss, recon_loss, kl_loss = vae_loss_function(
                        x_hat, x, mu, logvar, beta=beta
                    )

                batch_size = x.size(0)
                total_loss_sum += total_loss.item() * batch_size
                recon_loss_sum += recon_loss.item() * batch_size
                kl_loss_sum += kl_loss.item() * batch_size
                total_samples += batch_size

            if shard_idx % 10 == 0 or shard_idx == len(shard_paths):
                print(f"  Processed validation shard {shard_idx}/{len(shard_paths)}")

            del loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    epoch_total = total_loss_sum / total_samples
    epoch_recon = recon_loss_sum / total_samples
    epoch_kl = kl_loss_sum / total_samples

    return epoch_total, epoch_recon, epoch_kl


def plot_vae_losses(history, save_path):
    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, history["train_total"], label="Train Total")
    plt.plot(epochs, history["val_total"], label="Val Total")
    plt.ylabel("Total Loss")
    plt.title("VAE Training Curves")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, history["train_recon"], label="Train Recon")
    plt.plot(epochs, history["val_recon"], label="Val Recon")
    plt.ylabel("Recon Loss")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, history["train_kl"], label="Train KL")
    plt.plot(epochs, history["val_kl"], label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def binarize_piano_roll(prob_roll, primary_threshold=0.5):
    for threshold in [primary_threshold, 0.4, 0.3, 0.2]:
        binary = (prob_roll >= threshold).astype(np.uint8)
        if binary.sum() > 0:
            return binary
    return (prob_roll > 0.0).astype(np.uint8)


def generate_samples(model, device, num_samples=8, seq_len=128):
    model.eval()

    for i in range(num_samples):
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, device=device)
            x_hat = model.decode(z, seq_len=seq_len)

        piano_roll = x_hat.squeeze(0).cpu().numpy()
        piano_roll = binarize_piano_roll(piano_roll, primary_threshold=THRESHOLD)

        output_path = GENERATED_DIR / f"task2_vae_sample_{i+1}.mid"
        save_piano_roll_as_midi(piano_roll, str(output_path), tempo=120)
        print(f"Saved generated MIDI: {output_path}")


def save_interpolation_samples(model, device, seq_len=128, num_steps=5):
    model.eval()

    with torch.no_grad():
        z1 = torch.randn(1, LATENT_DIM, device=device)
        z2 = torch.randn(1, LATENT_DIM, device=device)

        alphas = np.linspace(0.0, 1.0, num_steps)

        for i, alpha in enumerate(alphas, start=1):
            z = (1.0 - alpha) * z1 + alpha * z2
            x_hat = model.decode(z, seq_len=seq_len)

            piano_roll = x_hat.squeeze(0).cpu().numpy()
            piano_roll = binarize_piano_roll(piano_roll, primary_threshold=THRESHOLD)

            output_path = GENERATED_DIR / f"task2_vae_interp_{i}.mid"
            save_piano_roll_as_midi(piano_roll, str(output_path), tempo=120)
            print(f"Saved interpolation MIDI: {output_path}")


def save_history(history, best_val_loss):
    to_save = dict(history)
    to_save["best_val_total_loss"] = best_val_loss

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)


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

    train_shards, val_shards = load_manifest()

    model = LSTMVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": []
    }

    best_val_loss = float("inf")

    print("\nStarting VAE training...\n")
    print("Mixed precision enabled:", use_amp)
    print("Pin memory enabled:", pin_memory)

    for epoch in range(EPOCHS):
        train_total, train_recon, train_kl = train_one_epoch(
            model=model,
            shard_paths=train_shards,
            optimizer=optimizer,
            device=device,
            beta=BETA,
            scaler=scaler,
            use_amp=use_amp,
            pin_memory=pin_memory
        )

        val_total, val_recon, val_kl = validate_one_epoch(
            model=model,
            shard_paths=val_shards,
            device=device,
            beta=BETA,
            use_amp=use_amp,
            pin_memory=pin_memory
        )

        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)

        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "
            f"Train Total: {train_total:.6f} | "
            f"Train Recon: {train_recon:.6f} | "
            f"Train KL: {train_kl:.6f} || "
            f"Val Total: {val_total:.6f} | "
            f"Val Recon: {val_recon:.6f} | "
            f"Val KL: {val_kl:.6f}"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved to: {BEST_MODEL_PATH}")

    print("\nTraining complete.")
    print("Best validation total loss:", best_val_loss)

    save_history(history, best_val_loss)
    print("Training history saved to:", HISTORY_PATH)

    plot_vae_losses(history, LOSS_PLOT_PATH)
    print("Loss curve saved to:", LOSS_PLOT_PATH)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

    print("\nGenerating Task 2 VAE samples...\n")
    generate_samples(
        model=model,
        device=device,
        num_samples=NUM_GENERATED_SAMPLES,
        seq_len=SEQ_LEN
    )

    print("\nGenerating latent interpolation samples...\n")
    save_interpolation_samples(
        model=model,
        device=device,
        seq_len=SEQ_LEN,
        num_steps=NUM_INTERPOLATION_STEPS
    )

    print("\nDone.")
    print("Outputs saved:")
    print("-", BEST_MODEL_PATH)
    print("-", HISTORY_PATH)
    print("-", LOSS_PLOT_PATH)

    for i in range(NUM_GENERATED_SAMPLES):
        print("-", GENERATED_DIR / f"task2_vae_sample_{i+1}.mid")

    for i in range(NUM_INTERPOLATION_STEPS):
        print("-", GENERATED_DIR / f"task2_vae_interp_{i+1}.mid")


if __name__ == "__main__":
    main()