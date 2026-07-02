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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.autoencoder import LSTMAutoencoder
from src.generation.midi_export import save_piano_roll_as_midi


# =========================================================
# Configuration
# =========================================================
TRAIN_PATH = PROJECT_ROOT / "data" / "train_test_split" / "X_train.npy"
VAL_PATH = PROJECT_ROOT / "data" / "train_test_split" / "X_val.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
GENERATED_DIR = OUTPUT_DIR / "generated_midis"

BEST_MODEL_PATH = OUTPUT_DIR / "ae_best.pt"
HISTORY_PATH = OUTPUT_DIR / "ae_history.json"
LOSS_PLOT_PATH = PLOTS_DIR / "ae_loss_curve.png"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

INPUT_DIM = 88
HIDDEN_DIM = 256
LATENT_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

NUM_GENERATED_SAMPLES = 5
SEQ_LEN = 128
THRESHOLD = 0.5

SEED = 42


# =========================================================
# Utilities
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation data not found: {VAL_PATH}")

    X_train = np.load(TRAIN_PATH).astype(np.float32)
    X_val = np.load(VAL_PATH).astype(np.float32)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    train_dataset = TensorDataset(torch.from_numpy(X_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        x = batch[0].to(device)

        optimizer.zero_grad()

        x_hat, z = model(x)
        loss = criterion(x_hat, x)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            x_hat, z = model(x)
            loss = criterion(x_hat, x)

            running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Reconstruction Loss")
    plt.title("LSTM Autoencoder Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def estimate_latent_distribution(model, dataloader, device, max_batches=100):
    """
    Estimate a simple Gaussian over latent codes from training data.
    """
    model.eval()
    latents = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch[0].to(device)
            z = model.encode(x)
            latents.append(z.cpu())

            if batch_idx + 1 >= max_batches:
                break

    latents = torch.cat(latents, dim=0)
    z_mean = latents.mean(dim=0)
    z_std = latents.std(dim=0) + 1e-6

    print("Estimated latent mean/std from training data.")
    return z_mean, z_std


def binarize_piano_roll(prob_roll, primary_threshold=0.5):
    """
    Convert decoder probabilities to binary piano-roll.
    Fallback thresholds help avoid silent generations.
    """
    for threshold in [primary_threshold, 0.4, 0.3, 0.2]:
        binary = (prob_roll >= threshold).astype(np.uint8)
        if binary.sum() > 0:
            return binary

    return (prob_roll > 0.0).astype(np.uint8)


def generate_samples(model, train_loader, device, num_samples=5, seq_len=128):
    model.eval()

    z_mean, z_std = estimate_latent_distribution(model, train_loader, device)
    z_mean = z_mean.to(device)
    z_std = z_std.to(device)

    for i in range(num_samples):
        with torch.no_grad():
            z = z_mean + z_std * torch.randn(1, z_mean.shape[0], device=device)
            x_hat = model.decode(z, seq_len=seq_len)

        piano_roll = x_hat.squeeze(0).cpu().numpy()
        piano_roll = binarize_piano_roll(piano_roll, primary_threshold=THRESHOLD)

        output_path = GENERATED_DIR / f"task1_ae_sample_{i+1}.mid"
        save_piano_roll_as_midi(piano_roll, str(output_path), tempo=120)
        print(f"Saved generated MIDI: {output_path}")


def save_history(train_losses, val_losses, best_val_loss):
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = load_data()

    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved to: {BEST_MODEL_PATH}")

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)

    save_history(train_losses, val_losses, best_val_loss)
    print("Training history saved to:", HISTORY_PATH)

    plot_loss_curve(train_losses, val_losses, LOSS_PLOT_PATH)
    print("Loss curve saved to:", LOSS_PLOT_PATH)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

    print("\nGenerating Task 1 MIDI samples...\n")
    generate_samples(
        model=model,
        train_loader=train_loader,
        device=device,
        num_samples=NUM_GENERATED_SAMPLES,
        seq_len=SEQ_LEN
    )

    print("\nDone.")
    print("Outputs saved:")
    print("-", BEST_MODEL_PATH)
    print("-", HISTORY_PATH)
    print("-", LOSS_PLOT_PATH)
    for i in range(NUM_GENERATED_SAMPLES):
        print("-", GENERATED_DIR / f"task1_ae_sample_{i+1}.mid")


if __name__ == "__main__":
    main()