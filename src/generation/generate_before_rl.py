import os
import sys
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.transformer import MusicTransformer
from src.generation.midi_export import save_piano_roll_as_midi


# =========================================================
# Paths
# =========================================================
VOCAB_PATH = PROJECT_ROOT / "data" / "train_test_split" / "lmd_tokenized" / "token_vocab.json"
MODEL_PATH = PROJECT_ROOT / "outputs" / "transformer_best.pt"

SURVEY_ROOT = PROJECT_ROOT / "outputs" / "survey_results"
BEFORE_RL_DIR = SURVEY_ROOT / "task4_before_rl"
SURVEY_TEMPLATE_PATH = SURVEY_ROOT / "task4_listening_survey_template.csv"

# =========================================================
# Generation settings
# =========================================================
LOWEST_PITCH = 21
HIGHEST_PITCH = 108
N_PITCHES = HIGHEST_PITCH - LOWEST_PITCH + 1

MAX_SEQ_LEN = 256
MAX_NEW_TOKENS = 1024
NUM_SAMPLES = 10

TEMPERATURE = 1.0
TOP_K = 50


D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

NUM_PARTICIPANTS = 10

SEED = 42


# =========================================================
# Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    SURVEY_ROOT.mkdir(parents=True, exist_ok=True)
    BEFORE_RL_DIR.mkdir(parents=True, exist_ok=True)


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

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find trained Transformer model: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loaded Transformer model from:", MODEL_PATH)
    return model


def token_ids_to_piano_roll(token_ids, id_to_token):
    """
    Decode generated token ids back into a binary piano-roll.
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
def generate_before_rl_samples(model, device, bos_id, eos_id, id_to_token, num_samples=10):
    sample_ids = []

    for i in range(num_samples):
        sample_name = f"before_rl_{i + 1:02d}"
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

        midi_path = BEFORE_RL_DIR / f"{sample_name}.mid"
        save_piano_roll_as_midi(piano_roll, str(midi_path), tempo=120)

        print(f"Saved survey sample: {midi_path}")

    return sample_ids


def create_survey_template(sample_ids, num_participants=10):
    """
    Create a simple listening survey CSV.
    The PDF requires human listening scores in [1, 5].
    """
    rows = []

    for p in range(1, num_participants + 1):
        participant_id = f"P{p:02d}"

        for sample_id in sample_ids:
            rows.append({
                "participant_id": participant_id,
                "sample_id": sample_id,
                "overall_score_1_to_5": "",
                "comments_optional": ""
            })

    with open(SURVEY_TEMPLATE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "sample_id",
                "overall_score_1_to_5",
                "comments_optional"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved survey template:", SURVEY_TEMPLATE_PATH)


def main():
    set_seed(SEED)
    ensure_dirs()

    device = get_device()
    token_to_id, id_to_token, pad_id, bos_id, eos_id, vocab_size = load_vocab()
    model = build_model(vocab_size, device)

    print("\nGenerating Task 4 before-RL survey samples...\n")
    sample_ids = generate_before_rl_samples(
        model=model,
        device=device,
        bos_id=bos_id,
        eos_id=eos_id,
        id_to_token=id_to_token,
        num_samples=NUM_SAMPLES
    )

    print("\nCreating listening survey template...\n")
    create_survey_template(
        sample_ids=sample_ids,
        num_participants=NUM_PARTICIPANTS
    )

    print("\nDone.")
    print("Generated before-RL samples saved in:", BEFORE_RL_DIR)
    print("Survey CSV template saved at:", SURVEY_TEMPLATE_PATH)


if __name__ == "__main__":
    main()