import os
import json
import math
import hashlib
import warnings
import numpy as np
import pretty_midi
from tqdm import tqdm


# =========================================================
# Paths
# =========================================================
VALID_FILES_JSON = "data/processed/lmd_valid_files.json"
PROCESSED_DIR = "data/processed"
TOKEN_ROOT = "data/train_test_split/lmd_tokenized"

TRAIN_DIR = os.path.join(TOKEN_ROOT, "train")
VAL_DIR = os.path.join(TOKEN_ROOT, "validation")
TEST_DIR = os.path.join(TOKEN_ROOT, "test")

VOCAB_JSON = os.path.join(TOKEN_ROOT, "token_vocab.json")
MANIFEST_JSON = os.path.join(TOKEN_ROOT, "token_shard_manifest.json")
STATS_JSON = os.path.join(TOKEN_ROOT, "token_dataset_stats.json")
SAMPLE_TOKENS_NPY = os.path.join(PROCESSED_DIR, "sample_token_sequence.npy")
SAMPLE_TOKENS_TXT = os.path.join(PROCESSED_DIR, "sample_token_sequence.txt")


# =========================================================
# Preprocessing settings
# =========================================================
LOWEST_PITCH = 21
HIGHEST_PITCH = 108
N_PITCHES = HIGHEST_PITCH - LOWEST_PITCH + 1

STEPS_PER_BAR = 16
BEATS_PER_BAR = 4
STEPS_PER_BEAT = STEPS_PER_BAR // BEATS_PER_BAR

# Fixed-length token windows for Transformer training
TOKEN_SEQ_LEN = 256
TOKEN_HOP = 256

SHARD_SIZE = 5000
MAX_FILES = None


# =========================================================
# Warnings
# =========================================================
warnings.filterwarnings(
    "ignore",
    message="Tempo, Key or Time signature change events found on non-zero tracks.*",
    category=RuntimeWarning
)


# =========================================================
# Vocabulary
# =========================================================
def build_vocab():
    """
    Token vocabulary:
      0   -> PAD
      1   -> BOS
      2   -> EOS

      NOTE_ON_<pitch>
      NOTE_OFF_<pitch>
      TIME_SHIFT_<k> for k in [1..16]

    Duration is represented implicitly through NOTE_ON/NOTE_OFF
    plus TIME_SHIFT tokens.
    """
    token_to_id = {}
    id_to_token = {}

    def add_token(token_name):
        idx = len(token_to_id)
        token_to_id[token_name] = idx
        id_to_token[idx] = token_name

    add_token("PAD")
    add_token("BOS")
    add_token("EOS")

    for pitch in range(LOWEST_PITCH, HIGHEST_PITCH + 1):
        add_token(f"NOTE_ON_{pitch}")

    for pitch in range(LOWEST_PITCH, HIGHEST_PITCH + 1):
        add_token(f"NOTE_OFF_{pitch}")

    for shift in range(1, STEPS_PER_BAR + 1):
        add_token(f"TIME_SHIFT_{shift}")

    return token_to_id, id_to_token


TOKEN_TO_ID, ID_TO_TOKEN = build_vocab()
PAD_ID = TOKEN_TO_ID["PAD"]
BOS_ID = TOKEN_TO_ID["BOS"]
EOS_ID = TOKEN_TO_ID["EOS"]


# =========================================================
# Utils
# =========================================================
def safe_make_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TOKEN_ROOT, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def load_valid_files(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find valid file list: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        midi_files = json.load(f)

    if MAX_FILES is not None:
        midi_files = midi_files[:MAX_FILES]

    return midi_files


def assign_split(midi_path):
    """
    Deterministic split based on file path hash:
      80% train, 10% validation, 10% test
    """
    digest = hashlib.md5(midi_path.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF

    if value < 0.80:
        return "train"
    elif value < 0.90:
        return "validation"
    else:
        return "test"


def get_split_dir(split_name):
    if split_name == "train":
        return TRAIN_DIR
    elif split_name == "validation":
        return VAL_DIR
    elif split_name == "test":
        return TEST_DIR
    else:
        raise ValueError(f"Unknown split: {split_name}")


# =========================================================
# Beat-aware quantization (same idea as previous preprocessing)
# =========================================================
def get_beat_grid(pm, steps_per_bar=16, beats_per_bar=4):
    """
    Build a quantized time grid using beats from the MIDI.
    For 4/4 music, 16 steps/bar = 4 steps/beat.

    Returns grid_times with length T+1 so intervals are [grid[i], grid[i+1]).
    """
    beats = pm.get_beats()

    if len(beats) < 2:
        end_time = pm.get_end_time()
        if end_time <= 0:
            return np.array([0.0, 0.25], dtype=np.float32)

        return np.linspace(
            0.0,
            end_time,
            num=max(2, int(end_time / 0.25) + 1),
            dtype=np.float32
        )

    steps_per_beat = steps_per_bar // beats_per_bar
    grid_times = []

    for i in range(len(beats) - 1):
        start = beats[i]
        end = beats[i + 1]
        substeps = np.linspace(start, end, num=steps_per_beat, endpoint=False)
        grid_times.extend(substeps.tolist())

    last_beat = beats[-1]
    beat_duration = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
    final_step = beat_duration / steps_per_beat

    for s in range(steps_per_beat + 1):
        grid_times.append(last_beat + s * final_step)

    grid_times = np.array(grid_times, dtype=np.float32)

    if len(grid_times) < 2:
        grid_times = np.array([0.0, 0.25], dtype=np.float32)

    return grid_times


def note_to_grid_index(note_time, grid_times):
    idx = np.searchsorted(grid_times, note_time, side="right") - 1
    idx = max(0, min(idx, len(grid_times) - 2))
    return idx


# =========================================================
# MIDI -> token events
# =========================================================
def midi_to_event_list(midi_path):
    """
    Convert one MIDI file into a list of quantized symbolic events.

    Output format:
      [(step_idx, event_type, pitch), ...]

    where:
      event_type in {"note_off", "note_on"}

    We ignore drum instruments.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    grid_times = get_beat_grid(pm, steps_per_bar=STEPS_PER_BAR, beats_per_bar=BEATS_PER_BAR)

    events = []
    usable_note_count = 0

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            if note.pitch < LOWEST_PITCH or note.pitch > HIGHEST_PITCH:
                continue

            start_idx = note_to_grid_index(note.start, grid_times)
            end_idx = note_to_grid_index(note.end, grid_times)

            if end_idx < start_idx:
                end_idx = start_idx

            events.append((start_idx, "note_on", note.pitch))
            events.append((end_idx + 1, "note_off", note.pitch))
            usable_note_count += 1

    if usable_note_count == 0:
        raise ValueError("No usable non-drum notes in piano range.")

    # Sort by time step, then NOTE_OFF before NOTE_ON at same step
    priority = {"note_off": 0, "note_on": 1}
    events.sort(key=lambda x: (x[0], priority[x[1]], x[2]))

    return events


def event_list_to_token_ids(events):
    """
    Convert event list into token ids:
      - TIME_SHIFT_k moves time forward
      - NOTE_ON_pitch / NOTE_OFF_pitch mark note state changes

    We add BOS at the beginning and EOS at the end.
    """
    if len(events) == 0:
        return []

    token_ids = [BOS_ID]
    current_step = 0

    for step_idx, event_type, pitch in events:
        delta = step_idx - current_step

        while delta > 0:
            shift = min(delta, STEPS_PER_BAR)
            token_ids.append(TOKEN_TO_ID[f"TIME_SHIFT_{shift}"])
            delta -= shift

        if event_type == "note_on":
            token_ids.append(TOKEN_TO_ID[f"NOTE_ON_{pitch}"])
        elif event_type == "note_off":
            token_ids.append(TOKEN_TO_ID[f"NOTE_OFF_{pitch}"])
        else:
            raise ValueError(f"Unknown event type: {event_type}")

        current_step = step_idx

    token_ids.append(EOS_ID)
    return token_ids


def midi_to_token_ids(midi_path):
    events = midi_to_event_list(midi_path)
    token_ids = event_list_to_token_ids(events)

    if len(token_ids) < 2:
        raise ValueError("Token sequence too short.")

    return token_ids


# =========================================================
# Fixed-length token windows
# =========================================================
def create_token_windows(token_ids, seq_len=TOKEN_SEQ_LEN, hop=TOKEN_HOP):
    """
    Segment one token sequence into fixed-length token windows.

    Returns shape-ready Python lists of length seq_len.
    """
    windows = []

    if len(token_ids) < seq_len:
        return windows

    for start in range(0, len(token_ids) - seq_len + 1, hop):
        end = start + seq_len
        window = token_ids[start:end]
        windows.append(window)

    return windows


# =========================================================
# Saving shards
# =========================================================
def save_chunk(split_name, chunk_sequences, shard_idx, manifest):
    split_dir = get_split_dir(split_name)
    save_name = f"tokens_{split_name}_shard_{shard_idx:05d}.npy"
    save_path = os.path.join(split_dir, save_name)

    array_to_save = np.array(chunk_sequences, dtype=np.int32)
    np.save(save_path, array_to_save)

    manifest[split_name].append({
        "path": save_path,
        "shape": list(array_to_save.shape),
        "num_sequences": int(array_to_save.shape[0]),
        "seq_len": int(array_to_save.shape[1]) if array_to_save.ndim == 2 else None
    })

    return save_path, array_to_save.shape


def flush_full_shards(split_name, buffers, shard_counters, manifest):
    while len(buffers[split_name]) >= SHARD_SIZE:
        chunk = buffers[split_name][:SHARD_SIZE]
        buffers[split_name] = buffers[split_name][SHARD_SIZE:]

        save_chunk(
            split_name=split_name,
            chunk_sequences=chunk,
            shard_idx=shard_counters[split_name],
            manifest=manifest
        )
        shard_counters[split_name] += 1


def flush_remaining_shard(split_name, buffers, shard_counters, manifest):
    if len(buffers[split_name]) == 0:
        return

    chunk = buffers[split_name]
    buffers[split_name] = []

    save_chunk(
        split_name=split_name,
        chunk_sequences=chunk,
        shard_idx=shard_counters[split_name],
        manifest=manifest
    )
    shard_counters[split_name] += 1


# =========================================================
# Main
# =========================================================
def main():
    safe_make_dirs()

    midi_files = load_valid_files(VALID_FILES_JSON)
    print(f"Loaded {len(midi_files)} valid MIDI files from: {VALID_FILES_JSON}")

    vocab_payload = {
        "token_to_id": TOKEN_TO_ID,
        "id_to_token": {str(k): v for k, v in ID_TO_TOKEN.items()},
        "special_tokens": {
            "PAD_ID": PAD_ID,
            "BOS_ID": BOS_ID,
            "EOS_ID": EOS_ID
        }
    }
    with open(VOCAB_JSON, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2)

    buffers = {
        "train": [],
        "validation": [],
        "test": []
    }

    shard_counters = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    manifest = {
        "train": [],
        "validation": [],
        "test": []
    }

    stats = {
        "settings": {
            "lowest_pitch": LOWEST_PITCH,
            "highest_pitch": HIGHEST_PITCH,
            "steps_per_bar": STEPS_PER_BAR,
            "beats_per_bar": BEATS_PER_BAR,
            "token_seq_len": TOKEN_SEQ_LEN,
            "token_hop": TOKEN_HOP,
            "shard_size": SHARD_SIZE,
            "max_files": MAX_FILES
        },
        "vocab_size": len(TOKEN_TO_ID),
        "train": {"files": 0, "token_windows": 0, "skipped": 0},
        "validation": {"files": 0, "token_windows": 0, "skipped": 0},
        "test": {"files": 0, "token_windows": 0, "skipped": 0},
        "global": {
            "processed_files": 0,
            "skipped_files": 0,
            "saved_sample": False
        }
    }

    first_saved_sample = False

    for midi_path in tqdm(midi_files, desc="Tokenizing LMD for Transformer"):
        split = assign_split(midi_path)

        try:
            token_ids = midi_to_token_ids(midi_path)
            windows = create_token_windows(token_ids, seq_len=TOKEN_SEQ_LEN, hop=TOKEN_HOP)

            stats[split]["files"] += 1
            stats[split]["token_windows"] += len(windows)
            stats["global"]["processed_files"] += 1

            if not first_saved_sample and len(token_ids) > 0:
                sample_array = np.array(token_ids[: min(len(token_ids), 512)], dtype=np.int32)
                np.save(SAMPLE_TOKENS_NPY, sample_array)

                with open(SAMPLE_TOKENS_TXT, "w", encoding="utf-8") as f:
                    for token_id in sample_array.tolist():
                        f.write(f"{token_id}\t{ID_TO_TOKEN[token_id]}\n")

                first_saved_sample = True
                stats["global"]["saved_sample"] = True

            if len(windows) > 0:
                buffers[split].extend(windows)
                flush_full_shards(split, buffers, shard_counters, manifest)

        except Exception as e:
            stats[split]["skipped"] += 1
            stats["global"]["skipped_files"] += 1
            print(f"\nSkipped: {midi_path}")
            print(f"Reason: {e}")


    for split_name in ["train", "validation", "test"]:
        flush_remaining_shard(split_name, buffers, shard_counters, manifest)

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nTokenization complete.")
    print("Processed files:", stats["global"]["processed_files"])
    print("Skipped files during tokenization:", stats["global"]["skipped_files"])
    print("Vocabulary size:", len(TOKEN_TO_ID))

    print("\nSplit summary:")
    for split_name in ["train", "validation", "test"]:
        print(
            f"{split_name.capitalize()} -> "
            f"files: {stats[split_name]['files']}, "
            f"token_windows: {stats[split_name]['token_windows']}, "
            f"skipped: {stats[split_name]['skipped']}, "
            f"shards: {len(manifest[split_name])}"
        )

    print("\nSaved:")
    print(VOCAB_JSON)
    print(MANIFEST_JSON)
    print(STATS_JSON)
    print(SAMPLE_TOKENS_NPY)
    print(SAMPLE_TOKENS_TXT)
    print(TRAIN_DIR)
    print(VAL_DIR)
    print(TEST_DIR)


if __name__ == "__main__":
    main()