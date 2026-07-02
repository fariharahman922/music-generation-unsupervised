import os
import json
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
SPLIT_ROOT = "data/train_test_split/lmd_matched"

TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")
VAL_DIR = os.path.join(SPLIT_ROOT, "validation")
TEST_DIR = os.path.join(SPLIT_ROOT, "test")

MANIFEST_JSON = os.path.join(SPLIT_ROOT, "lmd_shard_manifest.json")
STATS_JSON = os.path.join(SPLIT_ROOT, "lmd_dataset_stats.json")
SAMPLE_PIANO_ROLL_PATH = os.path.join(PROCESSED_DIR, "sample_piano_roll_lmd.npy")

# =========================================================
# Preprocessing settings
# =========================================================
LOWEST_PITCH = 21
HIGHEST_PITCH = 108
N_PITCHES = HIGHEST_PITCH - LOWEST_PITCH + 1

STEPS_PER_BAR = 16
BEATS_PER_BAR = 4

WINDOW_BARS = 8
WINDOW_SIZE = STEPS_PER_BAR * WINDOW_BARS   # 128
HOP_SIZE = WINDOW_SIZE


SHARD_SIZE = 5000
MAX_FILES = None

warnings.filterwarnings(
    "ignore",
    message="Tempo, Key or Time signature change events found on non-zero tracks.*",
    category=RuntimeWarning
)


def safe_make_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLIT_ROOT, exist_ok=True)
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
    """
    Map note time to nearest valid grid cell index.
    """
    idx = np.searchsorted(grid_times, note_time, side="right") - 1
    idx = max(0, min(idx, len(grid_times) - 2))
    return idx


def midi_to_quantized_piano_roll(midi_path):
    """
    Convert MIDI to quantized binary piano-roll of shape [time_steps, 88].
    Uses a beat-aware grid approximating 16 steps/bar.
    Merges all non-drum instruments into one piano-roll.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    grid_times = get_beat_grid(pm, steps_per_bar=STEPS_PER_BAR, beats_per_bar=BEATS_PER_BAR)
    time_steps = len(grid_times) - 1

    if time_steps <= 0:
        raise ValueError("Invalid time grid produced zero time steps.")

    piano_roll = np.zeros((time_steps, N_PITCHES), dtype=np.uint8)
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

            pitch_idx = note.pitch - LOWEST_PITCH
            piano_roll[start_idx:end_idx + 1, pitch_idx] = 1
            usable_note_count += 1

    if usable_note_count == 0:
        raise ValueError("No usable non-drum notes in piano range.")

    return piano_roll


def create_windows(piano_roll, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    """
    Segment piano-roll into fixed-length windows.
    Skip fully silent windows.
    """
    windows = []
    total_steps = piano_roll.shape[0]

    if total_steps < window_size:
        return windows

    for start in range(0, total_steps - window_size + 1, hop_size):
        end = start + window_size
        window = piano_roll[start:end]

        if np.sum(window) == 0:
            continue

        windows.append(window)

    return windows


def process_single_file(midi_path):
    piano_roll = midi_to_quantized_piano_roll(midi_path)
    windows = create_windows(piano_roll)
    return piano_roll, windows


def assign_split(midi_path):
    """
    Deterministic split assignment based on file path hash.
    80% train, 10% validation, 10% test.
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


def save_chunk(split_name, chunk_windows, shard_idx, manifest):
    """
    Save one shard of windows for a split.
    """
    split_dir = get_split_dir(split_name)
    save_name = f"X_{split_name}_shard_{shard_idx:05d}.npy"
    save_path = os.path.join(split_dir, save_name)

    array_to_save = np.array(chunk_windows, dtype=np.uint8)
    np.save(save_path, array_to_save)

    manifest[split_name].append({
        "path": save_path,
        "shape": list(array_to_save.shape),
        "num_windows": int(array_to_save.shape[0])
    })

    return save_path, array_to_save.shape


def flush_full_shards(split_name, buffers, shard_counters, manifest):
    """
    Save full-size shards while buffer length >= SHARD_SIZE.
    """
    while len(buffers[split_name]) >= SHARD_SIZE:
        chunk = buffers[split_name][:SHARD_SIZE]
        buffers[split_name] = buffers[split_name][SHARD_SIZE:]

        save_chunk(
            split_name=split_name,
            chunk_windows=chunk,
            shard_idx=shard_counters[split_name],
            manifest=manifest
        )
        shard_counters[split_name] += 1


def flush_remaining_shard(split_name, buffers, shard_counters, manifest):
    """
    Save final leftover shard for a split.
    """
    if len(buffers[split_name]) == 0:
        return

    chunk = buffers[split_name]
    buffers[split_name] = []

    save_chunk(
        split_name=split_name,
        chunk_windows=chunk,
        shard_idx=shard_counters[split_name],
        manifest=manifest
    )
    shard_counters[split_name] += 1


def main():
    safe_make_dirs()

    midi_files = load_valid_files(VALID_FILES_JSON)
    print(f"Loaded {len(midi_files)} valid MIDI files from: {VALID_FILES_JSON}")

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
            "n_pitches": N_PITCHES,
            "steps_per_bar": STEPS_PER_BAR,
            "beats_per_bar": BEATS_PER_BAR,
            "window_bars": WINDOW_BARS,
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "shard_size": SHARD_SIZE,
            "max_files": MAX_FILES
        },
        "train": {"files": 0, "windows": 0, "skipped": 0},
        "validation": {"files": 0, "windows": 0, "skipped": 0},
        "test": {"files": 0, "windows": 0, "skipped": 0},
        "global": {
            "processed_files": 0,
            "skipped_files": 0,
            "saved_sample": False
        }
    }

    first_saved_sample = False

    for midi_path in tqdm(midi_files, desc="Preprocessing LMD piano-rolls"):
        split = assign_split(midi_path)

        try:
            piano_roll, windows = process_single_file(midi_path)

            if not first_saved_sample:
                np.save(SAMPLE_PIANO_ROLL_PATH, piano_roll)
                first_saved_sample = True
                stats["global"]["saved_sample"] = True

            stats[split]["files"] += 1
            stats[split]["windows"] += len(windows)
            stats["global"]["processed_files"] += 1

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

    print("\nLMD piano-roll preprocessing complete.")
    print("Processed files:", stats["global"]["processed_files"])
    print("Skipped files during piano-roll stage:", stats["global"]["skipped_files"])

    print("\nSplit summary:")
    for split_name in ["train", "validation", "test"]:
        print(
            f"{split_name.capitalize()} -> "
            f"files: {stats[split_name]['files']}, "
            f"windows: {stats[split_name]['windows']}, "
            f"skipped: {stats[split_name]['skipped']}, "
            f"shards: {len(manifest[split_name])}"
        )

    print("\nSaved:")
    print(SAMPLE_PIANO_ROLL_PATH)
    print(MANIFEST_JSON)
    print(STATS_JSON)
    print(TRAIN_DIR)
    print(VAL_DIR)
    print(TEST_DIR)


if __name__ == "__main__":
    main()