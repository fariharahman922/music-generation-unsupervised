import os
import glob
import json
import pretty_midi

DATA_DIR = "data/raw_midi/lmd_matched"
OUTPUT_DIR = "data/processed"
VALID_FILES_JSON = os.path.join(OUTPUT_DIR, "lmd_valid_files.json")
STATS_JSON = os.path.join(OUTPUT_DIR, "lmd_midi_stats.json")


def find_midi_files(data_dir):
    midi_files = glob.glob(os.path.join(data_dir, "**", "*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(data_dir, "**", "*.midi"), recursive=True)
    return sorted(midi_files)


def inspect_midi_file(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    num_instruments = len(midi_data.instruments)
    end_time = midi_data.get_end_time()
    note_count = sum(len(instr.notes) for instr in midi_data.instruments)
    non_drum_instruments = sum(1 for instr in midi_data.instruments if not instr.is_drum)

    return {
        "path": midi_path,
        "num_instruments": num_instruments,
        "non_drum_instruments": non_drum_instruments,
        "end_time_seconds": float(end_time),
        "note_count": int(note_count)
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    midi_files = find_midi_files(DATA_DIR)
    print(f"Found {len(midi_files)} MIDI files")

    valid_files = []
    stats = []

    skipped = 0

    for i, midi_path in enumerate(midi_files, start=1):
        try:
            info = inspect_midi_file(midi_path)
            valid_files.append(midi_path)
            stats.append(info)

            if i <= 5:
                print(f"\nSample valid file {i}:")
                print("Path:", midi_path)
                print("Instruments:", info["num_instruments"])
                print("Non-drum instruments:", info["non_drum_instruments"])
                print("Duration (sec):", info["end_time_seconds"])
                print("Total notes:", info["note_count"])

        except Exception as e:
            skipped += 1
            print(f"Skipping corrupted file: {midi_path}")
            print("Reason:", e)

    with open(VALID_FILES_JSON, "w", encoding="utf-8") as f:
        json.dump(valid_files, f, indent=2)

    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nFinished MIDI parsing.")
    print("Valid files:", len(valid_files))
    print("Skipped files:", skipped)
    print("Saved valid file list to:", VALID_FILES_JSON)
    print("Saved stats to:", STATS_JSON)


if __name__ == "__main__":
    main()