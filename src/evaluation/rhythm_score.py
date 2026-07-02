import numpy as np
import pretty_midi


def get_note_durations(midi_path, quantize_digits=2):
    """
    Collect quantized note durations from a MIDI file.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    durations = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            duration = max(0.0, note.end - note.start)
            duration = round(duration, quantize_digits)
            durations.append(duration)

    return durations


def rhythm_diversity_score(midi_path):
    """
    PDF metric:
        D_rhythm = #unique durations / #total notes

    Higher means more rhythmic diversity.
    """
    durations = get_note_durations(midi_path)

    if len(durations) == 0:
        return 0.0

    unique_durations = len(set(durations))
    total_notes = len(durations)

    return float(unique_durations / total_notes)


def repetition_ratio(midi_path, fs=8, pattern_length=4):
    """
    Approximate repetition ratio from piano-roll patterns:
        R = #repeated patterns / #total patterns

    We convert MIDI to a binary piano-roll and look at repeated
    short time-step patterns.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = pm.get_piano_roll(fs=fs)

    if piano_roll.shape[1] == 0:
        return 0.0

    piano_roll = (piano_roll > 0).astype(np.uint8).T  # shape: [time, pitch]

    if piano_roll.shape[0] < pattern_length:
        return 0.0

    patterns = []
    for i in range(piano_roll.shape[0] - pattern_length + 1):
        pattern = piano_roll[i:i + pattern_length].flatten()
        patterns.append(tuple(pattern.tolist()))

    total_patterns = len(patterns)
    if total_patterns == 0:
        return 0.0

    counts = {}
    for pattern in patterns:
        counts[pattern] = counts.get(pattern, 0) + 1

    repeated_count = sum(count for count in counts.values() if count > 1)

    return float(repeated_count / total_patterns)


if __name__ == "__main__":
    print("rhythm_score.py loaded successfully.")