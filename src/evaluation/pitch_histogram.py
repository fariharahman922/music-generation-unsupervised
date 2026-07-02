import pretty_midi
import numpy as np


def get_note_pitches(midi_path):
    """
    Collect non-drum note pitches from a MIDI file.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    pitches = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitches.append(note.pitch)

    return pitches


def compute_pitch_histogram(midi_path):
    """
    Compute normalized 12-bin pitch-class histogram for one MIDI file.
    Returns shape: (12,)
    """
    pitches = get_note_pitches(midi_path)

    hist = np.zeros(12, dtype=np.float32)

    if len(pitches) == 0:
        return hist

    for pitch in pitches:
        pitch_class = pitch % 12
        hist[pitch_class] += 1.0

    hist /= hist.sum()
    return hist


def aggregate_pitch_histogram(midi_paths):
    """
    Compute aggregate normalized pitch-class histogram across many MIDI files.
    Returns shape: (12,)
    """
    hist = np.zeros(12, dtype=np.float32)

    for midi_path in midi_paths:
        hist += compute_pitch_histogram(midi_path)

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


def pitch_histogram_distance(p, q):
    """
    PDF metric:
        H(p, q) = sum_i |p_i - q_i|

    Lower is better.
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return float(np.sum(np.abs(p - q)))


if __name__ == "__main__":
    print("pitch_histogram.py loaded successfully.")