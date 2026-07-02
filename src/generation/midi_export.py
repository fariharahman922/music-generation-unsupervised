import os
import numpy as np
import pretty_midi

LOWEST_PITCH = 21
DEFAULT_TEMPO = 120
STEPS_PER_BAR = 16
BEATS_PER_BAR = 4

def piano_roll_to_pretty_midi(piano_roll, tempo=DEFAULT_TEMPO, lowest_pitch=LOWEST_PITCH):
    """
    Convert binary piano-roll [time_steps, 88] to PrettyMIDI object.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    seconds_per_beat = 60.0 / tempo
    seconds_per_step = seconds_per_beat / (STEPS_PER_BAR / BEATS_PER_BAR)

    time_steps, n_pitches = piano_roll.shape

    for pitch_idx in range(n_pitches):
        pitch = lowest_pitch + pitch_idx
        active = False
        note_start = 0

        for t in range(time_steps):
            value = piano_roll[t, pitch_idx]

            if value == 1 and not active:
                active = True
                note_start = t
            elif value == 0 and active:
                active = False
                note_end = t

                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=note_start * seconds_per_step,
                    end=note_end * seconds_per_step
                )
                instrument.notes.append(note)

        if active:
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=note_start * seconds_per_step,
                end=time_steps * seconds_per_step
            )
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm


def save_piano_roll_as_midi(piano_roll, output_path, tempo=DEFAULT_TEMPO):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm = piano_roll_to_pretty_midi(piano_roll, tempo=tempo)
    pm.write(output_path)