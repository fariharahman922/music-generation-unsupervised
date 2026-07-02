import os
import numpy as np
from midi_export import save_piano_roll_as_midi

OUTPUT_DIR = "outputs/generated_midis"
NUM_SAMPLES = 5
TIME_STEPS = 128
N_PITCHES = 88

def generate_random_piano_roll(time_steps=128, n_pitches=88, note_prob=0.03):
    """
    Generate a random binary piano-roll.
    note_prob controls how dense the music is.
    """
    piano_roll = (np.random.rand(time_steps, n_pitches) < note_prob).astype(np.uint8)
    return piano_roll

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_SAMPLES):
        sample = generate_random_piano_roll(
            time_steps=TIME_STEPS,
            n_pitches=N_PITCHES,
            note_prob=0.03
        )

        output_path = os.path.join(OUTPUT_DIR, f"random_baseline_{i+1}.mid")
        save_piano_roll_as_midi(sample, output_path, tempo=120)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()