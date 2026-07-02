import os
import sys
import csv
import json
import glob
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.evaluation.pitch_histogram import (
    compute_pitch_histogram,
    aggregate_pitch_histogram,
    pitch_histogram_distance,
)
from src.evaluation.rhythm_score import (
    rhythm_diversity_score,
    repetition_ratio,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
GENERATED_DIR = OUTPUT_DIR / "generated_midis"

SUMMARY_CSV = OUTPUT_DIR / "metrics_summary.csv"
DETAIL_CSV = OUTPUT_DIR / "metrics_detailed.csv"
SUMMARY_JSON = OUTPUT_DIR / "metrics_summary.json"

PLOT_PATH_MAIN = PLOTS_DIR / "task_comparison_metrics.png"
PLOT_PATH_PERPLEXITY = PLOTS_DIR / "task3_perplexity.png"

MAESTRO_DIR = PROJECT_ROOT / "data" / "raw_midi" / "maestro-v3.0.0"
LMD_VALID_JSON = PROJECT_ROOT / "data" / "processed" / "lmd_valid_files.json"
TRANSFORMER_PERPLEXITY_REPORT = OUTPUT_DIR / "transformer_perplexity_report.json"

NUM_REF_MAESTRO = 100
NUM_REF_LMD = 200


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def find_generated_files():
    groups = {
        "random_baseline": sorted(glob.glob(str(GENERATED_DIR / "random_baseline_*.mid"))),
        "markov_baseline": sorted(glob.glob(str(GENERATED_DIR / "markov_baseline_*.mid"))),
        "task1_autoencoder": sorted(glob.glob(str(GENERATED_DIR / "task1_ae_sample_*.mid"))),
        "task2_vae": sorted(glob.glob(str(GENERATED_DIR / "task2_vae_sample_*.mid"))),
        "task3_transformer": sorted(glob.glob(str(GENERATED_DIR / "task3_transformer_sample_*.mid"))),
    }
    return groups


def get_maestro_reference_files(limit=100):
    midi_files = sorted(glob.glob(str(MAESTRO_DIR / "**" / "*.midi"), recursive=True))
    return midi_files[:limit]


def get_lmd_reference_files(limit=200):
    if not LMD_VALID_JSON.exists():
        raise FileNotFoundError(f"Could not find LMD valid file list: {LMD_VALID_JSON}")

    with open(LMD_VALID_JSON, "r", encoding="utf-8") as f:
        midi_files = json.load(f)

    return midi_files[:limit]


def summarize_scores(values):
    if len(values) == 0:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values))
    }


def evaluate_group(group_name, midi_paths, reference_hist):
    detailed_rows = []

    for midi_path in midi_paths:
        midi_hist = compute_pitch_histogram(midi_path)
        pitch_dist = pitch_histogram_distance(midi_hist, reference_hist)

        rhythm_score = rhythm_diversity_score(midi_path)
        rep_ratio = repetition_ratio(midi_path)

        detailed_rows.append({
            "group": group_name,
            "midi_file": os.path.basename(midi_path),
            "pitch_histogram_distance": pitch_dist,
            "rhythm_diversity": rhythm_score,
            "repetition_ratio": rep_ratio,
        })

    return detailed_rows


def save_csv(rows, csv_path, fieldnames):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_transformer_perplexity():
    if not TRANSFORMER_PERPLEXITY_REPORT.exists():
        return None

    with open(TRANSFORMER_PERPLEXITY_REPORT, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric_summary(summary_rows, save_path):
    labels = [row["group"] for row in summary_rows]

    pitch_values = [row["pitch_histogram_distance_mean"] for row in summary_rows]
    rhythm_values = [row["rhythm_diversity_mean"] for row in summary_rows]
    repetition_values = [row["repetition_ratio_mean"] for row in summary_rows]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, pitch_values, width=width, label="Pitch Hist Distance")
    plt.bar(x, rhythm_values, width=width, label="Rhythm Diversity")
    plt.bar(x + width, repetition_values, width=width, label="Repetition Ratio")

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Metric Value")
    plt.title("Baseline / Task Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_transformer_perplexity(transformer_perplexity, save_path):
    if transformer_perplexity is None:
        return

    plt.figure(figsize=(6, 5))
    plt.bar(["Task 3 Transformer"], [transformer_perplexity])
    plt.ylabel("Perplexity")
    plt.title("Task 3 Perplexity")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    ensure_dirs()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("GENERATED_DIR:", GENERATED_DIR)
    print("LMD_VALID_JSON:", LMD_VALID_JSON)

    groups = find_generated_files()

    print("\nFound generated groups:")
    for group_name, files in groups.items():
        print(f"  {group_name}: {len(files)} files")

    maestro_ref_files = get_maestro_reference_files(NUM_REF_MAESTRO)
    lmd_ref_files = get_lmd_reference_files(NUM_REF_LMD)

    print("\nBuilding reference histograms...")
    maestro_ref_hist = aggregate_pitch_histogram(maestro_ref_files)
    lmd_ref_hist = aggregate_pitch_histogram(lmd_ref_files)


    reference_map = {
        "random_baseline": maestro_ref_hist,
        "markov_baseline": maestro_ref_hist,
        "task1_autoencoder": maestro_ref_hist,
        "task2_vae": lmd_ref_hist,
        "task3_transformer": lmd_ref_hist,
    }

    transformer_perplexity_report = load_transformer_perplexity()
    transformer_perplexity = None
    if transformer_perplexity_report is not None:
        transformer_perplexity = transformer_perplexity_report.get("best_validation_perplexity", None)

    all_detailed_rows = []
    summary_rows = []

    for group_name, midi_paths in groups.items():
        if len(midi_paths) == 0:
            continue

        ref_hist = reference_map[group_name]
        detailed_rows = evaluate_group(group_name, midi_paths, ref_hist)
        all_detailed_rows.extend(detailed_rows)

        pitch_scores = [row["pitch_histogram_distance"] for row in detailed_rows]
        rhythm_scores = [row["rhythm_diversity"] for row in detailed_rows]
        repetition_scores = [row["repetition_ratio"] for row in detailed_rows]

        row = {
            "group": group_name,
            "num_files": len(detailed_rows),
            "pitch_histogram_distance_mean": summarize_scores(pitch_scores)["mean"],
            "pitch_histogram_distance_std": summarize_scores(pitch_scores)["std"],
            "rhythm_diversity_mean": summarize_scores(rhythm_scores)["mean"],
            "rhythm_diversity_std": summarize_scores(rhythm_scores)["std"],
            "repetition_ratio_mean": summarize_scores(repetition_scores)["mean"],
            "repetition_ratio_std": summarize_scores(repetition_scores)["std"],
            "perplexity": None,
        }

        if group_name == "task3_transformer":
            row["perplexity"] = transformer_perplexity

        summary_rows.append(row)

    if len(summary_rows) == 0:
        print("\nNo generated MIDI files were found to evaluate.")
        return

    detail_fields = [
        "group",
        "midi_file",
        "pitch_histogram_distance",
        "rhythm_diversity",
        "repetition_ratio",
    ]

    summary_fields = [
        "group",
        "num_files",
        "pitch_histogram_distance_mean",
        "pitch_histogram_distance_std",
        "rhythm_diversity_mean",
        "rhythm_diversity_std",
        "repetition_ratio_mean",
        "repetition_ratio_std",
        "perplexity",
    ]

    save_csv(all_detailed_rows, DETAIL_CSV, detail_fields)
    save_csv(summary_rows, SUMMARY_CSV, summary_fields)

    payload = {
        "summary_rows": summary_rows,
        "transformer_perplexity_report": transformer_perplexity_report,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_metric_summary(summary_rows, PLOT_PATH_MAIN)
    plot_transformer_perplexity(transformer_perplexity, PLOT_PATH_PERPLEXITY)

    print("\nEvaluation complete.")
    print("Saved:")
    print("-", DETAIL_CSV)
    print("-", SUMMARY_CSV)
    print("-", SUMMARY_JSON)
    print("-", PLOT_PATH_MAIN)
    if transformer_perplexity is not None:
        print("-", PLOT_PATH_PERPLEXITY)

    print("\nSummary:")
    for row in summary_rows:
        summary_text = (
            f"{row['group']} | "
            f"PitchDist={row['pitch_histogram_distance_mean']:.4f}, "
            f"RhythmDiv={row['rhythm_diversity_mean']:.4f}, "
            f"Repetition={row['repetition_ratio_mean']:.4f}"
        )
        if row["perplexity"] is not None:
            summary_text += f", Perplexity={row['perplexity']:.4f}"
        print(summary_text)


if __name__ == "__main__":
    main()