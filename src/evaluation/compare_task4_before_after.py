import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURVEY_DIR = PROJECT_ROOT / "outputs" / "survey_results"

BEFORE_SUMMARY_JSON = SURVEY_DIR / "task4_reward_summary.json"
AFTER_SUMMARY_JSON = SURVEY_DIR / "task4_after_rl_reward_summary.json"

BEFORE_REWARD_CSV = SURVEY_DIR / "task4_reward_scores.csv"
AFTER_REWARD_CSV = SURVEY_DIR / "task4_after_rl_reward_scores.csv"

OUTPUT_JSON = SURVEY_DIR / "task4_before_vs_after_summary.json"
OUTPUT_CSV = SURVEY_DIR / "task4_before_vs_after_per_sample.csv"
OUTPUT_PLOT = SURVEY_DIR / "task4_before_vs_after_comparison.png"


# =========================================================
# Helpers
# =========================================================
def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reward_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def ensure_parent_dirs():
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)


def sort_sample_ids(df):
    df = df.copy()
    df["sample_num"] = df["sample_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("sample_num").drop(columns=["sample_num"])
    return df


def plot_comparison(before_mean, after_mean, per_sample_df, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Overall comparison
    axes[0].bar(["Before RL", "After RL"], [before_mean, after_mean])
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel("Average Human Score")
    axes[0].set_title("Task 4 Before vs After RLHF")

    # Per-sample comparison
    x = range(len(per_sample_df))
    width = 0.35

    axes[1].bar(
        [i - width / 2 for i in x],
        per_sample_df["before_mean_human_score"],
        width=width,
        label="Before RL"
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        per_sample_df["after_mean_human_score"],
        width=width,
        label="After RL"
    )

    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(per_sample_df["pair_id"], rotation=30)
    axes[1].set_ylim(0, 5)
    axes[1].set_ylabel("Mean Human Score")
    axes[1].set_title("Per-Sample Before vs After RLHF")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    ensure_parent_dirs()

    before_summary = load_json(BEFORE_SUMMARY_JSON)
    after_summary = load_json(AFTER_SUMMARY_JSON)

    before_df = load_reward_csv(BEFORE_REWARD_CSV)
    after_df = load_reward_csv(AFTER_REWARD_CSV)

    before_df = sort_sample_ids(before_df)
    after_df = sort_sample_ids(after_df)

    before_mean = float(before_summary["overall_before_rl_mean_score"])
    after_mean = float(after_summary["overall_after_rl_mean_score"])
    improvement = after_mean - before_mean

    # Pair samples by order: before_rl_01 <-> after_rl_01, etc.
    pair_count = min(len(before_df), len(after_df))
    paired_rows = []

    for i in range(pair_count):
        paired_rows.append({
            "pair_id": f"sample_{i + 1:02d}",
            "before_sample_id": before_df.iloc[i]["sample_id"],
            "after_sample_id": after_df.iloc[i]["sample_id"],
            "before_mean_human_score": float(before_df.iloc[i]["mean_human_score"]),
            "after_mean_human_score": float(after_df.iloc[i]["mean_human_score"]),
            "improvement": float(after_df.iloc[i]["mean_human_score"] - before_df.iloc[i]["mean_human_score"]),
        })

    per_sample_df = pd.DataFrame(paired_rows)
    per_sample_df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "task": "Task 4 Before vs After RLHF Comparison",
        "before_num_participants": int(before_summary["num_participants"]),
        "after_num_participants": int(after_summary["num_participants"]),
        "before_mean_human_score": before_mean,
        "after_mean_human_score": after_mean,
        "absolute_improvement": improvement,
        "relative_improvement_percent": (improvement / before_mean * 100.0) if before_mean != 0 else None,
        "before_total_ratings": int(before_summary["total_ratings"]),
        "after_total_ratings": int(after_summary["total_ratings"]),
        "score_range": [1, 5]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_comparison(before_mean, after_mean, per_sample_df, OUTPUT_PLOT)

    print("Task 4 before-vs-after comparison complete.\n")
    print("Saved:")
    print("-", OUTPUT_JSON)
    print("-", OUTPUT_CSV)
    print("-", OUTPUT_PLOT)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    print("\nPer-sample comparison:")
    print(per_sample_df.to_string(index=False))


if __name__ == "__main__":
    main()