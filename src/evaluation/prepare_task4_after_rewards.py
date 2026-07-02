import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SURVEY_DIR = PROJECT_ROOT / "outputs" / "survey_results"
RAW_SURVEY_CSV = SURVEY_DIR / "After_RL_Survey.csv"

LONG_SURVEY_CSV = SURVEY_DIR / "task4_after_rl_survey_long_format.csv"
REWARD_CSV = SURVEY_DIR / "task4_after_rl_reward_scores.csv"
SUMMARY_JSON = SURVEY_DIR / "task4_after_rl_reward_summary.json"
AFTER_PLOT = SURVEY_DIR / "task4_after_rl_human_scores.png"


# =========================================================
# Helpers
# =========================================================
def validate_score(x):
    """
    Keep only valid human listening scores in [1, 5].
    """
    if pd.isna(x):
        return False

    try:
        value = float(x)
    except Exception:
        return False

    return value in {1, 2, 3, 4, 5}


def find_participant_column(df):
    """
    Try common participant identifier columns.
    """
    candidates = ["participant_id", "Participant ID", "ID", "Name", "name"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a participant identifier column. "
        "Expected one of: participant_id, Participant ID, ID, Name, name"
    )


def infer_after_rl_sample_columns(df, participant_col):
    """
    Infer sample rating columns from the Google Form export.

    We exclude obvious metadata columns and keep the remaining columns
    as the 10 after-RL sample columns, then map them in order to:
      after_rl_01 ... after_rl_10
    """
    excluded = {
        "Timestamp",
        "Email Address",
        participant_col,
    }

    sample_columns = [col for col in df.columns if col not in excluded]

    if len(sample_columns) != 10:
        raise ValueError(
            f"Expected exactly 10 after-RL sample columns, but found {len(sample_columns)}.\n"
            f"Detected columns: {sample_columns}"
        )

    sample_column_map = {}
    for i, column_name in enumerate(sample_columns, start=1):
        sample_id = f"after_rl_{i:02d}"
        sample_column_map[column_name] = sample_id

    return sample_column_map


def ensure_dirs():
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)


def plot_after_scores(reward_df, save_path):
    """
    Plot average human score per after-RL sample.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(reward_df["sample_id"], reward_df["mean_human_score"])
    plt.ylim(0, 5)
    plt.xlabel("Sample ID")
    plt.ylabel("Average Human Score")
    plt.title("Task 4 After-RL Human Scores")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    ensure_dirs()

    if not RAW_SURVEY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find survey CSV at: {RAW_SURVEY_CSV}\n"
            "Put your exported Google Form CSV there, or update RAW_SURVEY_CSV in this script."
        )

    df = pd.read_csv(RAW_SURVEY_CSV)
    print("Loaded survey CSV:", RAW_SURVEY_CSV)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    participant_col = find_participant_column(df)
    print("Using participant column:", participant_col)

    sample_column_map = infer_after_rl_sample_columns(df, participant_col)
    print("Detected after-RL sample columns:")
    for column_name, sample_id in sample_column_map.items():
        print(f"  {column_name} -> {sample_id}")

    # -----------------------------------------------------
    # Convert wide survey to long format
    # -----------------------------------------------------
    long_rows = []

    for _, row in df.iterrows():
        participant_id = str(row[participant_col]).strip()

        for column_name, sample_id in sample_column_map.items():
            score = row[column_name]

            if not validate_score(score):
                continue

            long_rows.append({
                "participant_id": participant_id,
                "sample_id": sample_id,
                "overall_score_1_to_5": int(float(score))
            })

    long_df = pd.DataFrame(long_rows)

    if long_df.empty:
        raise ValueError("No valid survey scores found in the CSV.")

    long_df.to_csv(LONG_SURVEY_CSV, index=False)
    print("Saved long-format survey data to:", LONG_SURVEY_CSV)

    # -----------------------------------------------------
    # Aggregate reward scores per sample
    # -----------------------------------------------------
    reward_df = (
        long_df.groupby("sample_id")["overall_score_1_to_5"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={
            "mean": "mean_human_score",
            "count": "num_ratings",
            "std": "std_human_score"
        })
    )

    reward_df["std_human_score"] = reward_df["std_human_score"].fillna(0.0)

    reward_df["sample_num"] = reward_df["sample_id"].str.extract(r"(\d+)").astype(int)
    reward_df = reward_df.sort_values("sample_num").drop(columns=["sample_num"])

    reward_df.to_csv(REWARD_CSV, index=False)
    print("Saved reward scores to:", REWARD_CSV)

    # -----------------------------------------------------
    # Summary for after-RL human satisfaction
    # -----------------------------------------------------
    summary = {
        "num_participants": int(long_df["participant_id"].nunique()),
        "num_scored_samples": int(reward_df["sample_id"].nunique()),
        "total_ratings": int(len(long_df)),
        "overall_after_rl_mean_score": float(long_df["overall_score_1_to_5"].mean()),
        "overall_after_rl_std_score": float(long_df["overall_score_1_to_5"].std(ddof=0)),
        "reward_source": "Average human listening score per sample",
        "score_range": [1, 5]
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved reward summary to:", SUMMARY_JSON)

    # -----------------------------------------------------
    # Plot after-RL average human scores
    # -----------------------------------------------------
    plot_after_scores(reward_df, AFTER_PLOT)
    print("Saved after-RL human score plot to:", AFTER_PLOT)

    print("\nDone.")
    print("Summary:")
    print(json.dumps(summary, indent=2))

    print("\nPer-sample reward scores:")
    print(reward_df.to_string(index=False))


if __name__ == "__main__":
    main()