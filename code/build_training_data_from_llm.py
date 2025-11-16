import json
from pathlib import Path

import pandas as pd


# ---------- Paths ----------
TEAM_SUMMARY_PATH = Path("data/team_summary.csv")
TEAM_PAIRS_PATH = Path("results/team_pairs.csv")
LLM_ANSWERS_PATH = Path("results/llm_answers.jsonl")

OUT_LABELS_PATH = Path("results/llm_pair_labels.csv")
OUT_TRAIN_PATH = Path("results/training_data_for_model.csv")


def extract_choice(answer_text: str):
    """
    Try to detect whether the LLM chose Team A or Team B
    based on the free-form explanation text.

    Heuristic:
      - if only 'team a' appears  -> A
      - if only 'team b' appears  -> B
      - if both appear -> whichever appears first
      - if neither appears -> None (we'll keep but mark as unknown)
    """
    if not isinstance(answer_text, str):
        return None

    t = answer_text.lower()

    has_a = "team a" in t
    has_b = "team b" in t

    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    if has_a and has_b:
        idx_a = t.index("team a")
        idx_b = t.index("team b")
        return "A" if idx_a < idx_b else "B"

    return None


def main():
    print(f"📂 Loading team summary from: {TEAM_SUMMARY_PATH}")
    print(f"📂 Loading team pairs from:   {TEAM_PAIRS_PATH}")
    print(f"📂 Loading LLM answers from:  {LLM_ANSWERS_PATH}")

    team_summary = pd.read_csv(TEAM_SUMMARY_PATH)
    pairs = pd.read_csv(TEAM_PAIRS_PATH)

    # ---- Step 1: read LLM answers ----
    records = []
    with open(LLM_ANSWERS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Try to be robust to slightly different key names
            pair_id = obj.get("pair_id")
            qtype = obj.get("type") or obj.get("prompt_type") or obj.get("question_type")
            answer_text = (
                obj.get("answer")
                or obj.get("response")
                or obj.get("model_answer")
                or obj.get("content")
            )

            # We only use 'better_offense' prompts for labels
            if qtype != "better_offense":
                continue

            choice = extract_choice(answer_text)

            records.append(
                {
                    "pair_id": pair_id,
                    "question_type": qtype,
                    "answer_text": answer_text,
                    "choice": choice,  # "A" / "B" / None
                }
            )

    df_labels = pd.DataFrame(records)
    print("\n🧾 Raw label rows from LLM:")
    print(df_labels.head(5))

    unknown = df_labels["choice"].isna().sum()
    total = len(df_labels)
    print(f"\nℹ️ Parsed choices for {total - unknown}/{total} answers.")
    if unknown > 0:
        print("   Some answers did not clearly say 'Team A' or 'Team B'.")

    # ---- Step 2: join with pairs to know which teams A/B are ----
    df_labels = df_labels.merge(pairs, on="pair_id", how="left", validate="m:1")

    # Create binary target: 1 if LLM prefers teamA, 0 if it prefers teamB, NaN if unknown
    df_labels["llm_prefers_teamA"] = df_labels["choice"].map({"A": 1, "B": 0})

    print("\n✅ Saving pair-level labels to:", OUT_LABELS_PATH)
    df_labels.to_csv(OUT_LABELS_PATH, index=False)

    print("\n🔍 Preview of saved labels:")
    print(df_labels[["pair_id", "teamA", "teamB", "choice", "llm_prefers_teamA"]].head(5))

    # ---- Step 3: build ML-ready features using team_summary ----
    # team_summary currently has 'OffenseTeam' as team name
    teams = team_summary.rename(columns={"OffenseTeam": "team"})

    # Choose which numeric columns to use as features
    feature_cols = [
        "total_plays",
        "total_yards",
        "avg_yards_per_play",
        "rush_plays",
        "pass_plays",
        "touchdowns",
        "penalties",
        "rush_pct",
        "pass_pct",
        "yards_per_touchdown",
    ]

    teams_small = teams[["team"] + feature_cols].copy()

    # Create separate copies for teamA and teamB, then merge
    teams_A = teams_small.copy()
    teams_A.columns = [
        "teamA" if c == "team" else f"teamA_{c}" for c in teams_A.columns
    ]

    teams_B = teams_small.copy()
    teams_B.columns = [
        "teamB" if c == "team" else f"teamB_{c}" for c in teams_B.columns
    ]

    df_train = df_labels.merge(teams_A, on="teamA", how="left").merge(
        teams_B, on="teamB", how="left"
    )

    # Optional: add difference features (teamA_stat - teamB_stat)
    for c in feature_cols:
        df_train[f"diff_{c}"] = df_train[f"teamA_{c}"] - df_train[f"teamB_{c}"]

    # Keep a clean subset for modeling: differences + target
    diff_cols = [f"diff_{c}" for c in feature_cols]

    model_df = df_train[["pair_id", "teamA", "teamB", "llm_prefers_teamA"] + diff_cols]

    print("\n✅ Saving ML-ready training data to:", OUT_TRAIN_PATH)
    model_df.to_csv(OUT_TRAIN_PATH, index=False)

    print("\n🔍 Preview of training data:")
    print(model_df.head(5))

    print("\n🎉 Done. You now have:")
    print(f"   - Pair labels: {OUT_LABELS_PATH}")
    print(f"   - Training data: {OUT_TRAIN_PATH}")


if __name__ == "__main__":
    main()
