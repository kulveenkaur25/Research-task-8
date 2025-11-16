import os
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
TEAM_SUMMARY_PATH = os.path.join("data", "team_summary.csv")
OUTPUT_PATH = os.path.join("results", "team_pairs.csv")
# ----------------------------

def normalize(series: pd.Series) -> pd.Series:
    """
    Safe normalization: (x / max(x)).
    If max is 0 or NaN, returns 0 for all.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    max_val = s.max()
    if max_val is None or max_val == 0 or np.isnan(max_val):
        return pd.Series(0.0, index=s.index)
    return s / max_val

def main():
    print(f"📂 Loading team summary from: {TEAM_SUMMARY_PATH}")

    if not os.path.exists(TEAM_SUMMARY_PATH):
        raise FileNotFoundError(f"Could not find {TEAM_SUMMARY_PATH}. Make sure Step 3 ran successfully.")

    df = pd.read_csv(TEAM_SUMMARY_PATH)
    print("\nAvailable columns in team_summary.csv:")
    print(list(df.columns))

    # --- Check that OffenseTeam exists ---
    if "OffenseTeam" not in df.columns:
        raise KeyError("Column 'OffenseTeam' not found in team_summary.csv")

    # --- OPTIONAL: If win_pct or avg_points_for exist, use them, otherwise build our own strength ---
    strength_col = None

    if "win_pct" in df.columns:
        strength_col = "win_pct"
        print("\n✅ Using 'win_pct' as strength metric.")
    elif "avg_points_for" in df.columns:
        strength_col = "avg_points_for"
        print("\n✅ Using 'avg_points_for' as strength metric.")
    else:
        print("\n⚠️ No 'win_pct' or 'avg_points_for' found.")
        print("   ➜ Building a custom strength metric from offensive stats instead.")

        # Make sure required columns exist (otherwise treat missing as 0)
        for c in ["total_yards", "avg_yards_per_play", "touchdowns", "penalties"]:
            if c not in df.columns:
                print(f"   ⚠️ Column '{c}' not in file – treating it as 0.")
                df[c] = 0

        # Custom strength formula:
        # - More total_yards = stronger
        # - More avg_yards_per_play = stronger
        # - More touchdowns = stronger
        # - More penalties = weaker
        df["__strength__"] = (
            0.35 * normalize(df["total_yards"]) +
            0.30 * normalize(df["avg_yards_per_play"]) +
            0.25 * normalize(df["touchdowns"]) -
            0.10 * normalize(df["penalties"].abs())
        )

        strength_col = "__strength__"
        print("\n✅ Custom strength metric '__strength__' created using:")
        print("   35% total_yards, 30% avg_yards_per_play, 25% touchdowns, -10% penalties")

    # --- Prepare dataframe with just team + strength ---
    team_df = df[["OffenseTeam", strength_col]].copy()
    team_df = team_df.rename(columns={"OffenseTeam": "team", strength_col: "strength"})

    # Fill NaNs in strength (if any)
    team_df["strength"] = pd.to_numeric(team_df["strength"], errors="coerce").fillna(0.0)

    # Sort teams by strength (strongest first)
    team_df = team_df.sort_values("strength", ascending=False).reset_index(drop=True)

    print("\n🏋️ First few teams with strength:")
    print(team_df.head())

    # --- Create pairs ---
    pairs = []
    num_teams = len(team_df)

    if num_teams < 2:
        raise ValueError("Need at least 2 teams to create pairs.")

    print(f"\n🔗 Creating pairs for {num_teams} teams...")

    # Pair 0-1, 2-3, 4-5, ...
    pair_id = 1
    i = 0
    while i < num_teams - 1:
        teamA = team_df.iloc[i]
        teamB = team_df.iloc[i + 1]

        pairs.append({
            "pair_id": f"PAIR_{pair_id}",
            "teamA": teamA["team"],
            "teamB": teamB["team"],
            "teamA_strength": round(float(teamA["strength"]), 4),
            "teamB_strength": round(float(teamB["strength"]), 4),
        })

        pair_id += 1
        i += 2

    # If odd number of teams, last one is unpaired (we can drop or log it)
    if num_teams % 2 == 1:
        leftover_team = team_df.iloc[-1]["team"]
        print(f"\n⚠️ Odd number of teams. '{leftover_team}' has no pair and will be skipped.")

    pairs_df = pd.DataFrame(pairs)

    # Ensure results folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    pairs_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(pairs_df)} pairs to: {OUTPUT_PATH}\n")

    print("First few pairs:")
    print(pairs_df.head())

if __name__ == "__main__":
    main()
