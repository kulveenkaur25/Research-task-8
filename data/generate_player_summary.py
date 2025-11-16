import pandas as pd
from pathlib import Path

# === 1. Point this to your actual CSV file ===
DATA_PATH = Path(__file__).parent / "SHOT_ACCURACY.csv"  # change name if needed

# 2. Load data
df = pd.read_csv(DATA_PATH)

# 3. Drop useless 'Unnamed' columns
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

# 4. Quick sanity check – print columns once
print("Columns in dataset:")
print(df.columns.tolist())

# 5. Build TEAM-LEVEL SUMMARY using OffenseTeam
group_col = "OffenseTeam"  # main team column

team_summary = (
    df.groupby(group_col)
      .agg(
          total_plays=("GameId", "count"),
          total_yards=("Yards", "sum"),
          avg_yards_per_play=("Yards", "mean"),
          rush_plays=("IsRush", "sum"),
          pass_plays=("IsPass", "sum"),
          touchdowns=("IsTouchdown", "sum"),
          penalties=("IsPenalty", "sum"),
      )
      .reset_index()
)

# 6. Derived metrics
team_summary["rush_pct"] = team_summary["rush_plays"] / team_summary["total_plays"]
team_summary["pass_pct"] = team_summary["pass_plays"] / team_summary["total_plays"]
team_summary["yards_per_touchdown"] = team_summary["total_yards"] / team_summary["touchdowns"].replace(0, pd.NA)

# 7. Save output
OUT_PATH = Path(__file__).parent / "team_summary.csv"
team_summary.to_csv(OUT_PATH, index=False)

print(f"\nTeam summary saved to: {OUT_PATH}")

print("\nPreview:")
print(team_summary.head())
