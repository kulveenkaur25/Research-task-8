import pandas as pd
from pathlib import Path
import json

# ---------- PATHS ----------
ROOT = Path(".")  # assuming you run from: New folder (2)
TEAM_SUMMARY_PATH = ROOT / "data" / "team_summary.csv"
TEAM_PAIRS_PATH = ROOT / "results" / "team_pairs.csv"
OUTPUT_PATH = ROOT / "results" / "prompts_for_llm.jsonl"

print(f"📂 Loading team summary from: {TEAM_SUMMARY_PATH}")
team_df = pd.read_csv(TEAM_SUMMARY_PATH)

print(f"📂 Loading team pairs from: {TEAM_PAIRS_PATH}")
pairs_df = pd.read_csv(TEAM_PAIRS_PATH)

# Make sure expected columns exist
required_team_cols = [
    "OffenseTeam", "total_plays", "total_yards", "avg_yards_per_play",
    "rush_plays", "pass_plays", "touchdowns", "penalties",
    "rush_pct", "pass_pct", "yards_per_touchdown"
]

missing_cols = [c for c in required_team_cols if c not in team_df.columns]
if missing_cols:
    raise ValueError(f"These required columns are missing in team_summary.csv: {missing_cols}")

required_pair_cols = ["pair_id", "teamA", "teamB"]
missing_pairs = [c for c in required_pair_cols if c not in pairs_df.columns]
if missing_pairs:
    raise ValueError(f"These required columns are missing in team_pairs.csv: {missing_pairs}")

# ---------- HELPER: DESCRIBE A TEAM ----------
def describe_team(team_name, row):
    """
    Turn one row from team_summary into a human-readable summary string.
    Note: team_name is passed separately because OffenseTeam is used as index.
    """
    return (
        f"{team_name} ran {int(row['total_plays'])} plays, gaining "
        f"{int(row['total_yards'])} total yards "
        f"({row['avg_yards_per_play']:.2f} yards per play). "
        f"They rushed {int(row['rush_plays'])} times and passed {int(row['pass_plays'])} times "
        f"(rush_pct={row['rush_pct']:.1f}, pass_pct={row['pass_pct']:.1f}). "
        f"They scored {int(row['touchdowns'])} touchdowns, took {int(row['penalties'])} penalties, "
        f"and averaged {row['yards_per_touchdown']:.2f} yards per touchdown."
    )

# Index team_df by team name for fast lookup
team_lookup = {t: row for t, row in team_df.set_index("OffenseTeam").iterrows()}

records = []
num_pairs = 0
num_prompts = 0

for _, pair in pairs_df.iterrows():
    pair_id = pair["pair_id"]
    teamA = pair["teamA"]
    teamB = pair["teamB"]

    if teamA not in team_lookup or teamB not in team_lookup:
        print(f"⚠️ Skipping pair {pair_id}: missing stats for {teamA} or {teamB}")
        continue

    rowA = team_lookup[teamA]
    rowB = team_lookup[teamB]

    # 🔹 FIX: pass team name separately
    descA = describe_team(teamA, rowA)
    descB = describe_team(teamB, rowB)

    # ---------- PROMPT TYPE 1: Which offense is better? ----------
    prompt_better = f"""
You are an NFL offensive analytics expert.

Below are summaries for two teams' offenses from the same season.

Team A:
{descA}

Team B:
{descB}

Question:
Based ONLY on the numbers above (and not on reputation or history), which offense appears stronger overall, Team A or Team B? 
Choose one team and explain your reasoning in 3–5 sentences, citing specific stats (like yards, efficiency, or penalties) in your explanation.
"""

    records.append({
        "pair_id": pair_id,
        "prompt_type": "better_offense",
        "teamA": teamA,
        "teamB": teamB,
        "prompt": prompt_better.strip()
    })
    num_prompts += 1

    # ---------- PROMPT TYPE 2: Style comparison ----------
    prompt_style = f"""
You are a football strategy analyst.

Here are offensive summaries for two NFL teams.

Team A:
{descA}

Team B:
{descB}

Question:
Compare the offensive STYLES of Team A and Team B. 
Do they look more run-heavy or pass-heavy? 
Discuss how their play selection (rush vs pass), efficiency (yards per play), and discipline (penalties) might influence the kind of game plan each team prefers. 
Answer in 3–5 sentences.
"""

    records.append({
        "pair_id": pair_id,
        "prompt_type": "style_comparison",
        "teamA": teamA,
        "teamB": teamB,
        "prompt": prompt_style.strip()
    })
    num_prompts += 1

    num_pairs += 1

# ---------- WRITE JSONL ----------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"\n✅ Finished generating prompts.")
print(f"   Pairs used    : {num_pairs}")
print(f"   Total prompts : {num_prompts}")
print(f"   Saved to      : {OUTPUT_PATH}")

# Show a quick preview of the first few prompts
print("\n🔍 Preview of first 2 prompts:\n")
for rec in records[:2]:
    print(f"pair_id={rec['pair_id']} | type={rec['prompt_type']}")
    print(rec["prompt"])
    print("-" * 80)
