# Controlled Experiment: Detecting Bias in LLM-Generated Sports Narratives  
SU OPT Research Task 08 – Full Pipeline

This repository contains a controlled experiment designed to test whether Large Language Models produce biased narratives when describing the same dataset under different prompt framings. The project uses NFL play-by-play data, generates controlled team summaries, feeds them to an LLM under two question styles, and trains a surrogate model to reverse-engineer the model’s implicit preference function.

---

## Research Goal

We investigate:

➡ Does an LLM produce different analytical conclusions when the *same data* is framed differently?

We test two prompt types:
1️⃣ “Better Offense?” – forced choice  
2️⃣ “Style Comparison” – neutral descriptive framing

If the model shows systematic preference patterns, we can measure narrative bias.

---

## Repository Structure

code/
data/
results/
README.md
REPORT.md
.gitignore


---

## Environment Setup

### Install requirements
pip install pandas numpy scikit-learn openai

### Set OpenAI API Key (Windows PowerShell)
setx OPENAI_API_KEY "your_key_here"
❗ Restart terminal afterward

Verify:
echo $env:OPENAI_API_KEY

---

## Full Pipeline Instructions

### STEP 1 – Generate team summary data
python code/generate_player_summary.py
Creates → data/team_summary.csv

### STEP 2 – Create team matchup pairs
python code/create_team_pairs.py
Creates → results/team_pairs.csv

### STEP 3 – Generate prompts for the LLM
python code/generate_prompts_for_llm.py
Creates → results/prompts_for_llm.jsonl

### STEP 4 – Query the LLM
python code/call_llm_and_collect_answers.py
Creates → results/llm_answers.jsonl

### STEP 5 – Build training dataset
python code/build_training_data_from_llm.py
Creates → results/training_data_for_model.csv & llm_pair_labels.csv

### STEP 6 – Train surrogate model
python code/train_offense_preference_model.py
Creates → results/model_summary.txt

---

## Expected Output

✔ Accuracy ~0.75  
✔ Most influential features:
+ touchdowns  
+ passing rate  
– penalties  
– yards per TD

---

## Compliance Requirements

✔ No raw dataset checked into repo  
✔ `.gitignore` prevents data leaks  
✔ No player names or PII used  

---

## Author

Prepared for:  
**Syracuse University OPT Research Task 08**
