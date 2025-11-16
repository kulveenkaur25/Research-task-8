import json
import time
from pathlib import Path
from openai import OpenAI

# --------- PATHS ----------
ROOT = Path(".").parent  # so script in data/ can see project root
PROMPTS_PATH = ROOT / "results" / "prompts_for_llm.jsonl"
OUTPUT_PATH = ROOT / "results" / "llm_answers.jsonl"

print(f"📂 Reading prompts from: {PROMPTS_PATH}")

client = OpenAI()  # uses OPENAI_API_KEY from environment


def iter_prompts(path):
    """Yield each JSON record from the prompts file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def call_model(prompt_text: str) -> str:
    """
    Send one prompt to the LLM and return its answer text.
    You can change model name if needed (e.g. gpt-4.1, gpt-4o-mini).
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert NFL analytics writer. "
                    "Write clear, concise football analysis using the stats provided, "
                    "without inventing new statistics."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content


def main():
    prompts = list(iter_prompts(PROMPTS_PATH))
    total = len(prompts)
    print(f"🧮 Found {total} prompts to send to the model.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(prompts, start=1):
            prompt_text = rec["prompt"]

            print(f"⚙️  Calling model for prompt {i}/{total} "
                  f"(pair_id={rec['pair_id']}, type={rec['prompt_type']})")

            try:
                answer = call_model(prompt_text)
            except Exception as e:
                print(f"❌ Error on prompt {i}: {e}")
                # Save the error and continue
                rec_out = {**rec, "answer": None, "error": str(e)}
                out_f.write(json.dumps(rec_out) + "\n")
                out_f.flush()
                continue

            # Merge original prompt record + answer
            rec_out = {**rec, "answer": answer}
            out_f.write(json.dumps(rec_out) + "\n")
            out_f.flush()

            print(f"✅ Done {i}/{total}")
            # Optional small pause to be gentle with rate limits
            time.sleep(0.2)

    print(f"\n🎉 All done!")
    print(f"   Saved answers to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
