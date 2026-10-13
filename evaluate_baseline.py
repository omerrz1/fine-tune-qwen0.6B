
import json
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "Qwen/Qwen3-0.6B-MLX-4bit"

DATA_FILE = Path("data/valid.jsonl")
OUTPUT_FILE = Path("results/baseline_valid5.jsonl")

LIMIT = 5
MAX_NEW_TOKENS = 160

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Missing data/valid.jsonl. "
        "Run prepare_data.py first."
    )

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Load the pretrained model
model, tokenizer = load(MODEL_ID)

sampler = make_sampler(temp=0.0)

# Read validation examples
with DATA_FILE.open("r", encoding="utf-8") as f:
    examples = [
        json.loads(line)
        for _, line in zip(range(LIMIT), f)
    ]

# Evaluate each example
with OUTPUT_FILE.open("w", encoding="utf-8") as out:

    for i, example in enumerate(examples, start=1):

        messages = example["messages"]

        # The last message contains the correct answer.
        reference = messages[-1]["content"]

        # Give the model only system and user messages.
        input_messages = messages[:-1]

        prompt = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        prediction = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=MAX_NEW_TOKENS,
            sampler=sampler,
            verbose=False
        ).strip()

        result = {
            "example_id": i,
            "model": MODEL_ID,
            "prediction": prediction,
            "reference": reference,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0
        }

        out.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

        print(f"\nEXAMPLE {i}")
        print("MODEL:", prediction)
        print("HUMAN:", reference)
        print("-" * 60)

print(f"\nResults saved to {OUTPUT_FILE}")
