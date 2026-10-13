
import argparse
import json
from pathlib import Path
from statistics import mean

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from rouge_score import rouge_scorer

MODEL_ID = "Qwen/Qwen3-0.6B-MLX-4bit"

ADAPTERS = {
    "base": None,
    "step150": "adapters/dialogsum-v1-step150",
    "step200": "adapters/dialogsum-v1",
}

MAX_NEW_TOKENS = 160

# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    choices=list(ADAPTERS),
    required=True
)
parser.add_argument(
    "--limit",
    type=int,
    default=5
)
args = parser.parse_args()

if not 1 <= args.limit <= 100:
    parser.error("--limit must be between 1 and 100")

# Load the validation examples
data_file = Path("data/valid.jsonl")

with data_file.open("r", encoding="utf-8") as f:
    examples = [
        json.loads(line)
        for line in f
        if line.strip()
    ][:args.limit]

if len(examples) != args.limit:
    raise ValueError("Not enough validation examples")

# Load the selected model
print(f"\nLoading: {args.variant}")

adapter_path = ADAPTERS[args.variant]

if adapter_path is not None:
    if not Path(adapter_path, "adapters.safetensors").exists():
        raise FileNotFoundError(
            f"Missing adapter weights in {adapter_path}"
        )

model, tokenizer = load(
    MODEL_ID,
    adapter_path=adapter_path
)

sampler = make_sampler(temp=0.0)

# Initialize ROUGE
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

# Create output file
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output_file = (
    output_dir
    / f"eval_{args.variant}_{args.limit}.jsonl"
)

results = []

with output_file.open("w", encoding="utf-8") as out:

    for i, example in enumerate(examples):

        messages = example["messages"]

        # The reference is the final assistant answer
        reference = messages[-1]["content"]

        # Exclude the correct answer from model input
        input_messages = messages[:-1]

        # Format exactly as we did for baseline inference
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

        # Calculate ROUGE F1 scores
        scores = scorer.score(reference, prediction)

        rouge = {
            key: value.fmeasure
            for key, value in scores.items()
        }

        result = {
            "example_id": i,
            "variant": args.variant,
            "dialogue": input_messages[-1]["content"],
            "reference": reference,
            "prediction": prediction,
            "rouge": rouge
        }

        results.append(result)

        out.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        out.flush()

        print(
            f"[{i + 1}/{len(examples)}] "
            f"ROUGE-L: {rouge['rougeL']:.3f}"
        )

# Summarize the results
print(f"\nRESULTS: {args.variant}")

for metric in ["rouge1", "rouge2", "rougeL"]:
    average = mean(
        r["rouge"][metric]
        for r in results
    )
    print(f"{metric}: {average:.4f}")

print(f"\nSaved predictions to: {output_file}")
