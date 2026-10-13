
import argparse
import json
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=5)
args = parser.parse_args()

variants = ["base", "step150", "step200"]
metrics = ["rouge1", "rouge2", "rougeL"]

data = {}

for variant in variants:
    path = Path(
        f"results/eval_{variant}_{args.limit}.jsonl"
    )

    with path.open("r", encoding="utf-8") as f:
        data[variant] = [
            json.loads(line) for line in f
        ]

# Ensure we're comparing the same examples
for variant in variants:
    if len(data[variant]) != args.limit:
        raise ValueError("Unexpected number of examples")

    for base_row, row in zip(data["base"], data[variant]):
        if (
            base_row["example_id"] != row["example_id"]
            or base_row["reference"] != row["reference"]
            or base_row["dialogue"] != row["dialogue"]
        ):
            raise ValueError("Evaluation examples do not match")

# Print aggregate metric comparison
print("\nMODEL COMPARISON")
print(f"{'Model':<12} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
print("-" * 45)

averages = {}

for variant in variants:
    averages[variant] = {}

    for metric in metrics:
        averages[variant][metric] = mean(
            row["rouge"][metric]
            for row in data[variant]
        )

    values = averages[variant]

    print(
        f"{variant:<12} "
        f"{values['rouge1'] * 100:>9.2f} "
        f"{values['rouge2'] * 100:>9.2f} "
        f"{values['rougeL'] * 100:>9.2f}"
    )

# Compare against original Qwen
baseline = averages["base"]["rougeL"]

for variant in ["step150", "step200"]:
    delta = (
        averages[variant]["rougeL"] - baseline
    ) * 100

    print(
        f"\n{variant} ROUGE-L change "
        f"vs base: {delta:+.2f} points"
    )

# Show an actual side-by-side example
print("\n" + "=" * 60)
print("EXAMPLE COMPARISON")
print("=" * 60)

print("\nCONVERSATION:")
print(data["base"][0]["dialogue"])

print("\nHUMAN REFERENCE:")
print(data["base"][0]["reference"])

for variant in variants:
    print(f"\n{variant.upper()} SUMMARY:")
    print(data[variant][0]["prediction"])
