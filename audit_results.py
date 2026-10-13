
import json
import re
from collections import Counter
from pathlib import Path

VARIANTS = ["base", "step150", "step200"]

def load_results(variant):
    path = Path(f"results/eval_{variant}_100.jsonl")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["example_id"]: row for row in rows}

data = {v: load_results(v) for v in VARIANTS}

# Check that we have the same examples
ids = sorted(data["base"])

for variant in VARIANTS:
    assert sorted(data[variant]) == ids

# Measure repeated three-word sequences
def repetition_rate(text):
    words = re.findall(r"\w+", text.lower())

    if len(words) < 3:
        return 0.0

    trigrams = [
        tuple(words[i:i+3])
        for i in range(len(words) - 2)
    ]

    counts = Counter(trigrams)
    duplicates = sum(
        count - 1 for count in counts.values()
    )

    return duplicates / len(trigrams)

print("\n1. HOW OFTEN DOES FINE-TUNING WIN?\n")

for variant in ["step150", "step200"]:
    wins = 0
    losses = 0
    ties = 0

    for i in ids:
        base = data["base"][i]["rouge"]["rougeL"]
        tuned = data[variant][i]["rouge"]["rougeL"]

        if tuned > base:
            wins += 1
        elif tuned < base:
            losses += 1
        else:
            ties += 1

    print(
        f"{variant}: {wins} wins, "
        f"{losses} losses, {ties} ties"
    )

print("\n2. REPETITION DIAGNOSTICS\n")

for variant in VARIANTS:
    rates = [
        repetition_rate(
            data[variant][i]["prediction"]
        )
        for i in ids
    ]

    flagged = sum(r > 0.30 for r in rates)

    print(
        f"{variant}: "
        f"average repetition={sum(rates)/len(rates):.3f}, "
        f"flagged outputs={flagged}/100"
    )

print("\n3. MOST REPETITIVE STEP-200 OUTPUTS\n")

worst = sorted(
    ids,
    key=lambda i: repetition_rate(
        data["step200"][i]["prediction"]
    ),
    reverse=True
)[:3]

for i in worst:
    print("\n" + "=" * 65)
    print(f"EXAMPLE {i}")
    print("=" * 65)

    print("\nHUMAN REFERENCE:")
    print(data["base"][i]["reference"])

    for variant in VARIANTS:
        row = data[variant][i]

        print(f"\n{variant.upper()}:")
        print(row["prediction"])

        print(
            "ROUGE-L:",
            round(row["rouge"]["rougeL"], 3)
        )

        print(
            "Repetition rate:",
            round(repetition_rate(row["prediction"]), 3)
        )
