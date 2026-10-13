
from datasets import load_dataset
from pathlib import Path
import json

# 1. Download the dataset
dataset = load_dataset("knkarthick/dialogsum")

# 2. Inspect its structure
print("DATASET STRUCTURE")
print(dataset)

# 3. Look at a real training example
example = dataset["train"][0]

print("\nAVAILABLE COLUMNS:")
print(list(example.keys()))

print("\nCONVERSATION:")
print(example["dialogue"])

print("\nHUMAN-WRITTEN SUMMARY:")
print(example["summary"])



# 4. Create the output directory
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# 5. Define how each example should look
def format_example(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Summarize conversations accurately in "
                    "one or two sentences. Include only facts "
                    "from the conversation."
                )
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation:\n\n"
                    + example["dialogue"]
                )
            },
            {
                "role": "assistant",
                "content": example["summary"]
            }
        ]
    }

# 6. Choose the size of our first experiment
splits = {
    "train": 1000,
    "validation": 100,
    "test": 100
}

# 7. Convert and save the data
for split_name, count in splits.items():

    # Shuffle reproducibly
    selected = (
        dataset[split_name]
        .shuffle(seed=42)
        .select(range(count))
    )

    # MLX expects valid.jsonl, not validation.jsonl
    file_name = (
        "valid" if split_name == "validation"
        else split_name
    )

    file_path = output_dir / f"{file_name}.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for example in selected:
            formatted = format_example(example)
            f.write(
                json.dumps(formatted, ensure_ascii=False)
                + "\n"
            )

    print(f"Saved {count} examples to {file_path}")
