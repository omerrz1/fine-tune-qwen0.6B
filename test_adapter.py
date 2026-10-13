
import json

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = "Qwen/Qwen3-0.6B-MLX-4bit"
ADAPTER = "adapters/dialogsum-v1-step150"

# Load base model with our trained adapter
model, tokenizer = load(
    MODEL,
    adapter_path=ADAPTER
)

# Use the first validation example
with open("data/valid.jsonl", encoding="utf-8") as f:
    example = json.loads(f.readline())

messages = example["messages"]

reference = messages[-1]["content"]

prompt = tokenizer.apply_chat_template(
    messages[:-1],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

prediction = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=160,
    sampler=make_sampler(temp=0.9),
    verbose=False
)

print("\nFINE-TUNED SUMMARY:")
print(prediction)

print("\nHUMAN REFERENCE:")
print(reference)
