
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "Qwen/Qwen3-0.6B-MLX-4bit"

print("Loading model...")

model, tokenizer = load(MODEL_ID)

print("Model loaded successfully!")



# Our first example
conversation = """
Sarah: Is the website ready?
James: The homepage is finished.
Sarah: What about checkout?
James: Checkout still needs testing.
Sarah: Can we launch Friday?
James: Yes, if testing passes on Thursday.
"""

messages = [
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
            + conversation
        )
    }
]

# Apply Qwen's conversation template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

# Examine tokenization
token_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False
)

print("\nPROMPT TOKEN COUNT:", len(token_ids))
print("\nFORMATTED PROMPT:\n", prompt)

# Generate a summary
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=160,
    sampler=make_sampler(temp=0.0),
    verbose=False
)

print("\nMODEL SUMMARY:\n", response)
