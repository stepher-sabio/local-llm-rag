import os
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils import logging


logging.set_verbosity_error()  # hide warnings


MODEL_PATH = os.path.expanduser("~/Desktop/llm_project/Qwen3.5-0.8B")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the model
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16
).to(DEVICE)


def generate_reply(messages):
    """Get the model's answer."""
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(DEVICE)

    output = model.generate(**inputs, max_new_tokens=512)

    # Keep only the new tokens, then turn them into text
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# Chat loop
messages = []
while True:
    text = input("You: ").strip()
    if text.lower() == "exit":
        break

    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
    reply = generate_reply(messages)
    print("Qwen:", reply, "\n")
    messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})