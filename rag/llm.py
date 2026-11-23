import os
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils import logging

# Model folder inside the project: llm_project/models/Qwen3.5-0.8B
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "Qwen3.5-0.8B")

# Pick the fastest device on this computer
if torch.cuda.is_available():            # NVIDIA GPU (Windows / Linux)
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():  # Apple Silicon Mac
    DEVICE = "mps"
    DTYPE = torch.bfloat16
else:                                    # any CPU
    DEVICE = "cpu"
    DTYPE = torch.float32


def generate_reply(llm, messages, max_new_tokens=512, temperature=0):
    """Get the model's answer.

    messages: [{"role": "user", "content": "Hi"}, ...]
    """
    processor, model = llm

    # Put each text in the format the model expects
    formatted = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in messages
    ]

    inputs = processor.apply_chat_template(
        formatted, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(DEVICE)

    # temperature 0 = always the same answer (good for evaluation)
    if temperature > 0:
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=temperature)
    else:
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False)

    # Keep only the new tokens, then turn them into text
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    llm = load_llm()
    reply = generate_reply(llm, [{"role": "user", "content": "What is a GPIO pin?"}])
    print(reply)