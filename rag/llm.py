import os
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils import logging

logging.set_verbosity_error()  # hide warnings

MODEL_PATH = os.path.expanduser("~/Desktop/llm_project/Qwen3.5-0.8B")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_llm(model_path=MODEL_PATH):
    """Load the model once."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    return processor, model


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