from rag.llm import load_llm, generate_reply

llm = load_llm()
messages = []

while True:
    text = input("You: ").strip()
    if text.lower() == "exit":
        break

    messages.append({"role": "user", "content": text})
    reply = generate_reply(llm, messages)
    print("Qwen:", reply, "\n")
    messages.append({"role": "assistant", "content": reply})