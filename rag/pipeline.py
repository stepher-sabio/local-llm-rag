import os
import re

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import load_embedder, embed
from rag.vectordb import create_collection, add_chunks, search
from rag.llm import load_llm, generate_reply

TOP_K = 5  # number of chunks given to the LLM

SYSTEM_PROMPT = (
    "You answer questions about a technical document. "
    "Use ONLY the excerpts given to you. "
    "If the excerpts do not contain the answer, reply exactly: "
    "\"I don't know based on this document.\" "
    "Otherwise, give a short answer, then on a new line write "
    "'Evidence:' followed by ONE short quote (max 25 words) "
    "from the excerpts that supports it. Never copy a whole table."
)


def collection_name(path):
    """Turn a file name into a valid collection name."""
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r"[^a-z0-9_-]", "_", name)  # only allowed characters
    return ("doc_" + name)[:60].rstrip("_-")


def prepare_document(path, embedder):
    """PDF -> chunks -> vectors -> database."""
    pages = load_pdf(path)
    chunks = chunk_pages(pages)
    for c in chunks:
        c["source"] = os.path.basename(path)  # PDF name, for the citation
    vectors = embed(embedder, [c["text"] for c in chunks])

    collection = create_collection(collection_name(path))
    add_chunks(collection, chunks, vectors)
    return collection


def build_prompt(question, chunks):
    """Put the chunks and the question into chat messages."""
    excerpts = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)
    user_text = f"Excerpts:\n{excerpts}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def find_source(answer, chunks, embedder):
    """Pick the chunk the answer is most similar to."""
    vectors = embed(embedder, [answer] + [c["text"] for c in chunks])
    answer_vec = vectors[0]

    # Vectors are normalized, so the dot product = cosine similarity
    scores = [sum(a * b for a, b in zip(answer_vec, v)) for v in vectors[1:]]
    return chunks[scores.index(max(scores))]


def split_evidence(reply):
    """Split the model's reply into (answer, evidence)."""
    position = reply.lower().find("evidence:")
    if position == -1:
        return reply.strip(), ""
    answer = reply[:position].strip()
    evidence = reply[position + len("evidence:"):].strip()
    return answer, evidence


def ask(question, collection, embedder, llm, k=TOP_K):
    """Answer a question and cite the one chunk it came from."""
    q_vector = embed(embedder, [question])[0]
    chunks = search(collection, q_vector, k)

    reply = generate_reply(llm, build_prompt(question, chunks))

    # No answer -> no citation
    if "don't know" in reply.lower():
        return reply.strip(), None, None

    # The evidence is only used to find the source chunk, not shown.
    # A short answer like "True." says nothing about its source,
    # so without evidence we match on question + answer.
    answer, evidence = split_evidence(reply)
    source = find_source(evidence or (question + " " + answer), chunks, embedder)

    citation = f"{source['source']}, page {source['page']}"
    return answer, citation, source


if __name__ == "__main__":
    embedder = load_embedder()
    llm = load_llm()

    collection = prepare_document("data/ESP32 Datasheet.pdf", embedder)
    print(collection.count(), "chunks ready\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        answer, citation, source = ask(question, collection, embedder, llm)
        print("\nAnswer:", answer)
        print("Source:", citation, "\n")