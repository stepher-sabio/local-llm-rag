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
    "Be clear and concise."
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


def ask(question, collection, embedder, llm, k=TOP_K):
    """Answer a question using the most relevant chunks."""
    q_vector = embed(embedder, [question])[0]
    chunks = search(collection, q_vector, k)

    answer = generate_reply(llm, build_prompt(question, chunks))
    pages = sorted({c["page"] for c in chunks})
    return answer, pages, chunks


if __name__ == "__main__":
    embedder = load_embedder()
    llm = load_llm()

    collection = prepare_document("data/ESP32 Datasheet.pdf", embedder)
    print(collection.count(), "chunks ready\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        answer, pages, chunks = ask(question, collection, embedder, llm)
        print("\nAnswer:", answer)
        print("Pages:", pages, "\n")