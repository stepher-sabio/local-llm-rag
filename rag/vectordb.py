import chromadb

DB_PATH = "chroma_db"  # folder where the database is saved


def create_collection(name):
    """Make a new, empty collection (deletes the old one with the same name)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name)
    except Exception:
        pass  # it did not exist yet
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def add_chunks(collection, chunks, vectors):
    """Store the chunks, their vectors, PDF name and page numbers."""
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks],
    )


def search(collection, question_vector, k=5):
    """Return the k chunks closest to the question."""
    result = collection.query(query_embeddings=[question_vector], n_results=k)

    found = []
    for text, meta, dist in zip(result["documents"][0],
                                result["metadatas"][0],
                                result["distances"][0]):
        found.append({"source": meta["source"], "page": meta["page"],
                      "text": text, "distance": dist})
    return found


if __name__ == "__main__":
    from rag.pdf_loader import load_pdf
    from rag.chunker import chunk_pages
    from rag.embedder import load_embedder, embed

    chunks = chunk_pages(load_pdf("data/ESP32 Datasheet.pdf"))
    for c in chunks:
        c["source"] = "ESP32 Datasheet.pdf"
    model = load_embedder()

    collection = create_collection("esp32_datasheet")
    add_chunks(collection, chunks, embed(model, [c["text"] for c in chunks]))
    print(collection.count(), "chunks stored")

    question = "Which GPIO pins are input only?"
    q_vector = embed(model, [question])[0]
    for hit in search(collection, q_vector):
        print(f"{hit['source']}, page {hit['page']}  distance {hit['distance']:.3f}")
        print("  ", hit["text"][:120].replace("\n", " "))