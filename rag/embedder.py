from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2" # change this if you want to use a different model


def load_embedder(model_name=MODEL_NAME):
    """Load the embedding model."""
    return SentenceTransformer(model_name)


def embed(model, texts):
    """Turn a list of texts into vectors."""
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


if __name__ == "__main__":
    import time
    from pdf_loader import load_pdf
    from chunker import chunk_pages

    chunks = chunk_pages(load_pdf("data/ESP32 Datasheet.pdf"))
    texts = [c["text"] for c in chunks]

    model = load_embedder()
    start = time.perf_counter()
    vectors = embed(model, texts)
    seconds = time.perf_counter() - start

    print(len(vectors), "vectors of size", len(vectors[0]))
    print(f"Took {seconds:.1f} s")