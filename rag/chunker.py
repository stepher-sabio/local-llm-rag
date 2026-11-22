from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800     # max characters per chunk
CHUNK_OVERLAP = 150  # characters repeated between chunks


def chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split each page into chunks that keep their page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = []
    for page in pages:
        for text in splitter.split_text(page["text"]):
            chunks.append({"page": page["page"], "text": text})

    return chunks


if __name__ == "__main__":
    from pdf_loader import load_pdf

    pages = load_pdf("data/ESP32 Datasheet.pdf")
    chunks = chunk_pages(pages)
    print(len(pages), "pages ->", len(chunks), "chunks")
    print(chunks[0])