from langchain_text_splitters import RecursiveCharacterTextSplitter

# Why these values:
# - Recursive splitter: it cuts at blank lines first, then line breaks,
#   then spaces, so paragraphs and list items stay whole when possible.
#   Datasheets are mostly short paragraphs, lists and tables.
# - 800 characters (~200 tokens): small enough that each chunk is about
#   one topic (precise search), large enough to keep a full paragraph.
#   It also fits the input limit of small embedding models
#   (bge-small reads up to 512 tokens, all-MiniLM up to 256).
# - 150 characters overlap (~20%): a sentence cut at a chunk border
#   still appears whole in one of the two chunks.
# - Each page is split on its own, so every chunk has exactly one page
#   number for the citation.
# Experiment 2 (see evaluation.md): 800/150 found 6 of 8 answers in the
# top 5, while 400/80 found only 5 of 8, so we kept 800/150.
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
    from rag.pdf_loader import load_pdf

    pages = load_pdf("data/ESP32_Datasheet.pdf")
    chunks = chunk_pages(pages)
    print(len(pages), "pages ->", len(chunks), "chunks")
    print(chunks[0])