import pymupdf


def load_pdf(path):
    """Return the text of each page."""
    try:
        doc = pymupdf.open(path)
    except Exception:
        raise ValueError("This file is not a valid PDF.")

    if not doc.is_pdf:
        raise ValueError("This file is not a PDF.")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:  # skip empty pages
            pages.append({"page": i + 1, "text": text})

    if not pages:
        raise ValueError("This PDF has no text. It may be a scanned PDF.")

    return pages

# test the pdf_loader
if __name__ == "__main__":
    pages = load_pdf(f"data/ESP32 Datasheet.pdf")
    print(len(pages), "pages with text")
    print(pages[0])