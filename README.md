# local-llm-rag

Chat with Local **LLM** model about your PDF documents. Everything runs on your own machine: no API keys, and your documents are never sent to a cloud service.

The project uses **RAG** (Retrieval-Augmented Generation). It splits a PDF into small pieces, finds the pieces most relevant to your question, and gives only those to the model. The model answers based on provided document and doesn't use side sources at all.

---

## Features

- **Streamlit chat UI**: attach one or more PDFs and ask questions in the same chat.
- **Fully local**: Local language model and a BGE embedding model run on your machine.
- **Answers grounded in the document**: the model is told to use only the retrieved excerpts. If the answer isn't there, it replies *"I don't know based on this document."*
- **Page citations**: every answer lists the pages its excerpts came from.
- **Multiple PDFs**: each PDF gets its own answer, labeled with the file name.
- **Model selector**: switch between Qwen3.5 0.8B, 2B, and 4B in the sidebar (if models are not installed on your machine they will be installed automatically).
- **Automatic hardware detection**: uses an NVIDIA GPU (CUDA), Apple Silicon (MPS), or the CPU, whichever is available.

---

## How it works

```
PDF ──► pdf_loader ──► chunker ──► embedder ──► vectordb (Chroma)
                                                      │
Question ──► embedder ──► search top-5 chunks ◄───────┘
                                │
                                ▼
                     prompt + excerpts ──► LLM (Qwen) ──► answer + page numbers
```

| Step | File | What it does |
|------|------|--------------|
| 1. Load | [rag/pdf_loader.py](rag/pdf_loader.py) | Reads the text of each page with PyMuPDF and skips empty pages. Rejects files that aren't PDFs and scanned PDFs that have no text. |
| 2. Chunk | [rag/chunker.py](rag/chunker.py) | Splits each page into chunks of up to 800 characters, with 150 characters of overlap, using LangChain's `RecursiveCharacterTextSplitter`. Every chunk keeps its page number. |
| 3. Embed | [rag/embedder.py](rag/embedder.py) | Turns each chunk into a vector with [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). |
| 4. Store | [rag/vectordb.py](rag/vectordb.py) | Saves the chunks and their vectors in a persistent [Chroma](https://www.trychroma.com/) database (`chroma_db/`) using cosine distance. Each PDF gets its own collection. |
| 5. Retrieve and answer | [rag/pipeline.py](rag/pipeline.py) | Embeds the question, fetches the 5 closest chunks, builds the prompt, and asks the LLM. |
| LLM | [rag/llm.py](rag/llm.py) | Loads a Qwen3.5 model with Hugging Face Transformers and generates replies. By default it uses temperature 0, so the same input gives the same answer. |
| UI | [main.py](main.py) | The Streamlit chat app. |
| CLI chat | [run.py](run.py) | A plain terminal chat with the LLM, with no PDFs and no RAG. |

---

## Requirements

- **Python 3.10 or newer** (developed with Python 3.13)
- About **2–10 GB of free disk space** for the models, depending on which Qwen model you use
- **Optional:** an NVIDIA GPU with CUDA, or an Apple Silicon Mac, for much faster answers. The CPU works, but larger models will be slow.
- An internet connection **the first time you run it**, to download the models from Hugging Face

Python packages:

| Package | Tested version |
|---------|----------------|
| `streamlit` | 1.67 |
| `torch` | 2.11 |
| `transformers` | 5.7 |
| `sentence-transformers` | 6.3 |
| `chromadb` | 1.5 |
| `pymupdf` | 2.1 |
| `langchain-text-splitters` | 1.2 |

> The chat input relies on newer Streamlit features (`accept_file`, `st.badge`, `submit_mode`), and the LLM loader uses `AutoModelForMultimodalLM`, which needs a recent `transformers`. Use the versions above or newer.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/stepher-sabio/local-llm-rag.git
cd local-llm-rag
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

Install the build that matches your hardware. Follow the selector at <https://pytorch.org/get-started/locally/>. For example:

```bash
# NVIDIA GPU (CUDA 13.0)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# CPU only, or macOS
pip install torch
```

### 4. Install the other dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Use a local copy of the model

On first use, models download automatically from Hugging Face into its cache. To use a copy you already have, put it here:

```
local-llm-rag/
└── models/
    ├── Qwen3.5-0.8B/
    ├── Qwen3.5-2B/
    └── Qwen3.5-4B/
```

`rag/llm.py` checks `models/<model name>` first and falls back to downloading from Hugging Face.

---

## Usage

### Run the web app (recommended)

From the project root:

```bash
streamlit run main.py
```

Streamlit opens the app in your browser, usually at <http://localhost:8501>.

1. **Pick a model** in the sidebar. *Qwen3.5 0.8B (fast)* is the default and the best choice for a CPU.
2. **Attach PDFs** with the paperclip in the chat box. You can add several at once.
3. **Ask a question** in the same message or in a later one.
4. Read the answer and the **page numbers** listed under it.

Notes:
- The first question takes longer because the models are loading. After that they stay in memory.
- Uploaded PDFs are saved to the `PDFs/` folder, and their vectors are saved to `chroma_db/`.
- The chat history and the list of attached PDFs live only in the browser session. If you refresh the page, attach the PDFs again.
- Uploading a PDF with the same file name again replaces its old collection.
- Only text PDFs are supported. Scanned PDFs have no text and are rejected.

### Run the terminal chat (LLM only)

A plain chat with the model, without any documents:

```bash
python run.py
```

Type `exit` to quit.

## Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Models in the UI dropdown | `MODELS` in [main.py](main.py) | Qwen3.5 0.8B / 2B / 4B |
| Default LLM (CLI and tests) | `HUB_MODEL_ID` in [rag/llm.py](rag/llm.py) | `Qwen/Qwen3.5-0.8B` |
| Embedding model | `MODEL_NAME` in [rag/embedder.py](rag/embedder.py) | `BAAI/bge-small-en-v1.5` |
| Chunk size / overlap | `CHUNK_SIZE`, `CHUNK_OVERLAP` in [rag/chunker.py](rag/chunker.py) | 800 / 150 characters |
| Chunks sent to the LLM | `TOP_K` in [rag/pipeline.py](rag/pipeline.py) | 5 |
| System prompt | `SYSTEM_PROMPT` in [rag/pipeline.py](rag/pipeline.py) | Answer only from the excerpts |
| Max answer length | `max_new_tokens` in `generate_reply` ([rag/llm.py](rag/llm.py)) | 512 tokens |
| Database folder | `DB_PATH` in [rag/vectordb.py](rag/vectordb.py) | `chroma_db/` |

If you change the embedding model, upload your PDFs again so their vectors are rebuilt.

---

## Project structure

```
local-llm-rag/
├── main.py              # Streamlit chat app
├── run.py               # Terminal chat with the LLM (no RAG)
├── rag/
│   ├── pdf_loader.py    # PDF → page texts
│   ├── chunker.py       # pages → overlapping chunks
│   ├── embedder.py      # chunks → vectors (BGE)
│   ├── vectordb.py      # Chroma storage and search
│   ├── llm.py           # Qwen loading and generation
│   └── pipeline.py      # ties everything together (prepare_document, ask)
├── PDFs/                # uploaded PDFs (git-ignored)
├── chroma_db/           # vector database (git-ignored, created automatically)
├── data/                # sample PDFs for module tests (git-ignored)
└── models/              # optional local model copies (git-ignored)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Python was not found` on Windows | Use `py` instead of `python`, or install Python from python.org and tick *Add to PATH*. |
| `This PDF has no text. It may be a scanned PDF.` | The PDF contains only images. Run OCR on it first, for example with `ocrmypdf`. |
| Answers are very slow | Use the 0.8B model, or install the CUDA build of PyTorch if you have an NVIDIA GPU. |
| Out of memory | Pick a smaller model in the sidebar. |
| `AttributeError` mentioning `chat_input`, `badge`, or `AutoModelForMultimodalLM` | Upgrade the packages: `pip install -U streamlit transformers`. |
| `ModuleNotFoundError: rag` | Run commands from the project root, not from inside `rag/`. |

---

## License

See [LICENSE](LICENSE).
