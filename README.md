# local-llm-rag

Chat with your PDF documents using a local LLM. Everything runs on your machine: no API keys, and your documents never leave your computer.

The app uses **RAG** (Retrieval-Augmented Generation): it splits the PDF into small chunks, finds the chunks most relevant to your question, and gives only those to the model. The model answers from the document only.

---

## Features

**Core**
- **Chat UI** (Streamlit): attach one or more PDFs and ask questions.
- **Welcome summary**: a summary of the whole document after upload.
- **Cited answers**: each answer shows its source, e.g. *ESP32 Datasheet.pdf, page 45*. The source text opens in an expandable section.
- **"I don't know"**: if the answer is not in the document, the app says so.
- **Clear errors**: non-PDFs and scanned PDFs get a message, not a crash.

**Optional features (6 tokens)**
- **Settings** (1): temperature, answer language (English, Spanish, Catalan), answer style.
- **Conversation memory** (1): remembers the last 5 messages, so follow-ups work.
- **Hybrid search** (2): vector search + BM25 keyword search, merged with Reciprocal Rank Fusion.
- **Re-ranking** (2): a cross-encoder reorders 15 candidates and keeps the best 5.

---

## How it works

```
PDF ──► pdf_loader ──► chunker ──► embedder ──► Chroma
 └──► summary (map-reduce) ──► welcome summary

Question ──► vector + keyword search ──► re-rank ──► top 5 chunks
                                                         │
                         prompt + chunks + memory ──► LLM ──► answer + source
```

| File | What it does |
|---|---|
| `rag/pdf_loader.py` | Reads the text of each page. Rejects non-PDFs and scanned PDFs. |
| `rag/chunker.py` | Splits pages into 800-character chunks with 150 overlap. Each chunk keeps its page number. |
| `rag/embedder.py` | Turns chunks into vectors with **`BAAI/bge-small-en-v1.5`** (chosen in Experiment 1). |
| `rag/vectordb.py` | Stores the chunks in a local **Chroma** database with cosine distance. |
| `rag/retrieval.py` | Hybrid search and re-ranking. |
| `rag/summary.py` | Welcome summary: summarises each section, then combines the summaries. |
| `rag/llm.py` | Loads the LLM (Qwen3.5 or Gemma 4) and generates answers. |
| `rag/pipeline.py` | Connects everything: `prepare_document()` and `ask()`. |
| `main.py` | The Streamlit app. |

---

## Installation

**Requirements:** Python 3.12, internet the first time (to download models), 2–10 GB of disk per model. A GPU (NVIDIA or Apple Silicon) is optional but much faster.

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/stepher-sabio/local-llm-rag.git
cd local-llm-rag
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install PyTorch** for your hardware (see <https://pytorch.org/get-started/locally/>):

```bash
pip install torch                                                        # CPU or Mac
pip install torch --index-url https://download.pytorch.org/whl/cu130     # NVIDIA GPU
```

**3. Install the rest**

```bash
pip install -r requirements.txt
```

---

## Usage

From the project root:

```bash
streamlit run main.py
```

1. Pick a model in the sidebar (**Qwen3.5 0.8B** is fastest).
2. Attach a PDF with the paperclip and wait for the summary.
3. Ask questions.

Notes:
- The first run downloads the models, so it is slow.
- Summarising a long PDF takes a few minutes.
- Refreshing the page clears the chat. Attach the PDFs again.

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| Embedding model | `MODEL_NAME` in `rag/embedder.py` | `BAAI/bge-small-en-v1.5` |
| Chunk size / overlap | `rag/chunker.py` | 800 / 150 |
| Chunks given to the LLM | `TOP_K` in `rag/pipeline.py` | 5 |
| Messages remembered | `MAX_HISTORY` in `rag/pipeline.py` | 5 |
| System prompt | `SYSTEM_PROMPT` in `rag/pipeline.py` | Answer only from the excerpts |
| LLMs in the menu | `MODELS` in `main.py` | Qwen3.5 0.8B / 2B / 4B, Gemma 4 E2B |

If you change the embedding model, delete `chroma_db/` and upload the PDFs again.

---

## Evaluation

The experiments (embedding models, chunk sizes, LLMs, hybrid search and re-ranking), the results and the error analysis are in **[NN_Eval_Results.md](NN_Eval_Results.md)**.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt`, and run commands from the project root. |
| `KeyError: 'source'` | Old database. Delete `chroma_db/` and upload the PDFs again. |
| `AttributeError` with `chat_input` or `AutoModelForMultimodalLM` | Update: `pip install -U streamlit transformers`. |
| Answers are very slow | Use Qwen3.5 0.8B, or install the CUDA build of PyTorch on an NVIDIA GPU. |
| "This PDF has no text" | The PDF is scanned. Run OCR on it first (e.g. `ocrmypdf`). |