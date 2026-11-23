import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "rag"))
from pipeline import prepare_document, ask
from embedder import load_embedder
from llm import load_llm

PDF_DIR = Path(__file__).resolve().parent.parent / "PDFs"

st.set_page_config(page_title="Local LLM RAG", page_icon=":material/description:")


@st.cache_resource
def get_embedder():
    """Load the embedding model once and share it across reruns."""
    return load_embedder()


@st.cache_resource
def get_llm():
    """Load the LLM once and share it across reruns."""
    return load_llm()


st.title("Chat with your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdfs" not in st.session_state:
    st.session_state.pdfs = {}  # dictionary that holds file name and bites of the file
if "collections" not in st.session_state:
    st.session_state.collections = {}  # file name -> vector database collection


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        for name in msg.get("files", []):
            st.badge(name, icon=":material/picture_as_pdf:", color="gray")
        if msg.get("content"):
            st.markdown(msg["content"])


for msg in st.session_state.messages:
    render_message(msg)

prompt = st.chat_input(
    "Ask a question or attach PDFs",
    accept_file="multiple",
    file_type="pdf",
    submit_mode="disable",
)

if prompt:
    file_names = []
    for f in prompt.files:
        st.session_state.pdfs[f.name] = f.getvalue()
        file_names.append(f.name)

    user_msg = {"role": "user", "content": prompt.text, "files": file_names}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    parts = []
    if file_names:
        PDF_DIR.mkdir(exist_ok=True)
        with st.spinner("Indexing PDFs..."):
            embedder = get_embedder()
            for name in file_names:
                # load_pdf needs a file path, so save the bytes to disk first
                pdf_path = PDF_DIR / Path(name).name
                pdf_path.write_bytes(st.session_state.pdfs[name])
                st.session_state.collections[name] = prepare_document(str(pdf_path), embedder)
        parts.append(f"Indexed {len(file_names)} PDF(s): " + ", ".join(file_names))

    question = (prompt.text or "").strip()
    if question:
        if not st.session_state.collections:
            parts.append("Please attach a PDF first so I have something to answer from.")
        else:
            with st.spinner("Thinking..."):
                embedder = get_embedder()
                llm = get_llm()
                many = len(st.session_state.collections) > 1
                for name, collection in st.session_state.collections.items():
                    answer, pages, _chunks = ask(question, collection, embedder, llm)
                    pages_text = ", ".join(str(p) for p in pages)
                    header = f"**{name}**\n\n" if many else ""
                    parts.append(f"{header}{answer}\n\n_Pages: {pages_text}_")

    reply = "\n\n---\n\n".join(parts)
    assistant_msg = {"role": "assistant", "content": reply}
    st.session_state.messages.append(assistant_msg)
    render_message(assistant_msg)
