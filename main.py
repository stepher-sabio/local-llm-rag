from pathlib import Path
import streamlit as st
import os 

from rag.pipeline import *
from rag.embedder import *
from rag.llm import *


PDF_DIR = Path(__file__).resolve().parent / "PDFs"

MODELS = {
    "Qwen3.5 0.8B (fast)": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5 2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5 4B": "Qwen/Qwen3.5-4B",
    "Gemma 4 E2B": "google/gemma-4-E2B-it",
}


# Models
# Streamlit reruns your Python script whenever the user interacts with the application.
# So @st.cache_resource tells Streamlit to load a resource once and reuse it.
@st.cache_resource(max_entries=1)
def load_models(model_id):
    embedder = load_embedder()
    llm = load_llm(model_id)
    return embedder, llm


# State of the session
def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pdfs" not in st.session_state:
        st.session_state.pdfs = {}

    if "collections" not in st.session_state:
        st.session_state.collections = {}


# PDF handler
def index_pdfs(files):
    PDF_DIR.mkdir(exist_ok=True)

    embedder, _ = load_models(st.session_state.model_id)
    indexed_files = []

    for file in files:
        file_path = PDF_DIR / Path(file.name).name
        file_path.write_bytes(file.getvalue())

        collection = prepare_document(str(file_path), embedder)

        st.session_state.pdfs[file.name] = file.getvalue()
        st.session_state.collections[file.name] = collection

        indexed_files.append(file.name)

    return indexed_files


# Chat handling
def get_answer(question):
    embedder, llm = load_models(st.session_state.model_id)

    answers = []
    multiple_pdfs = len(st.session_state.collections) > 1

    for name, collection in st.session_state.collections.items():
        answer, citation, source = ask(question, collection, embedder, llm)

        if multiple_pdfs:
            answers.append(f"**{name}**\n\n{answer}")
        else:
            answers.append(answer)

        # One citation: the PDF and page the answer came from
        if citation:
            answers.append(f"_Source: {citation}_")

    return "\n\n".join(answers)


def process_message(prompt):
    parts = []

    if prompt.files:
        with st.spinner("Reading your PDFs..."):
            files = index_pdfs(prompt.files)

            for file in prompt.files:
                path = PDF_DIR / Path(file.name).name
                try:
                    os.remove(path)
                    print(f"File '{path}' deleted successfully.")
                except Exception as e:
                    print(f"File '{path}' wasn't deleted: {e}")

        parts.append(f"Added {len(files)} PDF(s): {', '.join(files)}")

    question = (prompt.text or "").strip()

    if question:
        if not st.session_state.collections:
            parts.append("Please attach a PDF before asking a question.")
        else:
            with st.spinner("Thinking..."):
                parts.append(get_answer(question))

    return "\n\n---\n\n".join(parts)


# Display
def display_message(message):
    with st.chat_message(message["role"]):
        for filename in message.get("files", []):
            st.badge(
                filename,
                icon=":material/picture_as_pdf:",
                color="gray"
            )

        if message.get("content"):
            st.markdown(message["content"])

# Main application
def main():
    st.set_page_config(
        page_title="Local LLM RAG",
        page_icon=":material/description:"
    )

    initialize_session()

    label = st.sidebar.selectbox("Model", list(MODELS))
    st.session_state.model_id = MODELS[label]

    st.title("Chat with your PDFs")

    for message in st.session_state.messages:
        display_message(message)

    prompt = st.chat_input(
        "Ask a question or attach PDFs",
        accept_file="multiple",
        file_type="pdf",
        submit_mode="disable"
    )

    if prompt:
        user_message = {
            "role": "user",
            "content": prompt.text or "",
            "files": [file.name for file in prompt.files]
        }

        st.session_state.messages.append(user_message)
        display_message(user_message)

        response = process_message(prompt)

        assistant_message = {
            "role": "assistant",
            "content": response
        }

        st.session_state.messages.append(assistant_message)
        display_message(assistant_message)


if __name__ == "__main__":
    main()