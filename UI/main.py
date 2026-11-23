import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Local LLM RAG", page_icon=":material/description:")

st.title("Chat with your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdfs" not in st.session_state:
    st.session_state.pdfs = {}  # dictionary that holds file name and bites of the file


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

    # Placeholder reply until the RAG pipeline is wired in.
    if file_names:
        reply = f"Received {len(file_names)} PDF(s): " + ", ".join(file_names)

        # Temporary solution will save pdf's in folder /PDFs
        Path("PDFs").mkdir(exist_ok=True)
        for name in file_names:
            Path("PDFs", Path(name).name).write_bytes(st.session_state.pdfs[name])
    else:
        reply = "PDF upload works. Answering questions isn't connected yet."
    assistant_msg = {"role": "assistant", "content": reply}
    st.session_state.messages.append(assistant_msg)
    render_message(assistant_msg)
