# local-llm-rag

Few notes:

pdf_loader -> chunker -> embedder -> vectordb (vector database) -> store

pdf_loader.py ingests valid pdf documents. 

chunker.py uses a recursive way of dividing a page into chunks. 

embedder.py uses bge-small-en-v1.5 model from hugging face, subject to change if deemed ineffective. 

vectordb.py uses Chroma with cosine distance to stores the chunks to its related chunks.
