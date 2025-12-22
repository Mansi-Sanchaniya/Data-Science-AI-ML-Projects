import os
import hashlib
import sqlite3
import numpy as np
import faiss
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from pdf_parser import parse_pdf_with_fitz, extract_elements_unstructured
from metadata_extractor import extract_metadata
from metadata_store import (
    init_db,
    save_metadata,
    is_pdf_already_processed,
    is_pdf_already_embedded,
    mark_as_embedded,
)
from enhanced_pdf_parser import parse_pdf_enhanced
from embedder import flatten_and_convert_structured_chunks, embed_pdf_structured_chunks, get_combined_embedding
import urllib3
urllib3.disable_warnings()

# ---- FAISS Config ----
FAISS_DB_PATH = "db/faiss"
VECTOR_DIM = 768  # Adjust if using different embedding model dimension
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# ==== Load or initialize FAISS vectorstore ====
def load_or_init_faiss(namespace: str):
    path = os.path.join(FAISS_DB_PATH, namespace)
    os.makedirs(path, exist_ok=True)
    faiss_path = os.path.join(path, "index.faiss")

    index = faiss.IndexFlatL2(VECTOR_DIM)
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}

    if os.path.exists(faiss_path):
        try:
            return FAISS.load_local(path, embedding_function=get_combined_embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")

    return FAISS(
        embedding_function=get_combined_embedding,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

# ==== Main pipeline ====
def run_pipeline():
    LOCAL_PDF_DIR = "data/pdfs"

    print("Initializing metadata DB...")
    init_db()

    # Load or initialize FAISS vectorstores
    vectorstore_compliance = load_or_init_faiss('compliance')
    vectorstore_research = load_or_init_faiss('research')

    for root, dirs, files in os.walk(LOCAL_PDF_DIR):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, start=LOCAL_PDF_DIR)
            already_processed = is_pdf_already_processed(rel_path)
            already_embedded = is_pdf_already_embedded(rel_path)

            if already_processed and already_embedded:
                print(f"Already processed and embedded: {rel_path} - skipping")
                continue

            # --- Document Parsing ---
            if not already_processed:
                print(f"Parsing (enhanced): {rel_path}")
                parsed = parse_pdf_enhanced(full_path)
                text = parsed.get("full_text", "")
                structured_chunks = parsed.get("tables") or []
                metadata = extract_metadata(text, structured_chunks)
                save_metadata(rel_path, metadata)
                print(f"Metadata saved for {rel_path}")
            else:
                print(f"Parsing (basic): {rel_path}")
                text = parse_pdf_with_fitz(full_path)
                structured_chunks = extract_elements_unstructured(full_path)
                metadata = extract_metadata(text, structured_chunks)
                save_metadata(rel_path, metadata)
                print(f"Metadata re-saved for {rel_path}")

            # Flatten chunks
            flat_structured_chunks = flatten_and_convert_structured_chunks(structured_chunks)
            print(f'flat_structured_chunks: {flat_structured_chunks}')

            # Determine namespace based on folder structure
            folder_namespace = rel_path.split(os.sep)[0].lower()
            print(f'folder namespace: {folder_namespace}')
            if folder_namespace == 'compliance':
                vectorstore = vectorstore_compliance
            elif folder_namespace == 'research':
                vectorstore = vectorstore_research
            else:
                print(f"Unknown namespace for {rel_path}; skipping")
                continue

            # Embed chunks using local multi-model embeddings
            docs_with_embeddings = embed_pdf_structured_chunks(rel_path, flat_structured_chunks, metadata)
            print(f'embedddings: {docs_with_embeddings}')

            if docs_with_embeddings:
                texts = [doc['text'] for doc in docs_with_embeddings]
                print(f'text: {texts}')
                metadatas = [doc['metadata'] for doc in docs_with_embeddings]
                print(f"Embedding {len(texts)} chunks for {rel_path}...")
                vectorstore.add_texts(texts, metadatas)
                vectorstore.save_local(os.path.join(FAISS_DB_PATH, folder_namespace))
                print(f"Added {len(texts)} chunks to FAISS index '{folder_namespace}'")
                mark_as_embedded(rel_path)
            else:
                print(f"No chunks to embed for {rel_path}.")

if __name__ == "__main__":
    run_pipeline()
