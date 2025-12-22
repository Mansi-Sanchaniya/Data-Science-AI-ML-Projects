import os
import faiss
import hashlib
import sqlite3
import numpy as np
import shutil
import gc
from contextlib import contextmanager
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
import urllib3
import fitz  # PyMuPDF

urllib3.disable_warnings()

# ==== Local Model Configuration ====
NOMIC_MODEL_PATH = "D:/models/nomic"
OLLAMA_MODEL_NAME = "llama3:latest"

# ==== SQLite Deduplication Database ====
HASH_DB_PATH = "db/chunk_hashes.db"
os.makedirs("db/faiss", exist_ok=True)
os.makedirs("db", exist_ok=True)


def init_hash_db():
    with sqlite3.connect(HASH_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedded_chunks (
                hash TEXT PRIMARY KEY,
                embedded INTEGER NOT NULL DEFAULT 0,
                indexed INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.commit()


def is_chunk_fully_processed(chunk_hash: str) -> bool:
    with sqlite3.connect(HASH_DB_PATH) as conn:
        row = conn.execute(
            "SELECT embedded, indexed FROM embedded_chunks WHERE hash = ?", (chunk_hash,)
        ).fetchone()
        return bool(row and row[0] == 1 and row[1] == 1)


def mark_chunk_embedded(chunk_hash: str):
    with sqlite3.connect(HASH_DB_PATH) as conn:
        conn.execute("""
            INSERT INTO embedded_chunks (hash, embedded, indexed)
            VALUES (?, 1, 0)
            ON CONFLICT(hash) DO UPDATE SET embedded=1
        """, (chunk_hash,))
        conn.commit()


def mark_chunk_indexed(chunk_hash: str):
    with sqlite3.connect(HASH_DB_PATH) as conn:
        conn.execute("UPDATE embedded_chunks SET indexed=1 WHERE hash=?", (chunk_hash,))
        conn.commit()


# ==== Local Embedding Class ====
class LocalNomicEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name=NOMIC_MODEL_PATH,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            **kwargs
        )


# ==== Local Ollama LLM ====
class LocalOllamaLLM(OllamaLLM):
    def __init__(self, **kwargs):
        super().__init__(model=OLLAMA_MODEL_NAME, **kwargs)


# ==== Initialize Embeddings & Detect Dimension ====
nomic_embedder = LocalNomicEmbeddings()

def flatten_embedding(emb):
    """Recursively flatten nested embeddings."""
    while isinstance(emb, (list, np.ndarray)) and len(emb) > 0 and isinstance(emb[0], (list, np.ndarray)):
        emb = emb[0]
    return np.array(emb, dtype=np.float32)

sample_emb = flatten_embedding(nomic_embedder.embed_query("test"))
VECTOR_DIM = len(sample_emb)
print(f"Detected embedding dimension: {VECTOR_DIM}")


# ==== Embedding Function ====
def get_combined_embedding(text: str):
    emb = nomic_embedder.embed_query(text)
    return flatten_embedding(emb)


# ==== Summarization Function ====
llama_llm = LocalOllamaLLM()


def summarize_text(text: str):
    prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
    try:
        response = llama_llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


# ==== Safe PDF Context Manager ====
@contextmanager
def safe_open_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    try:
        yield doc
    finally:
        doc.close()
        del doc
        gc.collect()


# ==== FAISS Loader / Creator ====
def safe_rmtree(path, retries=5, delay=0.5):
    import time
    for _ in range(retries):
        try:
            shutil.rmtree(path, ignore_errors=True)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"Failed to remove {path}")


def load_faiss_index(namespace: str):
    path = f"db/faiss/{namespace}"
    os.makedirs(path, exist_ok=True)
    index_file = os.path.join(path, "index.faiss")

    if os.path.exists(index_file):
        try:
            vectorstore = FAISS.load_local(path, embeddings=nomic_embedder, allow_dangerous_deserialization=True)
            if vectorstore.index.d != VECTOR_DIM:
                print(f"Dimension mismatch (existing: {vectorstore.index.d}, new: {VECTOR_DIM}), rebuilding...")
                del vectorstore
                gc.collect()
                safe_rmtree(path)
                os.makedirs(path, exist_ok=True)
            else:
                return vectorstore
        except Exception as e:
            print("Error loading FAISS index:", e)
            safe_rmtree(path)
            os.makedirs(path, exist_ok=True)

    index = faiss.IndexFlatL2(VECTOR_DIM)
    return FAISS(embedding_function=get_combined_embedding, index=index)


# ==== Add Documents to FAISS ====
def add_documents_to_faiss(docs_with_embeddings, namespace: str):
    if not docs_with_embeddings:
        print(f"No new docs for {namespace}")
        return

    vectorstore = load_faiss_index(namespace)
    texts = [doc["text"] for doc in docs_with_embeddings]
    metadatas = [doc["metadata"] for doc in docs_with_embeddings]
    embeddings = [doc["embedding"] for doc in docs_with_embeddings]

    vectorstore.add_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        metadatas=metadatas
    )

    vectorstore.save_local(f"db/faiss/{namespace}")

    for doc in docs_with_embeddings:
        mark_chunk_indexed(doc["chunk_hash"])

    print(f"Added {len(docs_with_embeddings)} docs to FAISS '{namespace}'")


# ==== Flatten PDF Chunks ====
def flatten_and_convert_structured_chunks(elements):
    flat_list = []
    for el in elements:
        text = str(el).strip()
        if text:
            flat_list.append({
                "text": text,
                "type": getattr(el, "category", "Paragraph"),
                "page_num": getattr(el, "metadata", {}).get("page_number", None)
            })
    return flat_list


def embed_pdf_structured_chunks(pdf_path, structured_chunks, metadata=None, force_reembed=False, batch_size=50):
    metadata = metadata or {}
    print(f'metadata: {metadata}')
    init_hash_db()
    docs = []

    for i in range(0, len(structured_chunks), batch_size):
        batch = structured_chunks[i:i+batch_size]
        batch_docs = []

        for chunk in batch:
            chunk_text = chunk["text"]
            chunk_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            file_name = os.path.basename(pdf_path)
            folder_namespace = os.path.normpath(pdf_path).split(os.sep)[-2].lower()
            relative_path = f"{folder_namespace}/{file_name}"
            doc_metadata = {**metadata, "chunk_hash": chunk_hash, "file_path": relative_path}
            print('docs metadata')


            if force_reembed or not is_chunk_fully_processed(chunk_hash):
                batch_docs.append({"text": chunk_text, "metadata": doc_metadata, "chunk_hash": chunk_hash})

            # Generate summary for each chunk in the batch
            summary = metadata.get("summary_map", {}).get(chunk_hash) or summarize_text(chunk_text)
            print('summary generated')

            if summary:
                summary_hash = hashlib.md5(summary.encode("utf-8")).hexdigest()
                summary_metadata = {**doc_metadata, "is_summary": True, "chunk_hash": summary_hash}
                if force_reembed or not is_chunk_fully_processed(summary_hash):
                    batch_docs.append({"text": summary, "metadata": summary_metadata, "chunk_hash": summary_hash})
            print('sumary embedded')


        # Embed after batch
        for doc in batch_docs:
            doc["embedding"] = get_combined_embedding(doc["text"])
            mark_chunk_embedded(doc["chunk_hash"])

        docs.extend(batch_docs)
        gc.collect()  # Free memory after each batch

    print(f"Embedded {len(docs)} chunks for {pdf_path}")
    return docs
