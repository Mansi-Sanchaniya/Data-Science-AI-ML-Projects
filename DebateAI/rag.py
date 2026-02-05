import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_docs(base):
    docs = []
    for root, _, files in os.walk(base):
        for f in files:
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                docs.append(
                    Document(
                        page_content=file.read(),
                        metadata={"source": path}
                    )
                )
    return docs

def build_vectorstore():
    print(">>> Building vectorstore...")
    docs = load_docs("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, EMBEDDINGS, persist_directory="./chroma")

def retrieve_context(vstore, query):
    results = vstore.similarity_search(query, k=5)

    text = []
    sources = set()

    for r in results:
        text.append(r.page_content)
        src = r.metadata.get("source", "")
        if "general" in src:
            sources.add("General")
        if "domains" in src:
            sources.add("Domain")
        if "topics" in src:
            sources.add("Topic")

    return {
        "text": "\n".join(text),
        "sources": list(sources)
    }
