import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# === Custom Prompt ===
SMART_AGENT_PROMPT = """
You are an intelligent assistant specializing in answering user questions based on internal documents.
- Think deeply before answering.
- Reference multiple documents if needed.
- Ask a clarification question first if the query is vague.
- Quote directly from the source if the question demands an exact policy or circular.
- Otherwise, respond in your own accurate, concise language.

Context:
----------
{context}
----------

Question: {question}
Answer:
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SMART_AGENT_PROMPT.strip()
)

# === Embedder ===
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="models/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# === LLM ===
def load_openai_llm():
    return ChatOpenAI(
        model="gpt-4o",  # or "gpt-4o-mini" if on lower tier
        temperature=0.3,
        max_tokens=512
    )

# === Agent Builder ===
def build_agent(agent_name, vector_dir):
    index_path = os.path.join(vector_dir, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index missing: {index_path}")

    print(f" Loading agent: {agent_name} from {vector_dir}")
    embedder = load_embedder()
    vectorstore = FAISS.load_local(vector_dir, embedder, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = load_openai_llm()

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=True
    )

    return chain
