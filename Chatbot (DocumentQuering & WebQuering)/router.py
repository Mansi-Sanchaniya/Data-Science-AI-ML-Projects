

import os
from typing import List
from langchain.schema import Document
import re
import httpx
from urllib.parse import quote
from markdown import markdown
from pydantic import BaseModel, Field
from langchain.schema import BaseRetriever, Document
from typing import List, Optional, TypedDict, Dict, Tuple
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.output_parsers.regex import RegexParser
from difflib import SequenceMatcher
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from langchain.schema import AIMessage, HumanMessage
from difflib import SequenceMatcher
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ==== Build Tools ====

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever
from langchain.tools import Tool
from difflib import SequenceMatcher
from markdown import markdown
from langchain_ollama.llms import OllamaLLM
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
PDF_FOLDER = "data/pdfs"  # folder where PDFs are stored locally
os.makedirs(PDF_FOLDER, exist_ok=True)
app.mount("/pdfs", StaticFiles(directory=PDF_FOLDER), name="pdfs")

SERPAPI_API_KEY = os.getenv("SERPAPI_KEY")

transport = httpx.HTTPTransport(verify=False)
http_client = httpx.Client(transport=transport)

llm = OllamaLLM(    
    model="llama3:latest",
    temperature=0.2)

from langchain_huggingface import HuggingFaceEmbeddings

# Load local model via sentence-transformers
local_model_path = "D:/models/nomic"
gemini_embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"trust_remote_code": True}
)

# ==== Prompt templates ====

rephrase_prompt = PromptTemplate.from_template("""
Rewrite the following user question to be clearer, more detailed, and explicit for an information agent.
Keep the meaning of the question the same don't add too much.
Don't hallucinate by adding too much on your own and rewrite by keeping the same meaning.
Use the chat history below to resolve any vague terms or references.

Chat history:
\"\"\"{chat_history}\"\"\"

Question: "{question}"

Rewritten question:
""")

clarify_prompt = PromptTemplate.from_template("""
The conversation so far:
{chat_history}

The user asked: "{question}"

If you've already asked for clarification about this topic and the user did not provide new details, respond:
"I'm sorry, but I couldn't find any information about this topic"

Otherwise, ask one clear, polite clarifying question...
""")

clarifier_with_topics_prompt = PromptTemplate.from_template("""
User query: "{question}"

I found these topics in the documents related to the query:
{topics_summary}

Please ask the user one clear, polite clarifying question to help specify which topic or detail they want.

Respond ONLY with the clarifying question.
""")

smart_answer_prompt = PromptTemplate.from_template("""
You are an intelligent and expert assistant.

Here is the full chat history with the user:
\"\"\"{chat_history}\"\"\"

The user's latest question:
\"\"\"{question}\"\"\"

Here are the live web search results relevant to the question:
\"\"\"{web_results}\"\"\"

Here are the most relevant document excerpts retrieved for this question:
\"\"\"{retrieved_content}\"\"\"

Please prioritize information from the web results especially if they contain newer or more relevant information, but answer using both sources.

Your job is to provide a **clear, intelligent, and helpful answer** based on:
- The user's question
- The context from the previous conversation
- The information above

Use the previous conversation context thoroughly to answer the latest user question.
If the question refers to a previous topic or entity, answer with continuity.
If you are unsure about the answer but have partial information from the documents, provide the best possible answer and then politely ask the user if they want more details or clarification.
Only ask for clarification if it is impossible to answer based on context and documents.

Be smart, helpful, and complete. Format the answer nicely understand the formatting and format it properly which makes it readable. Follow these rules:
- Provide a clear, accurate, and comprehensive answer to the user's question.
- Synthesize the retrieved information along with conversational context.
- Understand vague or short questions; reason through the content to infer intent.
- Summarize or extract key points when asked for summaries or highlights.
- Explain next steps or implications if the user asks such questions.
- Mention the page or section if relevant.
- Be precise and user-friendly; write like a human expert, not a bot.
- Use the previous conversation context thoroughly to answer the latest user question.
- Only ask for clarification if it is impossible to answer based on the context.
- Respond with continuity to previous topics or entities in the conversation.
- If you don't find the answer, politely say "The document doesn't seem to mention this."

Answer:
""")

rephrasing_chain = LLMChain(llm=llm, prompt=rephrase_prompt)
clarifier_chain = LLMChain(llm=llm, prompt=clarify_prompt)
clarifier_topics_chain = LLMChain(llm=llm, prompt=clarifier_with_topics_prompt)
answer_generation_chain = LLMChain(llm=llm, prompt=smart_answer_prompt)

# ==== Helper functions ====

def rephrase_question(question: str, chat_history) -> str:
    try:
        chat_text = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
        rewritten = rephrasing_chain.run({
            "chat_history": chat_text,
            "question": question
        }).strip()
        print(f"[Rephrased Question]: {rewritten}")
        return rewritten if rewritten else question
    except Exception as e:
        print(f"[Rephrase Error]: {e}")
        return question

def summarize_topics_from_docs(docs: List) -> str:
    if not docs:
        return ""
    summaries = []
    for idx, doc in enumerate(docs[:10], 1):
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        summaries.append(f"{idx}. {snippet}...")
    return "\n".join(summaries)

memory_circulars = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key="answer", k=10)
memory_research = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key="answer", k=10)
memory_master = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key="answer", k=10)

def build_tool(agent_name: str, memory: ConversationBufferWindowMemory) -> Tool:
    vector_dir = f"db/faiss/{'compliance' if agent_name == 'CircularsAgent' else 'research'}"
    vectorstore = FAISS.load_local(vector_dir, gemini_embeddings, allow_dangerous_deserialization=True)

    # === METADATA BOOSTING FUNCTION ===
    def metadata_filter_boost(query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        query_terms = set(re.findall(r'\w+', query.lower()))
        boosted_docs = []

        for doc in docs:
            metadata = doc.metadata or {}
            score = 0

            # Boost docs marked as summaries
            if metadata.get("is_summary") in [True, "true", "True"]:
                score += 10

            # Section title/type matching
            section_title = str(metadata.get("section_title", "")).lower()
            section_type = str(metadata.get("section_type", "")).lower()
            
            section_title_terms = set(re.findall(r'\w+', section_title))
            section_type_terms = set(re.findall(r'\w+', section_type))
            
            if query_terms.intersection(section_title_terms):
                score += 5
            if query_terms.intersection(section_type_terms):
                score += 3

            # Entities & keyphrases matching
            entities = set()
            for ent in metadata.get("entities", []):
                if isinstance(ent, (list, tuple)) and ent:
                    entities.add(ent[0].lower())
                elif isinstance(ent, str):
                    entities.add(ent.lower())

            keyphrases = set()
            for phrase in metadata.get("keyphrases", []):
                if isinstance(phrase, str):
                    keyphrases.add(phrase.lower())

            if query_terms.intersection(entities):
                score += 8  # strong boost for entity matches
            if query_terms.intersection(keyphrases):
                score += 6  # boost for keyphrase matches

            boosted_docs.append((score, doc))

        # Sort by score (descending), then return documents
        boosted_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in boosted_docs[:20]]

    # Base retriever
    base_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 30})
    
    # Wrapper that applies metadata boosting
    class MetadataBoostedRetriever:
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
            
        def invoke(self, query: str) -> List[Document]:
            try:
                # Get initial results
                docs = self.base_retriever.invoke(query)
                # Apply metadata boosting
                boosted_docs = metadata_filter_boost(query, docs)
                return boosted_docs[:8]  # Return top 8 after boosting
            except Exception as e:
                print(f"[Retriever Error]: {e}")
                return []
    
    retriever = MetadataBoostedRetriever(base_retriever)
    
    def find_dominant_source(answer_text: str, source_docs: List[Document]) -> tuple:
        if not source_docs:
            return None, 0
        best_doc = None
        best_score = -float('inf')
        for doc in source_docs:
            similarity = SequenceMatcher(None, answer_text, doc.page_content).ratio()
            search_score = doc.metadata.get("@search.reranker_score") or doc.metadata.get("@search.score") or 0
            combined_score = search_score * 0.7 + similarity * 30
            if combined_score > best_score:
                best_score = combined_score
                best_doc = doc
        return best_doc, best_score

    def run_qa(question: str, chat_history: List) -> dict:
        try:
            # Step 1: rephrase question if needed
            q_lower = question.strip().lower()
            if len(question.split()) < 5 or q_lower in {"what next?", "how so?"} or "this" in q_lower:
                question_rephrased = rephrase_question(question, chat_history)
            else:
                question_rephrased = question
            print(f'Question resphrased: {question_rephrased}')

            # Step 2: Use invoke() directly on FAISS retriever
            retrieved_docs = retriever.invoke(question_rephrased)
            print(f'retrieved docs count: {len(retrieved_docs) if retrieved_docs else 0}')
            
            top_docs = retrieved_docs[:6] if retrieved_docs else []
            print(f'top docs count: {len(top_docs)}')

            # Prepare concatenated context for final answer
            retrieved_content = "\n\n".join([doc.page_content for doc in top_docs])
            print(f'retrieved_content length: {len(retrieved_content)}')

            # Step 3: Generate final answer using best chunks
            try:
                final_answer = answer_generation_chain.run({
                    "chat_history": "\n".join([f"{m.type}: {m.content}" for m in chat_history]),
                    "question": question,
                    "retrieved_content": retrieved_content,
                    "web_results": ""
                })

                print(f"[DEBUG] Raw LLM Response: {repr(final_answer)}")
                print(f"[DEBUG] Response type: {type(final_answer)}")
                print(f"[DEBUG] Response length: {len(final_answer) if final_answer else 'None'}")

                final_answer = final_answer.strip() if final_answer else "No response generated"
            except Exception as e:
                print(f"[DEBUG] Answer generation failed: {e}")
                traceback.print_exc()
                final_answer = "Sorry, I'm having trouble generating an answer."
            
            # Prepare local PDF links for user
            formatted_sources = ""
            dominant_doc = None
            if top_docs:
                dominant_doc, _ = find_dominant_source(final_answer, top_docs)
                if dominant_doc:
                    src = dominant_doc.metadata.get("file_path")
                    if src:
                        file_url = f"/pdfs/{src.replace(os.sep, '/')}"
                        display_name = os.path.basename(src)
                        formatted_sources += f"📎 <a href='{file_url}' target='_blank'>{display_name}</a><br>"

            html_answer = markdown(final_answer, extensions=['extra'])
            formatted_result = f"<div style='white-space: pre-line;'>{html_answer}</div><br><b>Dominant Source:</b><br>{formatted_sources}"

            return {"output": formatted_result, "source_documents": [dominant_doc] if dominant_doc else []}
            
        except Exception as e:
            print(f"[{agent_name} QA Error]: {e}")
            traceback.print_exc()
            return {"output": "Sorry, I'm having trouble answering your query right now.", "source_documents": []}

    return Tool(name=agent_name, func=run_qa, description=f"Use this tool for {agent_name} documents.")

def build_master_tool() -> Tool:
    vectorstore_compliance = FAISS.load_local("db/faiss/compliance", gemini_embeddings, allow_dangerous_deserialization=True)
    vectorstore_research = FAISS.load_local("db/faiss/research", gemini_embeddings, allow_dangerous_deserialization=True)

    # === METADATA BOOSTING FUNCTION ===
    def metadata_filter_boost(query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        query_terms = set(re.findall(r'\w+', query.lower()))
        boosted_docs = []

        for doc in docs:
            metadata = doc.metadata or {}
            score = 0

            if metadata.get("is_summary") in [True, "true", "True"]:
                score += 10

            section_title = str(metadata.get("section_title", "")).lower()
            section_type = str(metadata.get("section_type", "")).lower()
            
            section_title_terms = set(re.findall(r'\w+', section_title))
            section_type_terms = set(re.findall(r'\w+', section_type))
            
            if query_terms.intersection(section_title_terms):
                score += 5
            if query_terms.intersection(section_type_terms):
                score += 3

            entities = set()
            for ent in metadata.get("entities", []):
                if isinstance(ent, (list, tuple)) and ent:
                    entities.add(ent[0].lower())
                elif isinstance(ent, str):
                    entities.add(ent.lower())

            keyphrases = set()
            for phrase in metadata.get("keyphrases", []):
                if isinstance(phrase, str):
                    keyphrases.add(phrase.lower())

            if query_terms.intersection(entities):
                score += 8
            if query_terms.intersection(keyphrases):
                score += 6

            boosted_docs.append((score, doc))

        boosted_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in boosted_docs[:15]]

    retriever_compliance = vectorstore_compliance.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15})
    retriever_research = vectorstore_research.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15})

    def find_dominant_source(answer_text: str, source_docs: List[Document]) -> tuple:
        if not source_docs:
            return None, 0
        best_doc = None
        best_score = -float('inf')
        for doc in source_docs:
            similarity = SequenceMatcher(None, answer_text, doc.page_content).ratio()
            search_score = doc.metadata.get("@search.reranker_score") or doc.metadata.get("@search.score") or 0
            combined_score = search_score * 0.7 + similarity * 30
            if combined_score > best_score:
                best_score = combined_score
                best_doc = doc
        return best_doc, best_score

    def run_qa(question: str, chat_history: List) -> dict:
        try:
            question_rephrased = question
            
            # Get docs from both retrievers
            docs1 = retriever_compliance.invoke(question_rephrased)
            docs2 = retriever_research.invoke(question_rephrased)
            combined_docs = (docs1 or []) + (docs2 or [])
            
            # Apply metadata boosting to combined results
            boosted_docs = metadata_filter_boost(question_rephrased, combined_docs)
            top_docs = boosted_docs[:4]
            
            retrieved_content = "\n\n".join([doc.page_content for doc in top_docs])

            try:
                final_answer = answer_generation_chain.run({
                    "chat_history": "\n".join([f"{m.type}: {m.content}" for m in chat_history]),
                    "question": question,
                    "retrieved_content": retrieved_content,
                    "web_results": ""
                })
                final_answer = final_answer.strip() if final_answer else "No response generated"
            except Exception as e:
                print(f"[Master Tool Answer Generation Error]: {e}")
                traceback.print_exc()
                final_answer = "Sorry, I'm having trouble generating an answer."

            # Format source links
            formatted_sources = ""
            dominant_doc = None
            if top_docs:
                dominant_doc, _ = find_dominant_source(final_answer, top_docs)
                if dominant_doc:
                    src = dominant_doc.metadata.get("file_path")
                    if src:
                        file_url = f"/pdfs/{src.replace(os.sep, '/')}"
                        display_name = os.path.basename(src)
                        formatted_sources += f"📎 <a href='{file_url}' target='_blank'>{display_name}</a><br>"

            html_answer = markdown(final_answer, extensions=['extra'])
            formatted_result = f"<div style='white-space: pre-line;'>{html_answer}</div><br><b>Dominant Source:</b><br>{formatted_sources}"

            return {"output": formatted_result, "source_documents": [dominant_doc] if dominant_doc else []}

        except Exception as e:
            print(f"[MasterAgent QA Chain Error]: {e}")
            traceback.print_exc()
            return {"output": "Sorry, I am having trouble answering that right now.", "source_documents": []}

    return Tool(name="MasterAgent", func=run_qa, description="Use this tool to answer questions combining compliance and research knowledge bases.")

# ==== Clarifier Tool Functions ====

def clarify_tool_func(
    question: str,
    retrieved_docs: Optional[List] = None,
    chat_history: Optional[List] = None,
    max_clarify_turns: int = 2
) -> dict:
    """
    Clarify user question based on context and prevent infinite clarification loops.
    """
    if chat_history is None:
        chat_history = []

    def same_clarification_loop(history: List, target_phrase: str) -> bool:
        clarifications = [
            m.content for m in history[-4:]
            if hasattr(m, "content") and isinstance(m, AIMessage)
        ]
        rep_count = sum(target_phrase.lower() in c.lower() for c in clarifications)
        return rep_count >= max_clarify_turns

    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history]) if chat_history else ""
    print(f"Chat history:\n{chat_history_str}")

    # If relevant docs exist, use clarifier with topics
    if retrieved_docs and len(retrieved_docs) > 0:
        topics_summary = summarize_topics_from_docs(retrieved_docs)
        try:
            clarification = clarifier_topics_chain.run({
                "question": question,
                "topics_summary": topics_summary,
                "chat_history": chat_history_str
            }).strip()
            print(f"[Clarifier with topics] Asked: {clarification}")
        except Exception as e:
            print(f"[Clarifier topics error]: {e}")
            clarification = "Could you please specify which aspect you are interested in?"
        return {"output": clarification, "source_documents": []}

    # Generic fallback clarifier with loop prevention
    fallback_message = "Sorry, I couldn't find information relevant to your query in our data."
    try:
        if len(chat_history) > 4:
            if same_clarification_loop(chat_history, question):
                print("[Clarifier] Breaking loop, returning fallback message.")
                return {"output": fallback_message, "source_documents": []}

        clarification = clarifier_chain.run({
            "question": question,
            "chat_history": chat_history_str
        }).strip()
        print(f"[Clarifier fallback] Asked: {clarification}")
    except Exception as e:
        print(f"[Clarifier fallback error]: {e}")
        clarification = "Sorry, could you please rephrase your question?"

    return {"output": clarification, "source_documents": []}

clarify_tool = Tool(
    name="ClarifyAgent",
    func=lambda question, retrieved_docs=None, chat_history=None: clarify_tool_func(
        question,
        retrieved_docs=retrieved_docs,
        chat_history=chat_history
    ),
    description="Ask user for clarification with contextual options."
)

# ==== Agents dictionary ====

tools = {
    "CircularsAgent": build_tool("CircularsAgent", memory_circulars),
    "ResearchAgent": build_tool("ResearchAgent", memory_research),
    "MasterAgent": build_master_tool(),
    "ClarifyAgent": clarify_tool,
}

def call_tool(state: Dict) -> Dict:
    tool_name = state["tool_to_use"]
    tool = tools.get(tool_name)
    print(f"Chosen tool: {tool_name}")

    try:
        if tool_name == "ClarifyAgent":
            result = clarify_tool_func(
                state["question"],
                retrieved_docs=state.get("retrieved_documents", []),
                chat_history=state.get("chat_history", [])
            )
        else:
            result = tool.func(state["question"], state.get("chat_history", []))
            answer_text = result.get("output", "").strip()
            source_docs = result.get("source_documents", [])
            if (answer_text.lower() in {"i don't know.", "unknown", ""} or not source_docs):
                print("[call_tool] No valid answer found, invoking clarify tool.")
                result = clarify_tool_func(
                    state["question"],
                    retrieved_docs=source_docs,
                    chat_history=state.get("chat_history", [])
                )
                answer_text = result.get("output", answer_text)
                source_docs = result.get("source_documents", [])
        
        chat_history: List = state.get("chat_history", [])

        chat_history.append(HumanMessage(content=state["question"]))
        chat_history.append(AIMessage(content=answer_text))

        source_paths = [doc.metadata.get("source") for doc in source_docs if doc.metadata.get("source")]

        return {
            "result": answer_text,
            "sources": source_paths,
            "chat_history": chat_history
        }
    except Exception as e:
        print(f"[call_tool Error]: {e}")
        traceback.print_exc()
        return {
            "result": "Sorry, I'm having trouble processing your request right now.",
            "sources": [],
            "chat_history": state.get("chat_history", [])
        }

# ==== Global conversation history ====

conversation_history: List = []

# ==== SerpAPI Web Search function (returns snippet and URL) ====

def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }
    try:
        with httpx.Client(timeout=10) as client:
            res = client.get("https://serpapi.com/search", params=params)
            res.raise_for_status()
            data = res.json()
            if "organic_results" in data:
                results = []
                for r in data["organic_results"][:num_results]:
                    snippet = r.get("snippet", "")
                    link = r.get("link", "")
                    results.append({"snippet": snippet, "link": link})
                return results
            return []
    except Exception as e:
        print(f"[SerpAPI Error]: {e}")
        return []

def load_memory_from_db_messages(messages: List[Dict]) -> ConversationBufferWindowMemory:

    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key="answer", k=10)
    for msg in messages:
        if msg['role'] == 'user':
            memory.chat_memory.add_user_message(msg['content'])
        elif msg['role'] == 'bot':
            memory.chat_memory.add_ai_message(msg['content'])
    return memory

# ==== Main interface function ====
def ask_question(
    question: str,
    user_access_modules: Optional[List[str]] = None,
    browser_mode: bool = False
) -> dict:
    global conversation_history
    if user_access_modules is None:
        user_access_modules = []

    print(f"User_access_Modules: {user_access_modules}, browser_mode: {browser_mode}")

    try:

        if "compliance" in user_access_modules and "research" in user_access_modules:
            chosen_tool = "MasterAgent"
            memory = memory_master
            
            # Simple function-based approach instead of custom class
            vectorstore_compliance = FAISS.load_local("db/faiss/compliance", gemini_embeddings, allow_dangerous_deserialization=True)
            vectorstore_research = FAISS.load_local("db/faiss/research", gemini_embeddings, allow_dangerous_deserialization=True)
            
            retriever_compliance = vectorstore_compliance.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
            retriever_research = vectorstore_research.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
            
            # Use a simple function instead of custom class
            def combined_invoke(query: str):
                try:
                    docs1 = retriever_compliance.invoke(query)
                    docs2 = retriever_research.invoke(query)
                    return (docs1 or []) + (docs2 or [])
                except Exception as e:
                    print(f"[Combined retriever error]: {e}")
                    return []
            
            retriever = type('SimpleRetriever', (), {'invoke': combined_invoke})()

        elif "compliance" in user_access_modules:
            chosen_tool = "CircularsAgent"
            memory = memory_circulars
            vectorstore_compliance = FAISS.load_local("db/faiss/compliance", gemini_embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore_compliance.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

        elif "research" in user_access_modules:
            chosen_tool = "ResearchAgent"
            memory = memory_research
            vectorstore_research = FAISS.load_local("db/faiss/research", gemini_embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore_research.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})
        else:
            return {"answer": "You do not have access to any knowledge base.", "sources": []}
        
        print(f'chosen tool: {chosen_tool}')


        chat_history_text = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])
        
        decider_prompt = f"""
            You are an intelligent assistant.
            You can answer in two ways:
            1. If the question clearly needs factual/specific data from the knowledge base, reply:
            MODE: RETRIEVE
            SEARCH_QUERY: <what to search for>
            2. If you can answer from your own knowledge/chat context without documents, reply:
            MODE: DIRECT
            ANSWER: <your answer>

            The conversation so far:
            {chat_history_text}

            The user's question:
            \"\"\"{question}\"\"\"

            Respond ONLY in the above format.
            """

        try:
            decision = llm.invoke(decider_prompt).strip()
        except Exception as e:
            print(f"[LLM Invoke Error]: {e}")
            # Fallback to RETRIEVE mode if LLM fails
            decision = "MODE: RETRIEVE\nSEARCH_QUERY: " + question
            
        print(f"[DECIDER RAW OUTPUT] {decision}")

        mode = ""
        content_lines = []
        capture_answer = False

        for line in decision.splitlines():
            if line.startswith("MODE:"):
                mode = line.split(":", 1)[1].strip().upper()
            elif line.startswith("ANSWER:"):
                capture_answer = True
                content_lines.append(line.split(":", 1)[1].strip())
            elif line.startswith("SEARCH_QUERY:"):
                content_lines.append(line.split(":", 1)[1].strip())
            elif capture_answer:
                # Keep appending all lines until next section or end
                content_lines.append(line.strip())

        content = "\n".join(content_lines).strip()

        if browser_mode and mode == "DIRECT":
            print("[INFO] Browser mode enabled – overriding MODE:DIRECT to MODE:RETRIEVE")
            mode = "RETRIEVE"
            # Use the question itself if SEARCH_QUERY was not provided
            if not content:
                content = question

        memory.chat_memory.add_user_message(question)

        if mode == "DIRECT":
            # LLM answers without retrieval
            direct_answer = content  # return raw LLM output (Markdown/plain text)

            memory.chat_memory.add_ai_message(content)
            conversation_history.extend([HumanMessage(content=question), AIMessage(content=content)])
            return {"answer": direct_answer, "sources": []}

        elif mode == "RETRIEVE":
            search_query = content
            print(f"[DECIDER] Selected RETRIEVE with query: {search_query}")

            # Use invoke() consistently
            retrieved_docs = retriever.invoke(search_query)
            print(f'retrieved docs count: {len(retrieved_docs) if retrieved_docs else 0}')
            print(f'chat history messages count: {len(memory.chat_memory.messages)}')
            
            if not retrieved_docs:
                print('No documents found, using clarify tool')
                clarification = clarify_tool_func(question, chat_history=memory.chat_memory.messages)["output"]
                memory.chat_memory.add_ai_message(clarification)
                return {"answer": clarification, "sources": []}

            if not browser_mode:
                input_state = {
                    "tool_to_use": chosen_tool,
                    "question": search_query,
                    "chat_history": memory.chat_memory.messages,
                    "retrieved_documents": [],
                }

                response = call_tool(input_state)
                print(f'Tool response: {response}')
                answer = response["result"].strip()
                sources = response.get("sources", [])
                print(f'Final sources: {sources}')

                memory.chat_memory.add_ai_message(answer)
                conversation_history.extend([HumanMessage(content=question), AIMessage(content=answer)])

                return {"answer": answer, "sources": sources}

            else:
                # Browser mode retrieval
                web_results = search_web(search_query, num_results=5)
                web_context = "\n".join([res['snippet'] for res in web_results])
                doc_results = retriever.invoke(search_query)
                docs_context = "\n\n".join([doc.page_content for doc in doc_results[:5]])

                prompt_input = {
                    "chat_history": chat_history_text,
                    "question": question,
                    "web_results": web_context,
                    "retrieved_content": docs_context,
                }

                try:
                    final_answer = answer_generation_chain.run(prompt_input).strip()
                except Exception as e:
                    print(f"[Browser mode answer generation error]: {e}")
                    final_answer = "Sorry, I'm having trouble answering your query right now."

                sources = [" Web search results from Google via SerpAPI:"]
                for idx, res in enumerate(web_results, 1):
                    snippet = res.get('snippet', '')
                    link = res.get('link', '')
                    if link:
                        sources.append(f"{idx}. {snippet} [Source]({link})")
                    else:
                        sources.append(f"{idx}. {snippet}")

                for doc in doc_results[:3]:
                    src = doc.metadata.get("file_path")  # relative path like "compliance/file1.pdf"
                    if src:
                        file_url = f"/pdfs/{src.replace(os.sep, '/')}"  
                        display_name = os.path.basename(src)
                        sources.append(f"📎 Source: {display_name} ({file_url})")

                html_answer = markdown(final_answer, extensions=['extra'])
                formatted_result = f"<div style='white-space: pre-line;'>{html_answer}</div>"

                memory.chat_memory.add_ai_message(final_answer)
                conversation_history.extend([HumanMessage(content=question), AIMessage(content=final_answer)])

                return {"answer": formatted_result, "sources": sources}

        else:
            # fallback if mode is not understood
            answer = "Sorry, I couldn't determine how to answer your question."
            memory.chat_memory.add_ai_message(answer)
            conversation_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
            return {"answer": answer, "sources": []}
            
    except Exception as e:
        print(f"[ask_question Error]: {e}")
        traceback.print_exc()
        return {"answer": "Sorry, I'm experiencing technical difficulties. Please try again.", "sources": []}