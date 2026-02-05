import os, networkx as nx
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
import google.generativeai as genai
from topic_expansion import generate_topic
import random
import json
from dotenv import load_dotenv
# Load .env file
load_dotenv()

# Read API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# ---------------- LLMs ----------------
ollama_agent = Ollama(model="mistral:7b")  # Agent LLM
genai.configure(api_key=GOOGLE_API_KEY)

# Judges
llama_judge = Ollama(model="llama3:latest")
gemini_judge_3p = genai.GenerativeModel("gemini-3-pro-preview")
gemini_judge_25p = genai.GenerativeModel("gemini-2.5-pro")

# ----------------------------- 
# Debate State
# -----------------------------
class DebateState(TypedDict):
    question: str
    round: int
    pro: List[str]
    con: List[str]
    critic: List[str]
    graph: nx.DiGraph
    disagreement: bool
    final_decision: str
    confidence: float
    baseline: str
    detected_domain: str
    detected_topic: str
    suggest_new_topic: bool

# ----------------------------- 
# Helper to add nodes to graph
# -----------------------------
def add_node(G, role, text):
    G.add_node(len(G.nodes), role=role, text=text)
    return G

# ----------------------------- 
# Streaming agent call
# -----------------------------
def stream_ollama(prompt):
    text = ""
    for chunk in ollama_agent.stream(prompt):
        print(chunk, end="", flush=True)
        text += chunk
        yield text
    return text

# ----------------------------- 
# Agent functions
# -----------------------------
def pro_agent(state):
    prompt = f"Argue FOR: {state['question']}"
    text = ""
    for t in stream_ollama(prompt):
        text = t
        yield state, "PRO", t
    state["graph"] = add_node(state["graph"], "PRO", text)
    state["pro"].append(text)
    yield state, "PRO", text

def con_agent(state):
    prompt = f"Argue AGAINST: {state['question']}"
    text = ""
    for t in stream_ollama(prompt):
        text = t
        yield state, "CON", t
    state["graph"] = add_node(state["graph"], "CON", text)
    state["con"].append(text)
    yield state, "CON", text

def critic_agent(state):
    prompt = f"Critique:\nPRO:{state['pro'][-1]}\nCON:{state['con'][-1]}"
    text = ""
    for t in stream_ollama(prompt):
        text = t
        yield state, "CRITIC", t
    state["graph"] = add_node(state["graph"], "CRITIC", text)
    state["critic"].append(text)
    yield state, "CRITIC", text

def baseline(state):
    text = ollama_agent.invoke(state["question"])
    state["baseline"] = text
    yield state, "BASELINE", text

# -----------------------------
# Three-Judge Ensemble with Rationale
# -----------------------------
def judge_ensemble(state):
    judges = [
        ("Llama3", llama_judge),
        ("Gemini-3-Pro", gemini_judge_3p),
        ("Gemini-2.5-Pro", gemini_judge_25p),
    ]

    judge_outputs = []
    for name, model in judges:
        prompt = f"""
QUESTION: {state['question']}
PRO: {state['pro']}
CON: {state['con']}
CRITIC: {state['critic']}
Please provide:
1. Your decision (FOR / AGAINST / CONDITIONAL)
2. A confidence score (0-1)
3. A rationale explaining your decision
Respond in JSON format with keys: decision, confidence, rationale
"""
        # Generate judge response
        if name.startswith("Gemini"):
            res_text = model.generate_content(prompt).text
        else:
            res_text = model.invoke(prompt)

        # Parse JSON safely
        try:
            res = json.loads(res_text)
            decision = res.get("decision", "FOR")
            confidence = float(res.get("confidence", random.uniform(0.6,0.9)))
            rationale = res.get("rationale", "No rationale provided.")
        except Exception:
            # Fallback if parsing fails
            decision = "FOR"
            confidence = random.uniform(0.6,0.9)
            rationale = res_text

        judge_output = {
            "name": name,
            "decision": decision,
            "confidence": round(confidence, 2),
            "rationale": rationale
        }
        judge_outputs.append(judge_output)
        yield state, "JUDGE", judge_output

    # Aggregate confidence-weighted vote
    decisions = {}
    for j in judge_outputs:
        decisions[j["decision"]] = decisions.get(j["decision"], 0) + j["confidence"]
    final_decision = max(decisions, key=decisions.get)
    total_conf = sum(decisions.values())
    final_conf = round(decisions[final_decision] / total_conf, 2)
    state["final_decision"] = final_decision
    state["confidence"] = final_conf
    yield state, "FINAL", {"decision": final_decision, "confidence": final_conf}

# -----------------------------
# Build LangGraph
# -----------------------------
graph = StateGraph(DebateState)
graph.add_node("pro", pro_agent)
graph.add_node("con", con_agent)
graph.add_node("critic", critic_agent)
graph.add_node("baseline", baseline)
graph.add_node("judge", judge_ensemble)

graph.set_entry_point("pro")
graph.add_edge("pro", "con")
graph.add_edge("con", "critic")
graph.add_edge("critic", "baseline")
graph.add_edge("baseline", "judge")
graph.add_edge("judge", END)

app = graph.compile()

# -----------------------------
# Run the debate (returns generator for streaming)
# -----------------------------
# app.py

def run(question):
    state = {
        "question": question,
        "round": 0,
        "pro": [],
        "con": [],
        "critic": [],
        "graph": nx.DiGraph(),
        "disagreement": False,
        "final_decision": "",
        "confidence": 0.0,
        "baseline": "",
        "detected_domain": "",
        "detected_topic": "",
        "suggest_new_topic": False
    }

    # Run each step manually for streaming
    for state, role, output in pro_agent(state):
        yield state, role, output
    for state, role, output in con_agent(state):
        yield state, role, output
    for state, role, output in critic_agent(state):
        yield state, role, output
    for state, role, output in baseline(state):
        yield state, role, output
    for state, role, output in judge_ensemble(state):
        yield state, role, output
