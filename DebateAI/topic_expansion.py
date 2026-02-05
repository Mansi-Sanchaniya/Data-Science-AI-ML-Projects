import os
from langchain_community.llms import Ollama

ollama = Ollama(model="mistral:7b")

def generate_topic(topic_name: str, question: str):
    prompt = f"""
Create a neutral, factual topic knowledge file.

Topic: {topic_name}

Include:
- Definition
- Benefits
- Risks
- Trade-offs
- Uncertainty

No opinions. No instructions.
"""
    content = ollama.invoke(prompt)

    path = f"data/topics/{topic_name}.txt"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    return path
