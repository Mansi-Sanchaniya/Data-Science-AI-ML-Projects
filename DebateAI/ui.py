import streamlit as st
from app import run
from report import export_report
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.title("🧠 Multi-Agent Debate AI")

# -------------------- SESSION STATE --------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "question" not in st.session_state:
    st.session_state.question = ""

if "timeline_text" not in st.session_state:
    st.session_state.timeline_text = {
        "PRO": "", "CON": "", "CRITIC": "", "BASELINE": ""
    }

if "judge_outputs" not in st.session_state:
    st.session_state.judge_outputs = []

if "final_output" not in st.session_state:
    st.session_state.final_output = None

if "last_state" not in st.session_state:
    st.session_state.last_state = None

# -------------------- QUESTION INPUT --------------------
q = st.text_area(
    "Enter a decision question",
    value=st.session_state.question
)
st.session_state.question = q

# -------------------- RUN BUTTON --------------------
if st.button("Run Debate"):
    if not q.strip():
        st.warning("Please enter a question")
    elif not st.session_state.running:
        st.session_state.running = True
        st.session_state.timeline_text = {
            "PRO": "", "CON": "", "CRITIC": "", "BASELINE": ""
        }
        st.session_state.judge_outputs = []
        st.session_state.final_output = None
        st.session_state.last_state = None

# -------------------- PLACEHOLDERS --------------------
timeline_placeholder = st.empty()
judge_table_placeholder = st.empty()
chart_placeholder = st.empty()
final_placeholder = st.empty()

# -------------------- STREAMING --------------------
if st.session_state.running:
    debate_gen = run(st.session_state.question)

    for state, role, output in debate_gen:

        if role in ["PRO", "CON", "CRITIC", "BASELINE"]:
            # Update the text for this specific role (accumulates word by word)
            st.session_state.timeline_text[role] = output
            
            # Render entire timeline after each update
            combined_text = ""
            for r in ["PRO", "CON", "CRITIC", "BASELINE"]:
                if st.session_state.timeline_text[r]:
                    combined_text += f"**{r}**:\n{st.session_state.timeline_text[r]}\n\n"
            
            timeline_placeholder.markdown(combined_text)

        elif role == "JUDGE":
            st.session_state.judge_outputs.append(output)

        elif role == "FINAL":
            st.session_state.final_output = output
            st.session_state.last_state = state
            st.session_state.running = False

# -------------------- RENDER FINAL STATE --------------------
if not st.session_state.running:
    combined_text = ""
    for r in ["PRO", "CON", "CRITIC", "BASELINE"]:
        if st.session_state.timeline_text[r]:
            combined_text += f"**{r}**:\n{st.session_state.timeline_text[r]}\n\n"
    
    if combined_text:
        timeline_placeholder.markdown(combined_text)

if st.session_state.judge_outputs:
    df = pd.DataFrame(st.session_state.judge_outputs)
    df = df.rename(columns={
        "name": "Judge",
        "decision": "Decision",
        "confidence": "Confidence",
        "rationale": "Rationale"
    })

    judge_table_placeholder.table(
        df[["Judge", "Decision", "Confidence", "Rationale"]]
    )

    chart_data = df.groupby("Decision")["Confidence"].sum().reset_index()
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Decision", sort=None),
        y="Confidence",
        color="Decision",
        tooltip=["Decision", "Confidence"]
    )
    chart_placeholder.altair_chart(chart, use_container_width=True)

if st.session_state.final_output:
    final_placeholder.subheader("✅ Final Decision")
    final_placeholder.markdown(
        f"**{st.session_state.final_output['decision']}** "
        f"(Confidence: {st.session_state.final_output['confidence']})"
    )

# -------------------- DETECTED DOMAIN / TOPIC --------------------
if st.session_state.last_state:
    state = st.session_state.last_state
    if state.get("detected_domain") and state.get("detected_topic"):
        st.subheader("Detected Domain & Topic")
        st.write(f"{state['detected_domain']} → {state['detected_topic']}")

# -------------------- PDF EXPORT --------------------
if st.session_state.final_output and st.button("📄 Export PDF"):
    result = {
        "question": st.session_state.question,
        "final_decision": st.session_state.final_output["decision"],
        "confidence": st.session_state.final_output["confidence"],
        "baseline": st.session_state.last_state.get("baseline", ""),
        "pro": st.session_state.last_state.get("pro", []),
        "con": st.session_state.last_state.get("con", []),
        "critic": st.session_state.last_state.get("critic", []),
        "judges": st.session_state.judge_outputs
    }
    export_report(result)
    st.success("📄 PDF exported successfully!")