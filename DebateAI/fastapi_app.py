# fastapi_app.py
import uvicorn
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware

from report import export_report
from app import run

app = FastAPI()

# ------------- CORS -------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- SESSION STORE -------------
SESSIONS = {}

def get_session(sid):
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "question": "",
            "timeline_text": {"PRO": "", "CON": "", "CRITIC": "", "BASELINE": ""},
            "judge_outputs": [],
            "final_output": None,
            "last_state": None
        }
    return SESSIONS[sid]

# ------------- UI HOME PAGE -------------
@app.get("/ui")
def ui_page():
    html = """
    <html>
    <head>
        <title>Multi-Agent AI Debate</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px;
                max-width: 1200px;
            }
            textarea { 
                width: 100%; 
                height: 100px; 
                padding: 10px;
                font-size: 14px;
            }
            .output-section {
                margin-top: 20px;
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                min-height: 200px;
            }
            .role-section {
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }
            .role-title {
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                color: #333;
            }
            .role-content {
                white-space: pre-wrap;
                line-height: 1.6;
                color: #555;
            }
            button { 
                padding: 12px 24px; 
                margin-top: 10px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #45a049;
            }
            .loading {
                color: #666;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <h2> Multi-Agent Debate System</h2>
        <textarea id="question" placeholder="Enter your question here..."></textarea>
        <br>
        <button onclick="start()">▶ Run Debate</button>
        
        <div class="output-section" id="output">
            <div class="loading">Waiting for debate to start...</div>
        </div>

        <script>
            // Store accumulated text for each role
            let roleTexts = {
                "PRO": "",
                "CON": "",
                "CRITIC": "",
                "BASELINE": ""
            };

            function renderOutput() {
                let html = '';
                
                for (let role of ["PRO", "CON", "CRITIC", "BASELINE"]) {
                    if (roleTexts[role]) {
                        html += `
                            <div class="role-section">
                                <div class="role-title">${role}</div>
                                <div class="role-content">${roleTexts[role]}</div>
                            </div>
                        `;
                    }
                }
                
                document.getElementById('output').innerHTML = html || '<div class="loading">Processing...</div>';
            }

            async function start() {
                let q = document.getElementById('question').value;
                if (!q) return alert("Enter a question!");
                
                // Reset
                roleTexts = {"PRO": "", "CON": "", "CRITIC": "", "BASELINE": ""};
                document.getElementById('output').innerHTML = '<div class="loading">Starting debate...</div>';

                // Create session
                let sidResp = await fetch('/session', { method: 'POST' });
                let sidJson = await sidResp.json();
                let sid = sidJson.session_id;

                // Stream
                let stream = new EventSource(`/stream?question=${encodeURIComponent(q)}&session_id=${sid}`);
                
                stream.onmessage = (event) => {
                    let data = JSON.parse(event.data);
                    
                    // Update the role's text (replacing, not appending)
                    if (data.role in roleTexts) {
                        roleTexts[data.role] = data.output;
                        renderOutput();
                    } else if (data.role === "JUDGE") {
                        console.log("Judge output:", data.output);
                    } else if (data.role === "FINAL") {
                        console.log("Final decision:", data.output);
                        stream.close();
                    }
                };
                
                stream.onerror = (error) => {
                    console.error("Stream error:", error);
                    stream.close();
                };
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ------------- HEALTH -------------
@app.get("/")
def home():
    return {"status": "ok", "message": "FastAPI Multi-Agent Debate API running 🎯"}

# ------------- CREATE SESSION -------------
@app.post("/session")
def create_session():
    sid = str(uuid4())
    SESSIONS[sid] = {
        "question": "",
        "timeline_text": {"PRO": "", "CON": "", "CRITIC": "", "BASELINE": ""},
        "judge_outputs": [],
        "final_output": None,
        "last_state": None
    }
    return {"session_id": sid}

# ------------- STREAM DEBATE -------------
@app.get("/stream")
def stream(question: str, session_id: str):

    session = get_session(session_id)
    session["question"] = question

    def event_gen():
        debate_gen = run(question)
        for state, role, output in debate_gen:
            if role in session["timeline_text"]:
                session["timeline_text"][role] = output
            elif role == "JUDGE":
                session["judge_outputs"].append(output)
            elif role == "FINAL":
                session["final_output"] = output
                session["last_state"] = state

            yield f"data: {json.dumps({'role': role, 'output': output})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ------------- STATE -------------
@app.get("/state")
def get_state(session_id: str):
    return get_session(session_id)

# ------------- EXPORT PDF -------------
@app.get("/export")
def export_pdf(session_id: str):
    session = get_session(session_id)
    final = session.get("final_output") or {}
    state = session.get("last_state") or {}

    result = {
        "question": session["question"],
        "final_decision": final.get("decision", ""),
        "final_confidence": final.get("confidence", ""),
        "baseline": state.get("baseline", ""),
        "pro": state.get("pro", []),
        "con": state.get("con", []),
        "critic": state.get("critic", []),
        "judges": session.get("judge_outputs", [])
    }

    filename = f"report_{session_id}.pdf"
    export_report(result, filename)
    return FileResponse(filename, filename=filename, media_type="application/pdf")

# ------------- RUN SERVER -------------
if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)