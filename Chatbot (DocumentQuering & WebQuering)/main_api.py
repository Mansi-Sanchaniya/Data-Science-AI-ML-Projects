import os
import uuid
import markdown
import io
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from PIL import Image
import pytesseract
from fpdf import FPDF
import imghdr
import snowflake.connector

from models import RegisterRequest, LoginRequest
from auth import register_user, login_user
from router import ask_question, load_memory_from_db_messages
from convert import pdf_to_word, word_to_pdf, image_to_pdf_with_ocr, ocr_pdf_to_text_pdf, cleanup_file
from dotenv import load_dotenv
load_dotenv()

# === Directories ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# === FastAPI app ===
app = FastAPI(title="ARG DocumentAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(".")), name="static")

executor = ThreadPoolExecutor(max_workers=4)

def get_connection():

    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )

def save_message(session_id: str, sender: str, message: str):
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Convert UUID to string if needed
        session_id_str = str(session_id)

        # Verify session exists
        cursor.execute("SELECT COUNT(*) FROM UserSessions WHERE SessionId = %s", (session_id_str,))
        if cursor.fetchone()[0] == 0:
            raise HTTPException(status_code=400, detail=f"Session ID does not exist: {session_id}")

        # Insert message
        cursor.execute(
            "INSERT INTO SessionMessages (SessionId, Sender, MessageText) VALUES (%s, %s, %s)",
            (session_id_str, sender, message)
        )

        # Update last updated
        cursor.execute(
            "UPDATE UserSessions SET LastUpdated = CURRENT_TIMESTAMP WHERE SessionId = %s",
            (session_id_str,)
        )

        conn.commit()
    finally:
        if conn:
            conn.close()

# ===================== Session Title Update =====================
def update_session_title_if_needed(session_id: str, first_message: str):
    conn = get_connection()
    cursor = conn.cursor()

    session_id_str = str(session_id)
    cursor.execute("SELECT SessionTitle FROM UserSessions WHERE SessionId = %s", (session_id_str,))
    row = cursor.fetchone()
    current_title = row[0] if row else None

    if not current_title or current_title.strip() == "" or current_title == "New Chat":
        new_title = (first_message[:50] + "...") if len(first_message) > 50 else first_message
        cursor.execute(
            "UPDATE UserSessions SET SessionTitle = %s WHERE SessionId = %s",
            (new_title, session_id_str)
        )
        conn.commit()
    conn.close()

# ===================== Models =====================
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    access_modules: Optional[List[str]] = []
    browser_mode: Optional[bool] = False

class SaveSessionRequest(BaseModel):
    user_id: str
    title: Optional[str] = None


@app.post("/convert-format")
async def convert_format(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    sourceFormat: str = Form(...),
    targetFormat: str = Form(...),
    extract_text_from_pdf_images: bool = Form(False),
    extract_images_only: bool = Form(False),  # new param for image extraction
):
    temp_output_path = None

    # 2. Multiple images to single PDF with OCR
    if sourceFormat == "image" and targetFormat == "pdf" and files:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="Please upload at least one image file.")
        temp_dir = os.path.join(TMP_DIR, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        image_paths = []
        try:
            for upload_file in files:
                ext = os.path.splitext(upload_file.filename)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")
                temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
                contents = await upload_file.read()
                with open(temp_path, "wb") as f:
                    f.write(contents)
                image_paths.append(temp_path)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            for i, img_path in enumerate(image_paths):
                image = Image.open(img_path).convert('L')
                text = pytesseract.image_to_string(image, config='--oem 3 --psm 6', lang='eng')
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=f"Image {i+1} Extracted Text:\n\n{text.strip()}")
                pdf.image(img_path, x=10, y=pdf.get_y() + 10, w=pdf.w - 20)

            temp_output_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_multi_image_ocr.pdf")
            pdf.output(temp_output_path)

            with open(temp_output_path, "rb") as f:
                converted_bytes = f.read()
            return StreamingResponse(
                io.BytesIO(converted_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={os.path.basename(temp_output_path)}"},
            )
        finally:
            for path in image_paths:
                try:
                    os.remove(path)
                except:
                    pass
            try:
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass
            try:
                if temp_output_path and os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
            except:
                pass

    # Single file upload processing
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a file.")

    contents = await file.read()
    temp_input_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_input_path, "wb") as f:
        f.write(contents)

    try:
        # 3. PDF → Word
        if sourceFormat == "pdf" and targetFormat == "word":
            temp_output_path = pdf_to_word(temp_input_path)

        # 4. Word → PDF
        elif sourceFormat == "word" and targetFormat == "pdf":
            temp_output_path = word_to_pdf(temp_input_path)

        # 1. PDF images OCR → Text PDF
        elif sourceFormat == "pdf" and targetFormat == "pdf" and extract_text_from_pdf_images:
            temp_output_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_ocr_text.pdf")
            temp_output_path = ocr_pdf_to_text_pdf(temp_input_path, temp_output_path, dpi=300)

        # 5. Extract images only from PDF (output zipped or handled externally)
        elif sourceFormat == "pdf" and extract_images_only:
            # Extract images and return as a zip or list of images (implementation depends)
            raise HTTPException(status_code=501, detail="Extract images only functionality not implemented yet.")

        # Single image → PDF conversion fallback
        elif sourceFormat == "image" and targetFormat == "pdf":
            if not imghdr.what(temp_input_path):
                raise HTTPException(status_code=400, detail="Uploaded file is not a supported image.")
            temp_output_path = image_to_pdf_with_ocr(temp_input_path)

        else:
            raise HTTPException(status_code=400, detail="Unsupported conversion types.")

        with open(temp_output_path, "rb") as f:
            converted_bytes = f.read()

        output_filename = os.path.basename(temp_output_path)
        return StreamingResponse(
            io.BytesIO(converted_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"},
        )
    finally:
        cleanup_file(temp_input_path)
        if temp_output_path:
            cleanup_file(temp_output_path)


# ===================== Basic Routes =====================
@app.get("/", response_class=HTMLResponse)
def root_page():
    index_path = Path("register.html")
    if index_path.exists():
        return index_path.read_text()
    return "<h1>ARG DocumentAI is running (register.html not found)</h1>"

@app.post("/register")
def register(payload: RegisterRequest):
    message = register_user(payload.email, payload.password)
    return {"message": message}

@app.post("/login")
def login(payload: LoginRequest):
    user_info = login_user(payload.email, payload.password)
    if user_info:
        return {
            "redirect": "/static/chat.html",
            "message": "Redirecting to chatbot...",
            "name": user_info["name"],
            "email": user_info["email"],
            "access_modules": user_info.get("access_modules", [])
        }
    return {"detail": "Unauthorized"}

@app.post("/logout")
async def logout(request: Request):
    return {"detail": "Logged out"}

# ===================== Session Routes =====================
@app.post("/save-session")
async def save_session(payload: SaveSessionRequest):
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # Check existing "New Chat" sessions
        cursor.execute(
            """
            SELECT SessionId FROM UserSessions
            WHERE UserId = %s AND (SessionTitle IS NULL OR SessionTitle = '' OR SessionTitle = 'New Chat')
            """,
            (payload.user_id,)
        )
        existing_session = cursor.fetchone()
        if existing_session:
            return {"session_id": str(existing_session[0])}

        # Create new session
        session_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO UserSessions (SessionId, UserId, SessionTitle) VALUES (%s, %s, %s)",
            (session_id, payload.user_id, payload.title or None)
        )
        conn.commit()
        return {"session_id": session_id}
    finally:
        if conn:
            conn.close()

@app.get("/get-sessions")
def get_sessions(user_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT SessionId, SessionTitle, CreatedAt, LastUpdated
            FROM UserSessions
            WHERE UserId = %s
            ORDER BY LastUpdated DESC
            """,
            (user_id,)
        )
        rows = cursor.fetchall()
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "title": row[1] or "",
                "created_at": row[2],
                "last_updated": row[3],
            })
        return sessions
    finally:
        conn.close()

@app.get("/history")
def get_history(user_id: str, session_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Verify session belongs to user
        cursor.execute(
            "SELECT COUNT(*) FROM UserSessions WHERE SessionId = %s AND UserId = %s",
            (session_id, user_id)
        )
        if cursor.fetchone()[0] == 0:
            raise HTTPException(status_code=403, detail="Session does not belong to user")

        cursor.execute(
            "SELECT Sender, MessageText, CreatedAt FROM SessionMessages WHERE SessionId = %s ORDER BY CreatedAt ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]
    finally:
        conn.close()

# ===================== Chat Endpoint =====================
@app.post("/chat")
async def chat(payload: ChatRequest, background_tasks: BackgroundTasks):
    try:
        if not payload.session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        # Verify session exists
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM UserSessions WHERE SessionId = %s", (payload.session_id,))
        if cursor.fetchone()[0] == 0:
            raise HTTPException(status_code=400, detail="Invalid or non-existent session_id")
        conn.close()

        save_message(payload.session_id, "user", payload.question)
        update_session_title_if_needed(payload.session_id, payload.question)

        result = ask_question(
            payload.question,
            user_access_modules=payload.access_modules,
            browser_mode=payload.browser_mode
        )
        print(f'Result: {result}')
        print("result fetched")

        save_message(payload.session_id, "bot", result["answer"])

        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except HTTPException as he:
        print(f"[ERROR] Chat HTTPException: {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        return JSONResponse(status_code=500, content={"answer": "Error processing request."})
