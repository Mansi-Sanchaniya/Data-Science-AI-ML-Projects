import sqlite3
import os

DB_PATH = "db/metadata.db"

def init_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS pdf_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        circular_no TEXT,
        title TEXT,
        date TEXT,
        links TEXT,
        embedded INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    conn.close()

def save_metadata(filename, metadata):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO pdf_metadata (filename, circular_no, title, date, links, embedded)
        VALUES (?, ?, ?, ?, ?, 0)
    """, (
        filename,
        metadata.get('circular_no'),
        metadata.get('title'),
        metadata.get('date'),
        ','.join(metadata.get('links', []))
    ))
    conn.commit()
    conn.close()

def is_pdf_already_processed(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM pdf_metadata WHERE filename = ?", (filename,))
    result = c.fetchone()[0]
    conn.close()
    return result > 0

def is_pdf_already_embedded(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT embedded FROM pdf_metadata WHERE filename = ?", (filename,))
    row = c.fetchone()
    conn.close()
    return row is not None and row[0] == 1

def mark_as_embedded(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE pdf_metadata SET embedded = 1 WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()
