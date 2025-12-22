import pyodbc
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

conn_str = os.getenv("AZURE_SQL_CONN_STR")


def get_db_conn():
    return pyodbc.connect(conn_str)


def store_session(session_id: str, email: str, timeout_minutes: int = 30):
    created_on = datetime.utcnow()
    expiry_time = created_on + timedelta(minutes=timeout_minutes)

    with get_db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            MERGE INTO sessions WITH (HOLDLOCK) AS target
            USING (SELECT ? AS session_id) AS source
            ON target.session_id = source.session_id
            WHEN MATCHED THEN
                UPDATE SET expiry_time = ?, email = ?
            WHEN NOT MATCHED THEN
                INSERT (session_id, email, created_on, expiry_time)
                VALUES (?, ?, ?, ?);
        """, session_id, expiry_time, email, session_id, email, created_on, expiry_time)
        conn.commit()



def is_session_valid(session_id: str, timeout_minutes: int = 30) -> bool:
    with get_db_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT expiry_time FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return False

        expiry = row[0]
        if datetime.utcnow() > expiry:
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return False

        # Extend session
        new_expiry = datetime.utcnow() + timedelta(minutes=timeout_minutes)
        cursor.execute("UPDATE sessions SET expiry_time = ? WHERE session_id = ?", (new_expiry, session_id))
        conn.commit()
        return True
