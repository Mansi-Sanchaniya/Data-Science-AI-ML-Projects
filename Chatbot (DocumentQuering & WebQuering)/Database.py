import os
from dotenv import load_dotenv
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise RuntimeError(" AZURE_STORAGE_CONNECTION_STRING not set in .env")

os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_STORAGE_CONNECTION_STRING

import pyodbc

def get_connection():
    server = os.getenv("SQL_SERVER_HOST")
    database = os.getenv("SQL_DATABASE")
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    driver = os.getenv("SQL_DRIVER")

    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=30;"
    )
    print(" Connecting to DB...")
    conn = pyodbc.connect(conn_str)
    print(" Connected to DB")

    return conn
