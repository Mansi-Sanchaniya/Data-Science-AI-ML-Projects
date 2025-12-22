from fastapi import HTTPException
import snowflake.connector
from hashlib import sha256
from dotenv import load_dotenv
import os

load_dotenv()

# ==== Snowflake config ====
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")


def get_connection():
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    return conn

def hash_password(password: str) -> str:
    return sha256(password.encode()).hexdigest()

def register_user(email: str, password: str) -> str:
    conn = get_connection()
    cursor = conn.cursor()

    # Check domain
    if not email.lower().endswith("@rathi.com"):
        raise HTTPException(status_code=400, detail="Use official email ending with @rathi.com.")

    # Check user exists in users table
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user_row = cursor.fetchone()
    if not user_row:
        raise HTTPException(status_code=403, detail="Ask IT team to grant access before registering.")

    # Check already registered
    cursor.execute("SELECT * FROM registration WHERE email = %s", (email,))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="User already registered.")

    # Unpack fields from users table
    (
        _id,
        eid,
        name,
        email,
        role,
        department,
        sub_department,
        compliance_access,
        hr_access,
        research_access,
        access_read,
        access_write,
        access_update,
        access_delete,
        access_view,
        created_on
    ) = user_row

    # Prepare access_modules
    access_modules = []
    if compliance_access and compliance_access.lower() == "yes":
        access_modules.append("compliance")
    if hr_access and hr_access.lower() == "yes":
        access_modules.append("hr")
    if research_access and research_access.lower() == "yes":
        access_modules.append("research")
    access_str = ",".join(access_modules)

    # Insert into registration table
    cursor.execute("""
        INSERT INTO registration (
            email, username, password, role,
            access_modules, access_read, access_write,
            access_update, access_view,
            department, sub_department, access_delete
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        email,
        name,
        hash_password(password),
        role,
        access_str,
        access_read,
        access_write,
        access_update,
        access_view,
        department,
        sub_department,
        access_delete
    ))

    conn.commit()
    cursor.close()
    conn.close()
    return "User registered successfully."


def login_user(email: str, password: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    # Fetch hashed password and username from registration table
    cursor.execute("SELECT username, password FROM registration WHERE email = %s", (email,))
    reg_row = cursor.fetchone()
    if not reg_row:
        raise HTTPException(status_code=404, detail="User not found.")
    username, hashed_pw = reg_row

    if hashed_pw != hash_password(password):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    # Fetch access flags from users table
    cursor.execute("""
        SELECT compliance_access, hr_access, research_access 
        FROM users WHERE email = %s
    """, (email,))
    access_row = cursor.fetchone()
    if not access_row:
        raise HTTPException(status_code=403, detail="Access details missing for user.")

    compliance_access, hr_access, research_access = access_row

    access_modules = []
    if compliance_access and compliance_access.lower() == "yes":
        access_modules.append("compliance")
    if hr_access and hr_access.lower() == "yes":
        access_modules.append("hr")
    if research_access and research_access.lower() == "yes":
        access_modules.append("research")

    cursor.close()
    conn.close()

    return {
        "name": username,
        "email": email,
        "access_modules": access_modules
    }
