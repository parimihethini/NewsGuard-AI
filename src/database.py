"""
src/database.py
===============
Phase 5 — Lightweight SQLite user database for NewsGuard AI.

Design decisions:
  - SQLite is the simplest reliable storage for a student project with no external DB.
  - The database file (users.db) is created automatically on first run.
  - Passwords are NEVER stored in plain text (see auth.py for hashing).
  - No hardcoded credentials in this file.
  - The DB path is configurable via environment variable NEWSGUARD_DB_PATH,
    falling back to <project_root>/data/users.db.
  - This module is ONLY responsible for CRUD on the users table.
    All hashing logic lives in auth.py.

Schema:
    users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE NOT NULL,
        email     TEXT UNIQUE NOT NULL,
        name      TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
"""
import os
import sqlite3
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# DB path configuration  (never hardcoded to a specific user path)
# ---------------------------------------------------------------------------
def _get_db_path() -> str:
    env_path = os.environ.get("NEWSGUARD_DB_PATH", "")
    if env_path:
        return env_path
    # Default: project_root/data/users.db
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "users.db")


DB_PATH = _get_db_path()


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------
CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    UNIQUE NOT NULL,
    email         TEXT    UNIQUE NOT NULL,
    name          TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
"""


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory for dict-like access."""
    path = db_path if db_path is not None else DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = None) -> None:
    """Create tables if they do not exist. Safe to call multiple times."""
    conn = get_connection(db_path)
    try:
        conn.execute(CREATE_USERS_TABLE)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(
    username: str,
    email: str,
    name: str,
    password_hash: str,
    db_path: str = None,
) -> bool:
    """
    Insert a new user.  password_hash must already be a bcrypt hash string.
    Returns True on success, False if username or email already exists.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO users (username, email, name, password_hash, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                username.strip().lower(),
                email.strip().lower(),
                name.strip(),
                password_hash,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False   # duplicate username or email
    finally:
        conn.close()


def get_user_by_username(username: str, db_path: str = None):
    """Return the user row dict or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(email: str, db_path: str = None):
    """Return the user row dict or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def username_exists(username: str, db_path: str = None) -> bool:
    return get_user_by_username(username, db_path) is not None


def email_exists(email: str, db_path: str = None) -> bool:
    return get_user_by_email(email, db_path) is not None


def user_count(db_path: str = None) -> int:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) as n FROM users").fetchone()
        return row["n"]
    finally:
        conn.close()

