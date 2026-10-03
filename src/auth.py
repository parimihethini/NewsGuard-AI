"""
src/auth.py
===========
Phase 5 — Authentication logic for NewsGuard AI.

Responsibilities:
  - Password hashing with bcrypt (industry-standard, adaptive work factor).
  - Login verification (compare candidate password against stored hash).
  - Registration validation (field checks, duplicate detection).
  - NO hardcoded credentials anywhere in this file.
  - NO plain-text passwords stored at any point.

bcrypt is used because:
  1. It is a recognised, peer-reviewed password hashing algorithm.
  2. It includes a built-in random salt (no manual salt management).
  3. The work factor (cost) can be tuned upward for future security hardening.
  4. It is available as a pure Python package with C backend.

This module is STATELESS.  All persistence is delegated to database.py.
"""
import re
import bcrypt

from database import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    init_db,
    username_exists,
    email_exists,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BCRYPT_ROUNDS = 12          # work factor — 12 is the recommended default
MIN_PASSWORD_LEN = 8        # minimum password length requirement
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Return a bcrypt hash of the plain-text password.

    The salt is generated automatically by bcrypt.hashpw.
    The returned string is safe to store in the database.
    """
    pwd_bytes = plain_password.encode("utf-8")
    hashed = bcrypt.hashpw(pwd_bytes, bcrypt.gensalt(rounds=BCRYPT_ROUNDS))
    return hashed.decode("utf-8")


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Constant-time comparison of plain_password against stored bcrypt hash.

    Returns True if the password matches, False otherwise.
    Catches all exceptions to prevent oracle attacks from error messages.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            stored_hash.encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def validate_registration(
    name: str,
    username: str,
    email: str,
    password: str,
    confirm_password: str,
) -> tuple[bool, str]:
    """Validate registration fields.

    Returns (ok: bool, error_message: str).
    ok=True means all validations passed and registration can proceed.
    """
    # Empty-field checks
    if not name or not name.strip():
        return False, "Full name is required."
    if not username or not username.strip():
        return False, "Username is required."
    if not email or not email.strip():
        return False, "Email address is required."
    if not password:
        return False, "Password is required."
    if not confirm_password:
        return False, "Please confirm your password."

    # Username format: alphanumeric + underscores, 3-30 chars
    username = username.strip()
    if not re.match(r"^[a-zA-Z0-9_]{3,30}$", username):
        return False, (
            "Username must be 3-30 characters and contain only letters, "
            "digits, or underscores."
        )

    # Email format
    email = email.strip()
    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address."

    # Password length
    if len(password) < MIN_PASSWORD_LEN:
        return False, "Password must be at least %d characters." % MIN_PASSWORD_LEN

    # Password confirmation match
    if password != confirm_password:
        return False, "Passwords do not match."

    # Duplicate checks
    if username_exists(username):
        return False, "That username is already taken. Please choose another."
    if email_exists(email):
        return False, "That email address is already registered."

    return True, ""


def register_user(
    name: str,
    username: str,
    email: str,
    password: str,
    confirm_password: str,
) -> tuple[bool, str]:
    """Full registration pipeline: validate -> hash -> persist.

    Returns (ok: bool, message: str).
    """
    # Ensure DB exists
    init_db()

    ok, error = validate_registration(name, username, email, password, confirm_password)
    if not ok:
        return False, error

    password_hash = hash_password(password)
    created = create_user(
        username=username.strip(),
        email=email.strip(),
        name=name.strip(),
        password_hash=password_hash,
    )
    if not created:
        return False, "Registration failed due to a database conflict. Please try again."

    return True, "Account created successfully. You may now log in."


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login_user(username_or_email: str, password: str) -> tuple[bool, str, dict | None]:
    """Authenticate a user by username or email.

    Returns (ok: bool, message: str, user_dict or None).
    user_dict contains: id, username, email, name, created_at.
    The password_hash field is STRIPPED from the returned dict.
    """
    init_db()

    if not username_or_email or not username_or_email.strip():
        return False, "Username or email is required.", None
    if not password:
        return False, "Password is required.", None

    identifier = username_or_email.strip().lower()

    # Try username first, then email
    user = get_user_by_username(identifier)
    if user is None:
        user = get_user_by_email(identifier)

    if user is None:
        # Generic message to prevent user enumeration
        return False, "Invalid username/email or password.", None

    if not verify_password(password, user["password_hash"]):
        return False, "Invalid username/email or password.", None

    # Return a safe copy without the hash
    safe_user = {
        "id":         user["id"],
        "username":   user["username"],
        "email":      user["email"],
        "name":       user["name"],
        "created_at": user["created_at"],
    }
    return True, "Login successful.", safe_user
