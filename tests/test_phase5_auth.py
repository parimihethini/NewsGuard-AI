"""
Phase 5 Test Suite -- Authentication & Database
Tests registration, login, validation, duplicate checks, and password hashing.
"""
import os
import sys
import tempfile
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import auth
import database

print("=" * 60)
print("PHASE 5: AUTHENTICATION AND DATABASE TESTS")
print("=" * 60)

# Create a temporary DB file for testing
temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
os.close(temp_db_fd)

# Override the database path for tests
database.DB_PATH = temp_db_path

all_passed = True

def check(name, condition, detail=""):
    global all_passed
    if condition:
        print(f"  [PASS]  {name}")
    else:
        print(f"  [FAIL]  {name}  {detail}")
        all_passed = False

try:
    # -----------------------------------------------------------------
    # A. Initialization
    # -----------------------------------------------------------------
    database.init_db()
    check("A1 Database initialized", os.path.exists(temp_db_path))

    # -----------------------------------------------------------------
    # B. Registration Validation
    # -----------------------------------------------------------------
    ok, msg = auth.validate_registration("Test User", "testuser", "test@test.com", "pass1234", "pass1234")
    check("B1 Valid registration data accepted", ok, msg)

    ok, msg = auth.validate_registration("", "testuser", "test@test.com", "pass1234", "pass1234")
    check("B2 Empty name rejected", not ok)

    ok, msg = auth.validate_registration("Test User", "us", "test@test.com", "pass1234", "pass1234")
    check("B3 Short username rejected", not ok)

    ok, msg = auth.validate_registration("Test User", "testuser", "invalidemail", "pass1234", "pass1234")
    check("B4 Invalid email rejected", not ok)

    ok, msg = auth.validate_registration("Test User", "testuser", "test@test.com", "short", "short")
    check("B5 Short password rejected", not ok)

    ok, msg = auth.validate_registration("Test User", "testuser", "test@test.com", "pass1234", "pass5678")
    check("B6 Password mismatch rejected", not ok)

    # -----------------------------------------------------------------
    # C. Registration & Hashing
    # -----------------------------------------------------------------
    ok, msg = auth.register_user("Alice Smith", "alice", "alice@example.com", "securepassword", "securepassword")
    check("C1 User registration succeeds", ok, msg)

    # Verify hash is stored, not plain text
    user = database.get_user_by_username("alice")
    check("C2 Password stored as hash", user["password_hash"] != "securepassword" and user["password_hash"].startswith("$2b$"))

    # Duplicates
    ok, msg = auth.register_user("Alice Clone", "alice", "alice2@example.com", "securepassword", "securepassword")
    check("C3 Duplicate username rejected", not ok)

    ok, msg = auth.register_user("Alice Clone", "alice2", "alice@example.com", "securepassword", "securepassword")
    check("C4 Duplicate email rejected", not ok)

    # -----------------------------------------------------------------
    # D. Login
    # -----------------------------------------------------------------
    # By username
    ok, msg, u = auth.login_user("alice", "securepassword")
    check("D1 Login by username succeeds", ok)
    if u:
        check("D2 User object safe (no hash)", "password_hash" not in u and "password" not in u)

    # By email
    ok, msg, u = auth.login_user("alice@example.com", "securepassword")
    check("D3 Login by email succeeds", ok)

    # Invalid cases
    ok, msg, u = auth.login_user("alice", "wrongpassword")
    check("D4 Wrong password rejected", not ok)

    ok, msg, u = auth.login_user("bob", "securepassword")
    check("D5 Unknown user rejected", not ok)

    ok, msg, u = auth.login_user("", "securepassword")
    check("D6 Empty username rejected", not ok)

finally:
    # Cleanup
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

print()
print("=" * 60)
print("OVERALL RESULT:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
print("=" * 60)
if not all_passed:
    sys.exit(1)
