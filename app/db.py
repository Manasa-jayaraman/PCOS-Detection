import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
from pathlib import Path

# Use users.db as requested
DB_PATH = Path(__file__).parent / "users.db"

try:
    import bcrypt  # type: ignore
    _HAS_BCRYPT = True
except Exception:
    bcrypt = None  # type: ignore
    _HAS_BCRYPT = False


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            age INTEGER,
            dob TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            p_pcos REAL,
            p_no_pcos REAL,
            confidence REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            weight REAL,
            sleep REAL,
            exercise_minutes INTEGER,
            stress TEXT,
            mood TEXT,
            water INTEGER,
            symptoms TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    # Create a compatibility view named progress_log as requested
    try:
        cur.execute("CREATE VIEW IF NOT EXISTS progress_log AS SELECT * FROM daily_logs;")
    except Exception:
        pass
    
    # Safe migrations: add columns if missing
    try:
        cur.execute("PRAGMA table_info(users);")
        cols = {row[1] for row in cur.fetchall()}
        if 'age' not in cols:
            cur.execute("ALTER TABLE users ADD COLUMN age INTEGER;")
        if 'dob' not in cols:
            cur.execute("ALTER TABLE users ADD COLUMN dob TEXT;")
    except Exception:
        pass
    
    # Migrate legacy DB if present (pcos_app.db -> users.db)
    try:
        legacy = Path(__file__).parent / "pcos_app.db"
        if legacy.exists():
            # Only migrate if current users table is empty
            cur.execute("SELECT COUNT(1) FROM users;")
            count = cur.fetchone()[0]
            if count == 0:
                lconn = sqlite3.connect(legacy)
                lcur = lconn.cursor()
                # Check legacy users table shape
                lcur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
                if lcur.fetchone():
                    try:
                        lcur.execute("SELECT email, name, password_hash, salt, created_at FROM users;")
                        rows = lcur.fetchall()
                        for r in rows:
                            cur.execute(
                                "INSERT OR IGNORE INTO users (email, name, password_hash, salt, age, dob, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (r[0], r[1], r[2], r[3], None, None, r[4])
                            )
                        conn.commit()
                    except Exception:
                        pass
                lconn.close()
    except Exception:
        pass
    conn.commit()
    conn.close()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash password using bcrypt if available, otherwise PBKDF2-SHA256.
    Returns (password_hash, salt). For bcrypt, password_hash is the UTF-8 string like '$2b$12$...'.
    """
    if _HAS_BCRYPT:
        # bcrypt embeds salt into the hash; keeping salt string is optional.
        salt_bytes = bcrypt.gensalt(rounds=12)
        pw_hash_bytes = bcrypt.hashpw(password.encode("utf-8"), salt_bytes)
        return pw_hash_bytes.decode("utf-8"), salt_bytes.decode("utf-8")
    # Fallback to PBKDF2
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000)
    return h.hex(), salt


def create_user(email: str, name: str, password: str, age: int | None = None, dob: str | None = None) -> tuple[bool, str]:
    try:
        pwd_hash, salt = hash_password(password)
        conn = get_conn()
        conn.execute(
            "INSERT INTO users (email, name, password_hash, salt, age, dob, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (email.lower().strip(), name.strip(), pwd_hash, salt, age, dob, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


essential_user_fields = "id, email, name, created_at"

def _try_bcrypt_verify(password: str, stored_hash_text: str, salt_text: str | None) -> bool:
    try:
        # Case A: stored as proper bcrypt text starting with $2
        candidate = stored_hash_text
        if candidate.startswith("$2"):
            return bcrypt.checkpw(password.encode("utf-8"), candidate.encode("utf-8"))
        # Case B: legacy hex-encoded bcrypt string -> decode to text
        try:
            candidate_bytes = bytes.fromhex(stored_hash_text)
            candidate_text = candidate_bytes.decode("utf-8")
            if candidate_text.startswith("$2"):
                return bcrypt.checkpw(password.encode("utf-8"), candidate_text.encode("utf-8"))
        except Exception:
            pass
        # Case C: if we incorrectly stored salt as hex/text, recreate and compare
        if salt_text:
            try:
                # salt_text might be hex or already text like $2b$...
                if salt_text.startswith("$2"):
                    salt_bytes = salt_text.encode("utf-8")
                else:
                    salt_bytes = bytes.fromhex(salt_text)
                test_hash = bcrypt.hashpw(password.encode("utf-8"), salt_bytes)
                return test_hash.decode("utf-8") == stored_hash_text or test_hash.hex() == stored_hash_text
            except Exception:
                pass
    except Exception:
        return False
    return False


def verify_user(email: str, password: str) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, name, password_hash, salt, created_at FROM users WHERE email=?", (email.lower().strip(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if _HAS_BCRYPT:
        ok = _try_bcrypt_verify(password, row[3], row[4])
        if ok:
            return {"id": row[0], "email": row[1], "name": row[2], "created_at": row[5]}
        # Fall back to PBKDF2 for legacy accounts
        try:
            calc_hash, _ = hash_password(password, row[4])
            if calc_hash == row[3]:
                return {"id": row[0], "email": row[1], "name": row[2], "created_at": row[5]}
        except Exception:
            pass
        return None
    # PBKDF2 fallback
    calc_hash, _ = hash_password(password, row[4])
    if calc_hash == row[3]:
        return {"id": row[0], "email": row[1], "name": row[2], "created_at": row[5]}
    return None


def save_prediction(user_id: int, input_json: str, prediction: int, p_pcos: float | None, p_no_pcos: float | None, confidence: float | None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO predictions (user_id, input_json, prediction, p_pcos, p_no_pcos, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, input_json, prediction, p_pcos, p_no_pcos, confidence, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_predictions(user_id: int, limit: int = 100):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, input_json, prediction, p_pcos, p_no_pcos, confidence, created_at FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_daily_log(user_id: int, date: str, weight: float, sleep: float, exercise_minutes: int, stress: str, mood: str, water: int, symptoms: str):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO daily_logs (user_id, date, weight, sleep, exercise_minutes, stress, mood, water, symptoms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, date, weight, sleep, exercise_minutes, stress, mood, water, symptoms, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_daily_logs(user_id: int, limit: int = 365):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, date, weight, sleep, exercise_minutes, stress, mood, water, symptoms, created_at FROM daily_logs WHERE user_id=? ORDER BY date DESC, id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# Ensure database exists when module is imported
if not DB_PATH.exists():
    init_db()
else:
    # Attempt to migrate/ensure tables
    init_db()
