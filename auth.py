"""
רישום והתחברות של מועדונים (multi-tenant).

כל מועדון:
  - נרשם עם שם מועדון, שם משתמש וסיסמה. הסיסמה נשמרת מוצפנת (PBKDF2-HMAC-SHA256
    עם salt ייחודי) בקובץ data/clubs.json - הסיסמה עצמה לא נשמרת.
  - מקבל קובץ נתונים משלו: data/clubs/<club_id>/players_raw.csv
    (דאטה סינתטית שנוצרת אוטומטית בעת הרישום).

הערה: מנגנון קליל לשימוש מקומי על נתונים סינתטיים - לא תחליף למערכת
אימות בענן לנתונים אמיתיים.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time

import pandas as pd

from generate_data import generate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLUBS_DIR = os.path.join(DATA_DIR, "clubs")
CLUBS_FILE = os.path.join(DATA_DIR, "clubs.json")
PBKDF2_ROUNDS = 200_000
DEFAULT_N = 650


# ----------------------------- אחסון -----------------------------
def load_clubs() -> dict:
    if not os.path.exists(CLUBS_FILE):
        return {}
    with open(CLUBS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_clubs(clubs: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CLUBS_FILE, "w", encoding="utf-8") as f:
        json.dump(clubs, f, ensure_ascii=False, indent=2)


# ----------------------------- עזרי סיסמה / מזהה -----------------------------
def _hash_pw(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ROUNDS).hex()


def _slugify(name: str) -> str:
    """מזהה תיקייה בטוח (ASCII). לשם בעברית - נופלים למזהה מבוסס-hash."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not s:
        s = "club-" + hashlib.sha1(name.encode()).hexdigest()[:8]
    return s


def _unique_club_id(base: str, clubs: dict) -> str:
    cid, i = base, 2
    while cid in clubs:
        cid, i = f"{base}-{i}", i + 1
    return cid


# ----------------------------- קובץ נתונים למועדון -----------------------------
def club_data_path(club_id: str) -> str:
    return os.path.join(CLUBS_DIR, club_id, "players_raw.csv")


def ensure_club_dataset(club_id: str, n: int | None = None, reseed: bool = False) -> str:
    """יוצר את קובץ הנתונים של המועדון אם חסר. reseed=True מגריל דאטה חדשה."""
    path = club_data_path(club_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not reseed and os.path.exists(path):
        return path
    if n is None:
        n = len(pd.read_csv(path)) if os.path.exists(path) else DEFAULT_N
    extra = secrets.token_hex(4) if reseed else ""
    seed = int(hashlib.sha256((club_id + extra).encode()).hexdigest(), 16) % (2**32)
    generate(n=n, seed=seed).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ----------------------------- API ציבורי -----------------------------
def register_club(name: str, username: str, password: str, n: int = DEFAULT_N) -> dict:
    name = name.strip()
    username = username.strip().lower()
    if not name or not username or not password:
        raise ValueError("יש למלא שם מועדון, שם משתמש וסיסמה.")
    if len(password) < 6:
        raise ValueError("הסיסמה חייבת להיות באורך 6 תווים לפחות.")

    clubs = load_clubs()
    if any(c["username"] == username for c in clubs.values()):
        raise ValueError("שם המשתמש כבר תפוס, בחרו אחר.")

    club_id = _unique_club_id(_slugify(name), clubs)
    salt = secrets.token_hex(16)
    clubs[club_id] = {
        "club_id": club_id,
        "name": name,
        "username": username,
        "salt": salt,
        "password_hash": _hash_pw(password, salt),
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    ensure_club_dataset(club_id, n=n, reseed=True)
    save_clubs(clubs)
    return clubs[club_id]


def authenticate(username: str, password: str) -> dict | None:
    username = username.strip().lower()
    for c in load_clubs().values():
        if c["username"] == username and secrets.compare_digest(
            c["password_hash"], _hash_pw(password, c["salt"])
        ):
            return c
    return None


# ----------------------------- מנהל מערכת (admin) -----------------------------
def is_admin(club: dict) -> bool:
    return club.get("role") == "admin"


def create_admin(username: str = "admin", password: str = "12345") -> dict:
    """יוצר/מעדכן משתמש מנהל בעל גישה לכל המועדונים. אין לו קובץ נתונים משלו."""
    username = username.strip().lower()
    clubs = load_clubs()
    for cid in [c for c, v in clubs.items() if v["username"] == username]:
        del clubs[cid]
    salt = secrets.token_hex(16)
    clubs["admin"] = {
        "club_id": "admin",
        "name": "מנהל מערכת",
        "username": username,
        "salt": salt,
        "password_hash": _hash_pw(password, salt),
        "role": "admin",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_clubs(clubs)
    return clubs["admin"]


def create_demo(username: str = "demo", password: str = "12345", n: int = DEFAULT_N) -> dict:
    """מועדון הדגמה - חוויית משתמש קצה מלאה, בלי שום יכולת ניהול.
    מאפשר למנהל להתחבר ולראות בדיוק מה שמועדון רגיל רואה."""
    username = username.strip().lower()
    clubs = load_clubs()
    for cid in [c for c, v in clubs.items() if v["username"] == username]:
        del clubs[cid]
    salt = secrets.token_hex(16)
    clubs["demo"] = {
        "club_id": "demo",
        "name": "מועדון הדגמה (DEMO)",
        "username": username,
        "salt": salt,
        "password_hash": _hash_pw(password, salt),
        "role": "demo",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    ensure_club_dataset("demo", n=n, reseed=True)
    save_clubs(clubs)
    return clubs["demo"]


def list_member_clubs() -> list[dict]:
    """מועדונים אמיתיים בלבד (ללא המנהל וללא מועדון ההדגמה)."""
    return [c for c in load_clubs().values() if c.get("role") not in ("admin", "demo")]


def club_summaries() -> pd.DataFrame:
    """טבלת סיכום למנהל: שורה למועדון עם מדדים בסיסיים."""
    rows = []
    for c in list_member_clubs():
        path = club_data_path(c["club_id"])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        rows.append({
            "מועדון": c["name"],
            "שם משתמש": c["username"],
            "נוצר": c["created_at"],
            "שחקנים": len(df),
            "אחוז הצלחה ממוצע": round(df["success_rate"].mean(), 1),
            "גיל ממוצע": round(df["age"].mean(), 1),
            "% שמאליים": round((df["dominant_hand"] == "left").mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def load_all_players() -> pd.DataFrame:
    """איחוד כל קובצי הנתונים של כל המועדונים, עם עמודת 'club'."""
    frames = []
    for c in list_member_clubs():
        path = club_data_path(c["club_id"])
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.insert(0, "club", c["name"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
