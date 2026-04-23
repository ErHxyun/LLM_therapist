"""SQLite persistence helpers shared by the CaiTI text and voice entrypoints."""
import json
import os
import sqlite3

from src.utils.log_util import get_logger

logger = get_logger("DBManager")

_SQLITE_TIMEOUT = 5.0


class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=_SQLITE_TIMEOUT)

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                turn_index INTEGER,
                speaker TEXT,
                text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                meta_data TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                summary_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER,
                key TEXT,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, key),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                turn_index INTEGER,
                rating TEXT,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS safety_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                turn_index INTEGER,
                flag_type TEXT,
                raw_text TEXT,
                severity INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )"""
        )
        conn.commit()
        conn.close()
        logger.info("Database initialized at %s", self.db_path)

    def get_user_id(self, subject_id: str) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE subject_id=?", (subject_id,))
        row = cur.fetchone()
        if row:
            user_id = row[0]
        else:
            cur.execute("INSERT INTO users (subject_id) VALUES (?)", (subject_id,))
            conn.commit()
            user_id = cur.lastrowid
        conn.close()
        return int(user_id)

    def create_session(self, user_id: int) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
        conn.commit()
        session_id = cur.lastrowid
        conn.close()
        return int(session_id)

    def add_turn(self, session_id: int, turn_index: int, speaker: str, text: str, meta_data=None):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO turns (session_id, turn_index, speaker, text, meta_data) VALUES (?, ?, ?, ?, ?)",
            (session_id, turn_index, speaker, text, json.dumps(meta_data) if meta_data else None),
        )
        conn.commit()
        conn.close()

    def get_session_history(self, session_id: int):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT speaker, text, meta_data FROM turns WHERE session_id=? ORDER BY turn_index, id",
            (session_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "speaker": speaker,
                "text": text,
                "meta_data": json.loads(meta_data) if meta_data else None,
            }
            for speaker, text, meta_data in rows
        ]

    def add_summary(self, session_id: int, summary_text: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO summaries (session_id, summary_text) VALUES (?, ?)", (session_id, summary_text))
        conn.commit()
        conn.close()

    def get_user_context_string(self, user_id: int, limit: int = 3) -> str:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM user_preferences WHERE user_id=?", (user_id,))
        prefs = cur.fetchall()
        cur.execute(
            """SELECT summaries.summary_text, summaries.created_at
               FROM summaries
               JOIN sessions ON summaries.session_id = sessions.id
               WHERE sessions.user_id=?
               ORDER BY sessions.start_time DESC
               LIMIT ?""",
            (user_id, limit),
        )
        summaries = cur.fetchall()
        conn.close()

        chunks = []
        if prefs:
            chunks.append("[User Preferences]")
            chunks.extend(f"- {key}: {value}" for key, value in prefs)
        if summaries:
            chunks.append("[Recent Session Summaries]")
            chunks.extend(f"- {created_at}: {summary}" for summary, created_at in summaries)
        return "\n".join(chunks).strip()
