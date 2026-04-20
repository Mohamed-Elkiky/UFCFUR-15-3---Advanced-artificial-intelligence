"""Lightweight SQLite store for prediction interactions."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).resolve().parents[1] / "logs" / "interactions.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            user_id         TEXT,
            endpoint        TEXT NOT NULL,
            input_summary   TEXT,
            prediction      TEXT,
            confidence      REAL,
            grade           TEXT,
            was_overridden  INTEGER DEFAULT 0,
            metadata        TEXT
        )"""
    )
    conn.commit()
    return conn


def log_interaction(
    endpoint: str,
    prediction: str,
    confidence: Optional[float] = None,
    grade: Optional[str] = None,
    user_id: Optional[str] = None,
    input_summary: Optional[str] = None,
    was_overridden: bool = False,
    metadata: Optional[Dict] = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO interactions
           (timestamp, user_id, endpoint, input_summary, prediction,
            confidence, grade, was_overridden, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            user_id,
            endpoint,
            input_summary,
            prediction,
            confidence,
            grade,
            int(was_overridden),
            json.dumps(metadata) if metadata else None,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_interactions() -> List[Dict]:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM interactions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_by_user(user_id: str) -> List[Dict]:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM interactions WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]