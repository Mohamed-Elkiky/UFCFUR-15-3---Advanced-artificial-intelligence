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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint    TEXT NOT NULL,
            input_hash  TEXT,
            prediction  TEXT,
            confidence  REAL,
            grade       TEXT,
            metadata    TEXT,
            created_at  TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def log_interaction(
    endpoint: str,
    prediction: str,
    confidence: Optional[float] = None,
    grade: Optional[str] = None,
    input_hash: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO interactions
           (endpoint, input_hash, prediction, confidence, grade, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            endpoint,
            input_hash,
            prediction,
            confidence,
            grade,
            json.dumps(metadata) if metadata else None,
            datetime.now().isoformat(),
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