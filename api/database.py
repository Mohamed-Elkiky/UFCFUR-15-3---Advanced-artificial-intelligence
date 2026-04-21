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
            override_value  TEXT,
            metadata        TEXT
        )"""
    )
    # Migration: add override_value if an older DB exists without the column
    try:
        conn.execute("ALTER TABLE interactions ADD COLUMN override_value TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
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
    override_value: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO interactions
           (timestamp, user_id, endpoint, input_summary, prediction,
            confidence, grade, was_overridden, override_value, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            user_id,
            endpoint,
            input_summary,
            prediction,
            confidence,
            grade,
            int(was_overridden),
            override_value,
            json.dumps(metadata) if metadata else None,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def mark_overridden(interaction_id: int, override_value: str) -> bool:
    """Mark an interaction as overridden and log the correction (AA-51).

    Parameters
    ----------
    interaction_id : int
        The ID of the interaction to mark.
    override_value : str
        The corrected prediction value provided by the user.

    Returns
    -------
    bool
        True if a row was updated, False if the interaction was not found.
    """
    conn = _get_conn()
    cur = conn.execute(
        """UPDATE interactions
           SET was_overridden = 1, override_value = ?
           WHERE id = ?""",
        (override_value, interaction_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


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