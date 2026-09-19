from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DATABASE_PATH


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, problem_statement TEXT NOT NULL,
              status TEXT NOT NULL, spec_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
              role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS datasets (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, filename TEXT NOT NULL,
              stored_path TEXT NOT NULL, profile_json TEXT NOT NULL, training_ready INTEGER NOT NULL,
              blockers_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
              status TEXT NOT NULL, result_json TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflow_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
              step TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )


def dump(value: object) -> str:
    return json.dumps(value, default=str)


def load(value: str | None, fallback: object = None):
    return json.loads(value) if value else fallback


def fetch_one(query: str, params: tuple = ()):
    with connection() as conn:
        return conn.execute(query, params).fetchone()


def fetch_all(query: str, params: tuple = ()):
    with connection() as conn:
        return conn.execute(query, params).fetchall()


def record_workflow_event(project_id: str, step: str, detail: str) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO workflow_events (project_id, step, detail, created_at) VALUES (?, ?, ?, ?)",
            (project_id, step, detail, now()),
        )
