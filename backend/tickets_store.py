"""Postgres persistence for escalation tickets.

The only module that talks to the database. Everything degrades to a no-op
when no connection string is configured, so the app and the chat run
unchanged with no database at all (local dev, or before the integration is
added). Nothing here ever raises into a request: errors are caught and logged
by exception type only (a message or DSN could leak a secret).
"""
import os

import psycopg
from psycopg.types.json import Json

_CREATE = """
CREATE TABLE IF NOT EXISTS tickets (
    id           TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL,
    question     TEXT NOT NULL,
    category     TEXT NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL,
    reason       TEXT NOT NULL,
    conversation JSONB NOT NULL,
    username     TEXT,
    signed_in_at TIMESTAMPTZ
)
"""

# CREATE TABLE IF NOT EXISTS won't touch a table that already exists, so the
# who-raised-it columns are added separately. Both are nullable: tickets
# predating the login flow have no user, and that is not an error.
_ADD_COLUMNS = (
    "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS username TEXT",
    "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS signed_in_at TIMESTAMPTZ",
)

_COLUMNS = ["id", "created_at", "question", "category", "confidence", "reason",
            "conversation", "username", "signed_in_at"]

_TIMESTAMPS = ("created_at", "signed_in_at")


def _dsn() -> str | None:
    # Vercel's Neon integration injects POSTGRES_URL; a plain Neon setup uses
    # DATABASE_URL. Prefer DATABASE_URL when both are present.
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


def db_configured() -> bool:
    return bool(_dsn())


def init_db() -> None:
    if not db_configured():
        return
    try:
        with psycopg.connect(_dsn()) as conn:
            conn.execute(_CREATE)
            for statement in _ADD_COLUMNS:
                conn.execute(statement)
    except Exception as exc:
        print(f"init_db failed: {type(exc).__name__}")


def save_ticket(ticket: dict) -> bool:
    if not db_configured():
        return False
    try:
        with psycopg.connect(_dsn()) as conn:
            conn.execute(
                "INSERT INTO tickets (id, created_at, question, category, confidence,"
                " reason, conversation, username, signed_in_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (id) DO NOTHING",
                (
                    ticket["id"],
                    ticket["created_at"],
                    ticket["question"],
                    ticket["category"],
                    ticket["confidence"],
                    ticket["reason"],
                    Json(ticket["conversation"]),
                    # absent for anything raised before the login flow existed
                    ticket.get("username"),
                    ticket.get("signed_in_at"),
                ),
            )
        return True
    except Exception as exc:
        print(f"save_ticket failed: {type(exc).__name__}")
        return False


def list_tickets(limit: int = 100) -> list[dict]:
    if not db_configured():
        return []
    try:
        with psycopg.connect(_dsn()) as conn:
            rows = conn.execute(
                "SELECT id, created_at, question, category, confidence, reason,"
                " conversation, username, signed_in_at"
                " FROM tickets ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
    except Exception as exc:
        print(f"list_tickets failed: {type(exc).__name__}")
        return []
    out = []
    for row in rows:
        d = dict(zip(_COLUMNS, row))
        for field in _TIMESTAMPS:
            value = d[field]
            d[field] = value.isoformat() if hasattr(value, "isoformat") else value
        out.append(d)
    return out
