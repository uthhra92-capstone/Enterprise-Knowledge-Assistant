"""Persistent conversation history in a local SQLite file.

Complements the in-session ``ConversationMemory`` (which only holds the recent
window used for query-condensing and prompting). Every turn is also written here
so past chats survive a browser refresh or an app restart, and can be reopened
from the sidebar.

Storage: ``storage/conversations.db`` — stdlib ``sqlite3``, no extra dependency.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

DB_FILENAME = "conversations.db"
_DEFAULT_TITLE = "New chat"
_SHARED_OWNER = "shared"  # used for local single-user runs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ConversationSummary:
    id: str
    title: str
    updated_at: str
    message_count: int


@dataclass
class StoredMessage:
    role: str  # "user" | "assistant"
    content: str
    standalone_question: str | None = None
    sources: list[str] | None = None
    created_at: str = ""


class ConversationStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or (settings.storage_dir / DB_FILENAME))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New chat',
                    owner       TEXT NOT NULL DEFAULT 'shared',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id      TEXT NOT NULL
                                         REFERENCES conversations(id) ON DELETE CASCADE,
                    role                 TEXT NOT NULL,
                    content              TEXT NOT NULL,
                    standalone_question  TEXT,
                    sources              TEXT,          -- JSON array of file names
                    created_at           TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, id);
                """
            )
            # migrate older databases that predate the `owner` column, THEN index it
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
            if "owner" not in columns:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT 'shared'"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_owner "
                "ON conversations(owner, updated_at)"
            )

    # --------------------------------------------------------------------- write
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _title_from(text: str, limit: int = 60) -> str:
        clean = " ".join(text.split())
        if not clean:
            return _DEFAULT_TITLE
        return clean[:limit] + ("…" if len(clean) > limit else "")

    def ensure_conversation(self, conversation_id: str, owner: str = _SHARED_OWNER) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, owner, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, _DEFAULT_TITLE, owner or _SHARED_OWNER, now, now),
            )

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        standalone_question: str | None = None,
        sources: list[str] | None = None,
        owner: str = _SHARED_OWNER,
    ) -> None:
        now = _now()
        self.ensure_conversation(conversation_id, owner)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, standalone_question, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    role,
                    content,
                    standalone_question,
                    json.dumps(sources) if sources else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )
            if role == "user":
                row = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if row is not None and row["title"] == _DEFAULT_TITLE:
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (self._title_from(content), conversation_id),
                    )

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (self._title_from(title) or _DEFAULT_TITLE, conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # ---------------------------------------------------------------------- read
    def list_conversations(
        self, owner: str | None = None, limit: int = 50
    ) -> list[ConversationSummary]:
        """List non-empty conversations, newest first. If *owner* is given, only
        that owner's conversations are returned (per-visitor isolation)."""
        where = "WHERE c.owner = ?" if owner else ""
        params: tuple = (owner, limit) if owner else (limit,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.title, c.updated_at, COUNT(m.id) AS n
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                {where}
                GROUP BY c.id, c.title, c.updated_at
                HAVING n > 0
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            ConversationSummary(r["id"], r["title"], r["updated_at"], r["n"]) for r in rows
        ]

    def load_messages(self, conversation_id: str) -> list[StoredMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, standalone_question, sources, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            StoredMessage(
                role=r["role"],
                content=r["content"],
                standalone_question=r["standalone_question"],
                sources=json.loads(r["sources"]) if r["sources"] else None,
                created_at=r["created_at"],
            )
            for r in rows
        ]
