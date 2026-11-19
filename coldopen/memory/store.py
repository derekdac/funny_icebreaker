"""Persistent SQLite store for team game history, session state, and optional GCS draft sync."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coldopen.observability.logging import get_logger
from coldopen.observability.pii import scrub_pii

_LOGGER = get_logger("coldopen.memory.store")


def resolve_db_path(db_path: str | Path | None = None) -> str:
    """Resolve SQLite database path, honoring COLDOPEN_DB_PATH when omitted."""
    if db_path is None:
        return os.getenv("COLDOPEN_DB_PATH", "coldopen_state.db")
    return str(db_path)


def get_export_bucket() -> str:
    """Return configured GCS export bucket name, or empty string if unset."""
    return os.getenv("COLDOPEN_EXPORT_BUCKET", "").strip()


def _get_gcs_client() -> Any:
    from google.cloud import storage

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or None
    return storage.Client(project=project)


def upload_draft_manifest_to_gcs(
    draft_id: str,
    manifest_dict: dict[str, Any],
    bucket_name: str | None = None,
) -> str | None:
    """Persist a draft manifest JSON to GCS for cross-instance handoff."""
    bucket = (bucket_name if bucket_name is not None else get_export_bucket()).strip()
    if not bucket:
        return None
    blob_name = f"coldopen-drafts/{draft_id}/manifest.json"
    try:
        client = _get_gcs_client()
        blob = client.bucket(bucket).blob(blob_name)
        blob.upload_from_string(
            json.dumps(manifest_dict),
            content_type="application/json",
        )
        return f"gs://{bucket}/{blob_name}"
    except Exception as exc:
        _LOGGER.warning("GCS draft manifest upload skipped for %s: %s", draft_id, exc)
        return None


def download_draft_manifest_from_gcs(
    draft_id: str,
    bucket_name: str | None = None,
) -> dict[str, Any] | None:
    """Load a draft manifest JSON from GCS when missing in local SQLite."""
    bucket = (bucket_name if bucket_name is not None else get_export_bucket()).strip()
    if not bucket:
        return None
    blob_name = f"coldopen-drafts/{draft_id}/manifest.json"
    try:
        client = _get_gcs_client()
        blob = client.bucket(bucket).blob(blob_name)
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_text())
        return data if isinstance(data, dict) else None
    except Exception as exc:
        _LOGGER.warning("GCS draft manifest download failed for %s: %s", draft_id, exc)
        return None


def upload_draft_image_to_gcs(
    draft_id: str,
    idx: int,
    local_path: str | Path,
    bucket_name: str | None = None,
) -> str | None:
    """Upload a fetched specimen PNG to GCS under the draft prefix."""
    bucket = (bucket_name if bucket_name is not None else get_export_bucket()).strip()
    if not bucket:
        return None
    blob_name = f"coldopen-drafts/{draft_id}/images/{idx}.png"
    try:
        client = _get_gcs_client()
        blob = client.bucket(bucket).blob(blob_name)
        blob.upload_from_filename(str(local_path), content_type="image/png")
        return f"gs://{bucket}/{blob_name}"
    except Exception as exc:
        _LOGGER.warning("GCS draft image upload skipped for %s[%d]: %s", draft_id, idx, exc)
        return None


def download_gcs_uri_to_file(gcs_uri: str, dest_path: Path) -> Path | None:
    """Download a gs://bucket/object URI to a local file path."""
    if not gcs_uri.startswith("gs://"):
        return None
    remainder = gcs_uri[5:]
    if "/" not in remainder:
        return None
    bucket, blob_name = remainder.split("/", 1)
    if not bucket or not blob_name:
        return None
    try:
        client = _get_gcs_client()
        blob = client.bucket(bucket).blob(blob_name)
        if not blob.exists():
            return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest_path))
        return dest_path
    except Exception as exc:
        _LOGGER.warning("GCS image download failed for %s: %s", gcs_uri, exc)
        return None


class ColdOpenStore:
    """SQLite-backed persistence for played trivia items, sessions, and consolidated memories."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = resolve_db_path(db_path)
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        with closing(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            with conn:
                yield conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS played_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT NOT NULL,
                    theme TEXT NOT NULL,
                    term TEXT NOT NULL,
                    category TEXT NOT NULL,
                    played_at TEXT NOT NULL,
                    UNIQUE(team_id, term)
                );

                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS team_memories (
                    team_id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS draft_manifests (
                    draft_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deck_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT NOT NULL,
                    deck_title TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_played_items(self, team_id: str, theme: str, items: list[dict[str, str]]) -> int:
        """Persist played terms for a team so future decks never repeat them."""
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        with self._connect() as conn:
            for item in items:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO played_items (team_id, theme, term, category, played_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        scrub_pii(team_id),
                        scrub_pii(theme),
                        scrub_pii(item["term"]),
                        scrub_pii(item["category"]),
                        now,
                    ),
                )
                inserted += cur.rowcount
        return inserted

    def get_played_terms(self, team_id: str) -> list[str]:
        """Return all previously played terms for a given team."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT term FROM played_items WHERE team_id = ? ORDER BY id ASC",
                (scrub_pii(team_id),),
            ).fetchall()
        return [row["term"] for row in rows]

    def append_session_message(self, session_id: str, role: str, content: str) -> None:
        """Store a PII-scrubbed conversational turn in persistent session state."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, scrub_pii(content), now),
            )

    def get_session_messages(self, session_id: str) -> list[dict[str, str]]:
        """Load ordered conversational history for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM session_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_team_memory(self, team_id: str, summary: dict[str, Any]) -> None:
        """Save consolidated team preferences and favorite themes."""
        now = datetime.now(timezone.utc).isoformat()
        safe_json = scrub_pii(json.dumps(summary))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_memories (team_id, summary_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    summary_json = excluded.summary_json,
                    updated_at = excluded.updated_at
                """,
                (scrub_pii(team_id), safe_json, now),
            )

    def get_team_memory(self, team_id: str) -> dict[str, Any]:
        """Retrieve consolidated memory for a team, or an empty dict if none exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM team_memories WHERE team_id = ?",
                (scrub_pii(team_id),),
            ).fetchone()
        return json.loads(row["summary_json"]) if row else {}

    def save_draft_manifest(self, draft_id: str, manifest_dict: dict[str, Any]) -> None:
        """Persist an audited DeckManifest under a short draft_id for zero-copy handoff."""
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(manifest_dict)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO draft_manifests (draft_id, manifest_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(draft_id) DO UPDATE SET
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                (draft_id, payload, now),
            )

    def get_draft_manifest(self, draft_id: str) -> dict[str, Any] | None:
        """Load a persisted DeckManifest by draft_id, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM draft_manifests WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
        return json.loads(row["manifest_json"]) if row else None

    def record_feedback(self, team_id: str, deck_title: str, rating: int, notes: str) -> None:
        """Persist post-game host feedback to improve future theme and item generation."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO deck_feedback (team_id, deck_title, rating, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (scrub_pii(team_id), scrub_pii(deck_title), int(rating), scrub_pii(notes), now),
            )

    def get_recent_feedback(self, team_id: str = "default", limit: int = 5) -> list[dict[str, Any]]:
        """Return recent post-game feedback entries for a team."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT deck_title, rating, notes, created_at FROM deck_feedback WHERE team_id = ? ORDER BY id DESC LIMIT ?",
                (scrub_pii(team_id), limit),
            ).fetchall()
        return [dict(row) for row in rows]
