"""Database-backed session history and Noogler state persistence (SQLite default)."""

from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService

from agent.infra.backends import set_profile_store
from agent.config import PROJECT_ROOT, AppConfig, get_config
from agent.mocks import MockNooglerProfileStore

NOOGLER_PROFILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS noogler_profiles (
    persona_id TEXT PRIMARY KEY,
    ldap TEXT NOT NULL,
    full_name TEXT NOT NULL,
    onboarding_day INTEGER NOT NULL,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def resolve_sqlite_path(url: str) -> Path:
    """Resolve a SQLite URL ('sqlite:///rel/path.db', 'sqlite:////abs.db') or path."""
    cleaned = url.strip()
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    path = Path(cleaned)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class SqliteNooglerProfileStore:
    """SQLite-backed NooglerProfileStore co-located alongside ADK session tables."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(NOOGLER_PROFILES_TABLE_SQL)

    def count_profiles(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM noogler_profiles").fetchone()
            return int(row["cnt"]) if row else 0

    def list_persona_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT persona_id FROM noogler_profiles ORDER BY persona_id ASC"
            ).fetchall()
            return [str(r["persona_id"]) for r in rows]

    def list_persona_summaries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT persona_id, full_name, onboarding_day "
                "FROM noogler_profiles ORDER BY persona_id ASC"
            ).fetchall()
            return [
                {
                    "persona_id": str(r["persona_id"]),
                    "full_name": str(r["full_name"]),
                    "onboarding_day": int(r["onboarding_day"]),
                }
                for r in rows
            ]

    def has_persona(self, persona_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM noogler_profiles WHERE persona_id = ?",
                (persona_id,),
            ).fetchone()
            return row is not None

    def get_profile(self, persona_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM noogler_profiles WHERE persona_id = ?",
                (persona_id,),
            ).fetchone()
        if row is None:
            valid = ", ".join(self.list_persona_ids())
            raise KeyError(f"Unknown persona '{persona_id}'. Valid personas: {valid}")
        return json.loads(row["profile_json"])

    def save_profile(self, profile: dict[str, Any]) -> None:
        """Upsert a Noogler profile into SQLite."""
        stored = copy.deepcopy(profile)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO noogler_profiles (
                    persona_id, ldap, full_name, onboarding_day, profile_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(persona_id) DO UPDATE SET
                    ldap = excluded.ldap,
                    full_name = excluded.full_name,
                    onboarding_day = excluded.onboarding_day,
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (
                    stored["persona_id"],
                    stored["ldap"],
                    stored["full_name"],
                    int(stored["onboarding_day"]),
                    json.dumps(stored, ensure_ascii=False),
                    now,
                ),
            )

    def seed_from_mock_store(
        self,
        *,
        mock_store: MockNooglerProfileStore | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Seed SQLite with mock Noogler personas if missing (or overwrite all)."""
        source = mock_store or MockNooglerProfileStore()
        seeded: list[str] = []
        for persona_id in source.list_persona_ids():
            if overwrite or not self.has_persona(persona_id):
                self.save_profile(source.get_profile(persona_id))
                seeded.append(persona_id)
        return seeded


def create_session_service(config: AppConfig | None = None) -> BaseSessionService:
    """Create the configured ADK session service (SQLite, SQLAlchemy DB, or in-memory)."""
    cfg = config or get_config()
    backend = cfg.database.backend

    if backend == "memory":
        return InMemorySessionService()

    if backend == "sqlite":
        db_path = resolve_sqlite_path(cfg.database.url)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSessionService(db_path=str(db_path))

    if backend == "database":
        from google.adk.sessions import DatabaseSessionService

        return DatabaseSessionService(db_url=cfg.database.url)

    raise ValueError(f"Unsupported database backend: '{backend}'")


def configure_storage(
    config: AppConfig | None = None,
    *,
    auto_seed: bool = True,
) -> BaseSessionService:
    """Configure both the active profile store and ADK session service from AppConfig."""
    cfg = config or get_config()
    if cfg.database.backend == "sqlite":
        db_path = resolve_sqlite_path(cfg.database.url)
        profile_store = SqliteNooglerProfileStore(db_path)
        if auto_seed and profile_store.count_profiles() == 0:
            profile_store.seed_from_mock_store()
        set_profile_store(profile_store)
    elif cfg.database.backend == "memory":
        set_profile_store(MockNooglerProfileStore())

    return create_session_service(cfg)
