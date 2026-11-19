"""Local development SQLite setup and seeding script for FDE Duckie.

Creates/initializes ADK's SQLite session tables (`sessions`, `events`,
`app_states`, `user_states`) and seeds the `noogler_profiles` table (plus
optional starter sessions) from `agent/mocks/data/noogler_profiles.json`.

Usage:
    uv run python scripts/setup_db.py
    uv run python scripts/setup_db.py --reset
    uv run python scripts/setup_db.py --config config.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Allow running directly via `python scripts/setup_db.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.events import Event  # noqa: E402
from google.adk.sessions.sqlite_session_service import SqliteSessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.infra.db import SqliteNooglerProfileStore, resolve_sqlite_path  # noqa: E402
from agent.state import (  # noqa: E402
    PERSONA_ID_STATE_KEY,
    PROFILE_STATE_KEY,
    SUMMARY_STATE_KEY,
    format_profile_summary,
)


async def seed_local_db(
    *,
    db_path: str | Path,
    app_name: str = "agent",
    reset: bool = False,
    seed_sessions: bool = True,
) -> dict[str, Any]:
    """Initialize SQLite schema and seed mock Noogler profiles & starter sessions."""
    target = Path(db_path)
    if reset:
        for candidate in (
            target,
            Path(f"{target}-wal"),
            Path(f"{target}-shm"),
        ):
            if candidate.exists():
                candidate.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)

    # 1. Seed `noogler_profiles` table.
    profile_store = SqliteNooglerProfileStore(target)
    seeded_profiles = profile_store.seed_from_mock_store(overwrite=reset)

    # 2. Initialize ADK's SQLite session tables (`sessions`, `events`, `app_states`, `user_states`).
    session_service = SqliteSessionService(db_path=str(target))
    seeded_sessions: list[str] = []
    try:
        # Touching list_sessions ensures ADK initializes all 4 core tables even if seed_sessions=False.
        await session_service.list_sessions(app_name=app_name)

        if seed_sessions:
            for persona_id in profile_store.list_persona_ids():
                session_id = f"default-{persona_id}"
                existing = await session_service.get_session(
                    app_name=app_name,
                    user_id=persona_id,
                    session_id=session_id,
                )
                if existing is not None:
                    continue

                profile = profile_store.get_profile(persona_id)
                session = await session_service.create_session(
                    app_name=app_name,
                    user_id=persona_id,
                    session_id=session_id,
                    state={
                        PERSONA_ID_STATE_KEY: persona_id,
                        PROFILE_STATE_KEY: profile,
                        SUMMARY_STATE_KEY: format_profile_summary(profile),
                    },
                )
                welcome = Event(
                    invocation_id=Event.new_id(),
                    author="fde_duckie",
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=(
                                    f"Welcome to FDE Duckie, {profile['full_name']}! "
                                    f"You're on Day {profile['onboarding_day']} in {profile['pod']}. "
                                    "Ask about your L200/L300/L400/PCA enablement status, "
                                    "what to focus on next, or any general Google question."
                                )
                            )
                        ],
                    ),
                )
                await session_service.append_event(session, welcome)
                seeded_sessions.append(session_id)
    finally:
        await session_service.close()

    return {
        "db_path": str(target),
        "all_profiles": profile_store.list_persona_ids(),
        "seeded_profiles": seeded_profiles,
        "seeded_sessions": seeded_sessions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize and seed the local SQLite database for FDE Duckie."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file path or inline JSON string.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional explicit SQLite file path overriding database.url from config.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the SQLite DB from scratch before seeding.",
    )
    parser.add_argument(
        "--no-seed-sessions",
        action="store_true",
        help="Only seed noogler_profiles and schema; skip starter chat sessions.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    db_path = Path(args.db_path) if args.db_path else resolve_sqlite_path(cfg.database.url)

    summary = await seed_local_db(
        db_path=db_path,
        app_name="agent",
        reset=args.reset,
        seed_sessions=not args.no_seed_sessions,
    )

    print(f"SQLite DB ready: {summary['db_path']}")
    print(f"  Profiles in DB  : {', '.join(summary['all_profiles'])}")
    print(
        f"  Newly seeded    : {', '.join(summary['seeded_profiles']) or '(already seeded)'}"
    )
    print(
        f"  Starter sessions: {', '.join(summary['seeded_sessions']) or '(already exist / skipped)'}"
    )
    print("\nRun locally with persistent history & state:")
    print("  uv run python main.py --user priya_fde")
    print(f"  uv run adk web --session_service_uri sqlite:///{summary['db_path']}")


if __name__ == "__main__":
    asyncio.run(_main())
