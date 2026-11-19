"""Configuration constants for FDE Duckie."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.genai import types

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


# Default to Gemini Flash-Lite; override via FDE_DUCKIE_MODEL in .env.
MODEL_NAME: str = os.getenv("FDE_DUCKIE_MODEL", "gemini-2.5-flash-lite")

DEFAULT_PERSONA_ID: str = os.getenv("FDE_DUCKIE_DEFAULT_USER", "alex_noogler")

# Context compaction settings tuned for small models (Flash-Lite).
COMPACTION_ENABLED: bool = _env_bool("FDE_DUCKIE_COMPACTION_ENABLED", True)
COMPACTION_MODEL_NAME: str = os.getenv("FDE_DUCKIE_COMPACTION_MODEL", MODEL_NAME)
COMPACTION_INTERVAL: int = _env_int("FDE_DUCKIE_COMPACTION_INTERVAL", 5)
COMPACTION_OVERLAP_SIZE: int = _env_int("FDE_DUCKIE_COMPACTION_OVERLAP", 1)
COMPACTION_TOKEN_THRESHOLD: int = _env_int(
    "FDE_DUCKIE_COMPACTION_TOKEN_THRESHOLD", 8000
)
COMPACTION_EVENT_RETENTION_SIZE: int = _env_int(
    "FDE_DUCKIE_COMPACTION_RETENTION_SIZE", 2
)

# Low temperature improves routing and tool-call reliability on small models.
LOW_TEMP_CONFIG = types.GenerateContentConfig(temperature=0.1)
