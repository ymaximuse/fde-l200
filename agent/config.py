"""Centralized Pydantic JSON configuration for FDE Duckie.

All runtime configuration (model, compaction, structured logging, OpenTelemetry,
and default persona) lives in a single JSON config passed via `--config` (file
path or inline JSON), with `.env` reserved strictly for credentials/secrets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="gemini-2.5-flash-lite",
        min_length=1,
        description="Gemini model ID used by FDE Duckie agents.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (low values improve Flash-Lite routing reliability).",
    )

    def to_generate_content_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(temperature=self.temperature)


class CompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True, description="Enable ADK conversation context compaction."
    )
    model_name: str | None = Field(
        default=None,
        description="Optional summarizer model override; defaults to model.name.",
    )
    interval: int = Field(
        default=5, gt=0, description="Sliding-window compaction interval in user turns."
    )
    overlap_size: int = Field(
        default=1, ge=0, description="Turn overlap preserved across consecutive summaries."
    )
    token_threshold: int = Field(
        default=8000,
        gt=0,
        description="Prompt token threshold triggering post-invocation compaction.",
    )
    event_retention_size: int = Field(
        default=2,
        ge=0,
        description="Number of recent raw events retained uncompacted on token trigger.",
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Minimum structured JSON log severity."
    )


class OtelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(
        default="fde-duckie", min_length=1, description="OpenTelemetry service.name."
    )
    exporter: Literal["none", "console", "otlp"] = Field(
        default="none",
        description="OTel trace exporter mode ('none', 'console', or 'otlp').",
    )


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["sqlite", "memory", "database"] = Field(
        default="sqlite",
        description="Session & state persistence backend ('sqlite', 'memory', or 'database').",
    )
    url: str = Field(
        default="sqlite:///data/fde_duckie.db",
        min_length=1,
        description="Database URI (e.g., 'sqlite:///data/fde_duckie.db' or SQLAlchemy URL).",
    )


class SecretsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["vault", "env", "gcp"] = Field(
        default="vault",
        description="Secrets manager backend ('vault' HashiCorp Vault, 'env', or 'gcp' Secret Manager).",
    )
    vault_addr: str = Field(
        default="http://127.0.0.1:8200",
        min_length=1,
        description="HashiCorp Vault server URL (overridden by VAULT_ADDR env if set).",
    )
    vault_mount: str = Field(
        default="secret",
        min_length=1,
        description="HashiCorp Vault KV v2 mount path.",
    )
    vault_path: str = Field(
        default="fde-duckie",
        min_length=1,
        description="Secret path inside the Vault KV v2 mount.",
    )
    vault_namespace: str | None = Field(
        default=None,
        description="Optional Vault namespace (overridden by VAULT_NAMESPACE env if set).",
    )
    gcp_project_id: str | None = Field(
        default=None,
        description="GCP project ID when backend='gcp'.",
    )


class AppConfig(BaseModel):
    """Top-level validated JSON configuration for FDE Duckie."""

    model_config = ConfigDict(extra="forbid")

    default_user: str = Field(
        default="alex_noogler",
        min_length=1,
        description="Default FDE Noogler persona ID loaded at session start.",
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    otel: OtelConfig = Field(default_factory=OtelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)

    @property
    def compaction_model_name(self) -> str:
        return self.compaction.model_name or self.model.name


def load_config(config: str | Path | None = None) -> AppConfig:
    """Load AppConfig from a `--config` JSON file path, inline JSON string, or defaults.

    Resolution order when `config` is None:
    1. `FDE_DUCKIE_CONFIG` env pointer (useful for `adk web` / `adk run agent`).
    2. Project-root `config.json` if present.
    3. Built-in `AppConfig()` defaults.
    """
    source: str | Path | None = config
    if source is None:
        env_config = os.getenv("FDE_DUCKIE_CONFIG")
        if env_config:
            source = env_config
        elif DEFAULT_CONFIG_PATH.is_file():
            source = DEFAULT_CONFIG_PATH
        else:
            return AppConfig()

    if isinstance(source, Path):
        raw_text = source.read_text(encoding="utf-8")
        return AppConfig.model_validate(json.loads(raw_text))

    stripped = source.strip()
    if stripped.startswith("{"):
        return AppConfig.model_validate(json.loads(stripped))

    path = Path(stripped)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found for --config: '{stripped}'"
        )
    return AppConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


_active_config: AppConfig = load_config()


def get_config() -> AppConfig:
    """Return the active runtime AppConfig."""
    return _active_config


def set_config(config: AppConfig) -> AppConfig:
    """Set the active runtime AppConfig."""
    global _active_config
    _active_config = config
    return _active_config
