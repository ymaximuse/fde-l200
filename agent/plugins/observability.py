"""Simple structured JSON logging and OpenTelemetry tracing for FDE Duckie."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

from google.adk.plugins import BasePlugin
from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

from agent.config import AppConfig, get_config
from agent.plugins.pii import PiiScrubber

LOGGER_NAME = "fde_duckie"
logger = logging.getLogger(LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    """Single-line JSON log formatter with automatic OTel trace correlation and optional PII scrubbing."""

    def __init__(self, *, scrubber: PiiScrubber | None = None) -> None:
        super().__init__()
        self._scrubber = scrubber

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if self._scrubber is not None:
            message = self._scrubber.scrub_text(message)

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": message,
        }

        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            payload["trace_id"] = format(span_ctx.trace_id, "032x")
            payload["span_id"] = format(span_ctx.span_id, "016x")

        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            cleaned_fields = {k: v for k, v in fields.items() if v is not None}
            if self._scrubber is not None:
                cleaned_fields = self._scrubber.scrub_value(cleaned_fields)
            payload.update(cleaned_fields)

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            if self._scrubber is not None:
                exc_text = self._scrubber.scrub_text(exc_text)
            payload["exception"] = exc_text

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    config: AppConfig | None = None,
    *,
    level: str | None = None,
    stream: TextIO | None = None,
    logger_name: str = LOGGER_NAME,
    scrubber: PiiScrubber | None = None,
) -> logging.Logger:
    """Configure the FDE Duckie logger to emit structured JSON lines to stderr."""
    cfg = config or get_config()
    numeric_level = getattr(logging, (level or cfg.logging.level).upper(), logging.INFO)

    active_scrubber = scrubber
    if active_scrubber is None and cfg.pii.enabled:
        active_scrubber = PiiScrubber(cfg.pii)

    target = logging.getLogger(logger_name)
    target.setLevel(numeric_level)
    target.propagate = False

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(JsonFormatter(scrubber=active_scrubber))

    target.handlers.clear()
    target.addHandler(handler)
    return target


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    log: logging.Logger | None = None,
    **fields: Any,
) -> None:
    """Emit a structured JSON log entry with optional key-value fields."""
    (log or logger).log(level, event, extra={"fields": {"event": event, **fields}})


def configure_otel(
    config: AppConfig | None = None,
    *,
    exporter: str | None = None,
    span_exporter: SpanExporter | None = None,
    set_global: bool = True,
) -> TracerProvider | None:
    """Configure OpenTelemetry tracing (`none`, `console`, or `otlp`).

    ADK automatically creates OTel spans for agent, LLM, and tool calls once a
    TracerProvider is registered.
    """
    cfg = config or get_config()
    mode = (exporter or cfg.otel.exporter).lower()
    if mode == "none" and span_exporter is None:
        return None

    resource = Resource.create({"service.name": cfg.otel.service_name})

    if mode == "otlp" and span_exporter is None:
        maybe_set_otel_providers(
            otel_hooks_to_setup=[OTelHooks()],
            otel_resource=resource,
        )
        return None

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        SimpleSpanProcessor(span_exporter or ConsoleSpanExporter(out=sys.stderr))
    )
    if set_global and not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(provider)
    return provider


class ObservabilityPlugin(BasePlugin):
    """Lightweight ADK plugin for structured JSON lifecycle logging."""

    def __init__(
        self,
        *,
        name: str = "fde_duckie_observability",
        log: logging.Logger | None = None,
    ) -> None:
        super().__init__(name=name)
        self._log = log or logger

    async def before_agent_callback(
        self, *, agent: Any, callback_context: Any
    ) -> None:
        log_event(
            "agent.start",
            log=self._log,
            agent_name=getattr(agent, "name", "unknown"),
            invocation_id=getattr(callback_context, "invocation_id", None),
        )

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> None:
        log_event(
            "tool.call",
            log=self._log,
            tool_name=getattr(tool, "name", "unknown"),
        )

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> None:
        status = result.get("status", "ok") if isinstance(result, dict) else "ok"
        log_event(
            "tool.result",
            log=self._log,
            tool_name=getattr(tool, "name", "unknown"),
            tool_status=status,
        )

    async def on_tool_error_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        error: Exception,
    ) -> None:
        log_event(
            "tool.error",
            level=logging.ERROR,
            log=self._log,
            tool_name=getattr(tool, "name", "unknown"),
            error=str(error),
        )
