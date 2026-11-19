"""ADK App plugins and middleware builders for FDE Duckie."""

from __future__ import annotations

from google.adk.plugins import BasePlugin

from agent.config import AppConfig, get_config
from agent.plugins.compaction import (
    FDE_COMPACTION_PROMPT_TEMPLATE,
    build_compaction_config,
)
from agent.plugins.observability import (
    ObservabilityPlugin,
    configure_logging,
    configure_otel,
    log_event,
)
from agent.plugins.pii import PiiRedactionPlugin, PiiScrubber
from agent.plugins.routing import (
    ModelRoutingPlugin,
    RoutingDecision,
    build_routing_plugins,
    read_complexity_from_state,
    select_model,
)
from agent.plugins.security import ModelArmorPlugin, build_security_plugins


def build_plugins(config: AppConfig | None = None) -> list[BasePlugin]:
    """Assemble the default ADK plugin stack from AppConfig."""
    cfg = config or get_config()
    plugins: list[BasePlugin] = list(build_security_plugins(cfg))
    plugins.extend(build_routing_plugins(cfg))
    if cfg.pii.enabled:
        plugins.append(PiiRedactionPlugin(config=cfg.pii))
    plugins.append(ObservabilityPlugin())
    return plugins


__all__ = [
    "FDE_COMPACTION_PROMPT_TEMPLATE",
    "ModelArmorPlugin",
    "ModelRoutingPlugin",
    "ObservabilityPlugin",
    "PiiRedactionPlugin",
    "PiiScrubber",
    "RoutingDecision",
    "build_compaction_config",
    "build_plugins",
    "build_routing_plugins",
    "build_security_plugins",
    "configure_logging",
    "configure_otel",
    "log_event",
    "read_complexity_from_state",
    "select_model",
]
