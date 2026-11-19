"""State-based strategic model routing ADK plugin for FDE Duckie.

Step 1 of the root SequentialAgent (`complexity_analyzer_agent`, always running
on Flash-Lite) classifies each turn into `session.state["request_complexity"]`.
This plugin reads that state key in `before_model_callback` and routes:
- `complexity == "complex"` -> `model_routing.capable_model` (Gemini Flash)
- otherwise -> `model.name` (Gemini Flash-Lite)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins import BasePlugin
from opentelemetry import trace

from agent.config import AppConfig, ModelRoutingConfig, get_config
from agent.plugins.observability import log_event, logger
from agent.schemas import ComplexityAssessment
from agent.state import COMPLEXITY_STATE_KEY
from agent.sub_agents.complexity_analyzer_agent import COMPLEXITY_ANALYZER_NAME


@dataclass(frozen=True)
class RoutingDecision:
    """Result of state-based model routing for a single LLM call."""

    model: str
    tier: Literal["fast", "capable"]
    complexity: Literal["simple", "complex"]
    reason: str


def read_complexity_from_state(
    state: Mapping[str, Any] | None,
) -> tuple[Literal["simple", "complex"], str | None]:
    """Read `(complexity, reason)` from session state, defaulting to `'simple'`."""
    if not state:
        return "simple", None

    raw = state.get(COMPLEXITY_STATE_KEY)
    if raw is None:
        return "simple", None

    if isinstance(raw, ComplexityAssessment):
        return raw.complexity, raw.reason

    if isinstance(raw, Mapping):
        level = str(raw.get("complexity", "simple")).strip().lower()
        reason = raw.get("reason")
        return ("complex" if level == "complex" else "simple"), (
            str(reason) if reason else None
        )

    if isinstance(raw, str):
        level = raw.strip().lower()
        return ("complex" if level == "complex" else "simple"), None

    return "simple", None


def select_model(
    state: Mapping[str, Any] | None,
    *,
    fast_model: str,
    routing_config: ModelRoutingConfig,
    agent_name: str | None = None,
) -> RoutingDecision:
    """Select between `fast_model` (Flash-Lite) and `capable_model` (Flash) from state."""
    if agent_name == COMPLEXITY_ANALYZER_NAME:
        return RoutingDecision(
            model=fast_model,
            tier="fast",
            complexity="simple",
            reason="complexity_analyzer_always_fast",
        )

    complexity, reason = read_complexity_from_state(state)
    if complexity == "complex":
        return RoutingDecision(
            model=routing_config.capable_model,
            tier="capable",
            complexity="complex",
            reason=reason or "state_complex",
        )

    return RoutingDecision(
        model=fast_model,
        tier="fast",
        complexity="simple",
        reason=reason or "state_simple",
    )


class ModelRoutingPlugin(BasePlugin):
    """ADK plugin that routes LLM calls based on `state["request_complexity"]`."""

    def __init__(
        self,
        *,
        fast_model: str,
        config: ModelRoutingConfig | None = None,
        name: str = "fde_duckie_model_routing",
        log: logging.Logger | None = None,
    ) -> None:
        super().__init__(name=name)
        self._fast_model = fast_model
        self._config = config or ModelRoutingConfig()
        self._log = log or logger

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: LlmRequest
    ) -> LlmResponse | None:
        if not self._config.enabled:
            return None

        agent_name = getattr(callback_context, "agent_name", None)
        state = getattr(callback_context, "state", None)
        previous_model = llm_request.model
        decision = select_model(
            state,
            fast_model=self._fast_model,
            routing_config=self._config,
            agent_name=agent_name,
        )
        llm_request.model = decision.model

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("fde_duckie.model_routing.tier", decision.tier)
            span.set_attribute("fde_duckie.model_routing.model", decision.model)
            span.set_attribute(
                "fde_duckie.model_routing.complexity", decision.complexity
            )
            span.set_attribute("fde_duckie.model_routing.reason", decision.reason)

        log_event(
            "model.routed",
            level=logging.INFO if decision.tier != "fast" else logging.DEBUG,
            log=self._log,
            agent_name=agent_name,
            previous_model=previous_model,
            selected_model=decision.model,
            tier=decision.tier,
            complexity=decision.complexity,
            reason=decision.reason,
        )
        return None


def build_routing_plugins(
    config: AppConfig | None = None,
    *,
    log: logging.Logger | None = None,
) -> list[BasePlugin]:
    """Build the state-based model routing plugin when enabled in AppConfig."""
    cfg = config or get_config()
    if not cfg.model_routing.enabled:
        return []
    return [
        ModelRoutingPlugin(
            fast_model=cfg.model.name,
            config=cfg.model_routing,
            log=log,
        )
    ]
