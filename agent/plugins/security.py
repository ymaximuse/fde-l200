"""Barebones Gemini Model Armor ADK plugin (not included out of the box in ADK)."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins import BasePlugin
from google.genai import types

from agent.config import AppConfig, get_config
from agent.plugins.observability import log_event, logger

_BLOCKED_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
}


class ModelArmorPlugin(BasePlugin):
    """ADK plugin attaching Gemini inline Model Armor templates to every LLM call."""

    def __init__(
        self,
        *,
        prompt_template_name: str | None = None,
        response_template_name: str | None = None,
        name: str = "fde_duckie_model_armor",
        log: logging.Logger | None = None,
    ) -> None:
        if not (prompt_template_name or response_template_name):
            raise ValueError(
                "ModelArmorPlugin requires prompt_template_name and/or response_template_name."
            )
        super().__init__(name=name)
        self._model_armor_config = types.ModelArmorConfig(
            prompt_template_name=prompt_template_name,
            response_template_name=response_template_name,
        )
        self._log = log or logger

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: LlmRequest
    ) -> LlmResponse | None:
        if llm_request.config is None:
            llm_request.config = types.GenerateContentConfig()
        if llm_request.config.model_armor_config is None:
            llm_request.config.model_armor_config = self._model_armor_config
            log_event(
                "security.model_armor.attached",
                level=logging.DEBUG,
                log=self._log,
                prompt_template=self._model_armor_config.prompt_template_name,
                response_template=self._model_armor_config.response_template_name,
            )
        return None

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.finish_reason in _BLOCKED_FINISH_REASONS:
            reason = getattr(
                llm_response.finish_reason, "name", str(llm_response.finish_reason)
            )
            log_event(
                "security.model_armor.blocked",
                level=logging.WARNING,
                log=self._log,
                finish_reason=reason,
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "This response was blocked by safety / Model Armor policy. "
                                "Please rephrase your FDE onboarding question."
                            )
                        )
                    ],
                )
            )
        return None


def build_security_plugins(
    config: AppConfig | None = None,
    *,
    log: logging.Logger | None = None,
) -> list[BasePlugin]:
    """Build configured custom security plugins (Model Armor) from AppConfig."""
    cfg = config or get_config()
    armor = cfg.security.model_armor
    if not armor.enabled:
        return []

    return [
        ModelArmorPlugin(
            prompt_template_name=armor.prompt_template_name,
            response_template_name=armor.response_template_name,
            log=log,
        )
    ]
