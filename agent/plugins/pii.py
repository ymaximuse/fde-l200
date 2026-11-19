"""Minimal regex-based PII redaction & scrubbing plugin for FDE Duckie."""

from __future__ import annotations

import copy
import re
from typing import Any

from google.adk.plugins import BasePlugin
from google.genai import types

from agent.config import PiiConfig, get_config

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "API_KEY",
        re.compile(
            r"\b(?:AIza[0-9A-Za-z_-]{35}|Bearer\s+[A-Za-z0-9._~+/=-]{20,})\b",
            re.IGNORECASE,
        ),
    ),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)"
        ),
    ),
)


class PiiScrubber:
    """Simple regex PII scrubber for text, dicts/lists, and ADK Content."""

    def __init__(self, config: PiiConfig | None = None) -> None:
        self.config = config or get_config().pii

    def scrub_text(self, text: str) -> str:
        if not self.config.enabled or not text:
            return text
        for label, pattern in _PII_PATTERNS:
            text = pattern.sub(f"[REDACTED_{label}]", text)
        return text

    def scrub_value(self, value: Any) -> Any:
        if not self.config.enabled:
            return value
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            return {k: self.scrub_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub_value(item) for item in value]
        return value

    def scrub_content(self, content: types.Content | None) -> types.Content | None:
        if not self.config.enabled or content is None or not content.parts:
            return content
        changed = False
        new_parts: list[types.Part] = []
        for part in content.parts:
            updated = copy.deepcopy(part)
            part_changed = False
            if updated.text:
                scrubbed = self.scrub_text(updated.text)
                if scrubbed != updated.text:
                    updated.text = scrubbed
                    part_changed = True
            if updated.function_call and updated.function_call.args is not None:
                scrubbed_args = self.scrub_value(updated.function_call.args)
                if scrubbed_args != updated.function_call.args:
                    updated.function_call.args = scrubbed_args
                    part_changed = True
            if updated.function_response and updated.function_response.response is not None:
                scrubbed_resp = self.scrub_value(updated.function_response.response)
                if scrubbed_resp != updated.function_response.response:
                    updated.function_response.response = scrubbed_resp
                    part_changed = True
            changed = changed or part_changed
            new_parts.append(updated if part_changed else part)
        return content.model_copy(update={"parts": new_parts}) if changed else content


class PiiRedactionPlugin(BasePlugin):
    """ADK plugin that scrubs PII on user input, tool outputs, model responses, and events."""

    def __init__(
        self,
        *,
        config: PiiConfig | None = None,
        name: str = "fde_duckie_pii_redaction",
    ) -> None:
        super().__init__(name=name)
        self._scrubber = PiiScrubber(config)

    async def on_user_message_callback(
        self, *, invocation_context: Any, user_message: types.Content
    ) -> types.Content | None:
        scrubbed = self._scrubber.scrub_content(user_message)
        return scrubbed if scrubbed is not user_message else None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        scrubbed = self._scrubber.scrub_value(result)
        return scrubbed if scrubbed != result else None

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> Any:
        content = getattr(llm_response, "content", None)
        scrubbed = self._scrubber.scrub_content(content)
        if scrubbed is not content:
            return llm_response.model_copy(update={"content": scrubbed})
        return None

    async def on_event_callback(
        self, *, invocation_context: Any, event: Any
    ) -> Any:
        content = getattr(event, "content", None)
        scrubbed = self._scrubber.scrub_content(content)
        if scrubbed is not content:
            return event.model_copy(update={"content": scrubbed})
        return None
