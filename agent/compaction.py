"""ADK context compaction configuration for FDE Duckie."""

from __future__ import annotations

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

from agent.config import (
    COMPACTION_ENABLED,
    COMPACTION_EVENT_RETENTION_SIZE,
    COMPACTION_INTERVAL,
    COMPACTION_MODEL_NAME,
    COMPACTION_OVERLAP_SIZE,
    COMPACTION_TOKEN_THRESHOLD,
)

FDE_COMPACTION_PROMPT_TEMPLATE = (
    "You are compacting conversation history for FDE Duckie, an onboarding "
    "assistant for Google FDE Nooglers. Summarize the interaction concisely so "
    "a small model (Gemini Flash-Lite) can continue accurately on future turns.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. State the user's primary language at the top (e.g., 'Conversation Language: English').\n"
    "2. Preserve the active Noogler persona/day, any enablement progress updates "
    "(L200, L300 lab counts, L400 eligibility/blockers, PCA), setup status, and "
    "key go/ links already shared.\n"
    "3. List exact tool names called so tool grounding stays accurate.\n"
    "4. Highlight any unresolved questions or next steps.\n\n"
    "{conversation_history}"
)


def build_compaction_config(
    *,
    enabled: bool = COMPACTION_ENABLED,
    model_name: str = COMPACTION_MODEL_NAME,
    compaction_interval: int = COMPACTION_INTERVAL,
    overlap_size: int = COMPACTION_OVERLAP_SIZE,
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    event_retention_size: int = COMPACTION_EVENT_RETENTION_SIZE,
    summarizer: BaseEventsSummarizer | None = None,
) -> EventsCompactionConfig | None:
    """Build ADK EventsCompactionConfig combining sliding-window and token-threshold triggers."""
    if not enabled:
        return None

    active_summarizer = summarizer or LlmEventSummarizer(
        llm=Gemini(model=model_name),
        prompt_template=FDE_COMPACTION_PROMPT_TEMPLATE,
    )
    return EventsCompactionConfig(
        summarizer=active_summarizer,
        compaction_interval=compaction_interval,
        overlap_size=overlap_size,
        token_threshold=token_threshold,
        event_retention_size=event_retention_size,
    )


def build_app(
    *,
    root_agent,
    name: str = "agent",
    compaction_enabled: bool = COMPACTION_ENABLED,
    summarizer: BaseEventsSummarizer | None = None,
) -> App:
    """Construct the ADK App wrapping root_agent with context compaction."""
    return App(
        name=name,
        root_agent=root_agent,
        events_compaction_config=build_compaction_config(
            enabled=compaction_enabled,
            summarizer=summarizer,
        ),
    )
