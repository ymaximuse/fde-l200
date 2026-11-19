"""ADK context compaction configuration for FDE Duckie."""

from __future__ import annotations

from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

from agent.config import AppConfig, get_config

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
    config: AppConfig | None = None,
    summarizer: BaseEventsSummarizer | None = None,
) -> EventsCompactionConfig | None:
    """Build ADK EventsCompactionConfig from AppConfig."""
    cfg = config or get_config()
    if not cfg.compaction.enabled:
        return None

    active_summarizer = summarizer or LlmEventSummarizer(
        llm=Gemini(model=cfg.compaction_model_name),
        prompt_template=FDE_COMPACTION_PROMPT_TEMPLATE,
    )
    return EventsCompactionConfig(
        summarizer=active_summarizer,
        compaction_interval=cfg.compaction.interval,
        overlap_size=cfg.compaction.overlap_size,
        token_threshold=cfg.compaction.token_threshold,
        event_retention_size=cfg.compaction.event_retention_size,
    )
