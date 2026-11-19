"""ADK context compaction configuration and async background plugin for FDE Duckie."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.compaction import (
    _latest_compaction_end_timestamp,
    _longest_self_contained_prefix,
)
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.errors._stale_session_error import StaleSessionError
from google.adk.events import Event
from google.adk.events._rewind_events import _apply_rewinds
from google.adk.models import Gemini
from google.adk.plugins import BasePlugin
from opentelemetry import trace

from agent.config import AppConfig, get_config
from agent.plugins.observability import log_event

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions.base_session_service import BaseSessionService

_SessionKey = tuple[int, str, str, str]

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


def select_first_n_messages_to_compact(
    events: list[Event],
    first_n_messages: int,
) -> list[Event]:
    """Select the first N uncompacted message events safe to summarize.

    Filters out rewound events, isolated sub-agent events, existing compaction
    events, and events already covered by a prior compaction summary. Returns
    an empty list until at least ``first_n_messages`` uncompacted messages are
    available, and trims the selected prefix to a tool-call/response balanced
    boundary so no function response is ever orphaned.
    """
    if first_n_messages <= 0 or not events:
        return []

    live_events = [
        event for event in _apply_rewinds(events) if event.isolation_scope is None
    ]
    if not live_events:
        return []

    last_compacted_end_timestamp = _latest_compaction_end_timestamp(live_events)
    uncompacted_messages = [
        event
        for event in live_events
        if not (event.actions and event.actions.compaction)
        and event.timestamp > last_compacted_end_timestamp
        and event.content is not None
        and bool(event.content.parts)
    ]
    if len(uncompacted_messages) < first_n_messages:
        return []

    prefix = uncompacted_messages[:first_n_messages]
    return _longest_self_contained_prefix(prefix, all_events=live_events)


class AsyncCompactionPlugin(BasePlugin):
    """ADK plugin that compacts the first N uncompacted messages asynchronously in the background.

    Unlike ADK's built-in synchronous compaction (which blocks turn completion
    while waiting for the summarizer LLM), this plugin schedules summarization
    on the asyncio event loop in ``after_run_callback`` and returns immediately.
    When summarization completes, the resulting ``EventCompaction`` is persisted
    to the session service (or safely deferred and flushed on the active turn
    boundary if another turn is concurrently in flight, avoiding
    ``StaleSessionError`` on SQLite/DB backends).
    """

    def __init__(
        self,
        *,
        first_n_messages: int = 6,
        summarizer: BaseEventsSummarizer,
        shutdown_timeout: float = 10.0,
        log: logging.Logger | None = None,
        name: str = "fde_duckie_async_compaction",
    ) -> None:
        super().__init__(name=name)
        if first_n_messages <= 0:
            raise ValueError("first_n_messages must be > 0")
        self.first_n_messages = first_n_messages
        self.summarizer = summarizer
        self._shutdown_timeout = shutdown_timeout
        self._log = log
        self._tracer = trace.get_tracer(__name__)

        self._active_runs: dict[_SessionKey, int] = {}
        self._in_flight_tasks: dict[_SessionKey, asyncio.Task[None]] = {}
        self._pending_compactions: dict[_SessionKey, list[Event]] = {}
        self._session_locks: dict[_SessionKey, asyncio.Lock] = {}

    @staticmethod
    def _is_nested_ephemeral_run(invocation_context: InvocationContext) -> bool:
        plugin_manager = getattr(invocation_context, "plugin_manager", None)
        return bool(getattr(plugin_manager, "_skip_closing_plugins", False))

    @staticmethod
    def _session_key(invocation_context: InvocationContext) -> _SessionKey:
        session = invocation_context.session
        return (
            id(invocation_context.session_service),
            session.app_name,
            session.user_id,
            session.id,
        )

    def _get_lock(self, key: _SessionKey) -> asyncio.Lock:
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock

    async def _flush_pending_to_active_session(
        self,
        key: _SessionKey,
        invocation_context: InvocationContext,
    ) -> None:
        pending = self._pending_compactions.pop(key, None)
        if not pending:
            return
        session = invocation_context.session
        for compaction_event in pending:
            compaction_event.timestamp = max(
                compaction_event.timestamp,
                session.last_update_time + 1e-6,
            )
            await invocation_context.session_service.append_event(
                session=session,
                event=compaction_event,
            )
            comp = compaction_event.actions.compaction if compaction_event.actions else None
            log_event(
                "compaction.flushed",
                log=self._log,
                session_id=session.id,
                start_timestamp=comp.start_timestamp if comp else None,
                end_timestamp=comp.end_timestamp if comp else None,
            )

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        if self._is_nested_ephemeral_run(invocation_context):
            return None
        key = self._session_key(invocation_context)
        async with self._get_lock(key):
            self._active_runs[key] = self._active_runs.get(key, 0) + 1
            await self._flush_pending_to_active_session(key, invocation_context)
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        if self._is_nested_ephemeral_run(invocation_context):
            return None

        key = self._session_key(invocation_context)
        session = invocation_context.session
        async with self._get_lock(key):
            await self._flush_pending_to_active_session(key, invocation_context)
            remaining = max(0, self._active_runs.get(key, 1) - 1)
            if remaining == 0:
                self._active_runs.pop(key, None)
            else:
                self._active_runs[key] = remaining

            existing_task = self._in_flight_tasks.get(key)
            if existing_task is not None and not existing_task.done():
                return None

            events_to_compact = select_first_n_messages_to_compact(
                list(session.events),
                self.first_n_messages,
            )
            if not events_to_compact:
                return None

            task = asyncio.create_task(
                self._compact_in_background(
                    key=key,
                    app_name=session.app_name,
                    user_id=session.user_id,
                    session_id=session.id,
                    session_service=invocation_context.session_service,
                    events_to_compact=list(events_to_compact),
                ),
                name=f"async-compaction-{session.id}",
            )
            self._in_flight_tasks[key] = task

        log_event(
            "compaction.scheduled",
            log=self._log,
            session_id=session.id,
            message_count=len(events_to_compact),
            first_n_messages=self.first_n_messages,
            start_timestamp=events_to_compact[0].timestamp,
            end_timestamp=events_to_compact[-1].timestamp,
        )
        return None

    async def _persist_to_storage(
        self,
        *,
        key: _SessionKey,
        app_name: str,
        user_id: str,
        session_id: str,
        session_service: BaseSessionService,
        compaction_event: Event,
    ) -> None:
        for attempt in range(2):
            fresh_session = await session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if fresh_session is None:
                log_event(
                    "compaction.session_missing",
                    level=logging.WARNING,
                    log=self._log,
                    session_id=session_id,
                )
                return

            compaction_event.timestamp = max(
                compaction_event.timestamp,
                fresh_session.last_update_time + 1e-6,
            )
            try:
                await session_service.append_event(
                    session=fresh_session,
                    event=compaction_event,
                )
                return
            except StaleSessionError:
                if self._active_runs.get(key, 0) > 0:
                    self._pending_compactions.setdefault(key, []).append(
                        compaction_event
                    )
                    log_event(
                        "compaction.deferred",
                        log=self._log,
                        session_id=session_id,
                        reason="stale_session_during_active_turn",
                    )
                    return
                if attempt == 1:
                    self._pending_compactions.setdefault(key, []).append(
                        compaction_event
                    )
                    log_event(
                        "compaction.deferred",
                        level=logging.WARNING,
                        log=self._log,
                        session_id=session_id,
                        reason="stale_session_retry_exhausted",
                    )
                    return

    async def _compact_in_background(
        self,
        *,
        key: _SessionKey,
        app_name: str,
        user_id: str,
        session_id: str,
        session_service: BaseSessionService,
        events_to_compact: list[Event],
    ) -> None:
        try:
            with self._tracer.start_as_current_span(
                "fde_duckie.async_compaction"
            ) as span:
                span.set_attribute("fde_duckie.compaction.session_id", session_id)
                span.set_attribute(
                    "fde_duckie.compaction.message_count", len(events_to_compact)
                )
                span.set_attribute(
                    "fde_duckie.compaction.first_n_messages", self.first_n_messages
                )
                compaction_event = await self.summarizer.maybe_summarize_events(
                    events=events_to_compact
                )
                if compaction_event is None:
                    log_event(
                        "compaction.skipped",
                        level=logging.DEBUG,
                        log=self._log,
                        session_id=session_id,
                        message_count=len(events_to_compact),
                    )
                    return

                async with self._get_lock(key):
                    if self._active_runs.get(key, 0) > 0:
                        self._pending_compactions.setdefault(key, []).append(
                            compaction_event
                        )
                        log_event(
                            "compaction.deferred",
                            log=self._log,
                            session_id=session_id,
                            reason="active_turn_in_progress",
                        )
                    else:
                        await self._persist_to_storage(
                            key=key,
                            app_name=app_name,
                            user_id=user_id,
                            session_id=session_id,
                            session_service=session_service,
                            compaction_event=compaction_event,
                        )
                        comp = (
                            compaction_event.actions.compaction
                            if compaction_event.actions
                            else None
                        )
                        log_event(
                            "compaction.completed",
                            log=self._log,
                            session_id=session_id,
                            message_count=len(events_to_compact),
                            start_timestamp=comp.start_timestamp if comp else None,
                            end_timestamp=comp.end_timestamp if comp else None,
                        )
        except asyncio.CancelledError:
            log_event(
                "compaction.cancelled",
                level=logging.DEBUG,
                log=self._log,
                session_id=session_id,
            )
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event(
                "compaction.error",
                level=logging.WARNING,
                log=self._log,
                session_id=session_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            current = asyncio.current_task()
            if self._in_flight_tasks.get(key) is current:
                self._in_flight_tasks.pop(key, None)

    @property
    def pending_task_count(self) -> int:
        return sum(1 for task in self._in_flight_tasks.values() if not task.done())

    async def wait_for_pending(self, timeout: float | None = None) -> None:
        """Await all currently scheduled background compaction tasks."""
        tasks = [task for task in self._in_flight_tasks.values() if not task.done()]
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        del done
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        await self.wait_for_pending(timeout=self._shutdown_timeout)


def _create_default_summarizer(
    cfg: AppConfig,
    summarizer: BaseEventsSummarizer | None = None,
) -> BaseEventsSummarizer:
    return summarizer or LlmEventSummarizer(
        llm=Gemini(model=cfg.compaction_model_name),
        prompt_template=FDE_COMPACTION_PROMPT_TEMPLATE,
    )


def build_compaction_plugins(
    config: AppConfig | None = None,
    *,
    summarizer: BaseEventsSummarizer | None = None,
) -> list[BasePlugin]:
    """Build async background compaction plugin(s) from AppConfig."""
    cfg = config or get_config()
    if not cfg.compaction.enabled or not cfg.compaction.async_background:
        return []

    return [
        AsyncCompactionPlugin(
            first_n_messages=cfg.compaction.first_n_messages,
            summarizer=_create_default_summarizer(cfg, summarizer),
        )
    ]


def build_compaction_config(
    *,
    config: AppConfig | None = None,
    summarizer: BaseEventsSummarizer | None = None,
) -> EventsCompactionConfig | None:
    """Build ADK synchronous EventsCompactionConfig when async_background is disabled."""
    cfg = config or get_config()
    if not cfg.compaction.enabled or cfg.compaction.async_background:
        return None

    return EventsCompactionConfig(
        summarizer=_create_default_summarizer(cfg, summarizer),
        compaction_interval=cfg.compaction.interval,
        overlap_size=cfg.compaction.overlap_size,
        token_threshold=cfg.compaction.token_threshold,
        event_retention_size=cfg.compaction.event_retention_size,
    )
