"""Offline tests for ADK plugins: observability/OTel, PII redaction, and Model Armor."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows.context._compaction import _process_compaction_events
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from agent.agent import app as fde_app
from agent.agent import create_app
from agent.config import (
    AppConfig,
    ModelArmorSettings,
    ModelRoutingConfig,
    PiiConfig,
    SecurityConfig,
)
from agent.plugins.compaction import (
    AsyncCompactionPlugin,
    select_first_n_messages_to_compact,
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
    build_routing_plugins,
    read_complexity_from_state,
    select_model,
)
from agent.plugins.security import ModelArmorPlugin, build_security_plugins
from agent.schemas import ComplexityAssessment
from agent.state import COMPLEXITY_STATE_KEY
from agent.sub_agents.complexity_analyzer_agent import COMPLEXITY_ANALYZER_NAME
from tests.helpers import make_context


class TestObservabilityAndOTel(unittest.TestCase):
    def test_structured_json_logging_correlates_otel_trace_and_span(self) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.trace",
        )
        span_exporter = InMemorySpanExporter()
        provider = configure_otel(
            exporter="console",
            span_exporter=span_exporter,
            set_global=False,
        )
        self.assertIsNotNone(provider)
        try:
            tracer = provider.get_tracer("fde_duckie.test")
            with tracer.start_as_current_span("test_span") as span:
                expected_trace_id = format(span.get_span_context().trace_id, "032x")
                expected_span_id = format(span.get_span_context().span_id, "016x")
                log_event(
                    "tool.call",
                    log=test_logger,
                    tool_name="get_enablement_status",
                    persona_id="priya_fde",
                )
        finally:
            provider.shutdown()

        record = json.loads(buf.getvalue().strip())
        self.assertEqual(record["severity"], "INFO")
        self.assertEqual(record["event"], "tool.call")
        self.assertEqual(record["tool_name"], "get_enablement_status")
        self.assertEqual(record["persona_id"], "priya_fde")
        self.assertEqual(record["trace_id"], expected_trace_id)
        self.assertEqual(record["span_id"], expected_span_id)
        self.assertEqual(len(span_exporter.get_finished_spans()), 1)

    def test_observability_plugin_emits_structured_lifecycle_logs(self) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.plugin",
        )
        span_exporter = InMemorySpanExporter()
        provider = configure_otel(
            exporter="console",
            span_exporter=span_exporter,
            set_global=False,
        )
        plugin = ObservabilityPlugin(log=test_logger)

        async def _exercise() -> None:
            tracer = provider.get_tracer("fde_duckie.test")
            ctx = make_context("priya_fde")
            ctx.invocation_id = "inv_test_1"
            fake_agent = SimpleNamespace(name="enablement_agent")
            fake_tool = SimpleNamespace(name="get_enablement_status")

            with tracer.start_as_current_span("plugin_lifecycle"):
                await plugin.before_agent_callback(
                    agent=fake_agent, callback_context=ctx
                )
                await plugin.before_tool_callback(
                    tool=fake_tool,
                    tool_args={"track": "L300"},
                    tool_context=ctx,
                )
                await plugin.after_tool_callback(
                    tool=fake_tool,
                    tool_args={"track": "L300"},
                    tool_context=ctx,
                    result={"status": "ok"},
                )
                await plugin.on_tool_error_callback(
                    tool=fake_tool,
                    tool_args={"track": "L300"},
                    tool_context=ctx,
                    error=RuntimeError("boom"),
                )

        try:
            asyncio.run(_exercise())
        finally:
            provider.shutdown()

        log_records = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        self.assertEqual(
            [r["event"] for r in log_records],
            ["agent.start", "tool.call", "tool.result", "tool.error"],
        )
        self.assertEqual(log_records[-1]["severity"], "ERROR")
        self.assertTrue(all("trace_id" in r and "span_id" in r for r in log_records))


class TestPiiRedactionAndScrubbing(unittest.TestCase):
    def test_scrubs_common_pii_and_preserves_fde_text(self) -> None:
        scrubber = PiiScrubber(PiiConfig(enabled=True))
        sample = (
            "Contact alice@example.com or +1 (415) 555-0199, "
            "SSN 123-45-6789, card 4532-0151-1283-0366, "
            "key AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe."
        )
        scrubbed = scrubber.scrub_text(sample)
        for token in (
            "[REDACTED_EMAIL]",
            "[REDACTED_PHONE]",
            "[REDACTED_SSN]",
            "[REDACTED_CREDIT_CARD]",
            "[REDACTED_API_KEY]",
        ):
            self.assertIn(token, scrubbed)
        for secret in (
            "alice@example.com",
            "555-0199",
            "123-45-6789",
            "4532-0151-1283-0366",
            "AIzaSyDaGmWKa4JsXZ",
        ):
            self.assertNotIn(secret, scrubbed)

        safe = "Priya Nair (ldap=priyafde, manager=carlos_mendez@, Day 42, L300 4/7, go/fde-enablement-dash)"
        self.assertEqual(scrubber.scrub_text(safe), safe)

    def test_pii_redaction_plugin_hooks(self) -> None:
        plugin = PiiRedactionPlugin(config=PiiConfig(enabled=True))

        async def _exercise() -> None:
            ctx = make_context("alex_noogler")
            fake_tool = SimpleNamespace(name="query_duckie")

            user_msg = types.Content(
                role="user",
                parts=[types.Part.from_text(text="Email alex@personal.org SSN 123-45-6789")],
            )
            scrubbed_user = await plugin.on_user_message_callback(
                invocation_context=ctx, user_message=user_msg
            )
            self.assertIsNotNone(scrubbed_user)
            self.assertEqual(
                scrubbed_user.parts[0].text,
                "Email [REDACTED_EMAIL] SSN [REDACTED_SSN]",
            )

            tool_result = await plugin.after_tool_callback(
                tool=fake_tool,
                tool_args={},
                tool_context=ctx,
                result={"status": "ok", "answer": "Reach hr@external.org"},
            )
            self.assertEqual(tool_result, {"status": "ok", "answer": "Reach [REDACTED_EMAIL]"})

            llm_resp = LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Call 415-555-0142")],
                )
            )
            scrubbed_resp = await plugin.after_model_callback(
                callback_context=ctx, llm_response=llm_resp
            )
            self.assertEqual(scrubbed_resp.content.parts[0].text, "Call [REDACTED_PHONE]")

            ev = Event(
                author="fde_duckie",
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Key AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe")],
                ),
            )
            scrubbed_ev = await plugin.on_event_callback(
                invocation_context=ctx, event=ev
            )
            self.assertEqual(scrubbed_ev.content.parts[0].text, "Key [REDACTED_API_KEY]")

        asyncio.run(_exercise())

    def test_structured_logs_scrub_pii(self) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            AppConfig(pii=PiiConfig(enabled=True)),
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.pii_logs",
        )
        log_event("user.note", log=test_logger, note="Email user@example.com SSN 123-45-6789")
        try:
            raise RuntimeError("Key AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe")
        except RuntimeError:
            test_logger.exception("Phone 415-555-0199")

        raw = buf.getvalue()
        for secret in ("user@example.com", "123-45-6789", "AIzaSyDaGmWKa4JsXZ", "415-555-0199"):
            self.assertNotIn(secret, raw)

    def test_app_registers_pii_plugin_when_enabled_and_skips_when_disabled(self) -> None:
        self.assertTrue(any(isinstance(p, PiiRedactionPlugin) for p in fde_app.plugins))
        disabled_app = create_app(AppConfig(pii=PiiConfig(enabled=False)))
        self.assertFalse(any(isinstance(p, PiiRedactionPlugin) for p in disabled_app.plugins))
        self.assertTrue(any(isinstance(p, ObservabilityPlugin) for p in disabled_app.plugins))


class TestSecurityPlugins(unittest.TestCase):
    def test_app_registers_opt_in_model_armor_only_when_enabled(self) -> None:
        self.assertFalse(
            any(isinstance(p, ModelArmorPlugin) for p in fde_app.plugins)
        )
        self.assertEqual(build_security_plugins(AppConfig()), [])

        armor_cfg = AppConfig(
            security=SecurityConfig(
                model_armor=ModelArmorSettings(
                    enabled=True,
                    prompt_template_name="projects/demo/locations/us-central1/templates/prompt-t1",
                    response_template_name="projects/demo/locations/us-central1/templates/resp-t1",
                ),
            )
        )
        armor_app = create_app(armor_cfg)
        self.assertTrue(
            any(isinstance(p, ModelArmorPlugin) for p in armor_app.plugins)
        )

    def test_model_armor_config_requires_template_when_enabled(self) -> None:
        with self.assertRaises(ValidationError):
            AppConfig(
                security=SecurityConfig(
                    model_armor=ModelArmorSettings(enabled=True)
                )
            )

    def test_model_armor_plugin_attaches_config_and_handles_blocked_response(self) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.model_armor",
        )
        plugin = ModelArmorPlugin(
            prompt_template_name="projects/demo/locations/us-central1/templates/p1",
            response_template_name="projects/demo/locations/us-central1/templates/r1",
            log=test_logger,
        )

        async def _exercise() -> None:
            ctx = make_context("priya_fde")
            req = LlmRequest(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="Check my L300 status")],
                    )
                ]
            )
            res = await plugin.before_model_callback(
                callback_context=ctx, llm_request=req
            )
            self.assertIsNone(res)
            self.assertIsNotNone(req.config)
            self.assertIsNotNone(req.config.model_armor_config)
            self.assertEqual(
                req.config.model_armor_config.prompt_template_name,
                "projects/demo/locations/us-central1/templates/p1",
            )
            self.assertEqual(
                req.config.model_armor_config.response_template_name,
                "projects/demo/locations/us-central1/templates/r1",
            )

            normal_resp = LlmResponse(
                finish_reason=types.FinishReason.STOP,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="All good!")],
                ),
            )
            self.assertIsNone(
                await plugin.after_model_callback(
                    callback_context=ctx, llm_response=normal_resp
                )
            )

            blocked_resp = LlmResponse(
                finish_reason=types.FinishReason.PROHIBITED_CONTENT,
            )
            fallback = await plugin.after_model_callback(
                callback_context=ctx, llm_response=blocked_resp
            )
            self.assertIsNotNone(fallback)
            self.assertIn("Model Armor", fallback.content.parts[0].text)

        asyncio.run(_exercise())

        events = [json.loads(line)["event"] for line in buf.getvalue().splitlines() if line]
        self.assertEqual(
            events,
            ["security.model_armor.attached", "security.model_armor.blocked"],
        )


class TestStrategicModelRouting(unittest.TestCase):
    def _user_request(self, text: str, model: str = "gemini-2.5-flash-lite") -> LlmRequest:
        return LlmRequest(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            ],
        )

    def test_read_complexity_from_state_handles_missing_dict_model_and_string(self) -> None:
        self.assertEqual(read_complexity_from_state(None), ("simple", None))
        self.assertEqual(read_complexity_from_state({}), ("simple", None))
        self.assertEqual(
            read_complexity_from_state({COMPLEXITY_STATE_KEY: "complex"}),
            ("complex", None),
        )
        self.assertEqual(
            read_complexity_from_state(
                {COMPLEXITY_STATE_KEY: {"complexity": "simple", "reason": "single lookup"}}
            ),
            ("simple", "single lookup"),
        )
        self.assertEqual(
            read_complexity_from_state(
                {
                    COMPLEXITY_STATE_KEY: ComplexityAssessment(
                        complexity="complex", reason="multi-domain query"
                    )
                }
            ),
            ("complex", "multi-domain query"),
        )

    def test_select_model_routes_by_state_and_pins_complexity_analyzer(self) -> None:
        cfg = ModelRoutingConfig(capable_model="gemini-2.5-flash")

        d_simple = select_model(
            {COMPLEXITY_STATE_KEY: {"complexity": "simple", "reason": "greeting"}},
            fast_model="gemini-2.5-flash-lite",
            routing_config=cfg,
            agent_name="fde_duckie_coordinator",
        )
        self.assertEqual(d_simple.model, "gemini-2.5-flash-lite")
        self.assertEqual(d_simple.tier, "fast")

        complex_state = {
            COMPLEXITY_STATE_KEY: {
                "complexity": "complex",
                "reason": "spans enablement and general Duckie",
            }
        }
        d_complex = select_model(
            complex_state,
            fast_model="gemini-2.5-flash-lite",
            routing_config=cfg,
            agent_name="fde_duckie_coordinator",
        )
        self.assertEqual(d_complex.model, "gemini-2.5-flash")
        self.assertEqual(d_complex.tier, "capable")
        self.assertEqual(d_complex.reason, "spans enablement and general Duckie")

        # Step-1 complexity analyzer always stays on Flash-Lite even if prior turn was complex.
        d_analyzer = select_model(
            complex_state,
            fast_model="gemini-2.5-flash-lite",
            routing_config=cfg,
            agent_name=COMPLEXITY_ANALYZER_NAME,
        )
        self.assertEqual(d_analyzer.model, "gemini-2.5-flash-lite")
        self.assertEqual(d_analyzer.tier, "fast")
        self.assertEqual(d_analyzer.reason, "complexity_analyzer_always_fast")

    def test_plugin_mutates_llm_request_logs_and_sets_otel_attributes(self) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.routing",
        )
        span_exporter = InMemorySpanExporter()
        provider = configure_otel(
            exporter="console",
            span_exporter=span_exporter,
            set_global=False,
        )
        plugin = ModelRoutingPlugin(
            fast_model="gemini-2.5-flash-lite",
            config=ModelRoutingConfig(capable_model="gemini-2.5-flash"),
            log=test_logger,
        )

        async def _exercise() -> None:
            tracer = provider.get_tracer("fde_duckie.test.routing")
            ctx = make_context("alex_noogler")
            ctx.agent_name = "fde_duckie_coordinator"

            ctx.state[COMPLEXITY_STATE_KEY] = {
                "complexity": "simple",
                "reason": "single eligibility lookup",
            }
            simple_req = self._user_request("Can I register for L400 Bootcamp yet?")
            with tracer.start_as_current_span("simple_call"):
                res = await plugin.before_model_callback(
                    callback_context=ctx, llm_request=simple_req
                )
            self.assertIsNone(res)
            self.assertEqual(simple_req.model, "gemini-2.5-flash-lite")

            ctx.state[COMPLEXITY_STATE_KEY] = {
                "complexity": "complex",
                "reason": "multi-domain request",
            }
            complex_req = self._user_request(
                "Check my L200 status and how do I apply for a corporate credit card?"
            )
            with tracer.start_as_current_span("complex_call"):
                res = await plugin.before_model_callback(
                    callback_context=ctx, llm_request=complex_req
                )
            self.assertIsNone(res)
            self.assertEqual(complex_req.model, "gemini-2.5-flash")

            # Analyzer stays on Flash-Lite even with complex state left from prior turn.
            ctx.agent_name = COMPLEXITY_ANALYZER_NAME
            analyzer_req = self._user_request("Next turn prompt")
            with tracer.start_as_current_span("analyzer_call"):
                await plugin.before_model_callback(
                    callback_context=ctx, llm_request=analyzer_req
                )
            self.assertEqual(analyzer_req.model, "gemini-2.5-flash-lite")

        try:
            asyncio.run(_exercise())
        finally:
            provider.shutdown()

        records = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        self.assertEqual(
            [r["event"] for r in records],
            ["model.routed", "model.routed", "model.routed"],
        )
        self.assertEqual(records[0]["severity"], "DEBUG")
        self.assertEqual(records[0]["selected_model"], "gemini-2.5-flash-lite")
        self.assertEqual(records[1]["severity"], "INFO")
        self.assertEqual(records[1]["selected_model"], "gemini-2.5-flash")
        self.assertEqual(records[1]["tier"], "capable")
        self.assertEqual(records[2]["selected_model"], "gemini-2.5-flash-lite")

        spans = span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0].attributes["fde_duckie.model_routing.tier"], "fast")
        self.assertEqual(spans[1].attributes["fde_duckie.model_routing.tier"], "capable")
        self.assertEqual(
            spans[1].attributes["fde_duckie.model_routing.model"], "gemini-2.5-flash"
        )
        self.assertEqual(spans[2].attributes["fde_duckie.model_routing.tier"], "fast")

    def test_app_registers_routing_plugin_when_enabled_and_skips_when_disabled(self) -> None:
        self.assertTrue(
            any(isinstance(p, ModelRoutingPlugin) for p in fde_app.plugins)
        )
        self.assertEqual(
            build_routing_plugins(AppConfig(model_routing=ModelRoutingConfig(enabled=False))),
            [],
        )
        disabled_app = create_app(
            AppConfig(model_routing=ModelRoutingConfig(enabled=False))
        )
        self.assertFalse(
            any(isinstance(p, ModelRoutingPlugin) for p in disabled_app.plugins)
        )


class _FakeAsyncSummarizer(BaseEventsSummarizer):
    def __init__(
        self,
        *,
        gate: asyncio.Event | None = None,
        started: asyncio.Event | None = None,
        fail: bool = False,
    ) -> None:
        self.gate = gate
        self.started = started
        self.fail = fail
        self.calls: list[list[Event]] = []

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        self.calls.append(list(events))
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("summarizer boom")
        if not events:
            return None
        return Event(
            invocation_id=Event.new_id(),
            author="user",
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=f"Summary of {len(events)} messages"
                            )
                        ],
                    ),
                )
            ),
        )


def _msg_event(
    text: str,
    *,
    ts: float,
    author: str = "user",
    role: str = "user",
    inv_id: str = "inv_1",
) -> Event:
    return Event(
        invocation_id=inv_id,
        author=author,
        timestamp=ts,
        content=types.Content(
            role=role,
            parts=[types.Part.from_text(text=text)],
        ),
    )


class TestAsyncBackgroundCompaction(unittest.TestCase):
    def test_select_first_n_messages_respects_threshold_prior_compaction_and_tool_pairs(
        self,
    ) -> None:
        events = [
            _msg_event(f"msg {i}", ts=100.0 + i, inv_id=f"inv_{i}")
            for i in range(6)
        ]
        # Fewer than N -> nothing to compact.
        self.assertEqual(select_first_n_messages_to_compact(events[:3], 4), [])

        # >= N -> exact first N messages.
        selected = select_first_n_messages_to_compact(events, 4)
        self.assertEqual([e.timestamp for e in selected], [100.0, 101.0, 102.0, 103.0])

        # After a compaction covering [100.0, 103.0], only 2 uncompacted messages remain (< 4).
        compaction = Event(
            invocation_id="comp_1",
            author="user",
            timestamp=106.0,
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=100.0,
                    end_timestamp=103.0,
                    compacted_content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="summary")],
                    ),
                )
            ),
        )
        self.assertEqual(
            select_first_n_messages_to_compact(events + [compaction], 4), []
        )

        # Tool call at index 3 whose response is at index 4 must not be split across the boundary.
        tool_events = [
            _msg_event("u1", ts=1.0, inv_id="inv_1"),
            _msg_event("m1", ts=2.0, author="fde_duckie", role="model", inv_id="inv_1"),
            _msg_event("u2", ts=3.0, inv_id="inv_2"),
            Event(
                invocation_id="inv_2",
                author="fde_duckie",
                timestamp=4.0,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call_1", name="get_enablement_status", args={}
                            )
                        )
                    ],
                ),
            ),
            Event(
                invocation_id="inv_2",
                author="fde_duckie",
                timestamp=5.0,
                content=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="call_1",
                                name="get_enablement_status",
                                response={"status": "ok"},
                            )
                        )
                    ],
                ),
            ),
        ]
        safe_prefix = select_first_n_messages_to_compact(tool_events, 4)
        self.assertEqual([e.timestamp for e in safe_prefix], [1.0, 2.0, 3.0])

    def test_async_compaction_runs_in_background_without_blocking_turn(self) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.async_compaction",
        )

        async def _exercise() -> None:
            gate = asyncio.Event()
            started = asyncio.Event()
            summarizer = _FakeAsyncSummarizer(gate=gate, started=started)
            plugin = AsyncCompactionPlugin(
                first_n_messages=4,
                summarizer=summarizer,
                log=test_logger,
            )
            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name="agent", user_id="alex_noogler", session_id="s1"
            )
            for i in range(6):
                await session_service.append_event(
                    session,
                    _msg_event(
                        f"message {i}",
                        ts=1000.0 + i,
                        author="user" if i % 2 == 0 else "fde_duckie_coordinator",
                        role="user" if i % 2 == 0 else "model",
                        inv_id=f"inv_{i // 2}",
                    ),
                )

            ctx = SimpleNamespace(
                session=session,
                session_service=session_service,
                plugin_manager=SimpleNamespace(_skip_closing_plugins=False),
            )
            await plugin.before_run_callback(invocation_context=ctx)
            # after_run_callback must return immediately while summarizer is still blocked!
            await asyncio.wait_for(
                plugin.after_run_callback(invocation_context=ctx), timeout=0.5
            )
            await asyncio.wait_for(started.wait(), timeout=0.5)
            self.assertEqual(plugin.pending_task_count, 1)

            # Release the background summarizer and wait for completion.
            gate.set()
            await plugin.wait_for_pending(timeout=2.0)
            self.assertEqual(plugin.pending_task_count, 0)

            self.assertEqual(len(summarizer.calls), 1)
            self.assertEqual(
                [e.timestamp for e in summarizer.calls[0]],
                [1000.0, 1001.0, 1002.0, 1003.0],
            )

            updated = await session_service.get_session(
                app_name="agent", user_id="alex_noogler", session_id="s1"
            )
            self.assertIsNotNone(updated)
            compaction_events = [
                e for e in updated.events if e.actions and e.actions.compaction
            ]
            self.assertEqual(len(compaction_events), 1)
            comp = compaction_events[0].actions.compaction
            self.assertEqual(comp.start_timestamp, 1000.0)
            self.assertEqual(comp.end_timestamp, 1003.0)

            # ADK's prompt assembler replaces the first 4 messages with the summary
            # while keeping messages 4 and 5 raw.
            effective = _process_compaction_events(updated.events)
            texts = [
                e.content.parts[0].text
                for e in effective
                if e.content and e.content.parts
            ]
            self.assertEqual(
                texts, ["Summary of 4 messages", "message 4", "message 5"]
            )

        asyncio.run(_exercise())

        events = [json.loads(line)["event"] for line in buf.getvalue().splitlines() if line]
        self.assertEqual(events, ["compaction.scheduled", "compaction.completed"])

    def test_async_compaction_defers_and_flushes_on_sqlite_during_active_turn(
        self,
    ) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.async_compaction_sqlite",
        )

        async def _exercise() -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "compaction_test.db"
                session_service = SqliteSessionService(db_path=str(db_path))
                try:
                    gate = asyncio.Event()
                    started = asyncio.Event()
                    summarizer = _FakeAsyncSummarizer(gate=gate, started=started)
                    plugin = AsyncCompactionPlugin(
                        first_n_messages=4,
                        summarizer=summarizer,
                        log=test_logger,
                    )
                    session = await session_service.create_session(
                        app_name="agent", user_id="priya_fde", session_id="s_sql"
                    )
                    for i in range(4):
                        await session_service.append_event(
                            session,
                            _msg_event(f"turn1 msg {i}", ts=2000.0 + i, inv_id="inv_1"),
                        )

                    # Turn 1 completes and kicks off background compaction.
                    ctx1 = SimpleNamespace(
                        session=session,
                        session_service=session_service,
                        plugin_manager=SimpleNamespace(_skip_closing_plugins=False),
                    )
                    await plugin.before_run_callback(invocation_context=ctx1)
                    await plugin.after_run_callback(invocation_context=ctx1)
                    await asyncio.wait_for(started.wait(), timeout=0.5)

                    # Turn 2 begins while compaction is still running.
                    turn2_session = await session_service.get_session(
                        app_name="agent", user_id="priya_fde", session_id="s_sql"
                    )
                    ctx2 = SimpleNamespace(
                        session=turn2_session,
                        session_service=session_service,
                        plugin_manager=SimpleNamespace(_skip_closing_plugins=False),
                    )
                    await plugin.before_run_callback(invocation_context=ctx2)

                    # Background summarizer finishes mid-turn -> must defer instead of causing StaleSessionError.
                    gate.set()
                    await plugin.wait_for_pending(timeout=2.0)

                    # Turn 2 appends its own event cleanly with no StaleSessionError.
                    await session_service.append_event(
                        turn2_session,
                        _msg_event("turn2 reply", ts=2010.0, inv_id="inv_2"),
                    )

                    # Turn 2 ends -> deferred compaction is flushed cleanly to SQLite.
                    await plugin.after_run_callback(invocation_context=ctx2)
                    await plugin.close()

                    persisted = await session_service.get_session(
                        app_name="agent", user_id="priya_fde", session_id="s_sql"
                    )
                    self.assertIsNotNone(persisted)
                    compaction_events = [
                        e
                        for e in persisted.events
                        if e.actions and e.actions.compaction
                    ]
                    self.assertEqual(len(compaction_events), 1)
                    self.assertEqual(
                        compaction_events[0].actions.compaction.start_timestamp, 2000.0
                    )
                    self.assertEqual(
                        compaction_events[0].actions.compaction.end_timestamp, 2003.0
                    )
                finally:
                    await session_service.close()

        asyncio.run(_exercise())

        events = [json.loads(line)["event"] for line in buf.getvalue().splitlines() if line]
        self.assertIn("compaction.deferred", events)
        self.assertIn("compaction.flushed", events)

    def test_async_compaction_handles_errors_and_ignores_nested_ephemeral_runs(
        self,
    ) -> None:
        buf = io.StringIO()
        test_logger = configure_logging(
            level="DEBUG",
            stream=buf,
            logger_name="fde_duckie.test.async_compaction_err",
        )

        async def _exercise() -> None:
            summarizer = _FakeAsyncSummarizer(fail=True)
            plugin = AsyncCompactionPlugin(
                first_n_messages=2,
                summarizer=summarizer,
                log=test_logger,
            )
            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name="agent", user_id="alex_noogler", session_id="s_err"
            )
            for i in range(2):
                await session_service.append_event(
                    session, _msg_event(f"m{i}", ts=3000.0 + i, inv_id="inv_1")
                )

            # Nested ephemeral AgentTool run must be ignored completely.
            nested_ctx = SimpleNamespace(
                session=session,
                session_service=session_service,
                plugin_manager=SimpleNamespace(_skip_closing_plugins=True),
            )
            await plugin.before_run_callback(invocation_context=nested_ctx)
            await plugin.after_run_callback(invocation_context=nested_ctx)
            self.assertEqual(plugin.pending_task_count, 0)
            self.assertEqual(len(summarizer.calls), 0)

            # Real run whose summarizer raises must not crash and must clean up in-flight state.
            real_ctx = SimpleNamespace(
                session=session,
                session_service=session_service,
                plugin_manager=SimpleNamespace(_skip_closing_plugins=False),
            )
            await plugin.before_run_callback(invocation_context=real_ctx)
            await plugin.after_run_callback(invocation_context=real_ctx)
            await plugin.wait_for_pending(timeout=2.0)
            self.assertEqual(plugin.pending_task_count, 0)
            self.assertEqual(len(summarizer.calls), 1)

        asyncio.run(_exercise())

        records = [json.loads(line) for line in buf.getvalue().splitlines() if line]
        self.assertEqual(
            [r["event"] for r in records],
            ["compaction.scheduled", "compaction.error"],
        )
        self.assertEqual(records[-1]["severity"], "WARNING")


if __name__ == "__main__":
    unittest.main()
