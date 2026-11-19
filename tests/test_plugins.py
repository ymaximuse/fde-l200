"""Offline tests for ADK plugins: observability/OTel, PII redaction, and Model Armor."""

import asyncio
import io
import json
import unittest
from types import SimpleNamespace

from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
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


if __name__ == "__main__":
    unittest.main()
