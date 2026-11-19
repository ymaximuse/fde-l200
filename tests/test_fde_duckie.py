"""Offline deterministic and ADK graph integrity tests for FDE Duckie."""

import asyncio
import inspect
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from google.adk.apps import App
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.compaction import _run_compaction_for_sliding_window
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools import AgentTool, FunctionTool
from google.adk.tools import _function_tool_declarations as ftd
from google.genai import types
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import TypeAdapter, ValidationError

from agent.agent import app as fde_app
from agent.agent import create_app, root_agent
from agent.compaction import (
    FDE_COMPACTION_PROMPT_TEMPLATE,
    build_app,
    build_compaction_config,
)
from agent.db import (
    SqliteNooglerProfileStore,
    configure_storage,
    create_session_service,
)
from agent.observability import (
    ObservabilityPlugin,
    configure_logging,
    configure_otel,
    log_event,
)
from agent.backends import (
    get_duckie_client,
    get_knowledge_base,
    get_profile_store,
    set_duckie_client,
    set_profile_store,
)
from agent.config import (
    AppConfig,
    CompactionConfig,
    DatabaseConfig,
    LoggingConfig,
    ModelConfig,
    OtelConfig,
    SecretsConfig,
    get_config,
    load_config,
)
from agent.secrets import (
    EnvSecretManager,
    VaultSecretManager,
    create_secret_manager,
    hydrate_runtime_secrets,
)
from scripts.setup_db import seed_local_db
from agent.mocks import (
    MockFdeKnowledgeBase,
    MockNooglerProfileStore,
    StubDuckieClient,
)
from agent.schemas import DuckieResponse
from agent.state import (
    PERSONA_ID_STATE_KEY,
    PROFILE_STATE_KEY,
    SUMMARY_STATE_KEY,
    load_noogler_profile,
    switch_noogler_persona,
)
from agent.sub_agents import (
    duckie_proxy_agent,
    enablement_agent,
    onboarding_guide_agent,
)
from agent.tools.duckie_tools import query_duckie
from agent.tools.enablement_tools import (
    check_l400_eligibility,
    get_enablement_status,
    update_enablement_progress,
)
from agent.tools.onboarding_tools import (
    get_next_recommended_actions,
    get_setup_checklist,
    get_team_contacts,
)


def _make_context(persona_id: str | None = None) -> SimpleNamespace:
    state: dict = {}
    if persona_id is not None:
        state[PERSONA_ID_STATE_KEY] = persona_id
    ctx = SimpleNamespace(state=state)
    load_noogler_profile(ctx)
    return ctx


class TestSessionStateAndPersonas(unittest.TestCase):
    def test_before_agent_callback_loads_default_profile(self) -> None:
        ctx = _make_context()
        self.assertIn(PROFILE_STATE_KEY, ctx.state)
        self.assertIn(SUMMARY_STATE_KEY, ctx.state)
        self.assertEqual(ctx.state[PROFILE_STATE_KEY]["persona_id"], "alex_noogler")
        self.assertIn("Alex Rivera", ctx.state[SUMMARY_STATE_KEY])

    def test_switch_noogler_persona_updates_state(self) -> None:
        ctx = _make_context("alex_noogler")
        res = switch_noogler_persona("priya_fde", ctx)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(ctx.state[PROFILE_STATE_KEY]["persona_id"], "priya_fde")
        self.assertIn("Priya Nair", ctx.state[SUMMARY_STATE_KEY])

    def test_switch_unknown_persona_returns_error(self) -> None:
        ctx = _make_context("alex_noogler")
        res = switch_noogler_persona("does_not_exist", ctx)
        self.assertEqual(res["status"], "error")
        self.assertTrue(len(res["available_personas"]) >= 3)


class TestEnablementTools(unittest.TestCase):
    def test_priya_l400_blocked_until_six_l300_labs(self) -> None:
        ctx = _make_context("priya_fde")
        elig = check_l400_eligibility(ctx)
        self.assertEqual(elig["status"], "ok")
        self.assertFalse(elig["eligible_for_l400"])
        self.assertEqual(elig["l300_labs_completed"], 4)
        self.assertEqual(elig["l300_labs_remaining"], 2)

        # Attempting to complete L400 directly must be rejected while blocked.
        blocked = update_enablement_progress("L400", "completed", tool_context=ctx)
        self.assertEqual(blocked["status"], "blocked")

        # Updating L300 to 6 labs marks L300 completed and automatically unblocks L400.
        updated = update_enablement_progress(
            "L300", "in_progress", labs_completed=6, tool_context=ctx
        )
        self.assertEqual(updated["status"], "ok")
        self.assertEqual(updated["updated_track"]["status"], "completed")
        self.assertIn("unlocked", updated)

        elig_after = check_l400_eligibility(ctx)
        self.assertTrue(elig_after["eligible_for_l400"])
        self.assertEqual(elig_after["l400_status"], "not_started")

    def test_jordan_pca_is_at_risk_past_day_60(self) -> None:
        ctx = _make_context("jordan_fde")
        status = get_enablement_status("PCA", tool_context=ctx)
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["core_progress"]["completed"], 3)
        self.assertEqual(status["tracks"][0]["pacing"], "at_risk")

    def test_all_tracks_includes_dashboard_sync_note(self) -> None:
        ctx = _make_context("alex_noogler")
        status = get_enablement_status("all", tool_context=ctx)
        self.assertEqual(status["status"], "ok")
        self.assertEqual(len(status["tracks"]), 6)
        self.assertEqual(
            status["dashboard_sync"]["dashboard_url"], "go/fde-enablement-dash"
        )


class TestOnboardingGuideTools(unittest.TestCase):
    def test_day_12_recommendations_prioritize_setup_and_l200(self) -> None:
        ctx = _make_context("alex_noogler")
        recs = get_next_recommended_actions(ctx)
        self.assertEqual(recs["status"], "ok")
        self.assertEqual(len(recs["recommended_actions"]), 3)
        categories = [r["category"] for r in recs["recommended_actions"]]
        self.assertEqual(categories[:2], ["setup", "setup"])
        self.assertEqual(categories[2], "enablement")

    def test_day_42_recommendations_prioritize_l300(self) -> None:
        ctx = _make_context("priya_fde")
        recs = get_next_recommended_actions(ctx)
        self.assertEqual(recs["status"], "ok")
        self.assertIn("L300", recs["recommended_actions"][0]["action"])

    def test_setup_checklist_and_team_contacts(self) -> None:
        ctx = _make_context("alex_noogler")
        argolis = get_setup_checklist("argolis", tool_context=ctx)
        self.assertEqual(argolis["status"], "ok")
        self.assertIn("argolis", argolis["guides"])
        self.assertEqual(len(argolis["personal_setup_status"]), 1)

        contacts = get_team_contacts(ctx)
        self.assertEqual(contacts["status"], "ok")
        self.assertIn("maya_chen", contacts["noogler"]["manager"])
        self.assertTrue(len(contacts["chat_spaces"]) >= 2)


class TestDuckieStub(unittest.TestCase):
    def tearDown(self) -> None:
        set_duckie_client(StubDuckieClient())

    def test_known_general_topic_returns_stub_with_references(self) -> None:
        res = query_duckie("How do I enroll in health benefits on gBenefits?")
        self.assertTrue(res["is_mock"])
        self.assertEqual(res["source"], "duckie_stub")
        self.assertEqual(res["matched_topic"], "benefits")
        self.assertIn("go/gbenefits", res["references"])

    def test_unknown_general_topic_returns_clean_fallback(self) -> None:
        res = query_duckie("Where is the best coffee bar in Building 43?")
        self.assertTrue(res["is_mock"])
        self.assertEqual(res["matched_topic"], "general_fallback")
        self.assertIn("go/duckie", res["references"])

    def test_duckie_client_is_swappable(self) -> None:
        class CustomClient:
            def ask(self, question: str) -> DuckieResponse:
                return DuckieResponse(
                    source="custom_mock",
                    is_mock=False,
                    matched_topic="custom",
                    answer=f"Custom answer for: {question}",
                    references=["go/custom"],
                )

        set_duckie_client(CustomClient())
        res = query_duckie("test")
        self.assertEqual(res["source"], "custom_mock")
        self.assertFalse(res["is_mock"])

    def test_all_mocks_are_isolated_under_agent_mocks(self) -> None:
        self.assertIsInstance(get_duckie_client(), StubDuckieClient)
        self.assertIsInstance(get_profile_store(), MockNooglerProfileStore)
        self.assertIsInstance(get_knowledge_base(), MockFdeKnowledgeBase)

        agent_dir = Path(__file__).resolve().parent.parent / "agent"
        self.assertFalse((agent_dir / "data").exists())
        self.assertTrue((agent_dir / "mocks" / "data" / "noogler_profiles.json").exists())
        self.assertTrue((agent_dir / "mocks" / "data" / "fde_kb.json").exists())


class TestADKAgentGraphIntegrity(unittest.TestCase):
    def test_root_and_specialists_meet_flash_lite_constraints(self) -> None:
        expected_model = get_config().model.name
        self.assertEqual(root_agent.name, "fde_duckie")
        self.assertEqual(root_agent.model, expected_model)
        self.assertIsNotNone(root_agent.before_agent_callback)

        # Strict Flash-Lite tool budget: <= 4 tools on root and every specialist.
        self.assertLessEqual(len(root_agent.tools), 4)
        agent_tools = [t for t in root_agent.tools if isinstance(t, AgentTool)]
        wrapped_names = {t.agent.name for t in agent_tools}
        self.assertEqual(
            wrapped_names,
            {"enablement_agent", "onboarding_guide_agent", "duckie_proxy_agent"},
        )

        for specialist in (enablement_agent, onboarding_guide_agent, duckie_proxy_agent):
            self.assertEqual(specialist.model, expected_model)
            self.assertLessEqual(len(specialist.tools), 4)

    def test_tool_declarations_have_well_defined_json_schemas(self) -> None:
        def _assert_all_props_described(schema: dict, label: str) -> None:
            self.assertEqual(schema.get("type"), "object", f"{label}: expected object schema")
            props = schema.get("properties", {})
            self.assertTrue(props, f"{label}: expected non-empty properties")
            self.assertNotIn("tool_context", props, f"{label}: tool_context leaked")
            for prop_name, prop_schema in props.items():
                self.assertTrue(
                    prop_schema.get("description"),
                    f"{label}.{prop_name}: missing description in JSON schema",
                )
            for def_name, def_schema in schema.get("$defs", {}).items():
                if def_schema.get("type") == "object":
                    _assert_all_props_described(def_schema, f"{label}.$defs.{def_name}")

        for ag in (
            root_agent,
            enablement_agent,
            onboarding_guide_agent,
            duckie_proxy_agent,
        ):
            for tool in ag.tools:
                if isinstance(tool, AgentTool):
                    decl = tool._get_declaration()
                    self.assertIsNotNone(decl)
                    self.assertTrue(decl.description, f"{decl.name}: missing description")
                    self.assertIsNotNone(decl.parameters_json_schema)
                    _assert_all_props_described(
                        decl.parameters_json_schema, f"{ag.name}.{decl.name}.params"
                    )
                    continue

                wrapped = FunctionTool(tool) if callable(tool) else tool
                decl = wrapped._get_declaration()
                self.assertIsNotNone(decl)
                self.assertTrue(decl.description, f"{decl.name}: missing description")

                # Check parameter JSON schema whenever the function exposes LLM parameters.
                llm_params = [
                    name
                    for name in inspect.signature(tool).parameters
                    if name != "tool_context"
                ]
                if llm_params:
                    self.assertIsNotNone(
                        decl.parameters_json_schema,
                        f"{decl.name}: missing parameters_json_schema",
                    )
                    _assert_all_props_described(
                        decl.parameters_json_schema, f"{ag.name}.{decl.name}.params"
                    )
                else:
                    self.assertIsNone(decl.parameters_json_schema)

                # Verify full Pydantic response JSON schema directly via ADK's builder.
                full_decl = ftd.build_function_declaration_with_json_schema(
                    tool, ignore_params=["tool_context"]
                )
                self.assertIsNotNone(
                    full_decl.response_json_schema,
                    f"{decl.name}: missing response_json_schema",
                )
                _assert_all_props_described(
                    full_decl.response_json_schema, f"{ag.name}.{decl.name}.response"
                )

    def test_runtime_tool_outputs_conform_to_response_schemas(self) -> None:
        ctx = _make_context("priya_fde")
        checks = [
            (switch_noogler_persona, switch_noogler_persona("priya_fde", ctx)),
            (get_enablement_status, get_enablement_status("all", tool_context=ctx)),
            (check_l400_eligibility, check_l400_eligibility(ctx)),
            (
                update_enablement_progress,
                update_enablement_progress("L300", "in_progress", labs_completed=5, tool_context=ctx),
            ),
            (get_next_recommended_actions, get_next_recommended_actions(ctx)),
            (get_setup_checklist, get_setup_checklist("all", tool_context=ctx)),
            (get_team_contacts, get_team_contacts(ctx)),
            (query_duckie, query_duckie("How do I enroll in gBenefits?")),
        ]
        for func, result in checks:
            self.assertIsInstance(result, dict)
            return_type = func.__annotations__["return"]
            TypeAdapter(return_type).validate_python(result)


class _FakeSummarizer(BaseEventsSummarizer):
    def __init__(self) -> None:
        self.calls: list[list[Event]] = []

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        self.calls.append(list(events))
        return Event(
            author="user",
            invocation_id=Event.new_id(),
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=f"Compacted summary of {len(events)} events."
                            )
                        ],
                    ),
                )
            ),
        )


class TestContextCompaction(unittest.TestCase):
    def test_app_exposes_flash_lite_compaction_config(self) -> None:
        self.assertIsInstance(fde_app, App)
        self.assertEqual(fde_app.name, "agent")
        self.assertIs(fde_app.root_agent, root_agent)

        cfg = fde_app.events_compaction_config
        self.assertIsNotNone(cfg)
        self.assertIsInstance(cfg.summarizer, LlmEventSummarizer)
        self.assertGreater(cfg.compaction_interval, 0)
        self.assertGreaterEqual(cfg.overlap_size, 0)
        self.assertGreater(cfg.token_threshold, 0)
        self.assertGreaterEqual(cfg.event_retention_size, 0)
        self.assertIn("{conversation_history}", FDE_COMPACTION_PROMPT_TEMPLATE)

    def test_compaction_can_be_disabled(self) -> None:
        disabled_cfg = AppConfig(compaction=CompactionConfig(enabled=False))
        self.assertIsNone(build_compaction_config(config=disabled_cfg))
        disabled_app = build_app(
            root_agent=root_agent, name="agent", config=disabled_cfg
        )
        self.assertIsNone(disabled_app.events_compaction_config)

    def test_adk_agent_loader_discovers_app_with_compaction(self) -> None:
        project_root = str(Path(__file__).resolve().parent.parent)
        loaded = AgentLoader(agents_dir=project_root).load_agent("agent")
        self.assertIsInstance(loaded, App)
        self.assertIsNotNone(loaded.events_compaction_config)

    def test_sliding_window_compaction_appends_compaction_event(self) -> None:
        async def _run() -> None:
            fake_summarizer = _FakeSummarizer()
            comp_cfg = AppConfig(
                compaction=CompactionConfig(
                    interval=2,
                    overlap_size=1,
                    token_threshold=8000,
                    event_retention_size=2,
                )
            )
            test_app = App(
                name="agent",
                root_agent=root_agent,
                events_compaction_config=build_compaction_config(
                    config=comp_cfg,
                    summarizer=fake_summarizer,
                ),
            )
            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name="agent", user_id="alex_noogler"
            )

            # Simulate 2 full user/agent invocations.
            for idx in range(1, 3):
                inv_id = f"inv_{idx}"
                await session_service.append_event(
                    session,
                    Event(
                        invocation_id=inv_id,
                        author="user",
                        timestamp=float(idx * 10),
                        content=types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=f"Question {idx}")],
                        ),
                    ),
                )
                await session_service.append_event(
                    session,
                    Event(
                        invocation_id=inv_id,
                        author="fde_duckie",
                        timestamp=float(idx * 10 + 1),
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=f"Answer {idx}")],
                        ),
                    ),
                )

            compacted_events = []
            async for ev in _run_compaction_for_sliding_window(
                app=test_app,
                session=session,
                session_service=session_service,
            ):
                await session_service.append_event(session, ev)
                compacted_events.append(ev)

            self.assertEqual(len(fake_summarizer.calls), 1)
            self.assertEqual(len(fake_summarizer.calls[0]), 4)
            self.assertEqual(len(compacted_events), 1)
            comp = compacted_events[0].actions.compaction
            self.assertIsNotNone(comp)
            self.assertEqual(comp.start_timestamp, 10.0)
            self.assertEqual(comp.end_timestamp, 21.0)
            updated = await session_service.get_session(
                app_name="agent", user_id="alex_noogler", session_id=session.id
            )
            self.assertTrue(
                any(ev.actions and ev.actions.compaction for ev in updated.events)
            )

        asyncio.run(_run())


class TestObservabilityAndOTel(unittest.TestCase):
    def test_app_registers_observability_plugin(self) -> None:
        self.assertTrue(
            any(isinstance(p, ObservabilityPlugin) for p in fde_app.plugins)
        )

    def test_configure_otel_none_is_noop(self) -> None:
        self.assertIsNone(configure_otel(exporter="none", set_global=False))

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
            ctx = _make_context("priya_fde")
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


class TestJsonConfig(unittest.TestCase):
    def test_default_config_file_loads_cleanly(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        cfg = load_config(config_path)
        self.assertEqual(cfg.default_user, "alex_noogler")
        self.assertEqual(cfg.model.name, "gemini-2.5-flash-lite")
        self.assertTrue(cfg.compaction.enabled)
        self.assertEqual(cfg.logging.level, "INFO")
        self.assertEqual(cfg.otel.exporter, "none")

    def test_inline_json_config_and_dynamic_app_creation(self) -> None:
        raw = json.dumps(
            {
                "default_user": "priya_fde",
                "model": {"name": "gemini-2.5-flash-lite-custom", "temperature": 0.25},
                "compaction": {"enabled": False},
                "logging": {"level": "DEBUG"},
                "otel": {"service_name": "fde-test", "exporter": "console"},
            }
        )
        cfg = load_config(raw)
        self.assertEqual(cfg.default_user, "priya_fde")
        self.assertEqual(cfg.model.name, "gemini-2.5-flash-lite-custom")
        self.assertEqual(cfg.model.temperature, 0.25)
        self.assertFalse(cfg.compaction.enabled)

        custom_app = create_app(cfg)
        self.assertEqual(custom_app.root_agent.model, "gemini-2.5-flash-lite-custom")
        self.assertEqual(
            custom_app.root_agent.generate_content_config.temperature, 0.25
        )
        self.assertIsNone(custom_app.events_compaction_config)
        for tool in custom_app.root_agent.tools:
            if isinstance(tool, AgentTool):
                self.assertEqual(tool.agent.model, "gemini-2.5-flash-lite-custom")

    def test_unknown_json_config_key_fails_validation(self) -> None:
        with self.assertRaises(ValidationError):
            load_config('{"unknown_key": true}')

    def test_missing_config_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("/tmp/definitely_nonexistent_fde_duckie_config.json")


class TestSqlitePersistenceAndSeeding(unittest.TestCase):
    def tearDown(self) -> None:
        set_profile_store(MockNooglerProfileStore())

    def test_setup_script_seeds_profiles_sessions_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "local_dev.db"
            summary = asyncio.run(
                seed_local_db(db_path=db_path, app_name="agent", reset=True)
            )
            self.assertEqual(
                summary["all_profiles"], ["alex_noogler", "jordan_fde", "priya_fde"]
            )
            self.assertEqual(len(summary["seeded_sessions"]), 3)

            # Verify both ADK session tables and noogler_profiles table exist.
            with sqlite3.connect(db_path) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            self.assertTrue(
                {
                    "sessions",
                    "events",
                    "app_states",
                    "user_states",
                    "noogler_profiles",
                }.issubset(tables)
            )

            # Idempotent re-run does not duplicate rows or sessions.
            rerun = asyncio.run(
                seed_local_db(db_path=db_path, app_name="agent", reset=False)
            )
            self.assertEqual(rerun["seeded_profiles"], [])
            self.assertEqual(rerun["seeded_sessions"], [])

    def test_session_history_and_state_persist_across_sqlite_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "history.db"
            cfg = AppConfig(
                database=DatabaseConfig(backend="sqlite", url=f"sqlite:///{db_path}")
            )

            async def _roundtrip() -> None:
                svc1 = create_session_service(cfg)
                self.assertIsInstance(svc1, SqliteSessionService)
                try:
                    sess = await svc1.create_session(
                        app_name="agent",
                        user_id="priya_fde",
                        session_id="default-priya_fde",
                        state={"noogler_id": "priya_fde", "custom_flag": "persisted"},
                    )
                    await svc1.append_event(
                        sess,
                        Event(
                            invocation_id="inv_1",
                            author="user",
                            content=types.Content(
                                role="user",
                                parts=[types.Part.from_text(text="Check L400")],
                            ),
                        ),
                    )
                finally:
                    await svc1.close()

                # Open a brand-new SqliteSessionService instance pointing to the same DB.
                svc2 = create_session_service(cfg)
                try:
                    reloaded = await svc2.get_session(
                        app_name="agent",
                        user_id="priya_fde",
                        session_id="default-priya_fde",
                    )
                    self.assertIsNotNone(reloaded)
                    self.assertEqual(reloaded.state["custom_flag"], "persisted")
                    self.assertEqual(len(reloaded.events), 1)
                    self.assertEqual(
                        reloaded.events[0].content.parts[0].text, "Check L400"
                    )
                finally:
                    await svc2.close()

            asyncio.run(_roundtrip())

    def test_profile_updates_persist_to_sqlite_profile_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "profiles.db"
            cfg = AppConfig(
                database=DatabaseConfig(backend="sqlite", url=f"sqlite:///{db_path}")
            )
            session_service = configure_storage(cfg)
            try:
                self.assertIsInstance(get_profile_store(), SqliteNooglerProfileStore)
                ctx = _make_context("priya_fde")
                res = update_enablement_progress(
                    "L300", "in_progress", labs_completed=6, tool_context=ctx
                )
                self.assertEqual(res["status"], "ok")

                # Verify fresh store instance reads the updated 6/7 L300 progress.
                fresh_store = SqliteNooglerProfileStore(db_path)
                reloaded_profile = fresh_store.get_profile("priya_fde")
                self.assertEqual(
                    reloaded_profile["enablement"]["L300"]["labs_completed"], 6
                )
                self.assertEqual(
                    reloaded_profile["enablement"]["L400"]["status"], "not_started"
                )
            finally:
                asyncio.run(session_service.close())

    def test_memory_backend_returns_in_memory_session_service(self) -> None:
        cfg = AppConfig(database=DatabaseConfig(backend="memory"))
        svc = configure_storage(cfg)
        self.assertIsInstance(svc, InMemorySessionService)
        self.assertIsInstance(get_profile_store(), MockNooglerProfileStore)


class TestVaultSecretsAndTerraform(unittest.TestCase):
    def _make_fake_vault_transport(self) -> tuple[httpx.MockTransport, list[dict]]:
        versions: list[dict[str, str]] = []
        seen_headers: list[dict] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(dict(request.headers))
            if request.headers.get("x-vault-token") != "test-vault-token":
                return httpx.Response(403, json={"errors": ["permission denied"]})

            if request.url.path != "/v1/secret/data/fde-duckie":
                return httpx.Response(404, json={"errors": []})

            if request.method == "GET":
                if not versions:
                    return httpx.Response(404, json={"errors": []})
                ver_param = request.url.params.get("version")
                if ver_param is not None:
                    idx = int(ver_param) - 1
                    if idx < 0 or idx >= len(versions):
                        return httpx.Response(404, json={"errors": []})
                    payload = versions[idx]
                    ver = idx + 1
                else:
                    payload = versions[-1]
                    ver = len(versions)
                return httpx.Response(
                    200,
                    json={"data": {"data": payload, "metadata": {"version": ver}}},
                )

            if request.method == "POST":
                body = json.loads(request.content.decode("utf-8"))
                versions.append(dict(body.get("data", {})))
                return httpx.Response(200, json={"data": {"version": len(versions)}})

            return httpx.Response(405)

        return httpx.MockTransport(_handler), seen_headers

    def test_vault_kv_v2_crud_and_versioning(self) -> None:
        transport, seen_headers = self._make_fake_vault_transport()
        vault = VaultSecretManager(
            addr="http://127.0.0.1:8200",
            mount="secret",
            path="fde-duckie",
            token="test-vault-token",
            namespace="fde-ns",
            transport=transport,
        )

        self.assertEqual(vault.list_secrets(), [])
        self.assertIsNone(vault.get_secret("GOOGLE_API_KEY"))

        v1 = vault.set_secret("GOOGLE_API_KEY", "key-v1")
        v2 = vault.set_secret("GOOGLE_API_KEY", "key-v2")
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(vault.get_secret("GOOGLE_API_KEY"), "key-v2")
        self.assertEqual(vault.get_secret("GOOGLE_API_KEY", version="1"), "key-v1")
        self.assertEqual(vault.list_secrets(), ["GOOGLE_API_KEY"])

        self.assertTrue(vault.delete_secret("GOOGLE_API_KEY"))
        self.assertFalse(vault.delete_secret("GOOGLE_API_KEY"))
        self.assertIsNone(vault.get_secret("GOOGLE_API_KEY"))
        self.assertTrue(
            all(h.get("x-vault-namespace") == "fde-ns" for h in seen_headers)
        )

    def test_hydrate_runtime_secrets_populates_env_and_respects_existing(self) -> None:
        transport, _ = self._make_fake_vault_transport()
        vault = VaultSecretManager(token="test-vault-token", transport=transport)
        vault.set_secret("TEST_HYDRATE_KEY", "secret-from-vault")

        os.environ.pop("TEST_HYDRATE_KEY", None)
        try:
            loaded = hydrate_runtime_secrets(
                names=("TEST_HYDRATE_KEY",), manager=vault, strict=True
            )
            self.assertEqual(loaded, ["TEST_HYDRATE_KEY"])
            self.assertEqual(os.environ["TEST_HYDRATE_KEY"], "secret-from-vault")

            # Does not overwrite already-set env var when overwrite=False.
            vault.set_secret("TEST_HYDRATE_KEY", "new-secret")
            loaded_again = hydrate_runtime_secrets(
                names=("TEST_HYDRATE_KEY",), manager=vault, overwrite=False, strict=True
            )
            self.assertEqual(loaded_again, [])
            self.assertEqual(os.environ["TEST_HYDRATE_KEY"], "secret-from-vault")
        finally:
            os.environ.pop("TEST_HYDRATE_KEY", None)

    def test_create_secret_manager_backends_and_graceful_fallback(self) -> None:
        vault_cfg = AppConfig(secrets=SecretsConfig(backend="vault"))
        self.assertIsInstance(create_secret_manager(vault_cfg), VaultSecretManager)

        env_cfg = AppConfig(secrets=SecretsConfig(backend="env"))
        self.assertIsInstance(create_secret_manager(env_cfg), EnvSecretManager)

        # Unauthenticated Vault fails gracefully when strict=False.
        broken_vault = VaultSecretManager(token=None)
        old_token = os.environ.pop("VAULT_TOKEN", None)
        try:
            hydrated = hydrate_runtime_secrets(
                names=("NONEXISTENT_KEY_XYZ",), manager=broken_vault, strict=False
            )
            self.assertEqual(hydrated, [])
        finally:
            if old_token is not None:
                os.environ["VAULT_TOKEN"] = old_token

    def test_terraform_config_uses_single_vault_provider(self) -> None:
        tf_root = Path(__file__).resolve().parent.parent / "infra" / "terraform"
        self.assertFalse((tf_root / "gcp").exists())
        self.assertFalse((tf_root / "local").exists())
        for fname in ("main.tf", "variables.tf", "outputs.tf", "README.md"):
            self.assertTrue(
                (tf_root / fname).is_file(),
                f"Missing Terraform file: infra/terraform/{fname}",
            )
        tf_main = (tf_root / "main.tf").read_text(encoding="utf-8")
        self.assertIn("hashicorp/vault", tf_main)
        self.assertNotIn("hashicorp/local", tf_main)
        self.assertNotIn("hashicorp/null", tf_main)
        self.assertIn("vault_kv_secret_v2", tf_main)
        self.assertIn("terraform_data", tf_main)
        self.assertIn("scripts/setup_db.py", tf_main)


if __name__ == "__main__":
    unittest.main()
