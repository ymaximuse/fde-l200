"""Offline deterministic and ADK graph integrity tests for FDE Duckie."""

import asyncio
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

from google.adk.apps import App
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.compaction import _run_compaction_for_sliding_window
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool
from google.adk.tools import _function_tool_declarations as ftd
from google.genai import types
from pydantic import TypeAdapter

from agent.agent import app as fde_app
from agent.agent import root_agent
from agent.compaction import (
    FDE_COMPACTION_PROMPT_TEMPLATE,
    build_app,
    build_compaction_config,
)
from agent.backends import (
    get_duckie_client,
    get_knowledge_base,
    get_profile_store,
    set_duckie_client,
)
from agent.config import MODEL_NAME
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
        self.assertEqual(root_agent.name, "fde_duckie")
        self.assertEqual(root_agent.model, MODEL_NAME)
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
            self.assertEqual(specialist.model, MODEL_NAME)
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
        self.assertIsNone(build_compaction_config(enabled=False))
        disabled_app = build_app(
            root_agent=root_agent, name="agent", compaction_enabled=False
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
            test_app = App(
                name="agent",
                root_agent=root_agent,
                events_compaction_config=build_compaction_config(
                    compaction_interval=2,
                    overlap_size=1,
                    token_threshold=8000,
                    event_retention_size=2,
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


if __name__ == "__main__":
    unittest.main()
