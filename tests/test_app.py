"""Offline tests for ADK agent graph integrity, App wiring, compaction, and JSON config."""

import inspect
import json
import unittest
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps import App
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.tools import AgentTool, FunctionTool
from google.adk.tools import _function_tool_declarations as ftd
from pydantic import TypeAdapter

from agent.agent import app as fde_app
from agent.agent import build_app, create_app, root_agent
from agent.config import AppConfig, CompactionConfig, get_config, load_config
from agent.plugins.compaction import (
    FDE_COMPACTION_PROMPT_TEMPLATE,
    build_compaction_config,
)
from agent.schemas import ComplexityAssessment
from agent.state import COMPLEXITY_STATE_KEY
from agent.sub_agents import (
    COMPLEXITY_ANALYZER_NAME,
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
from agent.tools.persona_tools import switch_noogler_persona
from tests.helpers import make_context


class TestADKAgentGraphIntegrity(unittest.TestCase):
    def test_root_and_specialists_meet_flash_lite_constraints(self) -> None:
        expected_model = get_config().model.name
        self.assertIsInstance(root_agent, SequentialAgent)
        self.assertEqual(root_agent.name, "fde_duckie")
        self.assertIsNotNone(root_agent.before_agent_callback)
        self.assertEqual(len(root_agent.sub_agents), 2)

        analyzer, coordinator = root_agent.sub_agents
        self.assertIsInstance(analyzer, LlmAgent)
        self.assertEqual(analyzer.name, COMPLEXITY_ANALYZER_NAME)
        self.assertEqual(analyzer.model, expected_model)
        self.assertEqual(analyzer.output_schema, ComplexityAssessment)
        self.assertEqual(analyzer.output_key, COMPLEXITY_STATE_KEY)
        self.assertEqual(len(analyzer.tools), 0)

        self.assertIsInstance(coordinator, LlmAgent)
        self.assertEqual(coordinator.name, "fde_duckie_coordinator")
        self.assertEqual(coordinator.model, expected_model)

        # Strict Flash-Lite tool budget: <= 4 tools on coordinator and every specialist.
        self.assertLessEqual(len(coordinator.tools), 4)
        agent_tools = [t for t in coordinator.tools if isinstance(t, AgentTool)]
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

        coordinator = root_agent.sub_agents[1]
        for ag in (
            coordinator,
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
        ctx = make_context("priya_fde")
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


class TestJsonConfig(unittest.TestCase):
    def test_default_config_file_loads_cleanly(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        cfg = load_config(config_path)
        self.assertEqual(cfg.default_user, "alex_noogler")
        self.assertEqual(cfg.model.name, "gemini-2.5-flash-lite")
        self.assertTrue(cfg.model_routing.enabled)
        self.assertEqual(cfg.model_routing.capable_model, "gemini-2.5-flash")
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
        self.assertIsInstance(custom_app.root_agent, SequentialAgent)
        analyzer, coordinator = custom_app.root_agent.sub_agents
        self.assertEqual(analyzer.model, "gemini-2.5-flash-lite-custom")
        self.assertEqual(coordinator.model, "gemini-2.5-flash-lite-custom")
        self.assertEqual(
            coordinator.generate_content_config.temperature, 0.25
        )
        self.assertIsNone(custom_app.events_compaction_config)
        for tool in coordinator.tools:
            if isinstance(tool, AgentTool):
                self.assertEqual(tool.agent.model, "gemini-2.5-flash-lite-custom")


if __name__ == "__main__":
    unittest.main()
