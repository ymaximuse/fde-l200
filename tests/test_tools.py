"""Offline tests for session state, persona switching, specialist tools, and mocks."""

import unittest

from agent.infra.backends import set_duckie_client
from agent.mocks import StubDuckieClient
from agent.schemas import DuckieResponse
from agent.state import PROFILE_STATE_KEY, SUMMARY_STATE_KEY
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


class TestSessionStateAndPersonas(unittest.TestCase):
    def test_before_agent_callback_loads_default_profile(self) -> None:
        ctx = make_context()
        self.assertIn(PROFILE_STATE_KEY, ctx.state)
        self.assertIn(SUMMARY_STATE_KEY, ctx.state)
        self.assertEqual(ctx.state[PROFILE_STATE_KEY]["persona_id"], "alex_noogler")
        self.assertIn("Alex Rivera", ctx.state[SUMMARY_STATE_KEY])

    def test_switch_noogler_persona_updates_state(self) -> None:
        ctx = make_context("alex_noogler")
        res = switch_noogler_persona("priya_fde", ctx)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(ctx.state[PROFILE_STATE_KEY]["persona_id"], "priya_fde")
        self.assertIn("Priya Nair", ctx.state[SUMMARY_STATE_KEY])

    def test_switch_unknown_persona_returns_error(self) -> None:
        ctx = make_context("alex_noogler")
        res = switch_noogler_persona("does_not_exist", ctx)
        self.assertEqual(res["status"], "error")
        self.assertTrue(len(res["available_personas"]) >= 3)


class TestEnablementTools(unittest.TestCase):
    def test_priya_l400_blocked_until_six_l300_labs(self) -> None:
        ctx = make_context("priya_fde")
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
        ctx = make_context("jordan_fde")
        status = get_enablement_status("PCA", tool_context=ctx)
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["core_progress"]["completed"], 3)
        self.assertEqual(status["tracks"][0]["pacing"], "at_risk")


class TestOnboardingGuideTools(unittest.TestCase):
    def test_day_12_recommendations_prioritize_setup_and_l200(self) -> None:
        ctx = make_context("alex_noogler")
        recs = get_next_recommended_actions(ctx)
        self.assertEqual(recs["status"], "ok")
        self.assertEqual(len(recs["recommended_actions"]), 3)
        categories = [r["category"] for r in recs["recommended_actions"]]
        self.assertEqual(categories[:2], ["setup", "setup"])
        self.assertEqual(categories[2], "enablement")

    def test_day_42_recommendations_prioritize_l300(self) -> None:
        ctx = make_context("priya_fde")
        recs = get_next_recommended_actions(ctx)
        self.assertEqual(recs["status"], "ok")
        self.assertIn("L300", recs["recommended_actions"][0]["action"])

    def test_setup_checklist_and_team_contacts(self) -> None:
        ctx = make_context("alex_noogler")
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


if __name__ == "__main__":
    unittest.main()
