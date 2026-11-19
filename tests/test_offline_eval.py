"""Offline validation tests for the ADK golden evaluation dataset and config."""

from pathlib import Path
import unittest

from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
DATASET_PATH = EVALS_DIR / "golden_dataset.evalset.json"
EVAL_CONFIG_PATH = EVALS_DIR / "test_config.json"


class TestGoldenEvalDataset(unittest.TestCase):
    def test_dataset_and_config_match_adk_schemas(self) -> None:
        eval_set = EvalSet.model_validate_json(DATASET_PATH.read_text(encoding="utf-8"))
        eval_config = EvalConfig.model_validate_json(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(eval_set.eval_cases), 10)
        self.assertIn("tool_trajectory_avg_score", eval_config.criteria)
        self.assertIn("response_match_score", eval_config.criteria)

    def test_dataset_covers_all_personas_and_coordinator_tools(self) -> None:
        eval_set = EvalSet.model_validate_json(DATASET_PATH.read_text(encoding="utf-8"))

        personas = {
            case.session_input.user_id
            for case in eval_set.eval_cases
            if case.session_input
        }
        self.assertEqual(personas, {"alex_noogler", "priya_fde", "jordan_fde"})

        tools = {
            call.name
            for case in eval_set.eval_cases
            for inv in (case.conversation or [])
            for call in (getattr(inv.intermediate_data, "tool_uses", None) or [])
        }
        self.assertEqual(
            tools,
            {
                "enablement_agent",
                "onboarding_guide_agent",
                "duckie_proxy_agent",
                "switch_noogler_persona",
            },
        )


if __name__ == "__main__":
    unittest.main()
