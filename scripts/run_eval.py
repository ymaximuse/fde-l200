"""Run automated golden dataset offline evaluation using Google ADK's AgentEvaluator."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import create_app
from agent.config import load_config, set_config
from agent.infra.backends import set_profile_store
from agent.infra.secrets import hydrate_runtime_secrets
from agent.mocks.profile_store import MockNooglerProfileStore

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "golden_dataset.evalset.json"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "evals" / "test_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADK golden dataset offline evaluation for FDE Duckie."
    )
    parser.add_argument("--config", default=None, help="Path or inline JSON for AppConfig.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to ADK EvalSet JSON.")
    parser.add_argument(
        "--eval-config", default=str(DEFAULT_EVAL_CONFIG), help="Path to ADK EvalConfig JSON."
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Runs per eval case (default: 1).")
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    cfg = set_config(load_config(args.config))
    hydrate_runtime_secrets(cfg)
    set_profile_store(MockNooglerProfileStore())

    eval_set = EvalSet.model_validate_json(Path(args.dataset).read_text(encoding="utf-8"))
    eval_config = EvalConfig.model_validate_json(Path(args.eval_config).read_text(encoding="utf-8"))
    active_app = create_app(cfg, name="agent")

    await AgentEvaluator.evaluate_eval_set(
        agent_module=active_app,
        eval_set=eval_set,
        eval_config=eval_config,
        num_runs=args.num_runs,
        print_detailed_results=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
