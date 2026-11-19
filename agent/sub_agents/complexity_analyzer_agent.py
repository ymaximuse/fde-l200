"""Step-1 complexity analyzer agent that classifies each turn into session state."""

from google.adk.agents import LlmAgent

from agent.config import AppConfig, get_config
from agent.schemas import ComplexityAssessment
from agent.state import COMPLEXITY_STATE_KEY

COMPLEXITY_ANALYZER_NAME = "complexity_analyzer_agent"

COMPLEXITY_ANALYZER_INSTRUCTION = """You are a fast request-complexity classifier for FDE Duckie.
Analyze the user's latest message and classify it as either "simple" or "complex".

Return "simple" when the request is:
- A greeting, chitchat, or asking what FDE Duckie can do
- A single-domain lookup (enablement status, L400 eligibility, setup checklist, buddy/manager/chat contacts, or a single general Google/Duckie question)
- A straightforward progress update or persona switch

Return "complex" when the request is:
- A compound / multi-domain question spanning 2+ areas (e.g., enablement + setup, enablement + general Google/Duckie, or setup + general Duckie)
- Multi-part questions requiring synthesis across multiple tools
- Planning, prioritization, comparisons, tradeoffs, custom study/ramp plans, or troubleshooting/debugging

Respond ONLY with JSON matching the output schema.
"""


def create_complexity_analyzer_agent(config: AppConfig | None = None) -> LlmAgent:
    cfg = config or get_config()
    return LlmAgent(
        name=COMPLEXITY_ANALYZER_NAME,
        model=cfg.model.name,
        description=(
            "Classifies the complexity of the user's latest request ('simple' vs 'complex') "
            "and writes the result to session state for downstream model routing."
        ),
        instruction=COMPLEXITY_ANALYZER_INSTRUCTION,
        output_schema=ComplexityAssessment,
        output_key=COMPLEXITY_STATE_KEY,
        generate_content_config=cfg.model.to_generate_content_config(),
    )


complexity_analyzer_agent = create_complexity_analyzer_agent()
