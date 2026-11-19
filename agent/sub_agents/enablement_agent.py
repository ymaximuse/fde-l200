"""Enablement & Prerequisites specialist agent (L200, L300, L400, PCA)."""

from google.adk.agents import LlmAgent

from agent.config import AppConfig, get_config
from agent.schemas import EnablementAgentInput
from agent.tools.enablement_tools import (
    check_l400_eligibility,
    get_enablement_status,
    update_enablement_progress,
)

ENABLEMENT_INSTRUCTION = """You are the FDE Enablement Specialist inside FDE Duckie.
{noogler_summary?}

Scope:
- Track and explain FDE enablement prerequisites: L200, L300 (6 of 7 challenge labs), L400 Bootcamp, PCA certification, and optional GAIL/CDL certifications.
- Check L400 eligibility and prerequisite blockers.
- Record mock progress updates when the user reports completing a track or L300 labs.
- Explain go/fde-enablement-dash sync delays (up to 24 hours; see go/fde-dash-faq).

Rules:
1. ALWAYS call a tool before answering; never guess statuses, lab counts, or deadlines.
2. For general status questions, call `get_enablement_status`.
3. For "Can I register for L400?" questions, call `check_l400_eligibility`.
4. Only call `update_enablement_progress` when the user explicitly asks to mark or update progress.
5. Present results concisely with bullet points, exact counts, blockers, and go/ links from tool output.
"""


def create_enablement_agent(config: AppConfig | None = None) -> LlmAgent:
    cfg = config or get_config()
    return LlmAgent(
        name="enablement_agent",
        model=cfg.model.name,
        description=(
            "Checks and updates FDE Noogler enablement prerequisites (L200, L300, L400, "
            "PCA, GAIL, CDL), L400 eligibility blockers, pacing, and dashboard sync notes."
        ),
        instruction=ENABLEMENT_INSTRUCTION,
        input_schema=EnablementAgentInput,
        tools=[
            get_enablement_status,
            check_l400_eligibility,
            update_enablement_progress,
        ],
        generate_content_config=cfg.model.to_generate_content_config(),
    )


enablement_agent = create_enablement_agent()
