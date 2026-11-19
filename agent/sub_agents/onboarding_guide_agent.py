"""Day-1 Setup & 30/60/90-Day Ramp Guide specialist agent."""

from google.adk.agents import LlmAgent

from agent.config import AppConfig, get_config
from agent.schemas import OnboardingGuideInput
from agent.tools.onboarding_tools import (
    get_next_recommended_actions,
    get_setup_checklist,
    get_team_contacts,
)

ONBOARDING_GUIDE_INSTRUCTION = """You are the FDE Onboarding & Ramp Guide Specialist inside FDE Duckie.
{noogler_summary?}

Scope:
- Day-1 & Week-1 FDE setup: Linux Cloudtop (go/mytech), Argolis sandbox (go/argolis), FDE GitHub org (go/fde-github), and developer toolchain.
- 30/60/90-day FDE ramp milestones (Capstone, Field-Readiness Panel, Customer Co-Build).
- Team directory: pod manager, onboarding buddy, FDE Google Chat spaces, OOO calendar, and recurring rituals.
- Prioritized "What should I do next?" action planning.

Rules:
1. ALWAYS call a tool first; never invent names, contacts, or go/ links.
2. For "what should I focus on / do next?" questions, call `get_next_recommended_actions`.
3. For setup steps (Argolis, Cloudtop, GitHub, rituals, 30/60/90 plan), call `get_setup_checklist`.
4. For buddy, manager, pod, chat spaces, or calendars, call `get_team_contacts`.
5. Keep responses structured, warm, and concise with clear next steps and links from tool output.
"""


def create_onboarding_guide_agent(config: AppConfig | None = None) -> LlmAgent:
    cfg = config or get_config()
    return LlmAgent(
        name="onboarding_guide_agent",
        model=cfg.model.name,
        description=(
            "Helps FDE Nooglers with Day-1 setup (Cloudtop, Argolis, go/fde-github), "
            "30/60/90-day ramp milestones, next-action planning, and pod/buddy/chat contacts."
        ),
        instruction=ONBOARDING_GUIDE_INSTRUCTION,
        input_schema=OnboardingGuideInput,
        tools=[
            get_next_recommended_actions,
            get_setup_checklist,
            get_team_contacts,
        ],
        generate_content_config=cfg.model.to_generate_content_config(),
    )


onboarding_guide_agent = create_onboarding_guide_agent()
