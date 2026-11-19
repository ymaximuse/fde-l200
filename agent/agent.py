"""Root coordinator and ADK App (with context compaction & observability) for FDE Duckie."""

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.tools import AgentTool

from agent.compaction import build_app
from agent.config import AppConfig, get_config
from agent.state import load_noogler_profile, switch_noogler_persona
from agent.sub_agents import (
    create_duckie_proxy_agent,
    create_enablement_agent,
    create_onboarding_guide_agent,
)

COORDINATOR_INSTRUCTION = """You are **FDE Duckie**, the onboarding assistant for new Google Forward Deployed Engineers (FDE Nooglers).
{noogler_summary?}

Your job is to route requests to the right specialist tool(s) and present a clear, concise combined answer.
NEVER answer FDE enablement status, setup steps, contacts, or general Google questions from memory—ALWAYS call the matching tool(s).

Routing Table:
1. `enablement_agent`: L200, L300 (6/7 challenge labs), L400 Bootcamp eligibility/blockers, PCA/GAIL/CDL certifications, enablement status checks, marking progress, or go/fde-enablement-dash sync issues.
2. `onboarding_guide_agent`: "What should I do next?", 30/60/90-day ramp plan, Cloudtop/Argolis/go/fde-github setup, onboarding buddy, pod manager, FDE Chat spaces, or team rituals.
3. `duckie_proxy_agent`: Any general, non-FDE-specific Google question (Workday, gBenefits, PTO, corporate card, Pixel/go/stuff, Piper/CitC/Critique, general Noogler compliance, go/wtf acronyms).
4. `switch_noogler_persona`: ONLY if the user explicitly asks to switch demo user/persona (alex_noogler, priya_fde, jordan_fde).

Rules:
- If a user asks a compound question spanning multiple domains, call ALL relevant specialist tools and synthesize their answers into one well-structured response.
- If the user just says hello or asks what you can do, briefly introduce FDE Duckie's capabilities and mention their active onboarding day from the profile above.
- Keep responses friendly, structured, and concise.
"""


def create_root_agent(config: AppConfig | None = None) -> LlmAgent:
    cfg = config or get_config()
    return LlmAgent(
        name="fde_duckie",
        model=cfg.model.name,
        description=(
            "FDE Duckie root coordinator for Google FDE Noogler onboarding: routes "
            "between FDE Enablement, Setup/Ramp Guide, and General Duckie (stub)."
        ),
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            AgentTool(agent=create_enablement_agent(cfg)),
            AgentTool(agent=create_onboarding_guide_agent(cfg)),
            AgentTool(agent=create_duckie_proxy_agent(cfg)),
            switch_noogler_persona,
        ],
        before_agent_callback=load_noogler_profile,
        generate_content_config=cfg.model.to_generate_content_config(),
    )


def create_app(config: AppConfig | None = None, *, name: str = "agent") -> App:
    cfg = config or get_config()
    return build_app(root_agent=create_root_agent(cfg), name=name, config=cfg)


# Default module-level exports discovered by `adk run agent` and `adk web`.
root_agent = create_root_agent()
app = build_app(root_agent=root_agent, name="agent")
