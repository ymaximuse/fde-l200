"""Root coordinator and ADK App (with context compaction) for FDE Duckie."""

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from agent.compaction import build_app
from agent.config import LOW_TEMP_CONFIG, MODEL_NAME
from agent.state import load_noogler_profile, switch_noogler_persona
from agent.sub_agents import (
    duckie_proxy_agent,
    enablement_agent,
    onboarding_guide_agent,
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

root_agent = LlmAgent(
    name="fde_duckie",
    model=MODEL_NAME,
    description=(
        "FDE Duckie root coordinator for Google FDE Noogler onboarding: routes "
        "between FDE Enablement, Setup/Ramp Guide, and General Duckie (stub)."
    ),
    instruction=COORDINATOR_INSTRUCTION,
    tools=[
        AgentTool(agent=enablement_agent),
        AgentTool(agent=onboarding_guide_agent),
        AgentTool(agent=duckie_proxy_agent),
        switch_noogler_persona,
    ],
    before_agent_callback=load_noogler_profile,
    generate_content_config=LOW_TEMP_CONFIG,
)

# ADK CLI (`adk run agent` / `adk web`) and `main.py` load `app` so
# EventsCompactionConfig is active automatically across all run modes.
app = build_app(root_agent=root_agent, name="agent")
