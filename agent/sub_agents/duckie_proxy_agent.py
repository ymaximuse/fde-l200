"""General Google Q&A specialist that forwards non-FDE questions to Duckie (stub)."""

from google.adk.agents import LlmAgent

from agent.config import AppConfig, get_config
from agent.schemas import DuckieProxyInput
from agent.tools.duckie_tools import query_duckie

DUCKIE_PROXY_INSTRUCTION = """You are the General Duckie Proxy inside FDE Duckie.
{noogler_summary?}

Scope:
- Handle general, non-FDE-specific Google questions (Workday, gBenefits, PTO, corporate card, equipment/Pixel, general Piper/CitC/Critique, general Noogler compliance training, or internal acronyms).

Rules:
1. ALWAYS call `query_duckie` with the user's question before responding.
2. Base your answer strictly on the response returned by `query_duckie`.
3. Clearly note that the response was forwarded via Duckie (currently running in stub/mock mode) and include the returned go/ references.
4. Keep your answer concise.
"""


def create_duckie_proxy_agent(config: AppConfig | None = None) -> LlmAgent:
    cfg = config or get_config()
    return LlmAgent(
        name="duckie_proxy_agent",
        model=cfg.model.name,
        description=(
            "Forwards general, non-FDE-specific Google questions (HR, Workday, benefits, "
            "corporate card, Piper/CitC, general compliance, acronyms) to Duckie (stub)."
        ),
        instruction=DUCKIE_PROXY_INSTRUCTION,
        input_schema=DuckieProxyInput,
        tools=[query_duckie],
        generate_content_config=cfg.model.to_generate_content_config(),
    )


duckie_proxy_agent = create_duckie_proxy_agent()
