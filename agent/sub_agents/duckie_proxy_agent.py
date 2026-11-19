"""General Google Q&A specialist that forwards non-FDE questions to Duckie (stub)."""

from google.adk.agents import LlmAgent

from agent.config import LOW_TEMP_CONFIG, MODEL_NAME
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

duckie_proxy_agent = LlmAgent(
    name="duckie_proxy_agent",
    model=MODEL_NAME,
    description=(
        "Forwards general, non-FDE-specific Google questions (HR, Workday, benefits, "
        "corporate card, Piper/CitC, general compliance, acronyms) to Duckie (stub)."
    ),
    instruction=DUCKIE_PROXY_INSTRUCTION,
    input_schema=DuckieProxyInput,
    tools=[query_duckie],
    generate_content_config=LOW_TEMP_CONFIG,
)
