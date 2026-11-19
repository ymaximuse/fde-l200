"""ADK tool for switching the active FDE Noogler demo persona."""

from typing import Annotated

from google.adk.tools import ToolContext
from pydantic import Field

from agent.infra.backends import get_profile_store
from agent.schemas import PersonaId, SwitchPersonaResult
from agent.state import save_active_profile


def switch_noogler_persona(
    persona_id: Annotated[
        PersonaId,
        Field(
            description=(
                "Target FDE Noogler persona ID to activate for this session "
                "('alex_noogler' Day 12, 'priya_fde' Day 42, or 'jordan_fde' Day 70)."
            )
        ),
    ],
    tool_context: ToolContext,
) -> SwitchPersonaResult:
    """Switch the active FDE Noogler persona for this session.

    Use this ONLY when the user explicitly asks to switch users or demo another
    Noogler persona (valid values: 'alex_noogler', 'priya_fde', 'jordan_fde').

    Args:
        persona_id: Target persona ID ('alex_noogler', 'priya_fde', or 'jordan_fde').
    """
    store = get_profile_store()
    normalized = persona_id.strip().lower()
    if not store.has_persona(normalized):
        return {
            "status": "error",
            "message": f"Unknown persona '{persona_id}'.",
            "available_personas": store.list_persona_summaries(),
        }

    profile = store.get_profile(normalized)
    save_active_profile(tool_context, profile)
    return {
        "status": "ok",
        "message": f"Switched active Noogler to {profile['full_name']} (Day {profile['onboarding_day']}).",
        "active_profile": {
            "persona_id": profile["persona_id"],
            "full_name": profile["full_name"],
            "ldap": profile["ldap"],
            "onboarding_day": profile["onboarding_day"],
            "pod": profile["pod"],
        },
    }
