"""Session state helpers and ADK callbacks for FDE Duckie."""

import copy
from typing import Annotated, Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from pydantic import Field

from agent.backends import get_profile_store
from agent.config import DEFAULT_PERSONA_ID
from agent.schemas import PersonaId, SwitchPersonaResult

PROFILE_STATE_KEY = "noogler_profile"
SUMMARY_STATE_KEY = "noogler_summary"
PERSONA_ID_STATE_KEY = "noogler_id"


def list_persona_ids() -> list[str]:
    return get_profile_store().list_persona_ids()


def get_seed_profile(persona_id: str) -> dict[str, Any]:
    return get_profile_store().get_profile(persona_id)


def format_profile_summary(profile: dict[str, Any]) -> str:
    """Compact 1-line summary injected into agent instructions for Flash-Lite."""
    return (
        f"Active Noogler: {profile['full_name']} "
        f"(persona={profile['persona_id']}, ldap={profile['ldap']}, "
        f"Day {profile['onboarding_day']}, pod={profile['pod']}, "
        f"location={profile['location']})"
    )


def load_noogler_profile(callback_context: CallbackContext) -> None:
    """ADK before_agent_callback: ensure session state holds an active Noogler profile."""
    if PROFILE_STATE_KEY in callback_context.state:
        if SUMMARY_STATE_KEY not in callback_context.state:
            callback_context.state[SUMMARY_STATE_KEY] = format_profile_summary(
                callback_context.state[PROFILE_STATE_KEY]
            )
        return

    store = get_profile_store()
    requested_id = callback_context.state.get(PERSONA_ID_STATE_KEY, DEFAULT_PERSONA_ID)
    persona_id = requested_id if store.has_persona(requested_id) else DEFAULT_PERSONA_ID
    profile = store.get_profile(persona_id)

    callback_context.state[PERSONA_ID_STATE_KEY] = persona_id
    callback_context.state[PROFILE_STATE_KEY] = profile
    callback_context.state[SUMMARY_STATE_KEY] = format_profile_summary(profile)


def get_active_profile(tool_context: ToolContext) -> dict[str, Any]:
    """Return a mutable deep copy of the active Noogler profile from session state."""
    profile = tool_context.state.get(PROFILE_STATE_KEY)
    if profile is None:
        store = get_profile_store()
        persona_id = tool_context.state.get(PERSONA_ID_STATE_KEY, DEFAULT_PERSONA_ID)
        persona_id = persona_id if store.has_persona(persona_id) else DEFAULT_PERSONA_ID
        profile = store.get_profile(persona_id)
        save_active_profile(tool_context, profile)
        return copy.deepcopy(profile)
    return copy.deepcopy(profile)


def save_active_profile(tool_context: ToolContext, profile: dict[str, Any]) -> None:
    """Persist updated profile to session state (reassignment triggers ADK state_delta)."""
    stored = copy.deepcopy(profile)
    tool_context.state[PERSONA_ID_STATE_KEY] = stored["persona_id"]
    tool_context.state[PROFILE_STATE_KEY] = stored
    tool_context.state[SUMMARY_STATE_KEY] = format_profile_summary(stored)


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
