"""Session state helpers and ADK callbacks for FDE Duckie."""

import copy
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext

from agent.config import get_config
from agent.infra.backends import get_profile_store

PROFILE_STATE_KEY = "noogler_profile"
SUMMARY_STATE_KEY = "noogler_summary"
PERSONA_ID_STATE_KEY = "noogler_id"
COMPLEXITY_STATE_KEY = "request_complexity"


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
    default_user = get_config().default_user
    requested_id = callback_context.state.get(PERSONA_ID_STATE_KEY, default_user)
    persona_id = requested_id if store.has_persona(requested_id) else default_user
    profile = store.get_profile(persona_id)

    callback_context.state[PERSONA_ID_STATE_KEY] = persona_id
    callback_context.state[PROFILE_STATE_KEY] = profile
    callback_context.state[SUMMARY_STATE_KEY] = format_profile_summary(profile)


def get_active_profile(tool_context: ToolContext) -> dict[str, Any]:
    """Return a mutable deep copy of the active Noogler profile from session state."""
    profile = tool_context.state.get(PROFILE_STATE_KEY)
    if profile is None:
        store = get_profile_store()
        default_user = get_config().default_user
        persona_id = tool_context.state.get(PERSONA_ID_STATE_KEY, default_user)
        persona_id = persona_id if store.has_persona(persona_id) else default_user
        profile = store.get_profile(persona_id)
        save_active_profile(tool_context, profile)
        return copy.deepcopy(profile)
    return copy.deepcopy(profile)


def save_active_profile(tool_context: ToolContext, profile: dict[str, Any]) -> None:
    """Persist updated profile to session state and durable store if supported."""
    stored = copy.deepcopy(profile)
    tool_context.state[PERSONA_ID_STATE_KEY] = stored["persona_id"]
    tool_context.state[PROFILE_STATE_KEY] = stored
    tool_context.state[SUMMARY_STATE_KEY] = format_profile_summary(stored)

    store = get_profile_store()
    save_fn = getattr(store, "save_profile", None)
    if callable(save_fn):
        save_fn(stored)
