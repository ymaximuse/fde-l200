"""Shared offline test helpers for FDE Duckie."""

from types import SimpleNamespace

from agent.state import PERSONA_ID_STATE_KEY, load_noogler_profile


def make_context(persona_id: str | None = None) -> SimpleNamespace:
    state: dict = {}
    if persona_id is not None:
        state[PERSONA_ID_STATE_KEY] = persona_id
    ctx = SimpleNamespace(state=state)
    load_noogler_profile(ctx)
    return ctx
