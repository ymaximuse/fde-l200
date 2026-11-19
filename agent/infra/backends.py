"""Backend protocol interfaces and swappable provider registry for FDE Duckie.

By default, every provider points to a mock implementation in `agent.mocks`.
Swap in real integrations (Duckie MCP/RPC, Enablement Dashboard API, Vertex AI
Search) by calling the `set_*` helpers at startup without touching any agent
or tool code.
"""

from typing import Any, Protocol

from agent.mocks import (
    MockFdeKnowledgeBase,
    MockNooglerProfileStore,
    StubDuckieClient,
)
from agent.schemas import DuckieResponse


class DuckieClient(Protocol):
    """Interface for querying Google's internal Duckie assistant."""

    def ask(self, question: str) -> DuckieResponse:
        ...


class NooglerProfileStore(Protocol):
    """Interface for retrieving FDE Noogler profiles and enablement records."""

    def list_persona_ids(self) -> list[str]:
        ...

    def list_persona_summaries(self) -> list[dict[str, Any]]:
        ...

    def has_persona(self, persona_id: str) -> bool:
        ...

    def get_profile(self, persona_id: str) -> dict[str, Any]:
        ...


class FdeKnowledgeBase(Protocol):
    """Interface for retrieving FDE onboarding setup guides, rituals, and dashboard notes."""

    def get_dashboard_notes(self) -> dict[str, Any]:
        ...

    def get_setup_guides(self) -> dict[str, Any]:
        ...

    def get_team_directory(self) -> dict[str, Any]:
        ...


_duckie_client: DuckieClient = StubDuckieClient()
_profile_store: NooglerProfileStore = MockNooglerProfileStore()
_knowledge_base: FdeKnowledgeBase = MockFdeKnowledgeBase()


def get_duckie_client() -> DuckieClient:
    return _duckie_client


def set_duckie_client(client: DuckieClient) -> None:
    global _duckie_client
    _duckie_client = client


def get_profile_store() -> NooglerProfileStore:
    return _profile_store


def set_profile_store(store: NooglerProfileStore) -> None:
    global _profile_store
    _profile_store = store


def get_knowledge_base() -> FdeKnowledgeBase:
    return _knowledge_base


def set_knowledge_base(kb: FdeKnowledgeBase) -> None:
    global _knowledge_base
    _knowledge_base = kb
