"""Mocked/faked backends and seed data for local FDE Duckie development.

Everything in this package is a stand-in for real Google-internal services:
- `StubDuckieClient`: replaces live Duckie MCP/RPC (`go/duckie`).
- `MockNooglerProfileStore`: replaces `go/fde-enablement-dash` + Pod Onboarding Tracker.
- `MockFdeKnowledgeBase`: replaces grounded FDE g3doc / Vertex AI Search retrieval.
"""

from .duckie import StubDuckieClient
from .knowledge_base import MockFdeKnowledgeBase
from .profile_store import MockNooglerProfileStore

__all__ = [
    "MockFdeKnowledgeBase",
    "MockNooglerProfileStore",
    "StubDuckieClient",
]
