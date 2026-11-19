"""Mocked local FDE onboarding knowledge base backed by static JSON."""

import copy
import functools
import json
from pathlib import Path
from typing import Any

MOCK_DATA_DIR = Path(__file__).resolve().parent / "data"


@functools.lru_cache(maxsize=1)
def _load_kb_json() -> dict[str, Any]:
    path = MOCK_DATA_DIR / "fde_kb.json"
    return json.loads(path.read_text(encoding="utf-8"))


class MockFdeKnowledgeBase:
    """Static JSON stand-in for grounded FDE g3doc / Vertex AI Search retrieval."""

    def get_dashboard_notes(self) -> dict[str, Any]:
        return copy.deepcopy(_load_kb_json()["dashboard_notes"])

    def get_setup_guides(self) -> dict[str, Any]:
        return copy.deepcopy(_load_kb_json()["setup_guides"])

    def get_team_directory(self) -> dict[str, Any]:
        return copy.deepcopy(_load_kb_json()["team_directory"])
