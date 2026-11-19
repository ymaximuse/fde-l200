"""Mocked FDE Noogler profile & enablement data store backed by local JSON."""

import copy
import functools
import json
from pathlib import Path
from typing import Any

MOCK_DATA_DIR = Path(__file__).resolve().parent / "data"


@functools.lru_cache(maxsize=1)
def _load_seed_profiles() -> dict[str, dict[str, Any]]:
    path = MOCK_DATA_DIR / "noogler_profiles.json"
    return json.loads(path.read_text(encoding="utf-8"))


class MockNooglerProfileStore:
    """In-memory / JSON-backed mock replacing go/fde-enablement-dash & Pod tracker."""

    def list_persona_ids(self) -> list[str]:
        return sorted(_load_seed_profiles().keys())

    def list_persona_summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "persona_id": pid,
                "full_name": p["full_name"],
                "onboarding_day": p["onboarding_day"],
            }
            for pid, p in _load_seed_profiles().items()
        ]

    def has_persona(self, persona_id: str) -> bool:
        return persona_id in _load_seed_profiles()

    def get_profile(self, persona_id: str) -> dict[str, Any]:
        profiles = _load_seed_profiles()
        if persona_id not in profiles:
            valid = ", ".join(sorted(profiles.keys()))
            raise KeyError(f"Unknown persona '{persona_id}'. Valid personas: {valid}")
        return copy.deepcopy(profiles[persona_id])
