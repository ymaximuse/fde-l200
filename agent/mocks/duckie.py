"""Stubbed Duckie client returning canned mock responses for general Google Q&A."""

from agent.schemas import DuckieResponse


class StubDuckieClient:
    """Deterministic mock Duckie client for local development and testing."""

    _CANNED_TOPICS: tuple[tuple[tuple[str, ...], str, str, list[str]], ...] = (
        (
            ("benefit", "gbenefits", "health", "insurance", "dental", "401k"),
            "benefits",
            (
                "You can view and enroll in Google benefits (health, dental, 401k, "
                "and perks) through gBenefits. New hires typically have 30 days "
                "from their start date to complete initial benefit elections."
            ),
            ["go/gbenefits", "go/workday"],
        ),
        (
            ("workday", "pto", "vacation", "leave", "payroll", "paystub", "payslip"),
            "workday_hr",
            (
                "Administrative HR tasks—including onboarding checklists, payroll, "
                "and requesting PTO/leave—are managed in Workday."
            ),
            ["go/workday"],
        ),
        (
            ("credit card", "corp card", "corporate card", "expense", "travel", "getcard"),
            "corporate_card_and_expenses",
            (
                "Apply for your Google corporate credit card via go/getcard before "
                "booking travel (such as L400 Bootcamp travel). Expenses are submitted "
                "through the internal travel & expense portal."
            ),
            ["go/getcard"],
        ),
        (
            ("pixel", "phone", "google fi", "stuff", "laptop", "equipment", "headset"),
            "corp_hardware",
            (
                "You can order corporate accessories, a corporate Pixel + Google Fi, "
                "and standard equipment through go/stuff."
            ),
            ["go/stuff", "go/mytech"],
        ),
        (
            ("piper", "citc", "g4", "hg", "critique", "google3", "cl ", "changelist"),
            "google3_vcs",
            (
                "Google's monorepo (google3) uses Piper and CitC workspaces, accessed "
                "via Fig (hg) or Jujutsu (jj), with code review in Critique."
            ),
            ["go/citc", "go/fig", "go/critique"],
        ),
        (
            ("acronym", "wtf", "meaning of", "stand for", "moma"),
            "google_acronyms_and_moma",
            (
                "Use go/wtf to look up internal Google acronyms and jargon, and "
                "claim your personal shortlink go/[ldap] pointing to your Moma profile."
            ),
            ["go/wtf", "go/moma"],
        ),
        (
            ("compliance", "noogler training", "mainsail", "checktraining"),
            "general_noogler_training",
            (
                "Complete general Google Noogler onboarding at go/noogler and mandatory "
                "compliance modules at go/compliancetraining. Track general training "
                "status on Mainsail at go/checktraining."
            ),
            ["go/noogler", "go/compliancetraining", "go/checktraining"],
        ),
    )

    def ask(self, question: str) -> DuckieResponse:
        lowered = question.lower()
        for keywords, topic, answer, refs in self._CANNED_TOPICS:
            if any(kw in lowered for kw in keywords):
                return DuckieResponse(
                    source="duckie_stub",
                    is_mock=True,
                    matched_topic=topic,
                    answer=answer,
                    references=refs,
                    note="Mocked Duckie response (stub). Connect real Duckie MCP/RPC to replace.",
                )

        return DuckieResponse(
            source="duckie_stub",
            is_mock=True,
            matched_topic="general_fallback",
            answer=(
                f"[Duckie Stub] Received general Google question: '{question}'. "
                "In production, this query is forwarded directly to Google's internal "
                "Duckie assistant (go/duckie) grounded on g3doc, Moma, YAQS, and google3."
            ),
            references=["go/duckie"],
            note="Mocked Duckie response (stub). Connect real Duckie MCP/RPC to replace.",
        )
