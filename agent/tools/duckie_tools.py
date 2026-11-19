"""Tool wrapper for forwarding non-FDE general Google questions to Duckie."""

from typing import Annotated

from pydantic import Field

from agent.infra.backends import get_duckie_client
from agent.schemas import DuckieToolResult


def query_duckie(
    question: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The user's general, non-FDE-specific Google question to forward to Duckie "
                "(e.g., Workday, gBenefits, PTO, corporate card, Piper/CitC, go/wtf)."
            ),
        ),
    ],
) -> DuckieToolResult:
    """Forward a general, non-FDE-specific Google question to Duckie.

    Use this for HR/Workday, benefits, corporate card/equipment, general Piper/CitC,
    company-wide compliance training, or general Google acronyms.
    """
    response = get_duckie_client().ask(question)
    return response.model_dump()
