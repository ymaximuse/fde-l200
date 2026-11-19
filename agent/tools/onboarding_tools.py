"""Deterministic tools for FDE Day-1 setup, 30/60/90-day ramp, and team contacts."""

from typing import Annotated, Any

from google.adk.tools import ToolContext
from pydantic import Field

from agent.backends import get_knowledge_base
from agent.schemas import (
    NextActionsResult,
    SetupCategoryFilter,
    SetupChecklistResult,
    TeamContactsResult,
)
from agent.state import get_active_profile
from agent.tools.enablement_tools import _reconcile_statuses


def get_next_recommended_actions(
    tool_context: ToolContext | None = None,
) -> NextActionsResult:
    """Deterministically compute the top 3 prioritized next actions for the active Noogler.

    Uses onboarding day, incomplete Day-1 setup items, and enablement progress/blockers
    so small models do not have to guess priorities.
    """
    if tool_context is None:
        return {"status": "error", "message": "Missing ToolContext."}

    profile = get_active_profile(tool_context)
    _reconcile_statuses(profile)
    day = profile["onboarding_day"]
    enablement = profile["enablement"]

    recommendations: list[dict[str, Any]] = []

    # Priority 1: Incomplete Day-1 / Week-1 setup prerequisites.
    for item in profile["setup_items"]:
        if item["status"] != "completed":
            recommendations.append(
                {
                    "priority": len(recommendations) + 1,
                    "category": "setup",
                    "action": item["title"],
                    "current_status": item["status"],
                    "reason": item.get("notes") or "Foundational FDE setup item.",
                    "link": item.get("link", ""),
                }
            )

    # Priority 2: Core enablement tracks in dependency & pacing order.
    l200 = enablement["L200"]
    l300 = enablement["L300"]
    l400 = enablement["L400"]
    pca = enablement["PCA"]

    if l200["status"] != "completed":
        recommendations.append(
            {
                "priority": len(recommendations) + 1,
                "category": "enablement",
                "action": "Finish GEAP L200 and submit the Final Project Assessment",
                "current_status": l200["status"],
                "reason": f"Target window is Days 1-30 (you are on Day {day}). Required before L300.",
                "link": l200.get("link", "go/fde-enablement-dash"),
            }
        )

    if l300["status"] != "completed" and l200["status"] == "completed":
        labs_done = l300.get("labs_completed") or 0
        labs_req = l300.get("labs_required") or 6
        remaining = max(0, labs_req - labs_done)
        recommendations.append(
            {
                "priority": len(recommendations) + 1,
                "category": "enablement",
                "action": f"Complete {remaining} more GEAP L300 Challenge Lab(s) ({labs_done}/{labs_req} done)",
                "current_status": l300["status"],
                "reason": "Strict prerequisite to unlock L400 Bootcamp registration (target Days 31-60).",
                "link": l300.get("link", "goo.gle/cps"),
            }
        )

    if l400["status"] not in ("completed", "blocked"):
        recommendations.append(
            {
                "priority": len(recommendations) + 1,
                "category": "enablement",
                "action": "Register for L400 Agent Building Bootcamp",
                "current_status": l400["status"],
                "reason": "L300 prerequisite is satisfied; plan travel/cohort dates early.",
                "link": l400.get("link", "go/fde-enablement-dash"),
            }
        )

    if pca["status"] != "completed":
        urgency = (
            f"Past Day 60 pacing target (currently Day {day}); schedule exam soon before Day 90."
            if day > 60
            else "Target window is Days 1-60 (hard deadline Day 90). Remember OnVUE works best on a personal laptop."
        )
        recommendations.append(
            {
                "priority": len(recommendations) + 1,
                "category": "certification",
                "action": "Prepare for and schedule Professional Cloud Architect (PCA) exam",
                "current_status": pca["status"],
                "reason": urgency,
                "link": pca.get("link", "go/cloudu"),
            }
        )

    if day >= 45 and l400["status"] == "completed":
        recommendations.append(
            {
                "priority": len(recommendations) + 1,
                "category": "ramp",
                "action": "Prepare for FDE Capstone Project & Field-Readiness Panel Defense",
                "current_status": "upcoming",
                "reason": "Key Days 31-60 / 61-90 bridge before joining customer co-build sprints.",
                "link": "go/fde-noogler-guide",
            }
        )

    top_three = recommendations[:3]
    for idx, rec in enumerate(top_three, start=1):
        rec["priority"] = idx

    phase = (
        "Days 1-30: Setup & Foundations"
        if day <= 30
        else ("Days 31-60: Deep Enablement & Capstone" if day <= 60 else "Days 61-90: Customer Co-Build")
    )

    return {
        "status": "ok",
        "noogler": profile["full_name"],
        "onboarding_day": day,
        "current_phase": phase,
        "recommended_actions": top_three,
    }


def get_setup_checklist(
    category: Annotated[
        SetupCategoryFilter,
        Field(
            description=(
                "Onboarding setup category to fetch: 'workstation', 'argolis', 'github', "
                "'rituals', 'ramp_milestones', or 'all' for the complete checklist."
            )
        ),
    ] = "all",
    tool_context: ToolContext | None = None,
) -> SetupChecklistResult:
    """Return FDE onboarding setup guides and the Noogler's personal setup status.

    Args:
        category: One of 'all', 'workstation', 'argolis', 'github', 'rituals',
            or 'ramp_milestones'.
    """
    guides = get_knowledge_base().get_setup_guides()
    normalized = category.lower()

    if normalized != "all" and normalized not in guides:
        return {
            "status": "error",
            "message": f"Unknown category '{category}'. Valid categories: all, {', '.join(guides.keys())}.",
        }

    selected_guides = guides if normalized == "all" else {normalized: guides[normalized]}
    personal_status: list[dict[str, Any]] = []
    if tool_context is not None:
        profile = get_active_profile(tool_context)
        personal_status = [
            item
            for item in profile["setup_items"]
            if normalized == "all" or item["category"] == normalized
        ]

    return {
        "status": "ok",
        "category": normalized,
        "personal_setup_status": personal_status,
        "guides": selected_guides,
    }


def get_team_contacts(
    tool_context: ToolContext | None = None,
) -> TeamContactsResult:
    """Return the Noogler's pod manager, onboarding buddy, FDE Chat spaces, and rituals."""
    if tool_context is None:
        return {"status": "error", "message": "Missing ToolContext."}

    profile = get_active_profile(tool_context)
    directory = get_knowledge_base().get_team_directory()

    return {
        "status": "ok",
        "noogler": {
            "full_name": profile["full_name"],
            "pod": profile["pod"],
            "location": profile["location"],
            "manager": profile["manager"],
            "onboarding_buddy": profile["buddy"],
        },
        "chat_spaces": directory["chat_spaces"],
        "calendars_and_rituals": directory["calendars_and_rituals"],
    }
