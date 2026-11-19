"""Deterministic tools for checking and updating FDE Enablement prerequisites."""

from typing import Annotated, Any

from google.adk.tools import ToolContext
from pydantic import Field

from agent.backends import get_knowledge_base
from agent.schemas import (
    EnablementStatusResult,
    EnablementTrackFilter,
    EnablementTrackId,
    L400EligibilityResult,
    UpdateEnablementResult,
    WritableTaskStatus,
)
from agent.state import get_active_profile, save_active_profile

CORE_TRACKS = ("L200", "L300", "L400", "PCA")
ALL_TRACKS = ("L200", "L300", "L400", "PCA", "GAIL", "CDL")


def _compute_pacing(track: dict[str, Any], onboarding_day: int) -> str:
    """Deterministically classify pacing relative to target window and hard deadline."""
    if track["status"] == "completed":
        return "completed"
    target_end = track["target_window_days"][1]
    hard_deadline = track.get("hard_deadline_day")
    if hard_deadline is not None and onboarding_day > hard_deadline:
        return "overdue"
    if onboarding_day > target_end:
        return "at_risk"
    return "on_track"


def _compute_blockers(
    track_id: str, enablement: dict[str, dict[str, Any]]
) -> list[str]:
    """Return unmet prerequisite descriptions for a given track."""
    track = enablement[track_id]
    blockers: list[str] = []
    for prereq_id in track.get("prerequisite_tracks", []):
        prereq = enablement.get(prereq_id)
        if not prereq:
            continue
        if prereq_id == "L300":
            labs_done = prereq.get("labs_completed") or 0
            labs_req = prereq.get("labs_required") or 6
            if prereq["status"] != "completed" or labs_done < labs_req:
                remaining = max(0, labs_req - labs_done)
                blockers.append(
                    f"Requires L300 completion ({labs_done}/{labs_req} required labs done; {remaining} more needed)."
                )
        elif prereq["status"] != "completed":
            blockers.append(f"Requires {prereq_id} to be completed first.")
    return blockers


def _reconcile_statuses(profile: dict[str, Any]) -> None:
    """Keep derived statuses (L300 lab threshold, L400 blocked gate) consistent."""
    enablement = profile["enablement"]
    l300 = enablement.get("L300")
    if l300 is not None:
        labs_done = l300.get("labs_completed") or 0
        labs_req = l300.get("labs_required") or 6
        if labs_done >= labs_req:
            l300["status"] = "completed"
        elif labs_done > 0 and l300["status"] == "not_started":
            l300["status"] = "in_progress"

    l400 = enablement.get("L400")
    if l400 is not None:
        blockers = _compute_blockers("L400", enablement)
        if blockers and l400["status"] != "completed":
            l400["status"] = "blocked"
        elif not blockers and l400["status"] == "blocked":
            l400["status"] = "not_started"


def _summarize_track(
    track_id: str,
    enablement: dict[str, dict[str, Any]],
    onboarding_day: int,
) -> dict[str, Any]:
    track = enablement[track_id]
    blockers = _compute_blockers(track_id, enablement)
    summary: dict[str, Any] = {
        "track_id": track_id,
        "title": track["title"],
        "required": track["required"],
        "status": track["status"],
        "pacing": _compute_pacing(track, onboarding_day),
        "target_window": f"Days {track['target_window_days'][0]}-{track['target_window_days'][1]}",
        "hard_deadline_day": track.get("hard_deadline_day"),
        "blockers": blockers,
        "notes": track.get("notes", ""),
        "link": track.get("link", ""),
    }
    if track.get("labs_required") is not None:
        labs = track.get("labs", [])
        summary["labs_completed"] = track.get("labs_completed", 0)
        summary["labs_required"] = track["labs_required"]
        summary["labs_total"] = len(labs)
        summary["incomplete_labs"] = [
            lab["title"] for lab in labs if not lab.get("completed")
        ]
    return summary


def get_enablement_status(
    track: Annotated[
        EnablementTrackFilter,
        Field(
            description=(
                "Specific enablement track ID ('L200', 'L300', 'L400', 'PCA', 'GAIL', 'CDL') "
                "or 'all' to retrieve the complete enablement overview."
            )
        ),
    ] = "all",
    tool_context: ToolContext | None = None,
) -> EnablementStatusResult:
    """Check FDE enablement and prerequisite completion status for the active Noogler.

    Returns structured status for core FDE tracks (L200, L300, L400, PCA) and
    optional early certifications (GAIL, CDL), including pacing, blockers, and
    dashboard sync guidance.

    Args:
        track: Specific track ID ('L200', 'L300', 'L400', 'PCA', 'GAIL', 'CDL')
            or 'all' for the complete overview.
    """
    if tool_context is None:
        return {"status": "error", "message": "Missing ToolContext."}

    profile = get_active_profile(tool_context)
    _reconcile_statuses(profile)
    enablement = profile["enablement"]
    day = profile["onboarding_day"]
    dashboard_notes = get_knowledge_base().get_dashboard_notes()

    normalized = track.upper() if track.lower() != "all" else "all"
    if normalized != "all" and normalized not in enablement:
        return {
            "status": "error",
            "message": f"Unknown enablement track '{track}'. Valid tracks: {', '.join(ALL_TRACKS)}.",
        }

    selected_ids = list(ALL_TRACKS) if normalized == "all" else [normalized]
    tracks_output = [
        _summarize_track(tid, enablement, day)
        for tid in selected_ids
        if tid in enablement
    ]

    core_completed = sum(
        1 for tid in CORE_TRACKS if enablement[tid]["status"] == "completed"
    )
    return {
        "status": "ok",
        "noogler": {
            "full_name": profile["full_name"],
            "ldap": profile["ldap"],
            "onboarding_day": day,
        },
        "core_progress": {
            "completed": core_completed,
            "total": len(CORE_TRACKS),
            "percent": round((core_completed / len(CORE_TRACKS)) * 100),
            "core_tracks": list(CORE_TRACKS),
        },
        "tracks": tracks_output,
        "dashboard_sync": dashboard_notes,
    }


def check_l400_eligibility(
    tool_context: ToolContext | None = None,
) -> L400EligibilityResult:
    """Deterministically verify whether the Noogler is eligible to register for L400 Bootcamp.

    L400 Bootcamp requires GEAP L300 completion (at least 6 of 7 challenge labs).
    """
    if tool_context is None:
        return {"status": "error", "message": "Missing ToolContext."}

    profile = get_active_profile(tool_context)
    _reconcile_statuses(profile)
    enablement = profile["enablement"]
    l300 = enablement["L300"]
    l400 = enablement["L400"]

    labs_done = l300.get("labs_completed") or 0
    labs_req = l300.get("labs_required") or 6
    labs_remaining = max(0, labs_req - labs_done)
    eligible = l300["status"] == "completed" and labs_done >= labs_req
    incomplete_labs = [
        lab["title"] for lab in l300.get("labs", []) if not lab.get("completed")
    ]

    return {
        "status": "ok",
        "noogler": profile["full_name"],
        "eligible_for_l400": eligible,
        "l400_status": l400["status"],
        "l300_status": l300["status"],
        "l300_labs_completed": labs_done,
        "l300_labs_required": labs_req,
        "l300_labs_remaining": labs_remaining,
        "remaining_lab_options": incomplete_labs,
        "recommendation": (
            "You meet the L300 prerequisite and can register for the next L400 Bootcamp cohort."
            if eligible
            else f"Not eligible yet. Complete {labs_remaining} more L300 challenge lab(s) to reach {labs_req}/7 and unlock L400 registration."
        ),
    }


def update_enablement_progress(
    track: Annotated[
        EnablementTrackId,
        Field(
            description="Enablement track ID to update ('L200', 'L300', 'L400', 'PCA', 'GAIL', or 'CDL')."
        ),
    ],
    status: Annotated[
        WritableTaskStatus,
        Field(description="New completion status ('not_started', 'in_progress', or 'completed')."),
    ],
    labs_completed: Annotated[
        int | None,
        Field(
            ge=0,
            le=7,
            description=(
                "Optional number of L300 challenge labs completed (integer from 0 to 7; "
                "6 or more automatically completes L300 and unblocks L400)."
            ),
        ),
    ] = None,
    tool_context: ToolContext | None = None,
) -> UpdateEnablementResult:
    """Update the mock completion status of an FDE enablement track in session state.

    Automatically enforces L300 -> L400 prerequisite rules (e.g., reaching 6/7
    L300 labs marks L300 completed and unblocks L400).

    Args:
        track: Track ID ('L200', 'L300', 'L400', 'PCA', 'GAIL', 'CDL').
        status: New status ('not_started', 'in_progress', or 'completed').
        labs_completed: Optional integer count (0-7) when updating L300 labs.
    """
    if tool_context is None:
        return {"status": "error", "message": "Missing ToolContext."}

    profile = get_active_profile(tool_context)
    enablement = profile["enablement"]
    tid = track.upper()
    if tid not in enablement:
        return {
            "status": "error",
            "message": f"Unknown track '{track}'. Valid tracks: {', '.join(ALL_TRACKS)}.",
        }

    # Prevent bypassing the L300 -> L400 prerequisite gate.
    if tid == "L400" and status in ("in_progress", "completed"):
        blockers = _compute_blockers("L400", enablement)
        if blockers:
            return {
                "status": "blocked",
                "message": "Cannot start or complete L400 while prerequisites are unmet.",
                "blockers": blockers,
            }

    target = enablement[tid]
    if tid == "L300":
        labs = target.get("labs", [])
        labs_req = target.get("labs_required") or 6
        if labs_completed is not None:
            clamped = max(0, min(len(labs), labs_completed))
            target["labs_completed"] = clamped
            for idx, lab in enumerate(labs):
                lab["completed"] = idx < clamped
            target["status"] = (
                "completed"
                if clamped >= labs_req
                else ("in_progress" if clamped > 0 else "not_started")
            )
        else:
            target["status"] = status
            if status == "completed" and (target.get("labs_completed") or 0) < labs_req:
                target["labs_completed"] = labs_req
                for idx, lab in enumerate(labs):
                    lab["completed"] = idx < labs_req
    else:
        target["status"] = status

    l400_before = enablement["L400"]["status"]
    _reconcile_statuses(profile)
    l400_after = enablement["L400"]["status"]
    save_active_profile(tool_context, profile)

    result: dict[str, Any] = {
        "status": "ok",
        "updated_track": _summarize_track(tid, enablement, profile["onboarding_day"]),
    }
    if l400_before == "blocked" and l400_after != "blocked":
        result["unlocked"] = (
            "L400 Agent Building Bootcamp is now UNBLOCKED and ready for registration!"
        )
    return result
