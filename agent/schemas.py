"""Typed schemas (inputs, outputs, and domain models) for FDE Duckie."""

from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared enum literals
# ---------------------------------------------------------------------------

TaskStatus = Literal["not_started", "in_progress", "completed", "blocked"]
WritableTaskStatus = Literal["not_started", "in_progress", "completed"]
PacingStatus = Literal["completed", "on_track", "at_risk", "overdue"]
EnablementTrackId = Literal["L200", "L300", "L400", "PCA", "GAIL", "CDL"]
EnablementTrackFilter = Literal["all", "L200", "L300", "L400", "PCA", "GAIL", "CDL"]
SetupCategoryFilter = Literal[
    "all", "workstation", "argolis", "github", "rituals", "ramp_milestones"
]
PersonaId = Literal["alex_noogler", "priya_fde", "jordan_fde"]
ToolOutcome = Literal["ok", "error", "blocked"]
ComplexityLevel = Literal["simple", "complex"]


# ---------------------------------------------------------------------------
# Domain Pydantic models
# ---------------------------------------------------------------------------


class LabRecord(BaseModel):
    id: str = Field(description="Unique lab identifier.")
    title: str = Field(description="Human-readable challenge lab title.")
    completed: bool = Field(default=False, description="Whether the lab is completed.")


class EnablementTrack(BaseModel):
    track_id: EnablementTrackId = Field(description="Enablement track identifier.")
    title: str = Field(description="Full title of the enablement track.")
    status: TaskStatus = Field(description="Current completion status.")
    target_window_days: tuple[int, int] = Field(
        description="Target onboarding day window [start_day, end_day]."
    )
    hard_deadline_day: int | None = Field(
        default=None, description="Hard onboarding day deadline, if applicable."
    )
    required: bool = Field(
        default=True, description="Whether this track is mandatory for FDE onboarding."
    )
    prerequisite_tracks: list[EnablementTrackId] = Field(
        default_factory=list, description="Tracks that must be completed first."
    )
    labs_required: int | None = Field(
        default=None, description="Minimum number of labs required (e.g., 6 for L300)."
    )
    labs_completed: int | None = Field(
        default=None, description="Number of labs completed so far."
    )
    labs: list[LabRecord] = Field(
        default_factory=list, description="Individual lab completion records."
    )
    notes: str = Field(default="", description="Additional track guidance or context.")
    link: str = Field(default="", description="Internal go/ or portal link.")


class SetupItem(BaseModel):
    id: str = Field(description="Unique setup item identifier.")
    category: str = Field(description="Setup category (workstation, argolis, github, rituals).")
    title: str = Field(description="Setup task title.")
    status: WritableTaskStatus = Field(description="Current setup status.")
    link: str = Field(default="", description="Relevant internal go/ link.")
    notes: str = Field(default="", description="Progress notes or caveats.")


class NooglerProfile(BaseModel):
    persona_id: str = Field(description="Unique persona identifier.")
    ldap: str = Field(description="Noogler Google LDAP.")
    full_name: str = Field(description="Noogler display name.")
    onboarding_day: int = Field(ge=1, description="Current day number in FDE onboarding.")
    pod: str = Field(description="Assigned FDE pod name.")
    manager: str = Field(description="Pod manager name and LDAP.")
    buddy: str = Field(description="Assigned onboarding buddy name and LDAP.")
    location: str = Field(description="Office location code.")
    setup_items: list[SetupItem] = Field(description="Day-1/Week-1 setup checklist items.")
    enablement: dict[str, EnablementTrack] = Field(
        description="Map of enablement track IDs to track progress records."
    )


class DuckieResponse(BaseModel):
    source: str = Field(description="Backend identifier that produced this answer.")
    is_mock: bool = Field(
        default=False, description="True if returned by a local stub/mock backend."
    )
    matched_topic: str = Field(description="Topic category matched by the Duckie backend.")
    answer: str = Field(description="Answer text from Duckie.")
    references: list[str] = Field(
        default_factory=list, description="Internal go/ links cited in the answer."
    )
    note: str = Field(default="", description="Optional provenance or stub disclaimer note.")


# ---------------------------------------------------------------------------
# Specialist AgentTool input schemas (explicit JSON Schema for root -> specialist)
# ---------------------------------------------------------------------------


class EnablementAgentInput(BaseModel):
    """Input payload for invoking the FDE Enablement Specialist agent."""

    request: str = Field(
        min_length=1,
        description=(
            "Specific FDE enablement question or progress update request "
            "(e.g., check L200/L300/L400/PCA status, verify L400 eligibility, "
            "update L300 lab count, or explain go/fde-enablement-dash sync delays)."
        ),
    )


class OnboardingGuideInput(BaseModel):
    """Input payload for invoking the FDE Onboarding & Ramp Guide Specialist agent."""

    request: str = Field(
        min_length=1,
        description=(
            "Specific FDE onboarding setup, 30/60/90-day ramp milestone, "
            "next-action prioritization, or pod/buddy/chat directory question."
        ),
    )


class DuckieProxyInput(BaseModel):
    """Input payload for forwarding a general Google question to the Duckie Proxy agent."""

    request: str = Field(
        min_length=1,
        description=(
            "General, non-FDE-specific Google question to forward to Duckie "
            "(e.g., Workday, gBenefits, PTO, corporate card, Piper/CitC, go/wtf)."
        ),
    )


class ComplexityAssessment(BaseModel):
    """Structured complexity classification written to session state by step 1."""

    complexity: ComplexityLevel = Field(
        description=(
            "'simple' for greetings or single-domain lookups/updates; "
            "'complex' for multi-domain compound queries, comparisons, planning, "
            "or troubleshooting."
        ),
    )
    reason: str = Field(
        min_length=1,
        description="Short one-line reason for the complexity classification.",
    )


# ---------------------------------------------------------------------------
# Tool response TypedDicts (dict at runtime + full response_json_schema in ADK)
# ---------------------------------------------------------------------------


class PersonaSummaryDict(TypedDict):
    persona_id: Annotated[str, Field(description="Persona identifier.")]
    full_name: Annotated[str, Field(description="Noogler full name.")]
    onboarding_day: Annotated[int, Field(ge=1, description="Current onboarding day.")]


class ActiveProfileDict(TypedDict):
    persona_id: Annotated[str, Field(description="Active persona identifier.")]
    full_name: Annotated[str, Field(description="Active Noogler full name.")]
    ldap: Annotated[str, Field(description="Active Noogler LDAP.")]
    onboarding_day: Annotated[int, Field(ge=1, description="Current onboarding day.")]
    pod: Annotated[str, Field(description="Assigned FDE pod.")]


class SwitchPersonaResult(TypedDict):
    status: Annotated[Literal["ok", "error"], Field(description="Operation status.")]
    message: Annotated[str, Field(description="Human-readable outcome message.")]
    active_profile: NotRequired[
        Annotated[ActiveProfileDict, Field(description="Newly activated Noogler profile.")]
    ]
    available_personas: NotRequired[
        Annotated[
            list[PersonaSummaryDict],
            Field(description="Valid personas available when an unknown ID is supplied."),
        ]
    ]


class NooglerBriefDict(TypedDict):
    full_name: Annotated[str, Field(description="Noogler full name.")]
    ldap: Annotated[str, Field(description="Noogler LDAP.")]
    onboarding_day: Annotated[int, Field(ge=1, description="Current onboarding day.")]


class CoreProgressDict(TypedDict):
    completed: Annotated[int, Field(ge=0, description="Number of core tracks completed.")]
    total: Annotated[int, Field(ge=1, description="Total number of core tracks.")]
    percent: Annotated[int, Field(ge=0, le=100, description="Completion percentage (0-100).")]
    core_tracks: Annotated[list[str], Field(description="List of core track IDs.")]


class TrackSummaryDict(TypedDict):
    track_id: Annotated[str, Field(description="Track identifier (e.g., L200, L300, L400, PCA).")]
    title: Annotated[str, Field(description="Full enablement track title.")]
    required: Annotated[bool, Field(description="Whether the track is mandatory.")]
    status: Annotated[TaskStatus, Field(description="Current completion status.")]
    pacing: Annotated[PacingStatus, Field(description="Pacing classification vs. target window.")]
    target_window: Annotated[str, Field(description="Target onboarding day window.")]
    hard_deadline_day: Annotated[
        int | None, Field(description="Hard onboarding day deadline, if any.")
    ]
    blockers: Annotated[list[str], Field(description="Unmet prerequisite blockers, if any.")]
    notes: Annotated[str, Field(description="Track notes and context.")]
    link: Annotated[str, Field(description="Relevant internal go/ or portal link.")]
    labs_completed: NotRequired[
        Annotated[int, Field(ge=0, description="Completed challenge lab count (for lab tracks).")]
    ]
    labs_required: NotRequired[
        Annotated[int, Field(ge=1, description="Minimum challenge labs required.")]
    ]
    labs_total: NotRequired[
        Annotated[int, Field(ge=1, description="Total challenge labs in the track.")]
    ]
    incomplete_labs: NotRequired[
        Annotated[list[str], Field(description="Titles of remaining incomplete labs.")]
    ]


class DashboardSyncDict(TypedDict):
    dashboard_url: Annotated[str, Field(description="Enablement dashboard go/ link.")]
    faq_url: Annotated[str, Field(description="Dashboard FAQ go/ link.")]
    sync_delay_note: Annotated[str, Field(description="Guidance on dashboard credit sync lag.")]
    known_l300_bugs_url: Annotated[str, Field(description="Known L300 lab issues go/ link.")]


class EnablementStatusResult(TypedDict):
    status: Annotated[Literal["ok", "error"], Field(description="Operation status.")]
    message: NotRequired[Annotated[str, Field(description="Error message when status='error'.")]]
    noogler: NotRequired[
        Annotated[NooglerBriefDict, Field(description="Active Noogler metadata.")]
    ]
    core_progress: NotRequired[
        Annotated[CoreProgressDict, Field(description="Aggregate core enablement progress.")]
    ]
    tracks: NotRequired[
        Annotated[list[TrackSummaryDict], Field(description="Requested enablement track details.")]
    ]
    dashboard_sync: NotRequired[
        Annotated[DashboardSyncDict, Field(description="Enablement dashboard sync reference.")]
    ]


class L400EligibilityResult(TypedDict):
    status: Annotated[Literal["ok", "error"], Field(description="Operation status.")]
    message: NotRequired[Annotated[str, Field(description="Error message when status='error'.")]]
    noogler: NotRequired[Annotated[str, Field(description="Active Noogler full name.")]]
    eligible_for_l400: NotRequired[
        Annotated[bool, Field(description="True if L300 prerequisite (>=6/7 labs) is satisfied.")]
    ]
    l400_status: NotRequired[Annotated[TaskStatus, Field(description="Current L400 status.")]]
    l300_status: NotRequired[Annotated[TaskStatus, Field(description="Current L300 status.")]]
    l300_labs_completed: NotRequired[
        Annotated[int, Field(ge=0, description="Completed L300 challenge labs.")]
    ]
    l300_labs_required: NotRequired[
        Annotated[int, Field(ge=1, description="Required L300 challenge labs (6).")]
    ]
    l300_labs_remaining: NotRequired[
        Annotated[int, Field(ge=0, description="Additional L300 labs needed to unlock L400.")]
    ]
    remaining_lab_options: NotRequired[
        Annotated[list[str], Field(description="Titles of incomplete L300 labs to choose from.")]
    ]
    recommendation: NotRequired[
        Annotated[str, Field(description="Actionable eligibility summary.")]
    ]


class UpdateEnablementResult(TypedDict):
    status: Annotated[ToolOutcome, Field(description="Outcome ('ok', 'blocked', or 'error').")]
    message: NotRequired[Annotated[str, Field(description="Explanation when blocked or errored.")]]
    blockers: NotRequired[
        Annotated[list[str], Field(description="Prerequisite blockers preventing the update.")]
    ]
    updated_track: NotRequired[
        Annotated[TrackSummaryDict, Field(description="Updated track summary after saving.")]
    ]
    unlocked: NotRequired[
        Annotated[str, Field(description="Notice emitted if this update unblocked L400.")]
    ]


class RecommendedActionDict(TypedDict):
    priority: Annotated[int, Field(ge=1, le=3, description="Priority rank (1 = highest).")]
    category: Annotated[
        Literal["setup", "enablement", "certification", "ramp"],
        Field(description="Onboarding workstream category."),
    ]
    action: Annotated[str, Field(description="Concrete recommended action.")]
    current_status: Annotated[str, Field(description="Current item status.")]
    reason: Annotated[str, Field(description="Why this action is prioritized now.")]
    link: Annotated[str, Field(description="Relevant internal go/ link.")]


class NextActionsResult(TypedDict):
    status: Annotated[Literal["ok", "error"], Field(description="Operation status.")]
    message: NotRequired[Annotated[str, Field(description="Error message when status='error'.")]]
    noogler: NotRequired[Annotated[str, Field(description="Active Noogler full name.")]]
    onboarding_day: NotRequired[Annotated[int, Field(ge=1, description="Current onboarding day.")]]
    current_phase: NotRequired[
        Annotated[str, Field(description="Active 30/60/90-day onboarding phase.")]
    ]
    recommended_actions: NotRequired[
        Annotated[
            list[RecommendedActionDict],
            Field(description="Top 3 prioritized next actions."),
        ]
    ]


class SetupItemStatusDict(TypedDict):
    id: Annotated[str, Field(description="Setup item identifier.")]
    category: Annotated[str, Field(description="Setup category.")]
    title: Annotated[str, Field(description="Setup task title.")]
    status: Annotated[WritableTaskStatus, Field(description="Personal completion status.")]
    link: Annotated[str, Field(description="Relevant internal go/ link.")]
    notes: Annotated[str, Field(description="Personal progress notes.")]


class RampPhaseDict(TypedDict):
    window: Annotated[str, Field(description="Onboarding day window heading.")]
    items: Annotated[list[str], Field(description="Milestones in this ramp window.")]


class SetupGuideDict(TypedDict):
    title: Annotated[str, Field(description="Guide heading.")]
    steps: NotRequired[Annotated[list[str], Field(description="Step-by-step instructions.")]]
    phases: NotRequired[
        Annotated[list[RampPhaseDict], Field(description="30/60/90-day ramp phase breakdown.")]
    ]
    links: Annotated[list[str], Field(description="Reference internal go/ links.")]


class SetupChecklistResult(TypedDict):
    status: Annotated[Literal["ok", "error"], Field(description="Operation status.")]
    message: NotRequired[Annotated[str, Field(description="Error message when status='error'.")]]
    category: NotRequired[Annotated[str, Field(description="Requested setup category.")]]
    personal_setup_status: NotRequired[
        Annotated[
            list[SetupItemStatusDict],
            Field(description="Active Noogler's current completion status for matching items."),
        ]
    ]
    guides: NotRequired[
        Annotated[
            dict[str, SetupGuideDict],
            Field(description="Step-by-step setup or ramp guides keyed by category."),
        ]
    ]


class TeamContactProfileDict(TypedDict):
    full_name: Annotated[str, Field(description="Active Noogler full name.")]
    pod: Annotated[str, Field(description="Assigned FDE pod.")]
    location: Annotated[str, Field(description="Office location code.")]
    manager: Annotated[str, Field(description="Assigned pod manager.")]
    onboarding_buddy: Annotated[str, Field(description="Assigned onboarding buddy.")]


class ChatSpaceDict(TypedDict):
    name: Annotated[str, Field(description="Google Chat space name.")]
    purpose: Annotated[str, Field(description="What the Chat space is used for.")]


class RitualDict(TypedDict):
    name: Annotated[str, Field(description="Calendar or recurring meeting name.")]
    link: Annotated[str, Field(description="Internal go/ link.")]
    cadence: Annotated[str, Field(description="Frequency or sync behavior.")]


class TeamContactsResult(TypedDict):
    status: Annotated[Literal["ok", "error"], Field(description="Operation status.")]
    message: NotRequired[Annotated[str, Field(description="Error message when status='error'.")]]
    noogler: NotRequired[
        Annotated[TeamContactProfileDict, Field(description="Noogler pod, manager, and buddy.")]
    ]
    chat_spaces: NotRequired[
        Annotated[list[ChatSpaceDict], Field(description="Key FDE Google Chat spaces.")]
    ]
    calendars_and_rituals: NotRequired[
        Annotated[
            list[RitualDict],
            Field(description="FDE calendars and recurring team syncs."),
        ]
    ]


class DuckieToolResult(TypedDict):
    source: Annotated[str, Field(description="Backend identifier that produced this response.")]
    is_mock: Annotated[bool, Field(description="True when served by the local Duckie stub.")]
    matched_topic: Annotated[str, Field(description="Topic category matched by Duckie.")]
    answer: Annotated[str, Field(description="Answer returned by Duckie.")]
    references: Annotated[list[str], Field(description="Internal go/ links cited in the answer.")]
    note: Annotated[str, Field(description="Provenance or stub notice.")]
