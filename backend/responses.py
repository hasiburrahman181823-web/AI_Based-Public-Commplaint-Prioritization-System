"""Public response contracts, including deliberately omitted private fields.

Use ``response_model_exclude_unset=True`` on routes returning redacted records.
An absent private field then stays absent rather than appearing as ``null``.
The stored ML analysis remains extensible as research artifacts evolve.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ComplaintView(BaseModel):
    """Submission receipt or admin record; optional fields require read access."""

    id: str
    lang: Literal["bn", "en", "banglish"]
    category: str | None
    severity: str | None
    priority_score: float
    priority_label: str
    status: Literal["submitted", "in_review", "confirmed", "in_progress", "resolved", "rejected"]
    area_id: str | None
    group_id: str | None
    version: int = Field(ge=1, description="Use as expected_version in an admin PATCH.")
    group_version: int | None = Field(description="Current group version, if the complaint belongs to a group.")
    created_at: str = Field(description="UTC ISO-8601 creation timestamp.")
    updated_at: str = Field(description="UTC ISO-8601 update timestamp.")
    text: str | None = Field(default=None, description="Omitted from optional public reads.")
    location: str | None = Field(default=None, description="Submitted location; omitted from optional public reads.")
    contact: str | None = Field(default=None, description="Optional private contact. Only authenticated administrator reads include this field.")
    timestamp: str | None = Field(default=None, description="Submitted timestamp, with explicit timezone.")
    confidence: float | None = Field(default=None, description="Calibrated classification confidence, not priority or duplicate confidence.")
    review_required: bool | None = None
    analysis: dict[str, Any] | None = Field(default=None, description="Automatic analysis snapshot. Citizen receipts redact other reports' identifiers and metadata.")


class ComplaintPage(BaseModel):
    items: list[ComplaintView]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)


class CitizenAccountView(BaseModel):
    id: str
    email: str
    full_name: str | None
    phone: str | None
    created_at: str


class CitizenAuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(gt=0)
    account: CitizenAccountView


class GroupView(BaseModel):
    id: str
    status: Literal["proposed", "confirmed", "reviewed_split"]
    version: int = Field(ge=1, description="Use as expected_version for confirm or split.")
    size: int = Field(ge=0)
    created_at: str
    updated_at: str
    confirmed_by: str | None = Field(default=None, description="Admin identity; omitted from optional public reads.")
    complaint_ids: list[str] | None = Field(default=None, description="Up to 1,000 sorted member IDs; omitted from optional public reads.")
    members_truncated: bool | None = Field(default=None, description="True when the member-ID list is capped at 1,000.")


class GroupPage(BaseModel):
    items: list[GroupView]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)


class GroupSplitResult(BaseModel):
    original_group: GroupView
    new_group: GroupView


class AnalysisResponse(BaseModel):
    """Core triage contract with additional research/provenance fields preserved."""

    model_config = ConfigDict(extra="allow")

    category: str | None
    severity: str | None
    priority_score: float
    priority_label: str
    confidence: float | None = Field(description="Minimum calibrated category/severity top probability; not a calibrated joint, priority or duplicate probability.")
    explanation: dict[str, Any]
    review_required: bool
    duplicate_candidates: list[dict[str, Any]] = Field(description="Redacted score/guard suggestions. Other citizens' record IDs and metadata are omitted.")
    lang: Literal["bn", "en", "banglish"]


class StatsResponse(BaseModel):
    total_complaints: int = Field(ge=0)
    total_groups: int = Field(ge=0)
    needs_human_review: int = Field(ge=0)
    by_category: dict[str, int]
    by_priority: dict[str, int]
    by_status: dict[str, int]
    by_area: dict[str, int]
    by_language: dict[str, int]


class TrackingResponse(BaseModel):
    """Possession of the random tracking ID permits only this minimal summary."""

    id: str
    status: Literal["submitted", "in_review", "confirmed", "in_progress", "resolved", "rejected"]
    category: str | None
    severity: str | None
    priority_score: float
    priority_label: str
    review_required: bool
    area_id: str | None
    lang: Literal["bn", "en", "banglish"]
    created_at: str
    updated_at: str


class PriorityDay(BaseModel):
    date: str = Field(description="UTC creation date, YYYY-MM-DD.")
    priority: str
    count: int = Field(ge=0)


class ReviewWorkload(BaseModel):
    status: str
    review_required: bool
    count: int = Field(ge=0)


class AnalyticsResponse(StatsResponse):
    priority_over_time: list[PriorityDay]
    review_workload: list[ReviewWorkload]


class MLReadiness(BaseModel):
    ready: bool
    status: Literal["ready", "degraded"]
    classifiers_loaded: bool
    duplicate_model_loaded: bool
    review_policy_loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    database: str | None = None
    rate_limit_storage: str | None = None
    ml: MLReadiness | None = None
    detail: str | None = None
