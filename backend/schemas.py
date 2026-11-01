"""Validated public/admin payloads and conservative, content-neutral spam checks."""

from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
import re
import unicodedata
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from ml.normalize import clean_text

STATUS_VALUES = ("submitted", "in_review", "confirmed", "in_progress", "resolved", "rejected")
Status = Literal["submitted", "in_review", "confirmed", "in_progress", "resolved", "rejected"]


def validate_submission(text: str, max_length: int = 1000, max_urls: int = 2) -> str:
    if not isinstance(text, str):
        raise ValueError("Complaint text must be a string")
    normalized = "".join(" " if unicodedata.category(char) == "Cc" else char
                         for char in unicodedata.normalize("NFKC", text)).strip()
    if not normalized or len(normalized) > max_length:
        raise ValueError(f"Complaint text must contain 1–{max_length} characters after Unicode normalization")
    urls = re.findall(r"(?:https?://|www\.)\S+", normalized, re.IGNORECASE)
    if len(urls) > max_urls:
        raise ValueError(f"Complaint text contains too many URLs (maximum {max_urls})")
    cleaned = clean_text(normalized)
    if not cleaned or not any(char.isalnum() for char in cleaned):
        raise ValueError("Describe the complaint in text; URL-only or punctuation-only submissions are not accepted")
    if any(not match.group(1).isspace() for match in re.finditer(r"(.)\1{24,}", normalized)):
        raise ValueError("Complaint appears to contain excessive repeated characters")
    words = cleaned.split()
    counts = Counter(words)
    if len(words) >= 20 and (len(counts) <= 2 or max(counts.values()) / len(words) > 0.75):
        raise ValueError("Complaint appears to contain excessive repeated words")
    return normalized


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComplaintCreate(StrictPayload):
    text: str = Field(min_length=1, max_length=1000)
    lang: Literal["bn", "en", "banglish"] | None = None
    location: str | None = Field(default=None, max_length=256)
    contact: str | None = Field(default=None, max_length=256)
    timestamp: datetime | None = None

    @field_validator("text", mode="before")
    @classmethod
    def check_text(cls, value):
        return validate_submission(value)

    @field_validator("location")
    @classmethod
    def trim_location(cls, value):
        return value.strip() or None if value is not None else None

    @field_validator("contact")
    @classmethod
    def private_contact(cls, value):
        if value is None:
            return None
        value = unicodedata.normalize("NFKC", value)
        if len(value.strip()) > 256 or any(unicodedata.category(char) == "Cc" for char in value):
            raise ValueError("Contact must be at most 256 characters without control characters")
        return value.strip() or None

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_type(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("Timestamp must be an ISO-8601 datetime with an explicit timezone")
        return value

    @field_validator("timestamp")
    @classmethod
    def timezone_required(cls, value):
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Timestamp needs an explicit timezone offset")
            return value.astimezone(timezone.utc)
        return value


class AnalyzeRequest(ComplaintCreate):
    pass


class AdminLogin(StrictPayload):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("username")
    @classmethod
    def trim_username(cls, value):
        if not value.strip():
            raise ValueError("Username must be nonempty")
        return value.strip()


class TokenResponse(StrictPayload):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(gt=0)


class ReviewPayload(StrictPayload):
    expected_version: StrictInt = Field(ge=1)
    reason: str = Field(min_length=3, max_length=2000)

    @field_validator("reason")
    @classmethod
    def meaningful_reason(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Provide a reason of at least three non-whitespace characters")
        return value.strip()


class ComplaintPatch(ReviewPayload):
    status: Status | None = None
    priority_score: float | None = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
    priority_label: str | None = Field(default=None, min_length=1, max_length=32)
    category: str | None = Field(default=None, min_length=1, max_length=128)
    severity: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("priority_score", mode="before")
    @classmethod
    def numeric_priority(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Priority score must be a finite numeric value")
        return value

    @field_validator("priority_label", "category", "severity")
    @classmethod
    def nonblank_label(cls, value):
        if value is not None:
            if not value.strip():
                raise ValueError("Override labels must be nonempty")
            return value.strip()
        return value

    @model_validator(mode="after")
    def changes_required(self):
        changed = self.model_fields_set - {"expected_version", "reason"}
        if not changed:
            raise ValueError("Provide at least one override/status field")
        if any(getattr(self, field) is None for field in changed):
            raise ValueError("Override/status fields cannot be null; omitted fields remain unchanged")
        return self


def _unique_ids(values):
    try:
        identifiers = [str(UUID(value)) for value in values]
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Every complaint ID must be a UUID") from exc
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Complaint IDs must be unique")
    return identifiers


class GroupConfirm(ReviewPayload):
    complaint_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("complaint_ids")
    @classmethod
    def optional_identifiers(cls, values):
        return _unique_ids(values) if values is not None else None


class GroupSplit(ReviewPayload):
    complaint_ids: list[str] = Field(min_length=1, max_length=1000)

    @field_validator("complaint_ids")
    @classmethod
    def unique_identifiers(cls, values):
        return _unique_ids(values)


class DataDeletionRequest(StrictPayload):
    complaint_ids: list[str] = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=3, max_length=2000)
    apply: StrictBool = False

    @field_validator("complaint_ids")
    @classmethod
    def unique_identifiers(cls, values):
        return _unique_ids(values)

    @field_validator("reason")
    @classmethod
    def meaningful_reason(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Provide a reviewed deletion reason")
        return value.strip()
