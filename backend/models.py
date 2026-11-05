"""Portable relational persistence; all mutations are application transactions.

The four domain tables deliberately separate submitted analysis, human feedback,
and append-only audit evidence. Input validation additionally rejects non-finite
floats before SQL; SQL checks protect ordinary out-of-range direct writes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, text as sql_text, true
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class UTCDateTime(TypeDecorator):
    """Return aware UTC datetimes even when SQLite drops timezone information."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class CitizenAccount(Base):
    __tablename__ = "citizen_accounts"
    __table_args__ = (
        CheckConstraint("length(email) BETWEEN 6 AND 254", name="ck_citizen_accounts_email_length"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(254), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow, onupdate=utcnow)


class CitizenComplaintLink(Base):
    __tablename__ = "citizen_complaint_links"
    __table_args__ = (
        Index("ix_citizen_complaint_links_account_created", "account_id", "created_at"),
    )

    complaint_id: Mapped[str] = mapped_column(String(36), ForeignKey("complaints.id", ondelete="CASCADE"), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(36), ForeignKey("citizen_accounts.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class DuplicateGroup(Base):
    __tablename__ = "duplicate_groups"
    __table_args__ = (
        CheckConstraint("status IN ('proposed', 'confirmed', 'reviewed_split')", name="ck_duplicate_groups_status"),
        CheckConstraint("version >= 1", name="ck_duplicate_groups_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="proposed", server_default="proposed", index=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow, onupdate=utcnow)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    complaints: Mapped[list["Complaint"]] = relationship(back_populates="duplicate_group", passive_deletes=True)
    reviews: Mapped[list["AdminReview"]] = relationship(back_populates="duplicate_group", passive_deletes=True)


class Complaint(Base):
    __tablename__ = "complaints"
    __table_args__ = (
        CheckConstraint("length(text) BETWEEN 1 AND 4000", name="ck_complaints_text_length"),
        CheckConstraint("lang IN ('bn', 'en', 'banglish')", name="ck_complaints_lang"),
        CheckConstraint("priority_score >= 0", name="ck_complaints_priority_score"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_complaints_confidence"),
        CheckConstraint("status IN ('submitted', 'in_review', 'confirmed', 'in_progress', 'resolved', 'rejected')", name="ck_complaints_status"),
        CheckConstraint("version >= 1", name="ck_complaints_version"),
        Index("ix_complaints_filters", "category", "priority_label", "area_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    severity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    priority_label: Mapped[str] = mapped_column(String(32), nullable=False, default="low", server_default="low", index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="submitted", server_default="submitted", index=True)
    area_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Operational contact is private: never included in analysis, embedding
    # records, public tracking or feedback-export training data.
    contact: Mapped[str | None] = mapped_column(String(256), nullable=True)
    timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    group_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("duplicate_groups.id", ondelete="SET NULL"), nullable=True, index=True)
    source_report_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    analysis: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default=sql_text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow, onupdate=utcnow)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    duplicate_group: Mapped[DuplicateGroup | None] = relationship(back_populates="complaints")
    reviews: Mapped[list["AdminReview"]] = relationship(back_populates="complaint", passive_deletes=True)


class AdminReview(Base):
    __tablename__ = "admin_reviews"
    __table_args__ = (
        CheckConstraint("complaint_id IS NOT NULL OR group_id IS NOT NULL", name="ck_admin_reviews_target"),
        CheckConstraint("action IN ('complaint_update', 'group_confirm', 'group_split')", name="ck_admin_reviews_action"),
        CheckConstraint("length(reason) >= 1", name="ck_admin_reviews_reason"),
        Index("ix_admin_reviews_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    complaint_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("complaints.id", ondelete="RESTRICT"), nullable=True, index=True)
    group_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("duplicate_groups.id", ondelete="RESTRICT"), nullable=True, index=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    previous_values: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default=sql_text("'{}'"))
    new_values: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default=sql_text("'{}'"))
    feedback_exportable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    complaint: Mapped[Complaint | None] = relationship(back_populates="reviews")
    duplicate_group: Mapped[DuplicateGroup | None] = relationship(back_populates="reviews")


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_entity", "entity_type", "entity_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    changes: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default=sql_text("'{}'"))
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow, index=True)
