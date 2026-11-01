"""Create complaint, duplicate-group, review and audit persistence.

Revision ID: 0001_initial
Revises: none
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "duplicate_groups",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="proposed"),
        sa.Column("confirmed_by", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.CheckConstraint("status IN ('proposed', 'confirmed', 'reviewed_split')", name="ck_duplicate_groups_status"),
        sa.CheckConstraint("version >= 1", name="ck_duplicate_groups_version"),
    )
    op.create_index("ix_duplicate_groups_status", "duplicate_groups", ["status"])
    op.create_table(
        "complaints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("lang", sa.String(16), nullable=False),
        sa.Column("category", sa.String(128)),
        sa.Column("severity", sa.String(64)),
        sa.Column("priority_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("priority_label", sa.String(32), nullable=False, server_default="low"),
        sa.Column("confidence", sa.Float()),
        sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("status", sa.String(32), nullable=False, server_default="submitted"),
        sa.Column("area_id", sa.String(128)),
        sa.Column("location", sa.String(256)),
        sa.Column("timestamp", sa.DateTime(timezone=True)),
        sa.Column("group_id", sa.String(36), sa.ForeignKey("duplicate_groups.id", ondelete="SET NULL")),
        sa.Column("source_report_id", sa.String(128)),
        sa.Column("analysis", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.CheckConstraint("length(text) BETWEEN 1 AND 4000", name="ck_complaints_text_length"),
        sa.CheckConstraint("lang IN ('bn', 'en', 'banglish')", name="ck_complaints_lang"),
        sa.CheckConstraint("priority_score >= 0", name="ck_complaints_priority_score"),
        sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_complaints_confidence"),
        sa.CheckConstraint("status IN ('submitted', 'in_review', 'confirmed', 'in_progress', 'resolved', 'rejected')", name="ck_complaints_status"),
        sa.CheckConstraint("version >= 1", name="ck_complaints_version"),
    )
    for field in ("category", "priority_label", "status", "area_id", "group_id", "created_at"):
        op.create_index("ix_complaints_" + field, "complaints", [field])
    op.create_index("ix_complaints_filters", "complaints", ["category", "priority_label", "area_id", "status"])
    op.create_table(
        "admin_reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("complaint_id", sa.String(36), sa.ForeignKey("complaints.id", ondelete="RESTRICT")),
        sa.Column("group_id", sa.String(36), sa.ForeignKey("duplicate_groups.id", ondelete="RESTRICT")),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("previous_values", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("new_values", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("feedback_exportable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("complaint_id IS NOT NULL OR group_id IS NOT NULL", name="ck_admin_reviews_target"),
        sa.CheckConstraint("action IN ('complaint_update', 'group_confirm', 'group_split')", name="ck_admin_reviews_action"),
        sa.CheckConstraint("length(reason) >= 1", name="ck_admin_reviews_reason"),
    )
    for field in ("complaint_id", "group_id", "created_at"):
        op.create_index("ix_admin_reviews_" + field, "admin_reviews", [field])
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(36)),
        sa.Column("changes", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("request_id", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])
    op.create_index("ix_audit_log_entity", "audit_log", ["entity_type", "entity_id", "created_at"])


def downgrade():
    op.drop_table("audit_log")
    op.drop_table("admin_reviews")
    op.drop_table("complaints")
    op.drop_table("duplicate_groups")
