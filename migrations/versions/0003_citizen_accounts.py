"""Add citizen profiles and ownership links for signed-in submissions.

Revision ID: 0003_citizen_accounts
Revises: 0002_private_contact
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_citizen_accounts"
down_revision = "0002_private_contact"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "citizen_accounts",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length(email) BETWEEN 6 AND 254", name="ck_citizen_accounts_email_length"),
    )
    op.create_index("ix_citizen_accounts_email", "citizen_accounts", ["email"], unique=True)
    op.create_table(
        "citizen_complaint_links",
        sa.Column("complaint_id", sa.String(length=36), nullable=False),
        sa.Column("account_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["citizen_accounts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["complaint_id"], ["complaints.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("complaint_id"),
    )
    op.create_index("ix_citizen_complaint_links_account_id", "citizen_complaint_links", ["account_id"], unique=False)
    op.create_index(
        "ix_citizen_complaint_links_account_created",
        "citizen_complaint_links",
        ["account_id", "created_at"],
        unique=False,
    )


def downgrade():
    op.drop_index("ix_citizen_complaint_links_account_created", table_name="citizen_complaint_links")
    op.drop_index("ix_citizen_complaint_links_account_id", table_name="citizen_complaint_links")
    op.drop_table("citizen_complaint_links")
    op.drop_index("ix_citizen_accounts_email", table_name="citizen_accounts")
    op.drop_table("citizen_accounts")
