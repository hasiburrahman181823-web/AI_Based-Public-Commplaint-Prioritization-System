"""Add optional operational contact without modifying prior complaint content.

Revision ID: 0002_private_contact
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_private_contact"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("complaints", sa.Column("contact", sa.String(256), nullable=True))


def downgrade():
    # PostgreSQL and supported modern SQLite versions can drop this standalone
    # nullable column without rebuilding complaints and disturbing review FKs.
    op.drop_column("complaints", "contact")
