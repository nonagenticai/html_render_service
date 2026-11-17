"""Add verification-lane columns to ``redesign_jobs`` (status + report).

The automated verification lane gates assembly-mode completion: a design only
finishes if it passes render/console/a11y checks, else it fails loud with the
findings. These columns persist that outcome:

* ``verification_status`` — ``passed`` / ``failed`` / ``skipped``.
* ``verification_report`` — the structured findings + per-check results (JSONB).

Additive + nullable (NULL = not verified, e.g. legacy/freeform jobs), so existing
rows and non-assembly callers are unaffected. ``ADD COLUMN IF NOT EXISTS`` keeps
it re-runnable against a create_all-built test DB.

Revision ID: 0007_verification_report
Revises: 0006_assembly_fields
Create Date: 2026-07-01 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_verification_report"
down_revision: str | None = "0006_assembly_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS verification_status VARCHAR(20)")
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS verification_report JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS verification_report")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS verification_status")
