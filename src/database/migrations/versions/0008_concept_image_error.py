"""Add ``concept_image_error`` to ``redesign_jobs``.

Concept images are optional enrichment on top of a successful HTML redesign.
Previously a requested-but-FAILED generation (submit/poll error, timeout, failed
downstream batch) was swallowed to ``concept_image_assets = []/NULL`` and the job
still reported ``status = "completed"`` with no error — byte-identical to a job
that never requested concept images. This column records the failure reason so a
genuine failure is distinguishable from "none requested", WITHOUT downgrading the
job status (which gates artifact/result serving via many ``status == "completed"``
checks — the core job did complete).

Additive + nullable (NULL = not requested / succeeded / legitimately zero), so
existing rows and non-concept-image callers are unaffected. ``ADD COLUMN IF NOT
EXISTS`` keeps it re-runnable against a create_all-built test DB.

Revision ID: 0008_concept_image_error
Revises: 0007_verification_report
Create Date: 2026-07-17 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008_concept_image_error"
down_revision: str | None = "0007_verification_report"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS concept_image_error TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS concept_image_error")
