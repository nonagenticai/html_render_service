"""Widen the ``style`` columns from VARCHAR(50) to VARCHAR(255).

``style`` is a free-text design descriptor (e.g. the orchestrator's
website_redesign passes ``"bold editorial, large serif headlines, dark theme,
single accent"`` — 58 chars). The original 50-char cap silently overflowed:
single-style redesign happened to fit ("modern, generous whitespace, single
CTA" = 40), but realistic multi-style variant fan-out did not — every
``redesign_variants`` INSERT raised StringDataRightTruncation, which the
per-variant absorb-and-continue handler swallowed, so the job completed with
ZERO variants and an empty ``visuals`` list. (Caught by the website_review
variant E2E, the first to exercise the multi-style path end-to-end.)

Widen both ``redesign_jobs.style`` and ``redesign_variants.style``. ALTER
COLUMN TYPE to a wider varchar is a metadata-only, data-preserving, re-runnable
change; the service role owns both tables (created in 0001), so the ALTER's
ownership check passes.

Revision ID: 0003_widen_style
Revises: 0002_job_lease
Create Date: 2026-06-04 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_widen_style"
down_revision: str | None = "0002_job_lease"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs ALTER COLUMN style TYPE VARCHAR(255)")
    op.execute("ALTER TABLE redesign_variants ALTER COLUMN style TYPE VARCHAR(255)")


def downgrade() -> None:
    # Truncates to 50 chars on the way down so the narrower type accepts existing
    # rows; lossy by nature of the original (too-small) cap.
    op.execute(
        "ALTER TABLE redesign_jobs ALTER COLUMN style TYPE VARCHAR(50) USING left(style, 50)"
    )
    op.execute(
        "ALTER TABLE redesign_variants ALTER COLUMN style TYPE VARCHAR(50) USING left(style, 50)"
    )
