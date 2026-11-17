"""Durable claim-queue: lease/claim columns + job_options bundle.

Adds the columns the Postgres claim-queue (Phase 1 of the rearchitecture)
needs on ``redesign_jobs``:

- ``claimed_by``       which pod holds the job (for fast shutdown release)
- ``lease_expires_at`` when the claim expires; a ``processing`` row past this
                       (or with a NULL lease) is reclaimable by any worker
- ``attempts``         claim count, for the poison-job cap
- ``job_options``      JSONB bundle of the per-request render/processing params
                       so any pod/restart can reconstruct the work. This is also
                       the queue-ownership flag: a row with ``job_options IS NOT
                       NULL`` is owned by the worker pool (single ``create_redesign``
                       / ``refine`` jobs); batch child rows leave it NULL and keep
                       their existing in-process orchestration.

Idempotent raw SQL: a no-op on a DB that already has the columns, corrective on
a fresh CI DB. The service role owns ``redesign_jobs`` (created in 0001), so the
``ALTER ... IF [NOT] EXISTS`` ownership check passes.

NOTE (deferred): per-phase checkpointing (a ``current_phase`` column to skip
already-completed LLM phases on reclaim) is intentionally NOT added. The MVP
re-runs the whole job on reclaim, which is safe because the phases are
idempotent; add a checkpoint only if crash-time LLM-token waste proves material.

Revision ID: 0002_job_lease
Revises: 0001_initial
Create Date: 2026-06-04 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_job_lease"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS claimed_by VARCHAR(64)")
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ")
    op.execute(
        "ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE redesign_jobs ADD COLUMN IF NOT EXISTS job_options JSONB")
    # Partial index over only queue-owned rows keeps the claim query's
    # status/lease scan cheap as completed jobs accumulate.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_redesign_jobs_claimable "
        "ON redesign_jobs (status, lease_expires_at) "
        "WHERE job_options IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_redesign_jobs_claimable")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS job_options")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS attempts")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS lease_expires_at")
    op.execute("ALTER TABLE redesign_jobs DROP COLUMN IF EXISTS claimed_by")
