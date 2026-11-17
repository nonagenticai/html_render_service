"""Stale-batch reaper + a reason column on ``batch_jobs``.

``batch_jobs`` rows were IMMORTAL. The table has no lease columns and is never
claimed — a batch is driven by an in-process ``asyncio.create_task``, and the
only writers that advance it live in that task. So a pod restart or an unhandled
error strands the row in ``pending``/``processing`` with:

* no reaper — the lease reclaim added in ``0002_job_lease`` is ``redesign_jobs``
  only; the aggregate table was left out of a mechanism written for child rows;
* no prune — ``CleanupService`` has no caller anywhere in the repo, and
  ``RetentionSettings.enabled`` defaults to ``False``.

The batch status endpoint is then wedged forever for that client.

This revision:

1. adds ``batch_jobs.error_message`` so a terminal batch can carry a REASON
   (``BatchManager.mark_batch_failed`` accepted one and dropped it on the floor);
1b. adds ``batch_jobs.last_progress_at``, the progress clock that lets the sweep
   be a NO-PROGRESS WINDOW instead of a wall-clock ceiling. ``started_at`` is
   written once by ``mark_processing`` and never again, so a ceiling built on it
   must exceed the longest legitimate batch — 20 pages at concurrency 3 is 7
   sequential waves, and an in-process batch child has NO wall-clock cap at all
   (``max_job_lifetime_seconds`` bounds only the queue path). There is no honest
   number there, and one that is too small reaps a HEALTHY batch;
2. creates ``reap_stale_batch_jobs(int)``, a SECURITY DEFINER sweep in the same
   shape as ``claim_next_render_job`` — cross-tenant, RLS-bypassing, atomic —
   that moves a batch with no page progress (and its non-queue-owned children,
   and their variants) to ``failed`` with that reason. It clamps its argument up
   to a 3600s floor: it is SECURITY DEFINER and terminates work across every
   tenant, so it cannot take the caller's word for how aggressive to be;
3. re-issues ``claim_next_render_job`` so a RE-CLAIM retires the previous
   attempt's unfinished ``redesign_variants``, which otherwise sit ``pending``
   forever (``create_variant`` only inserts; nothing ever resets them).

RECLAIM, NOT DELETION. Nothing here removes a row: retention/pruning stays
disabled by default and untouched. The change is that a dead batch reaches a
terminal state instead of hanging.

``claim_next_render_job`` is CREATE OR REPLACE'd, not dropped: the signature and
return type are unchanged, so its ACL (and the out-of-band app-role GRANT that a
DROP would destroy — see ``0010``) survive.

Revision ID: 0011_batch_reap
Revises: 0010_tenant_id_uuid
Create Date: 2026-09-06 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

from src.database.rls_resolvers import (
    CLAIM_FN_SQL,
    GRANT_REAP_BATCH_FN,
    REAP_BATCH_FN_SQL,
    REVOKE_REAP_BATCH_FN,
)

revision: str = "0011_batch_reap"
down_revision: str | None = "0010_tenant_id_uuid"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


#: Era-correct (pre-0011) claim resolver, spelled out literally for the
#: downgrade — identical to ``0010``'s body without the variant retirement.
#: Never rendered from the live module: importing the CURRENT source into a
#: downgrade would replay the change it is meant to undo.
_LEGACY_CLAIM_FN_SQL = """
CREATE OR REPLACE FUNCTION claim_next_render_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id uuid, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant uuid; v_options jsonb; v_attempts int;
  BEGIN
    SELECT rj.id, rj.job_id, rj.tenant_id, rj.job_options, rj.attempts
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts
    FROM redesign_jobs rj
    WHERE rj.job_options IS NOT NULL
      AND (
        rj.status = 'pending'
        OR (rj.status = 'processing'
            AND (rj.lease_expires_at IS NULL OR rj.lease_expires_at < now()))
      )
    ORDER BY rj.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;  -- empty queue
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE redesign_jobs
        SET status = 'failed',
            error_message = 'exceeded max claim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now()
      WHERE id = v_id;
      RETURN;  -- poison job skipped
    END IF;

    UPDATE redesign_jobs
      SET status = 'processing',
          claimed_by = p_worker_id,
          started_at = COALESCE(started_at, now()),
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""


def upgrade() -> None:
    op.execute("ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS error_message TEXT")
    op.execute("ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS last_progress_at TIMESTAMPTZ")
    # Seed the progress clock for batches that are IN FLIGHT at deploy time. They
    # have no progress history, so "as of now" is the only truthful value — and
    # leaving it NULL would judge them on ``started_at``, i.e. the wall-clock
    # ceiling this column exists to replace, giving them a false-positive reap
    # window on the very first sweep after the rollout.
    op.execute(
        "UPDATE batch_jobs SET last_progress_at = now() "
        "WHERE status IN ('pending', 'processing') AND last_progress_at IS NULL"
    )
    # Keeps the sweep's scan cheap as terminal batches accumulate: the reaper only
    # ever looks at non-terminal rows.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_batch_jobs_reapable "
        "ON batch_jobs (status, last_progress_at, started_at, created_at) "
        "WHERE status IN ('pending', 'processing')"
    )
    op.execute(REAP_BATCH_FN_SQL)
    op.execute(REVOKE_REAP_BATCH_FN)
    op.execute(GRANT_REAP_BATCH_FN)
    op.execute(CLAIM_FN_SQL)


def downgrade() -> None:
    op.execute(_LEGACY_CLAIM_FN_SQL)
    op.execute("DROP FUNCTION IF EXISTS reap_stale_batch_jobs(int)")
    op.execute("DROP INDEX IF EXISTS ix_batch_jobs_reapable")
    op.execute("ALTER TABLE batch_jobs DROP COLUMN IF EXISTS last_progress_at")
    op.execute("ALTER TABLE batch_jobs DROP COLUMN IF EXISTS error_message")
