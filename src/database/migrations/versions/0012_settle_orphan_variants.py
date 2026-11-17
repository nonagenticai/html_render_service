"""Backfill the variants stranded ``pending`` under an already-terminal parent.

Measured on prod 2026-09-08, before this ran::

    SELECT j.status AS parent, count(*)
      FROM redesign_variants v JOIN redesign_jobs j ON j.job_id = v.job_id
     WHERE v.status NOT IN ('completed','failed')
     GROUP BY 1;
    -- failed    | 225
    -- completed |  92

317 rows, oldest ``created_at`` 2026-06-07, and **zero** under a parent still
``pending``/``processing`` — so none of them was live work. None carried an
``error_message`` either, which is why they never showed up in any ranking of
failures: to every reader, and to ``GET /jobs/{id}/variants``, they were work
still on its way.

Nothing could ever reclaim them. ``create_variant`` only INSERTs;
``mark_variant_completed``/``mark_variant_failed`` run only inside the variant's
own render; ``claim_next_render_job``'s variant retirement (``0011``) fires only
on a RE-CLAIM, and its claim predicate selects jobs whose status is
``pending``/``processing`` — a terminal parent is never claimed again, so the
sweep can never reach its children; ``reap_stale_batch_jobs`` reaches only
batch-owned rows. The code-side hole is closed in the same change by
``JobManager.settle_orphan_variants``, called from every write that gives the
PARENT a verdict (``mark_completed``, ``mark_failed``, ``cancel_job``). This
revision only cleans up the rows that predate it.

RECLAIM, NOT DELETION — nothing here removes a row.

The reason is composed from the PERSISTED parent row's status, per row, not from
any ambient request/tenant context: this walks every tenant's rows at once and
must not attribute one tenant's cause to another's job (nor fabricate a
provenance a row never had). ``failed`` is used because it is the only
non-``completed`` terminal variant status the schema has — no new state, and the
text names what happened rather than implying the tenant cancelled.

IDEMPOTENT + BATCHED + RE-RUNNABLE. The predicate is "non-terminal child under a
terminal parent", which the UPDATE itself falsifies, so a re-run matches zero
rows. The loop commits in bounded chunks rather than taking one lock over every
matching row: ``redesign_variants`` is written by live workers, and a single
statement over the whole table would hold row locks for the duration.

Revision ID: 0012_settle_orphan_variants
Revises: 0011_batch_reap
Create Date: 2026-09-08 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012_settle_orphan_variants"
down_revision: str | None = "0011_batch_reap"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Rows per statement. The literals below are spelled out rather than imported
#: from ``src.services.job_manager``: a migration must not import application
#: code that can be refactored out from under it (the same rule ``0011`` follows
#: for ``CLAIM_FN_SQL``). ``tests/unit/test_orphan_variants.py`` asserts the two
#: sets agree, so the duplication cannot drift silently.
_BATCH = 500

_BACKFILL = """
WITH doomed AS (
    SELECT v.id, j.status AS parent_status
      FROM redesign_variants v
      JOIN redesign_jobs j ON j.job_id = v.job_id
     WHERE v.status IN ('pending', 'processing')
       AND j.status IN ('completed', 'failed', 'cancelled')
     ORDER BY v.id
     LIMIT {batch}
)
UPDATE redesign_variants v
   SET status = 'failed',
       error_message = 'abandoned: the parent job reached terminal status '''
                       || doomed.parent_status
                       || ''' while this variant was still unfinished — the '
                       || 'variant render never reported an outcome '
                       || '(backfilled by migration 0012)',
       completed_at = now()
  FROM doomed
 WHERE v.id = doomed.id
"""


def upgrade() -> None:
    """Settle every stranded variant, ``_BATCH`` rows at a time."""
    conn = op.get_bind()
    statement = _BACKFILL.format(batch=_BATCH)
    total = 0
    while True:
        settled = conn.exec_driver_sql(statement).rowcount or 0
        total += settled
        if settled < _BATCH:
            break
    print(f"0012: settled {total} orphaned redesign_variants row(s)")


def downgrade() -> None:
    """Put the backfilled rows back to ``pending``.

    Scoped by the migration's own marker in ``error_message`` so it can only undo
    what ``upgrade()`` did: a variant that failed for a REAL reason, and one
    settled by the live ``settle_orphan_variants`` code path (whose text carries
    no ``migration 0012`` marker), are both left alone. Restoring those would
    manufacture the exact immortal rows this revision exists to remove.
    """
    op.execute(
        """
        UPDATE redesign_variants
           SET status = 'pending',
               error_message = NULL,
               completed_at = NULL
         WHERE status = 'failed'
           AND error_message LIKE '%(backfilled by migration 0012)'
        """
    )
