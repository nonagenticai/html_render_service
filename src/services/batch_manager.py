"""Batch job management for multi-page redesign processing."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import BatchJob, RedesignJob

logger = logging.getLogger(__name__)

#: The statuses a batch never leaves on its own — the same fence
#: ``cancel_batch`` already applies when it refuses to cancel a finished batch,
#: named once so ``finalize_batch`` can apply it too. A ``retry-failed`` reset is
#: the only sanctioned way back out, and it writes ``processing`` explicitly.
TERMINAL_BATCH_STATUSES: tuple[str, ...] = (
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
)


class BatchManager:
    """Manages batch redesign job lifecycle."""

    @staticmethod
    async def create_batch(
        db: AsyncSession,
        *,
        tenant_id: str,
        total_pages: int,
        shared_settings: dict | None = None,
    ) -> BatchJob:
        """Create a new batch job record."""
        batch = BatchJob(
            batch_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            status="pending",
            total_pages=total_pages,
            completed_pages=0,
            failed_pages=0,
            shared_settings=shared_settings,
        )
        db.add(batch)
        await db.flush()
        logger.info(
            "Created batch %s with %d pages for tenant %s", batch.batch_id, total_pages, tenant_id
        )
        return batch

    @staticmethod
    async def get_batch(
        db: AsyncSession, batch_id: str, *, tenant_id: str | None
    ) -> BatchJob | None:
        """Fetch a batch by ID scoped to ``tenant_id``, or None if not found.

        Keyword-only and required, same contract as ``JobManager.get_job``:
        ``None`` means "deliberately cross-tenant, authorised upstream" (the batch
        background processor, which already resolved the tenant at enqueue time).
        """
        conditions = [BatchJob.batch_id == batch_id]
        if tenant_id is not None:
            conditions.append(BatchJob.tenant_id == tenant_id)
        result = await db.execute(select(BatchJob).where(*conditions))
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_processing(db: AsyncSession, batch_id: str) -> bool:
        """Transition a batch to 'processing' status with a started timestamp.

        Fenced on ``cancelled`` for the same reason as
        ``JobManager.mark_processing``, and it is the FIRST thing
        ``process_batch`` does. Unfenced, a batch cancelled between enqueue and
        the background task starting was flipped straight back to
        ``processing`` here — after which ``process_page``'s per-page
        cancellation check, which reads exactly this field, could never fire and
        all N pages rendered and billed.

        ``failed``/``completed*`` are deliberately NOT fenced: ``retry-failed``
        legitimately reopens a finished batch.

        Returns:
            True if the batch is now processing; False if the tenant cancelled
            it and the caller must not process its pages.
        """
        logger.debug("Marking batch %s as processing", batch_id)
        now = datetime.now(UTC)
        result = await db.execute(
            update(BatchJob)
            .where(BatchJob.batch_id == batch_id, BatchJob.status != "cancelled")
            .values(status="processing", started_at=now, last_progress_at=now)
        )
        started: int = getattr(result, "rowcount", 0) or 0
        if not started:
            logger.info("Batch %s not started: it was cancelled before it began", batch_id)
        return bool(started)

    @staticmethod
    async def mark_batch_failed(db: AsyncSession, batch_id: str, error_message: str) -> None:
        """Mark a batch as failed with an error message.

        ``error_message`` is now PERSISTED. It was accepted and dropped: the row
        went terminal with no reason on it, so a client polling the batch status
        endpoint saw a bare ``failed`` and had nothing to act on.
        """
        logger.debug("Marking batch %s as failed: %s", batch_id, error_message[:100])
        await db.execute(
            update(BatchJob)
            .where(BatchJob.batch_id == batch_id)
            .values(
                status="failed",
                error_message=error_message,
                completed_at=datetime.now(UTC),
            )
        )

    @staticmethod
    async def reap_stale_batches(db: AsyncSession, no_progress_seconds: int) -> int:
        """Fail batches that have made no page progress for *no_progress_seconds*.

        The batch arm of the lease sweep the worker pool already runs. It exists
        because ``batch_jobs`` is the ONE non-terminal table with no owner: it
        carries no lease columns, no worker ever claims it, and the only writers
        that advance it (``mark_processing`` / ``finalize_batch`` /
        ``mark_batch_failed``) live inside the in-process ``asyncio`` task the
        request started. Kill the pod and the row is immortal — the reclaim in
        ``0002_job_lease`` never covered it, ``CleanupService`` has no caller, and
        ``RETENTION_ENABLED`` defaults false.

        Delegates to the SECURITY DEFINER ``reap_stale_batch_jobs`` resolver for
        the same reason ``claim_next_job`` delegates: the sweep is cross-tenant
        and the worker binds no ``app.current_tenant``, so under RLS enforce a
        plain UPDATE from this role would match 0 rows — silently, in production
        only.

        A NO-PROGRESS WINDOW, not a wall-clock ceiling on the batch: staleness is
        measured from ``last_progress_at``, which ``increment_completed`` /
        ``increment_failed`` bump on every committed page outcome. A ceiling on
        total batch age could not be set safely — ``started_at`` is written once
        and never again, and the longest legitimate batch (20 pages at
        concurrency 3, with no per-page wall-clock cap on this path) has no
        honest upper bound to sit above. This window only has to exceed the
        longest single PAGE.

        The resolver enforces its own floor on the argument, so a misconfigured
        (or hostile) value cannot be used to terminate live batches fleet-wide.

        This RECLAIMS; it does not delete. Returns the number of batches reaped.
        """
        result = await db.execute(
            text("SELECT reap_stale_batch_jobs(:secs) AS reaped"),
            {"secs": no_progress_seconds},
        )
        reaped = int(result.scalar_one() or 0)
        if reaped:
            logger.warning(
                "Reaped %d abandoned batch(es) with no page progress for %ds — moved "
                "to 'failed' with a recorded reason (their non-queue-owned child jobs "
                "and variants too). The batch status endpoint is no longer wedged.",
                reaped,
                no_progress_seconds,
            )
        return reaped

    @staticmethod
    async def touch_progress(db: AsyncSession, batch_id: str) -> None:
        """Bump the progress clock mid-page, without touching a counter.

        The sub-page progress signal. ``increment_completed`` / ``increment_failed``
        fire once per PAGE, and one page is unbounded (``generate_sections`` makes
        one sequential LLM call per DOM landmark), so per-page bumps alone left
        hours of real work looking identical to a dead orchestrator.

        Deliberately does NOT move a counter: nothing has completed, the batch is
        merely demonstrably alive. And it is fenced to non-terminal rows so a late
        report from a page still in flight cannot resurrect the progress clock of
        a batch that has already been finalized, cancelled, or reaped.
        """
        await db.execute(
            update(BatchJob)
            .where(
                BatchJob.batch_id == batch_id,
                BatchJob.status.in_(("pending", "processing")),
            )
            .values(last_progress_at=datetime.now(UTC))
        )

    @staticmethod
    async def increment_completed(db: AsyncSession, batch_id: str) -> None:
        """Increment the completed page counter for a batch."""
        logger.debug("Incrementing completed count for batch %s", batch_id)
        await db.execute(
            update(BatchJob)
            .where(BatchJob.batch_id == batch_id)
            .values(
                completed_pages=BatchJob.completed_pages + 1,
                # This is the progress signal the reaper reads. It has to be
                # written on the SAME statement as the counter it reports, in the
                # same transaction the page handler commits — a progress clock
                # bumped anywhere else can advance while nothing is progressing.
                last_progress_at=datetime.now(UTC),
            )
        )

    @staticmethod
    async def increment_failed(db: AsyncSession, batch_id: str) -> None:
        """Increment the failed page counter for a batch."""
        logger.debug("Incrementing failed count for batch %s", batch_id)
        await db.execute(
            update(BatchJob)
            .where(BatchJob.batch_id == batch_id)
            .values(
                failed_pages=BatchJob.failed_pages + 1,
                # A FAILED page is progress too: the batch is still being worked,
                # it is just producing failures. Reaping a batch that is actively
                # failing its way through 20 pages would be a false positive.
                last_progress_at=datetime.now(UTC),
            )
        )

    @staticmethod
    async def finalize_batch(db: AsyncSession, batch_id: str) -> str:
        """Derive final status from page results and update batch.

        Background processor path — the tenant was already resolved at enqueue
        time, so the read is deliberately cross-tenant.
        """
        batch = await BatchManager.get_batch(db, batch_id, tenant_id=None)
        if batch is None:
            raise ValueError(f"Batch not found: {batch_id}")

        accounted = (batch.completed_pages or 0) + (batch.failed_pages or 0)
        total = batch.total_pages or 0
        unaccounted = max(total - accounted, 0)

        if total and batch.failed_pages >= total:
            status = "failed"
        elif batch.failed_pages > 0 or unaccounted:
            # `unaccounted` was the second half of the "a cancelled batch reports
            # completed" defect. The status used to be derived from
            # `failed_pages` ALONE, so a batch whose pages never reached either
            # counter -- every page returning early, a task killed by a
            # BaseException that the per-page `except Exception` cannot see --
            # derived a clean `completed` out of 0 failures and 0 successes.
            # Silence is not success: an unaccounted page is reported.
            status = "completed_with_errors"
        else:
            status = "completed"

        unaccounted_reason = (
            f"{unaccounted} of {total} pages produced no result (neither completed nor failed)"
            if unaccounted
            else None
        )

        # THE FENCE. `finalize_batch` runs at the end of `process_batch`, which
        # races `POST /api/batch/{id}/cancel`: the pages stop, this still runs.
        # Unfenced it overwrote `cancelled` with a derived `completed`, cleared
        # the cancellation reason, and (via the caller) fired `batch.completed`
        # at the tenant for a batch they had just stopped.
        #
        # ``cancelled`` is the WHOLE fence, deliberately, and the other three
        # members of `TERMINAL_BATCH_STATUSES` are deliberately NOT in it:
        #
        # * `cancelled` is an INSTRUCTION from the tenant, not a derived verdict.
        #   Nothing this function computes can be more authoritative than "stop".
        # * `failed`/`completed*` written by the REAPER or by an earlier
        #   finalize ARE derived verdicts, and reaping does not stop the work
        #   (`process_page` aborts only on `cancelled`). A reaped batch whose
        #   pages then genuinely finish must be allowed to correct itself — that
        #   is `test_finalize_clears_a_reaper_reason_it_overwrites`, and blocking
        #   it would strand a real success as a false `failed`.
        result = await db.execute(
            update(BatchJob)
            .where(BatchJob.batch_id == batch_id, BatchJob.status != "cancelled")
            .values(
                status=status,
                # ``status`` and ``error_message`` are written TOGETHER, always.
                # Leaving a stale reason behind while overwriting the status makes
                # the row report a permanent contradiction (``status=completed``
                # alongside ``error='batch abandoned…'``) on the field the batch
                # status endpoint surfaces. So this is ``None`` on the clean paths
                # and the unaccounted-pages reason on the one path that has one —
                # never the previous owner's text.
                error_message=unaccounted_reason,
                completed_at=datetime.now(UTC),
            )
        )
        if not (getattr(result, "rowcount", 0) or 0):
            # Loud, and it returns the status the batch ACTUALLY has. The caller
            # dispatches `batch.completed` / `batch.failed` off this return value,
            # so handing back the derived-but-not-written status would fix the DB
            # row and leave the webhook lying — the half-fix that fixes nothing.
            current = await BatchManager.get_batch(db, batch_id, tenant_id=None)
            actual = current.status if current is not None else "unknown"
            logger.warning(
                "Batch %s NOT finalized as %s: the tenant cancelled it (status=%s). "
                "No batch-level webhook will be dispatched for this run.",
                batch_id,
                status,
                actual,
            )
            return actual
        logger.info("Batch %s finalized: %s", batch_id, status)
        return status

    @staticmethod
    async def get_batch_jobs(
        db: AsyncSession, batch_id: str, *, tenant_id: str | None
    ) -> list[RedesignJob]:
        """Get all child jobs for a batch, scoped to ``tenant_id``."""
        conditions = [RedesignJob.batch_id == batch_id]
        if tenant_id is not None:
            conditions.append(RedesignJob.tenant_id == tenant_id)
        result = await db.execute(
            select(RedesignJob)
            .where(*conditions)
            .order_by(RedesignJob.priority_score.desc().nullslast())
        )
        return list(result.scalars().all())

    @staticmethod
    async def cancel_batch(db: AsyncSession, batch_id: str, *, tenant_id: str | None) -> bool:
        """Cancel a batch and all its child jobs, scoped to ``tenant_id``.

        A cross-tenant id returns False, which the route turns into the same 409 an
        already-finished batch produces — no existence oracle. The UPDATEs carry the
        tenant predicate too, so a mismatch cannot write.
        """
        batch = await BatchManager.get_batch(db, batch_id, tenant_id=tenant_id)
        if batch is None:
            return False
        if batch.status in TERMINAL_BATCH_STATUSES:
            return False

        now = datetime.now(UTC)
        batch_conditions = [BatchJob.batch_id == batch_id]
        job_conditions = [
            RedesignJob.batch_id == batch_id,
            RedesignJob.status.in_(["pending", "processing"]),
        ]
        if tenant_id is not None:
            batch_conditions.append(BatchJob.tenant_id == tenant_id)
            job_conditions.append(RedesignJob.tenant_id == tenant_id)
        await db.execute(
            update(BatchJob)
            .where(*batch_conditions)
            # Same reason as ``finalize_batch``: a cancel supersedes any earlier
            # verdict, so it owns ``error_message`` too. Leaving a stale reaper
            # reason on a ``cancelled`` row reports a cause that is no longer true.
            .values(status="cancelled", error_message=None, completed_at=now)
        )
        # Cancel all pending/processing child jobs
        await db.execute(
            update(RedesignJob)
            .where(*job_conditions)
            .values(status="cancelled", cancelled_at=now, completed_at=now)
        )
        logger.info("Cancelled batch %s and its pending jobs", batch_id)
        return True

    @staticmethod
    async def list_batches(db: AsyncSession, tenant_id: str) -> list[BatchJob]:
        """List all batches for a tenant."""
        result = await db.execute(
            select(BatchJob)
            .where(BatchJob.tenant_id == tenant_id)
            .order_by(BatchJob.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_failed_jobs(
        db: AsyncSession, batch_id: str, *, tenant_id: str | None
    ) -> list[RedesignJob]:
        """Get failed child jobs for retry, scoped to ``tenant_id``."""
        conditions = [RedesignJob.batch_id == batch_id, RedesignJob.status == "failed"]
        if tenant_id is not None:
            conditions.append(RedesignJob.tenant_id == tenant_id)
        result = await db.execute(select(RedesignJob).where(*conditions))
        return list(result.scalars().all())

    @staticmethod
    async def reset_job_for_retry(db: AsyncSession, job_id: str) -> None:
        """Reset a failed job back to pending for retry."""
        logger.debug("Resetting job %s to pending for retry", job_id)
        await db.execute(
            update(RedesignJob)
            .where(RedesignJob.job_id == job_id)
            .values(
                status="pending",
                error_message=None,
                completed_at=None,
                started_at=None,
            )
        )
