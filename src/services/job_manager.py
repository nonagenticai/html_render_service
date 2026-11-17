"""Redesign job lifecycle management."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import date as date_type

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.dml import Update

from src.database.models import RedesignJob, RedesignUsage, RedesignVariant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claim ownership
# ---------------------------------------------------------------------------

_CLAIM_OWNER: ContextVar[str | None] = ContextVar("html_render_claim_owner", default=None)


@contextmanager
def claim_owner(worker_id: str) -> Iterator[None]:
    """Bind *worker_id* as the owner of the claim for the duration of this job.

    The queue's ownership token is ``redesign_jobs.claimed_by``, written by
    ``claim_next_render_job``. Every write that ends a job (``mark_failed``) or
    hands it back (``mark_retryable``) must predicate on that same token, or a
    worker that has already LOST the claim can still overwrite the row — see
    those two methods for the concrete double-render this prevents.

    The owner is bound here rather than threaded through ~6 pipeline signatures
    for the same reason the caller's identity is (``job_worker._run_claimed``):
    the worker task owns the whole dispatch, and every failure seam inside it is
    reached by ``await`` from that task, so a ContextVar reaches all of them.
    It is reset on the way out so the next claim on this reused loop cannot
    inherit the previous job's owner.
    """
    token = _CLAIM_OWNER.set(worker_id)
    try:
        yield
    finally:
        _CLAIM_OWNER.reset(token)


def get_claim_owner() -> str | None:
    """Return the worker id holding the current job's claim, if any.

    ``None`` outside a worker dispatch. A caller that cannot name an owner
    cannot prove it still holds the claim, so the fenced writes degrade to
    "unclaimed rows only" rather than to "no fence at all".
    """
    return _CLAIM_OWNER.get()


# ---------------------------------------------------------------------------
# Terminal-status fence
# ---------------------------------------------------------------------------

#: The statuses a job never leaves on its own. ``cancelled`` belongs here as much
#: as ``completed``/``failed`` does: the tenant asked for the work to stop, and a
#: later phase that stamps ``completed`` over it does not merely mislabel a row —
#: it bills for a redesign the tenant cancelled and fires ``job.completed`` at
#: their webhook. (A ``retry-failed`` reset is the ONE sanctioned way out, and it
#: goes through ``BatchManager.reset_job_for_retry``, not through these writes.)
TERMINAL_JOB_STATUSES: tuple[str, ...] = ("completed", "failed", "cancelled")

#: Every status a ``redesign_variants`` row can hold. ``create_variant`` writes
#: ``pending``; ``mark_variant_completed``/``mark_variant_failed`` write the two
#: terminal ones. ``processing`` is never written by this module but IS in the
#: SQL reclaim predicates (``rls_resolvers.CLAIM_FN_SQL``,
#: ``REAP_BATCH_FN_SQL``), so it is enumerated here too: the whole point of this
#: constant is that the Python settler and the SQL sweeps agree on the set, and
#: a state one of them knows about but the other does not is exactly the defect.
VARIANT_STATUSES: tuple[str, ...] = ("pending", "processing", "completed", "failed")

#: The variant statuses that are a VERDICT. Anything else means "still coming".
TERMINAL_VARIANT_STATUSES: tuple[str, ...] = ("completed", "failed")

#: Derived, never spelled out literally: a status added to ``VARIANT_STATUSES``
#: without a terminal classification lands here automatically and is therefore
#: swept by :meth:`JobManager.settle_orphan_variants` on day one, instead of
#: silently becoming the next immortal row.
NON_TERMINAL_VARIANT_STATUSES: tuple[str, ...] = tuple(
    status for status in VARIANT_STATUSES if status not in TERMINAL_VARIANT_STATUSES
)


def terminal_fence(statement: Update, new_status: str) -> Update:
    """Add "this row is not already terminal in some OTHER way" to *statement*.

    The invariant, stated once: **a terminal status may not be overwritten by a
    DIFFERENT terminal status.** Whoever reached a verdict first owns the row.

    The "different" is load bearing in both directions:

    * It blocks the real defects — ``completed`` over ``cancelled`` (which bills
      the tenant for work they stopped), ``failed`` over ``cancelled``, and
      either over the other worker's verdict in a lost-claim race.
    * It permits the same-status rewrite, which is not a race but a legitimate
      double-record: ``_record_background_failure`` is the OUTER of two failure
      seams and re-marks a job the inner ``_phase_finalize_failed`` already
      marked ``failed``. Blocking that would turn every ordinary failure into a
      "lost the race" warning and train the reader to ignore the one that means
      something.

    Fencing here rather than in each caller is deliberate: the DB row is the only
    place all workers agree, so the predicate must ride on the UPDATE itself, not
    on a read-then-write the next worker can interleave with.
    """
    blocked = tuple(s for s in TERMINAL_JOB_STATUSES if s != new_status)
    return statement.where(RedesignJob.status.not_in(blocked))


@dataclass(frozen=True)
class ClaimedJob:
    """A detached snapshot of a claimed job.

    Carries exactly what the worker needs to reconstruct the dispatch
    (``job_options`` bundle persisted at enqueue time) without touching an ORM
    object that would expire after the claim's commit.
    """

    job_id: str
    tenant_id: str
    options: dict


class JobManager:
    """Manages redesign job lifecycle."""

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        tenant_id: str,
        url: str,
        html_content: str | None = None,
        scope: str = "full_page",
        style: str = "modern",
        page_type: str | None = None,
        critique_findings: list[dict] | None = None,
        brand_info: dict | None = None,
        site_analysis: dict | None = None,
        user_context: dict | None = None,
        template_version: str | None = None,
        styles: list[str] | None = None,
        component_target: dict | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        batch_id: str | None = None,
        priority_score: float | None = None,
        mode: str = "freeform",
        interactivity: str = "static",
        component_set: str | None = None,
        embeds: list[dict] | None = None,
    ) -> RedesignJob:
        """Create a new redesign job record in the database.

        Args:
            db: Async database session.
            tenant_id: Tenant identifier.
            url: URL of the page to redesign.
            html_content: Optional pre-fetched HTML content.
            scope: Redesign scope (full_page, sections, component).
            style: Design style name or custom style slug.
            page_type: Page type classification.
            critique_findings: Critique findings to address.
            brand_info: Brand identity data.
            site_analysis: Site analysis context.
            user_context: Rich user context for targeting.
            template_version: Prompt template version.
            styles: Multiple styles for A/B variant generation.
            component_target: Component target for scope=component.
            webhook_url: Webhook notification URL.
            webhook_secret: Webhook HMAC secret.
            batch_id: Parent batch identifier.
            priority_score: Priority for batch ordering.

        Returns:
            The newly created RedesignJob ORM instance.
        """
        variant_count = len(styles) if styles else None
        job = RedesignJob(
            job_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            url=url,
            html_content=html_content,
            scope=scope,
            style=style,
            page_type=page_type,
            critique_findings=critique_findings,
            brand_info=brand_info,
            site_analysis=site_analysis,
            user_context=user_context,
            template_version=template_version,
            styles=styles,
            variant_count=variant_count,
            component_target=component_target,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            batch_id=batch_id,
            priority_score=priority_score,
            mode=mode,
            interactivity=interactivity,
            component_set=component_set,
            embeds=embeds,
            status="pending",
        )
        db.add(job)
        await db.flush()
        logger.info(
            "Created redesign job %s (url=%s, scope=%s, style=%s)", job.job_id, url, scope, style
        )
        return job

    @staticmethod
    async def create_refinement_job(
        db: AsyncSession,
        *,
        parent_job: RedesignJob,
        feedback: str,
        generate_screenshots: bool = True,
        generate_concept_images: bool = False,
    ) -> RedesignJob:
        """Create a refinement job linked to a parent."""
        root_id = parent_job.root_job_id or parent_job.job_id
        iteration = (parent_job.iteration_number or 0) + 1

        job = RedesignJob(
            job_id=str(uuid.uuid7()),
            tenant_id=parent_job.tenant_id,
            url=parent_job.url,
            scope=parent_job.scope,
            style=parent_job.style,
            page_type=parent_job.page_type,
            critique_findings=parent_job.critique_findings,
            brand_info=parent_job.brand_info,
            site_analysis=parent_job.site_analysis,
            user_context=parent_job.user_context,
            template_version=parent_job.template_version,
            parent_job_id=parent_job.job_id,
            root_job_id=root_id,
            iteration_number=iteration,
            feedback_text=feedback,
            webhook_url=parent_job.webhook_url,
            webhook_secret=parent_job.webhook_secret,
            status="pending",
        )
        db.add(job)
        await db.flush()
        logger.info(
            "Created refinement job %s (parent=%s, iteration=%d)",
            job.job_id,
            parent_job.job_id,
            iteration,
        )
        return job

    @staticmethod
    async def get_job(
        db: AsyncSession, job_id: str, *, tenant_id: str | None
    ) -> RedesignJob | None:
        """Fetch a job by its unique ID, scoped to ``tenant_id``.

        ``tenant_id`` is keyword-only and REQUIRED so every caller has to make the
        decision explicitly. Passing ``None`` means "deliberately cross-tenant,
        authorised upstream" and skips the predicate — the queue claim path
        (:meth:`claim_next_job`) runs before any tenant is bound and is the reason
        this cannot be a required non-null ``str``.

        A tenant mismatch is indistinguishable from a genuine miss (``None``), so
        callers return 404 rather than 403: a 403 would confirm the id exists,
        which is an enumeration oracle over ids that double as artifact URLs.
        """
        conditions = [RedesignJob.job_id == job_id]
        if tenant_id is not None:
            conditions.append(RedesignJob.tenant_id == tenant_id)
        result = await db.execute(select(RedesignJob).where(*conditions))
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_processing(db: AsyncSession, job_id: str) -> bool:
        """Transition a job to 'processing' status with a started timestamp.

        Fenced on ``cancelled``, and this is the cancel-BEFORE-START case rather
        than a nicety. Unfenced, this write ERASED the cancellation: a job
        cancelled while it sat in the queue was flipped back to ``processing``
        here, and every ``_check_cancelled`` gate after it — which reads
        ``status == 'cancelled'`` — was then structurally unable to see the
        cancellation it exists to catch. The whole render ran, and phase 11
        billed it.

        Only ``cancelled`` blocks. A ``failed`` row is legitimately re-run after
        ``reset_job_for_retry`` (which puts it back to ``pending`` first), and a
        re-claimed row is already ``processing``.

        Returns:
            True if the job is now processing; False if the tenant cancelled it
            and the caller must not run the pipeline.
        """
        logger.debug("Marking job %s as processing", job_id)
        result = await db.execute(
            update(RedesignJob)
            .where(RedesignJob.job_id == job_id, RedesignJob.status != "cancelled")
            .values(status="processing", started_at=datetime.now(UTC))
        )
        started: int = getattr(result, "rowcount", 0) or 0
        if not started:
            logger.info("Job %s not started: it was cancelled before it began", job_id)
        return bool(started)

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        *,
        redesigned_html_path: str | None = None,
        section_html_paths: list[str] | None = None,
        screenshot_paths: dict[str, str] | None = None,
        section_screenshot_paths: dict[str, str] | None = None,
        concept_image_assets: list[dict[str, object]] | None = None,
        concept_image_degraded: bool = False,
        concept_image_reason: str | None = None,
        pages_count: int = 1,
        accessibility_score: float | None = None,
        accessibility_report: dict | None = None,
        original_screenshot_paths: dict[str, str] | None = None,
        comparison_paths: dict[str, str] | None = None,
        diff_paths: dict[str, str] | None = None,
        component_html_path: str | None = None,
        html_validation_report: dict | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost: float | None = None,
        worker_id: str | None = None,
    ) -> bool:
        """Mark a job as completed and store all output artifact paths.

        Fenced exactly like :meth:`mark_failed`, and for a bigger reason. The
        last cancellation check in the pipeline is ``service.py``'s, and the
        a11y auto-fix LLM call, the screenshot renders and the concept-image
        poll all run AFTER it — minutes during which
        ``POST /api/designs/jobs/{id}/cancel`` can land. Unfenced, this write
        stamped ``completed`` straight over ``cancelled``, and its caller then
        ran ``record_usage(pages_redesigned=1, llm_tokens_used=…)``: the tenant
        was **billed for the job they cancelled** and got a ``job.completed``
        webhook for it.

        Two predicates, both of which must hold:

        * the terminal fence (see :func:`terminal_fence`) — nothing overwrites
          another verdict;
        * ``worker_id``, when given, must still hold the claim, same contract
          and same ``claimed_by IS NULL`` allowance as ``mark_failed`` (batch
          child rows are never claimed and must stay markable).

        Returns:
            True if the row was marked completed. **False means this completion
            did not happen** — the caller MUST NOT record usage or dispatch
            ``job.completed``, because someone else owns the outcome.
        """
        logger.debug("Marking job %s as completed", job_id)
        # The core job completed successfully; concept images are optional
        # enrichment. A requested-and-failed generation is recorded on a
        # dedicated column (not the job status) so it stays distinguishable from
        # "none requested" WITHOUT making the completed job's artifacts read as
        # not-ready to the many status=="completed" gates.
        values: dict = {
            "status": "completed",
            "redesigned_html_path": redesigned_html_path,
            "section_html_paths": section_html_paths,
            "screenshot_paths": screenshot_paths,
            "concept_image_assets": concept_image_assets,
            "concept_image_error": concept_image_reason if concept_image_degraded else None,
            "pages_count": pages_count,
            "completed_at": datetime.now(UTC),
        }
        optional_fields: dict[str, object] = {
            "section_screenshot_paths": section_screenshot_paths,
            "accessibility_score": accessibility_score,
            "accessibility_report": accessibility_report,
            "original_screenshot_paths": original_screenshot_paths,
            "comparison_paths": comparison_paths,
            "diff_paths": diff_paths,
            "component_html_path": component_html_path,
            "html_validation_report": html_validation_report,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
        }
        for key, value in optional_fields.items():
            if value is not None:
                values[key] = value

        statement = terminal_fence(
            update(RedesignJob).where(RedesignJob.job_id == job_id), "completed"
        )
        if worker_id is not None:
            statement = statement.where(
                or_(RedesignJob.claimed_by.is_(None), RedesignJob.claimed_by == worker_id)
            )
        result = await db.execute(statement.values(**values))
        marked: int = getattr(result, "rowcount", 0) or 0
        if marked:
            # Same transaction as the verdict: a variant left ``pending`` under a
            # terminal parent is reachable by NO reclaim path (see
            # ``settle_orphan_variants``). Gated on ``marked`` so a write that
            # lost the terminal fence does not settle the WINNER's children.
            await JobManager.settle_orphan_variants(db, job_id, parent_status="completed")
        if not marked:
            # Loud, not silent-success: this is the money path. "Could not
            # confirm the completion" is reported as a refusal so the caller
            # skips billing and the webhook, never as a quiet no-op.
            logger.warning(
                "Job %s NOT marked completed: the row is already terminal "
                "(cancelled or failed) or worker %s no longer owns the claim. "
                "Skipping usage + job.completed — this work is not billable.",
                job_id,
                worker_id,
            )
        return bool(marked)

    @staticmethod
    async def mark_failed(
        db: AsyncSession, job_id: str, error_message: str, *, worker_id: str | None = None
    ) -> bool:
        """Mark a job as failed with an error message.

        ``worker_id`` fences the write to the worker that still holds the claim.
        The claim predicate (``claim_next_render_job`` sets ``claimed_by``) and
        this terminal predicate must agree: a job whose lease lapsed, or that was
        requeued and re-claimed by a sibling, is being rendered by SOMEONE ELSE,
        and stamping ``failed`` on it from the losing worker both kills a live
        render and races that worker's own ``mark_completed``.

        ``claimed_by IS NULL`` is in the fence because a row that no worker holds
        is nobody's to lose — batch child rows (``job_options`` NULL) are never
        claimed at all and must still be markable.

        ``worker_id=None`` (the default) leaves the write UNFENCED, which is
        correct for the callers that are not a claim holder: the poison/lifetime
        gates inside the claim itself, cancellation, and the pre-queue paths.

        The SECOND fence is the terminal one (see :func:`terminal_fence`), and it
        is not covered by the claim fence above: a *batch child* row has
        ``claimed_by IS NULL``, so the claim predicate waves it through, and a
        cancelled variant job whose fan-out produced zero variant rows was being
        stamped ``failed`` over ``cancelled`` by ``phases.py``'s "all N variants
        failed" verdict. Cancelled is a verdict already.

        Returns:
            True if the row was marked failed; False if either fence rejected the
            write — this worker no longer owns the job, or someone else already
            reached a different verdict. **False means the error was DROPPED**:
            the caller must not record error usage or dispatch ``job.failed``.
        """
        logger.debug("Marking job %s as failed: %s", job_id, error_message[:100])
        statement = terminal_fence(
            update(RedesignJob).where(RedesignJob.job_id == job_id), "failed"
        )
        if worker_id is not None:
            statement = statement.where(
                or_(RedesignJob.claimed_by.is_(None), RedesignJob.claimed_by == worker_id)
            )
        result = await db.execute(
            statement.values(
                status="failed",
                error_message=error_message,
                completed_at=datetime.now(UTC),
            )
        )
        marked: int = getattr(result, "rowcount", 0) or 0
        if marked:
            # See ``mark_completed`` — and ``settle_orphan_variants`` for why a
            # ``failed`` parent's children are the LARGEST stranded population
            # (225 of the 317 rows measured on prod).
            await JobManager.settle_orphan_variants(db, job_id, parent_status="failed")
        if not marked:
            logger.warning(
                "Job %s NOT marked failed: it is already terminal in another way "
                "(cancelled/completed), or worker %s no longer owns the claim "
                "(another worker holds it and is still rendering). Dropped error: %s",
                job_id,
                worker_id,
                error_message,
            )
        return bool(marked)

    @staticmethod
    async def mark_retryable(
        db: AsyncSession, job_id: str, error_message: str, *, worker_id: str | None
    ) -> bool:
        """Return a queue-owned job to ``pending`` so the pool re-runs it.

        The alternative to :meth:`mark_failed` for a *transient* upstream failure
        (see ``src.services.upstream_failure``). ``failed`` is a verdict a job can
        never come back from; this hands the row back to the claim queue instead.

        Bounded by the mechanism that already exists — no new retry engine:
        ``claim_next_render_job`` charged ``attempts += 1`` when it claimed this
        row and fails any row whose ``attempts >= max_attempts`` as poison, and
        ``max_job_lifetime_seconds`` caps total running wall-clock from
        ``started_at``. So a job that keeps timing out still reaches a terminal
        ``failed``, just after ``REDESIGN_MAX_ATTEMPTS`` tries instead of one.
        ``attempts`` is deliberately NOT refunded here (unlike ``release_claims``,
        where the handoff was graceful and produced no verdict).

        Three guards decide whether the requeue is legal at all, and the caller
        falls back to ``mark_failed`` when they say no:

        * ``job_options IS NOT NULL`` — the queue-ownership flag. A batch child
          row carries its own orchestration and no worker would ever claim it, so
          setting it ``pending`` would strand it forever with the batch counters
          hanging.
        * ``status IN ('processing', 'pending')`` — never resurrect a row that was
          cancelled, or completed/failed out from under this handler. ``pending``
          is in the set because the failure travels through TWO seams: the phase
          handler requeues, re-raises, and the background-task wrapper's own
          ``except`` then sees the same exception on an already-``pending`` row.
          It must find the requeue idempotent — if this returned False there, the
          wrapper would fall back to ``mark_failed`` and undo it.
        * ``claimed_by IS NULL OR claimed_by = worker_id`` — the OWNERSHIP fence,
          and the reason the ``pending`` arm above is not a hole. Without it, this
          exact interleaving double-renders a paid job::

              A: inner seam requeues  -> pending,   claimed_by=NULL
              B: claims it (~2s poll) -> processing, claimed_by=B, renders
              A: outer seam unwinds   -> pending,   claimed_by=NULL   <-- stomp
              C: claims it            -> B and C now render the SAME job

          With the fence, A's second write sees ``claimed_by=B`` and matches zero
          rows. ``IS NULL`` keeps the honest idempotent case (A's own requeue,
          before anyone re-claims) working, and keeps never-claimed rows writable.

        ``error_message`` is still written: a caller polling mid-retry sees why
        the previous attempt did not finish. ``completed_at`` stays NULL because
        the job is not finished.

        Returns:
            True if the row was handed back to the queue; False if it is not
            eligible, in which case the caller must record a terminal failure.
            A lost race is information, so the False case is logged, not silent.
        """
        result = await db.execute(
            update(RedesignJob)
            .where(
                RedesignJob.job_id == job_id,
                RedesignJob.job_options.is_not(None),
                RedesignJob.status.in_(("processing", "pending")),
                or_(RedesignJob.claimed_by.is_(None), RedesignJob.claimed_by == worker_id),
            )
            .values(
                status="pending",
                error_message=error_message,
                claimed_by=None,
                lease_expires_at=None,
            )
        )
        requeued: int = getattr(result, "rowcount", 0) or 0
        if requeued:
            logger.warning("Job %s requeued after transient failure: %s", job_id, error_message)
        else:
            logger.warning(
                "Job %s requeue FENCED OFF (worker=%s): the row is not queue-owned, has left "
                "processing/pending, or is now claimed by another worker. Falling back to the "
                "terminal path. Transient failure was: %s",
                job_id,
                worker_id,
                error_message,
            )
        return bool(requeued)

    # ========================================================================
    # Durable claim-queue (Phase 1)
    # ========================================================================

    @staticmethod
    async def set_job_options(db: AsyncSession, job_id: str, options: dict) -> None:
        """Persist the per-request render/processing bundle on the job row.

        A non-NULL ``job_options`` is the queue-ownership flag: the worker pool
        only claims rows that carry it (single ``create_redesign`` / ``refine``
        jobs). Batch child rows leave it NULL and keep their own orchestration.
        """
        await db.execute(
            update(RedesignJob).where(RedesignJob.job_id == job_id).values(job_options=options)
        )

    @staticmethod
    async def set_verification(
        db: AsyncSession, job_id: str, status: str, report: dict | None
    ) -> None:
        """Persist the verification-lane outcome on the job row."""
        await db.execute(
            update(RedesignJob)
            .where(RedesignJob.job_id == job_id)
            .values(verification_status=status, verification_report=report)
        )

    @staticmethod
    async def claim_next_job(
        db: AsyncSession,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
        max_lifetime_seconds: int = 0,
    ) -> ClaimedJob | None:
        """Atomically claim the next runnable queue job, or return None.

        Delegates to the SECURITY DEFINER ``claim_next_render_job`` resolver, which
        does the whole atomic claim in one statement: pick one queue-owned
        (``job_options IS NOT NULL``) ``pending`` / expired-lease row via ``FOR UPDATE
        SKIP LOCKED`` (so concurrent workers/pods never claim the same job), fail
        poison jobs past ``max_attempts``, else move it to ``processing`` with a fresh
        lease + incremented attempt and return ``(job_id, tenant_id, job_options)``.

        Phase D RLS: the queue is cross-tenant and the worker holds no tenant context
        at claim time, so a plain SELECT under enforce (non-bypass role, no GUC) would
        see 0 rows. The resolver is owned by the table owner → bypasses ENABLE. The
        worker then binds the returned tenant for all subsequent per-job work.
        """
        result = await db.execute(
            text(
                "SELECT job_id, tenant_id, job_options "
                "FROM claim_next_render_job(:worker, :lease, :max_attempts)"
            ),
            {"worker": worker_id, "lease": lease_seconds, "max_attempts": max_attempts},
        )
        row = result.first()
        # Persist the resolver's claim/poison UPDATEs (they run in this transaction).
        await db.commit()
        # The resolver mutated rows out-of-band of the ORM identity map; detach cached
        # instances so any later read on this session re-queries fresh state (the
        # session uses expire_on_commit=False). expunge (not expire) avoids a sync
        # lazy-load in the async context.
        db.expunge_all()
        if row is None:
            return None

        # Wall-clock ceiling on a job's total *running* life.
        #
        # `attempts` is a crash counter and `release_claims` refunds the attempt a
        # graceful handoff consumed — otherwise a rollout / HPA scale-down would
        # poison healthy jobs. But that refund alone leaves no bound at all: under
        # sustained churn a job could be re-claimed forever, and every re-claim
        # replays the whole render (a 16k-token WM call), so the *spend* would be
        # unbounded too. This is the bound that replaces the one the refund removes.
        #
        # Measured from `started_at` (first claim), NOT `created_at`: a job sitting
        # in a backlog has not consumed anything, and ageing it out unstarted would
        # mass-fail a queue that is merely behind.
        if max_lifetime_seconds > 0:
            # tenant_id=None by design: the claim is cross-tenant and the worker
            # has not bound a tenant yet at this point.
            job = await JobManager.get_job(db, row.job_id, tenant_id=None)
            started = job.started_at if job else None
            if started is not None:
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                running_for = (datetime.now(UTC) - started).total_seconds()
                if running_for > max_lifetime_seconds:
                    logger.error(
                        "Job %s exceeded max lifetime (%.0fs running > %ds cap) — failing it",
                        row.job_id,
                        running_for,
                        max_lifetime_seconds,
                    )
                    await JobManager.mark_failed(
                        db,
                        row.job_id,
                        f"exceeded max job lifetime ({max_lifetime_seconds}s) — "
                        f"never completed across {job.attempts if job else '?'} attempt(s)",
                        # This worker just won the claim, so it is the owner; the
                        # fence makes that explicit rather than implicit.
                        worker_id=worker_id,
                    )
                    await db.commit()
                    db.expunge_all()
                    return None

        # ``str(...)`` on the tenant is load-bearing, not cosmetic. This is a raw
        # ``text()`` query, so it bypasses the SQLAlchemy type layer entirely: since
        # ``0010_tenant_id_uuid`` the resolver declares ``tenant_id uuid`` and psycopg
        # hands back a ``uuid.UUID`` object here, NOT a ``str``. ``uuid.UUID(t) == t``
        # is False for every string ``t``, so an uncoerced value would flow into
        # ``ClaimedJob.tenant_id`` (declared ``str``) and then into
        # ``_process_job_background`` -> the per-job scoped reads, silently matching
        # nothing. Not a type error anywhere, so nothing would catch it but production.
        claimed = ClaimedJob(
            job_id=row.job_id,
            tenant_id=str(row.tenant_id),
            options=dict(row.job_options or {}),
        )
        logger.info("Claimed job %s (worker=%s)", claimed.job_id, worker_id)
        return claimed

    @staticmethod
    async def renew_lease(db: AsyncSession, job_id: str, lease_seconds: int) -> None:
        """Extend a running job's lease (heartbeat). No-op once it leaves processing."""
        await db.execute(
            update(RedesignJob)
            .where(RedesignJob.job_id == job_id, RedesignJob.status == "processing")
            .values(lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds))
        )

    @staticmethod
    async def release_claims(db: AsyncSession, claimed_by: str) -> int:
        """Expire this pod's in-flight leases at shutdown so reclaim is immediate.

        Returns the number of rows released. A graceful shutdown can't finish a
        long render mid-flight; expiring the lease lets a sibling/restart reclaim
        it on the next poll instead of waiting out the full lease TTL.

        **The attempt is refunded.** ``claim_next_render_job`` charges
        ``attempts += 1`` on *every* claim, and poisons a job once ``attempts >=
        max_attempts``. That counter exists to stop a job whose payload keeps
        *killing* its worker from cycling forever — it is a crash counter. A
        graceful handoff is not a crash: the claim is surrendered without a
        verdict, so the job never got its fair try and must not be charged for it.

        Without the refund, every rollout / HPA scale-down / node drain that
        interrupts an in-flight render silently burns one of the job's three
        lives. A redesign runs for minutes (a 16k-token WM call plus multi-viewport
        Chromium), and this deployment is ``replicas:2 + HPA``, so a perfectly
        healthy job caught by three deploys dies as ``exceeded max claim
        attempts`` — a poison verdict on a job that never once misbehaved. That
        is the real source of the ``website_redesign`` poison failures.

        A genuine crash (OOMKill / SIGKILL) never reaches this code path — the
        lease simply expires unrefunded — so real poison detection is preserved
        exactly as designed.
        """
        result = await db.execute(
            update(RedesignJob)
            .where(RedesignJob.claimed_by == claimed_by, RedesignJob.status == "processing")
            .values(
                lease_expires_at=datetime.now(UTC),
                # Refund the attempt this claim consumed. GREATEST(...,0) keeps the
                # counter non-negative even if a row is somehow released twice.
                attempts=func.greatest(func.coalesce(RedesignJob.attempts, 1) - 1, 0),
            )
        )
        count: int = getattr(result, "rowcount", 0) or 0
        if count:
            logger.info(
                "Released %d in-flight lease(s) for %s at shutdown (attempt refunded)",
                count,
                claimed_by,
            )
        return count

    @staticmethod
    async def cancel_job(db: AsyncSession, job_id: str, *, tenant_id: str | None) -> bool:
        """Cancel a job owned by ``tenant_id``. Returns True if it was cancellable.

        A cross-tenant id returns False, which the route turns into the SAME 409 a
        genuinely already-finished job produces — no existence oracle, no contract
        change. The UPDATE is likewise tenant-scoped so a mismatch cannot write.
        """
        job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if job is None:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        now = datetime.now(UTC)
        conditions = [RedesignJob.job_id == job_id]
        if tenant_id is not None:
            conditions.append(RedesignJob.tenant_id == tenant_id)
        await db.execute(
            update(RedesignJob)
            .where(*conditions)
            .values(status="cancelled", cancelled_at=now, completed_at=now)
        )
        # A cancel landing mid-fan-out is one of the three ways a variant was
        # stranded: the render coroutine is not what writes the verdict, so
        # nothing else will ever touch those rows. See ``settle_orphan_variants``.
        await JobManager.settle_orphan_variants(db, job_id, parent_status="cancelled")
        return True

    @staticmethod
    async def list_jobs(
        db: AsyncSession,
        tenant_id: str,
        *,
        status: str | None = None,
        page_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[RedesignJob], int]:
        """List jobs for a tenant with optional filters. Returns (jobs, total_count)."""
        from sqlalchemy import func as sa_func

        conditions = [RedesignJob.tenant_id == tenant_id]
        if status:
            conditions.append(RedesignJob.status == status)
        if page_type:
            conditions.append(RedesignJob.page_type == page_type)

        count_result = await db.execute(
            select(sa_func.count()).select_from(RedesignJob).where(*conditions)
        )
        total = count_result.scalar() or 0

        result = await db.execute(
            select(RedesignJob)
            .where(*conditions)
            .order_by(RedesignJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        jobs = list(result.scalars().all())
        return jobs, total

    @staticmethod
    async def is_cancelled(db: AsyncSession, job_id: str) -> bool:
        """Check if a job has been cancelled.

        Worker-side poll on a job the worker already owns via the claim, so it is
        deliberately cross-tenant (``tenant_id=None``); the signature stays
        ``(db, job_id)`` for its callers.
        """
        job = await JobManager.get_job(db, job_id, tenant_id=None)
        return job is not None and job.status == "cancelled"

    @staticmethod
    async def get_iteration_history(
        db: AsyncSession, root_job_id: str, *, tenant_id: str | None
    ) -> list[RedesignJob]:
        """Get all iterations for a root job, ordered by iteration number.

        Scoped to ``tenant_id`` (``None`` = deliberately cross-tenant).
        """
        # Include the root job itself
        stmt = select(RedesignJob).where(
            (RedesignJob.job_id == root_job_id) | (RedesignJob.root_job_id == root_job_id)
        )
        if tenant_id is not None:
            stmt = stmt.where(RedesignJob.tenant_id == tenant_id)
        result = await db.execute(stmt.order_by(RedesignJob.iteration_number.asc().nullsfirst()))
        return list(result.scalars().all())

    # ========================================================================
    # Variant Management
    # ========================================================================

    @staticmethod
    async def create_variant(
        db: AsyncSession,
        *,
        job_id: str,
        style: str,
    ) -> RedesignVariant:
        """Create a variant record for an A/B test style.

        Args:
            db: Async database session.
            job_id: Parent job identifier.
            style: Style name for this variant.

        Returns:
            The newly created RedesignVariant instance.
        """
        variant = RedesignVariant(
            variant_id=str(uuid.uuid7()),
            job_id=job_id,
            style=style,
            status="pending",
        )
        db.add(variant)
        await db.flush()
        return variant

    @staticmethod
    async def get_variants(
        db: AsyncSession, job_id: str, *, tenant_id: str | None
    ) -> list[RedesignVariant]:
        """Get all variants for a job, ordered by creation time.

        ``RedesignVariant`` carries no ``tenant_id`` of its own, so ownership is
        established by joining back to the parent ``RedesignJob`` row.
        """
        stmt = select(RedesignVariant).where(RedesignVariant.job_id == job_id)
        if tenant_id is not None:
            stmt = stmt.join(RedesignJob, RedesignJob.job_id == RedesignVariant.job_id).where(
                RedesignJob.tenant_id == tenant_id
            )
        result = await db.execute(stmt.order_by(RedesignVariant.created_at))
        return list(result.scalars().all())

    @staticmethod
    async def mark_variant_completed(
        db: AsyncSession,
        variant_id: str,
        *,
        redesigned_html_path: str | None = None,
        screenshot_paths: dict[str, str] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> None:
        """Mark a variant as completed with output paths and token usage."""
        logger.debug("Marking variant %s as completed", variant_id)
        await db.execute(
            update(RedesignVariant)
            .where(RedesignVariant.variant_id == variant_id)
            .values(
                status="completed",
                redesigned_html_path=redesigned_html_path,
                screenshot_paths=screenshot_paths,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=estimated_cost,
                completed_at=datetime.now(UTC),
            )
        )

    @staticmethod
    async def mark_variant_failed(db: AsyncSession, variant_id: str, error_message: str) -> None:
        """Mark a variant as failed with an error message."""
        logger.debug("Marking variant %s as failed", variant_id)
        await db.execute(
            update(RedesignVariant)
            .where(RedesignVariant.variant_id == variant_id)
            .values(
                status="failed",
                error_message=error_message,
                completed_at=datetime.now(UTC),
            )
        )

    @staticmethod
    async def settle_orphan_variants(db: AsyncSession, job_id: str, *, parent_status: str) -> int:
        """Retire *job_id*'s unfinished variants once the parent reached a verdict.

        Measured on prod 2026-09-08: **317 ``redesign_variants`` rows sat
        ``pending`` under a parent job that was already terminal** — 225 under
        ``failed`` parents and 92 under ``completed`` ones, oldest 2026-06-07,
        and ZERO under a parent still ``pending``/``processing``. None of them
        carried an error string, so they were invisible to every ranking of
        ``error_message``; ``GET /jobs/{id}/variants`` simply advertised work
        that was never coming.

        Why nothing reclaimed them. ``create_variant`` only ever INSERTs, and the
        only writers that move a variant off ``pending`` are:

        * its own render (``mark_variant_completed``/``mark_variant_failed``),
          which does not run if the process died or the parent's verdict was
          reached elsewhere (a cancel landing mid-fan-out, an "all variants
          failed" verdict, a pod restart);
        * the two SQL sweeps in ``rls_resolvers`` — but ``claim_next_render_job``
          retires them only on a RE-CLAIM, and its claim predicate selects
          ``status IN ('pending','processing')`` jobs only, so a job that is
          already terminal is **never claimed again and never re-swept**; and
          ``reap_stale_batch_jobs`` reaches only batch-owned children.

        That is the reachability hole this closes, from the Python side: every
        write that gives the PARENT a verdict settles the children in the same
        transaction. ``failed`` is used because it is the only non-``completed``
        member of :data:`TERMINAL_VARIANT_STATUSES` — no new state is invented,
        and the reason string says what actually happened rather than blaming the
        tenant (a variant under a ``cancelled`` parent reads "parent job reached
        terminal status 'cancelled'", which is the truth, not "you cancelled it").

        Idempotent: the predicate is the non-terminal set, so a second call after
        the first matches zero rows and cannot overwrite a real verdict.

        Returns:
            The number of variants settled — 0 for the overwhelmingly common case
            of a job whose variants all reported their own outcome, and for every
            single-style job, which has no variant rows at all.
        """
        result = await db.execute(
            update(RedesignVariant)
            .where(
                RedesignVariant.job_id == job_id,
                RedesignVariant.status.in_(NON_TERMINAL_VARIANT_STATUSES),
            )
            .values(
                status="failed",
                error_message=(
                    f"abandoned: the parent job reached terminal status "
                    f"'{parent_status}' while this variant was still unfinished — "
                    f"the variant render never reported an outcome"
                ),
                completed_at=datetime.now(UTC),
            )
        )
        settled: int = getattr(result, "rowcount", 0) or 0
        if settled:
            logger.info(
                "Settled %d unfinished variant(s) of job %s: parent reached '%s'",
                settled,
                job_id,
                parent_status,
            )
        return settled

    # ========================================================================
    # Usage Tracking
    # ========================================================================

    @staticmethod
    async def record_usage(
        db: AsyncSession,
        *,
        tenant_id: str,
        pages_redesigned: int = 0,
        screenshots_rendered: int = 0,
        concept_images_generated: int = 0,
        llm_tokens_used: int = 0,
        variants_generated: int = 0,
        is_error: bool = False,
    ) -> None:
        """Record usage for daily aggregation (upsert)."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        today = date_type.today()
        stmt = pg_insert(RedesignUsage).values(
            tenant_id=tenant_id,
            date=today,
            request_count=1,
            pages_redesigned=pages_redesigned,
            screenshots_rendered=screenshots_rendered,
            concept_images_generated=concept_images_generated,
            llm_tokens_used=llm_tokens_used,
            variants_generated=variants_generated,
            error_count=1 if is_error else 0,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_redesign_usage_tenant_date",
            set_={
                "request_count": RedesignUsage.request_count + 1,
                "pages_redesigned": RedesignUsage.pages_redesigned + pages_redesigned,
                "screenshots_rendered": RedesignUsage.screenshots_rendered + screenshots_rendered,
                "concept_images_generated": RedesignUsage.concept_images_generated
                + concept_images_generated,
                "llm_tokens_used": RedesignUsage.llm_tokens_used + llm_tokens_used,
                "variants_generated": RedesignUsage.variants_generated + variants_generated,
                "error_count": RedesignUsage.error_count + (1 if is_error else 0),
            },
        )
        await db.execute(stmt)
