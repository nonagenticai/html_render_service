"""Job + batch + refinement lifecycle for ``RedesignService``."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.database.models import RedesignJob
from src.llm_credentials import LLMConfigLike
from src.services.batch_progress import batch_progress
from src.services.html_generator import estimate_cost
from src.services.job_manager import TERMINAL_JOB_STATUSES, get_claim_owner
from src.services.style_manager import StyleManager
from src.services.upstream_failure import describe_failure, is_transient_upstream

logger = logging.getLogger(__name__)


def _pkg() -> Any:
    from src.services import design_builder as _p

    return _p


class RedesignLifecycleMixin:
    """Mixin: create_job, refinement, and batch processing for ``RedesignService``."""

    _storage: Any
    _html_gen: Any
    _dispatch_webhook: Callable[..., Awaitable[None]]
    # Provided by RedesignHelpersMixin. Sync + returns None: the cost-bus emit is
    # fire-and-forget and is deliberately NOT awaited into the render's success.
    _emit_render_cost_event: Callable[..., None]
    process_job: Callable[..., Awaitable[None]]

    async def create_job(
        self,
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
        styles: list[str] | None = None,
        component_target: dict | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        mode: str = "freeform",
        interactivity: str = "static",
        component_set: str | None = None,
        embeds: list[dict] | None = None,
    ) -> str:
        """Create an async redesign job, returns job_id."""
        settings = get_settings()
        job = await _pkg().JobManager.create_job(
            db,
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
            template_version=settings.redesign.template_version,
            styles=styles,
            component_target=component_target,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            mode=mode,
            interactivity=interactivity,
            component_set=component_set,
            embeds=embeds,
        )
        return job.job_id

    async def create_refinement_job(
        self,
        db: AsyncSession,
        parent_job_id: str,
        feedback: str,
        generate_screenshots: bool = True,
        generate_concept_images: bool = False,
        *,
        tenant_id: str,
    ) -> str:
        """Create a refinement job from a completed parent owned by ``tenant_id``.

        ``tenant_id`` is required, not optional: the child row inherits
        ``parent.tenant_id``, so an unscoped parent lookup let any caller clone
        another tenant's job **into that tenant** and bill them for it. There is no
        legitimate cross-tenant refinement, hence a plain ``str``.
        """
        settings = get_settings()
        parent = await _pkg().JobManager.get_job(db, parent_job_id, tenant_id=tenant_id)
        if parent is None:
            raise ValueError(f"Parent job not found: {parent_job_id}")
        if parent.status != "completed":
            raise ValueError(f"Parent job is not completed (status: {parent.status})")

        root_id = parent.root_job_id or parent.job_id
        history = await _pkg().JobManager.get_iteration_history(
            db, root_id, tenant_id=parent.tenant_id
        )
        if len(history) >= settings.redesign.max_iterations:
            raise ValueError(f"Maximum iterations ({settings.redesign.max_iterations}) reached")

        job = await _pkg().JobManager.create_refinement_job(
            db,
            parent_job=parent,
            feedback=feedback,
            generate_screenshots=generate_screenshots,
            generate_concept_images=generate_concept_images,
        )
        return job.job_id

    async def _load_parent_html(self, db: AsyncSession, job: RedesignJob, tenant_id: str) -> str:
        """Return the parent design's stored HTML, or raise saying which link broke.

        Lifted out of ``process_refinement_job`` verbatim (no behaviour change)
        so the cancel-before-start gate could be added there without pushing it
        over the complexity ceiling. Every arm raises ``ValueError``, which the
        caller's recorder turns into a terminal failure with the reason on the
        row — so the three distinct causes stay distinguishable to the tenant.
        """
        if job.parent_job_id is None:
            raise ValueError("Refinement job missing parent_job_id")
        parent = await _pkg().JobManager.get_job(db, job.parent_job_id, tenant_id=tenant_id)
        if parent is None or not parent.redesigned_html_path:
            raise ValueError("Parent job HTML not found")
        previous_html = await self._storage.retrieve_text(parent.redesigned_html_path)
        if not previous_html:
            raise ValueError("Parent HTML file not found on storage")
        return previous_html

    async def process_refinement_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        generate_screenshots: bool = True,
        generate_concept_images: bool = False,
        focus_areas: list[str] | None = None,
        llm_config: LLMConfigLike | None = None,
    ) -> None:
        """Process a refinement job by applying feedback to a parent's HTML."""
        t0 = time.monotonic()
        logger.info("Processing refinement job %s for tenant=%s", job_id, tenant_id)
        job = await _pkg().JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        if not await _pkg().JobManager.mark_processing(db, job_id):
            # Cancelled before it started — same gate as ``process_job``. See
            # ``JobManager.mark_processing``: unfenced, this write erased the
            # cancellation and every later ``_check_cancelled`` went blind.
            logger.info(
                "Refinement job %s was cancelled before processing began; nothing to do", job_id
            )
            await db.commit()
            return

        await db.commit()

        settings = get_settings()
        screenshots_count = 0

        try:
            await self._dispatch_webhook(db, job, "job.started")

            # ONE generation budget for this refinement job — same shape as
            # ``RedesignService.process_job`` (service.py:207-230). Without it
            # ``refine_full_page``'s multi-minute LLM round trip ran with no
            # deadline and no call accounting at all (#131).
            from src.services.html_gen.section_run import GenerationBudget

            budget = GenerationBudget.from_settings()

            previous_html = await self._load_parent_html(db, job, tenant_id)

            # Resolve custom style description for refinement
            style_description: str | None = None
            if job.style not in ("modern", "minimalist", "bold", "elegant"):
                custom_style = await StyleManager.get_style_by_slug(db, tenant_id, job.style)
                if custom_style:
                    style_description = StyleManager.resolve_style_description(custom_style)

            # Release the connection before the long refine call: holding an
            # open transaction across it lets the idle backend be reaped, so the
            # post-refine write would hit a dead socket (see process_job).
            await db.commit()

            llm_result = await self._html_gen.refine_full_page(
                url=job.url,
                previous_html=previous_html,
                feedback=job.feedback_text or "",
                page_type=job.page_type,
                style=job.style,
                brand=job.brand_info,
                site_analysis=job.site_analysis,
                style_description=style_description,
                focus_areas=focus_areas,
                llm_config=llm_config,
                budget=budget,
            )

            job_dir = self._storage.get_job_dir(job_id)
            redesigned_html_path = await self._storage.store_html(
                job_id, llm_result.text, "redesign.html"
            )

            screenshot_paths: dict[str, str] | None = None
            if generate_screenshots and settings.screenshot.enabled:
                screenshots_dir = job_dir / "screenshots"
                screenshot_paths = await _pkg().render_from_html_string(
                    llm_result.text,
                    screenshots_dir,
                    settings.screenshot.viewports,
                    timeout_ms=settings.screenshot.timeout_ms,
                )
                screenshots_count = len(screenshot_paths)

            job_cost = estimate_cost(
                llm_result.input_tokens, llm_result.output_tokens, self._html_gen._model
            )
            refined = await _pkg().JobManager.mark_completed(
                db,
                job_id,
                redesigned_html_path=redesigned_html_path,
                screenshot_paths=screenshot_paths,
                pages_count=1,
                input_tokens=llm_result.input_tokens or None,
                output_tokens=llm_result.output_tokens or None,
                estimated_cost=job_cost if job_cost > 0 else None,
                worker_id=get_claim_owner(),
            )

            await self._settle_refinement_completion(
                db,
                job,
                job_id,
                tenant_id,
                refined=refined,
                screenshots_count=screenshots_count,
                llm_tokens_used=llm_result.input_tokens + llm_result.output_tokens,
            )

        except Exception as exc:
            # Catch EVERY exception type, deliberately — same reasoning as
            # ``RedesignService.process_job``. This block is a refinement job's
            # ONLY chance to record why it died: it is what calls ``mark_failed``
            # (moving the row out of ``processing``), ``record_usage(is_error)``
            # and the ``job.failed`` webhook. The previous narrow tuple
            # ``(OSError, ValueError, RuntimeError, TimeoutError)`` could not see
            # what the enclosed work really raises — ``httpx.HTTPError`` from the
            # ``worker_management`` refine call and the webhook,
            # ``playwright.async_api.Error``/``TimeoutError`` from
            # ``render_from_html_string``, ``redis.exceptions.RedisError`` from
            # the cache, ``sqlalchemy.exc.*`` from the storage/DB work. All of
            # those derive from ``Exception``, not ``OSError``, and Playwright's
            # and redis-py's ``TimeoutError`` merely share the builtin's *name*.
            # So the most common real failures skipped this handler and the
            # tenant's refinement job was left stuck in ``processing`` with no
            # recorded reason.
            #
            # Catching broadly is safe here precisely because this handler does
            # not swallow: it records and re-raises (below), so the caller's
            # behaviour — including ``CancelledError`` propagation on a shutdown
            # drain, which is a ``BaseException`` and never enters here — is
            # unchanged. Narrow tuples belong on per-phase degrade handlers that
            # choose to continue; this one only chooses to remember.
            logger.error("Refinement job %s failed: %s", job_id, exc, exc_info=True)
            error_msg = describe_failure(exc)
            # Same transient/terminal split as the redesign pipeline (see
            # ``RedesignFinalizeMixin._phase_finalize_failed``): an upstream
            # timeout goes back on the queue rather than becoming a verdict, and
            # neither the error-usage counter nor the ``job.failed`` webhook
            # fires for a job that is going to run again.
            worker_id = get_claim_owner()
            if is_transient_upstream(exc) and await _pkg().JobManager.mark_retryable(
                db, job_id, error_msg, worker_id=worker_id
            ):
                await db.commit()
                raise
            await self._persist_refinement_failure(db, job, job_id, tenant_id, error_msg)
            await db.commit()
            raise

        elapsed = time.monotonic() - t0
        logger.info("Refinement job %s completed in %.2fs", job_id, elapsed)
        await db.commit()

    async def _settle_refinement_completion(
        self,
        db: AsyncSession,
        job: RedesignJob,
        job_id: str,
        tenant_id: str,
        *,
        refined: bool,
        screenshots_count: int,
        llm_tokens_used: int,
    ) -> None:
        """Bill and notify a refinement — but ONLY if its completion actually landed.

        Same conditional ordering as the redesign pipeline's
        ``_phase_finalize_completed``, and for the same window: the
        ``refine_full_page`` call above is a multi-minute LLM round trip with no
        cancellation check after it, so ``POST .../cancel`` lands inside it
        routinely. ``mark_completed`` refuses to overwrite the resulting
        ``cancelled``, and *these two calls are the ones that reach the tenant* —
        ``record_usage`` bills them, the webhook tells them the refinement they
        stopped is ready. Fencing the row without gating these would have kept
        the invoice and the notification and merely stopped the row agreeing
        with them.
        """
        if not refined:
            logger.warning(
                "Refinement job %s completion was fenced off — not billing and not "
                "dispatching job.completed. Cancelled, or owned by another worker.",
                job_id,
            )
            return
        await _pkg().JobManager.record_usage(
            db,
            tenant_id=tenant_id,
            pages_redesigned=1,
            screenshots_rendered=screenshots_count,
            llm_tokens_used=llm_tokens_used,
        )
        # Same fence, same side: a refinement whose completion did not land
        # returned above and bills neither the usage row nor the cost bus.
        self._emit_render_cost_event(
            job_id=job_id,
            tenant_id=tenant_id,
            stage="refinement",
            screenshots=screenshots_count,
            llm_tokens=llm_tokens_used,
        )
        await self._dispatch_webhook(db, job, "job.completed")

    async def _persist_refinement_failure(
        self,
        db: AsyncSession,
        job: RedesignJob,
        job_id: str,
        tenant_id: str,
        error_msg: str,
    ) -> None:
        """Best-effort: record the failure and notify, without ever replacing
        the exception the caller is already in the middle of re-raising.

        Split out of ``process_refinement_job`` so the guard below has its own
        scope: the caller is inside ``except Exception as exc: ...; raise`` --
        the bare ``raise`` after this call is what makes ``exc`` reach the
        caller. A DB error out of ``mark_failed``/``record_usage``, or a
        network error out of the webhook, would otherwise propagate from here
        instead, replacing ``exc`` (the real cause) and the bare ``raise``
        would never run -- the worse failure of the two would silently win.
        Log-and-continue keeps ``exc`` intact for it.

        Cost: if ``mark_failed``/``record_usage`` genuinely fail, the
        refinement job is left in ``processing`` with no ``error_message``,
        same as the redesign pipeline's equivalent gap in
        ``RedesignFinalizeMixin._phase_finalize_failed``. Nothing else
        reconciles it besides the claim lease lapsing. The webhook failure is
        pure best-effort, same as that same sibling method: the notification
        is optional, the record is not.
        """
        # Tri-state, same as ``_phase_finalize_failed``: a FENCED-OFF write (the
        # job is cancelled/completed, or a sibling holds the claim) must produce
        # no error usage and no ``job.failed``, while a write that RAISED is a
        # real failure the tenant is still owed.
        fenced_off = False
        try:
            fenced_off = not await _pkg().JobManager.mark_failed(
                db, job_id, error_msg, worker_id=get_claim_owner()
            )
            if not fenced_off:
                await _pkg().JobManager.record_usage(db, tenant_id=tenant_id, is_error=True)
        except Exception:
            logger.exception(
                "Failed to persist failure state for refinement job %s (original error: %s)",
                job_id,
                error_msg,
            )
        if fenced_off:
            logger.warning(
                "Refinement job %s failure was fenced off — not recording error usage and "
                "not dispatching job.failed. Someone else owns this job's outcome.",
                job_id,
            )
            return
        try:
            await self._dispatch_webhook(db, job, "job.failed", error=error_msg)
        except Exception:  # noqa: BLE001 — the failure record outranks the notification
            logger.warning("Failed to dispatch failure webhook for refinement job %s", job_id)

    async def create_batch(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        pages: list[dict],
        style: str = "modern",
        scope: str = "full_page",
        generate_screenshots: bool = True,
        generate_concept_images: bool = False,
        critique_findings: list[dict] | None = None,
        brand_info: dict | None = None,
        site_analysis: dict | None = None,
        user_context: dict | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Create a batch job with child page jobs. Returns (batch_id, page_jobs)."""
        settings = get_settings()

        focus_pages = user_context.get("focus_pages") if user_context else None
        exclude_pages = user_context.get("exclude_pages") if user_context else None

        scored_pages = _pkg().select_pages(
            pages,
            critique_findings=critique_findings,
            max_pages=settings.redesign.max_pages,
            focus_pages=focus_pages,
            exclude_pages=exclude_pages,
        )

        batch = await _pkg().BatchManager.create_batch(
            db,
            tenant_id=tenant_id,
            total_pages=len(scored_pages),
            shared_settings={
                "style": style,
                "scope": scope,
                "generate_screenshots": generate_screenshots,
                "generate_concept_images": generate_concept_images,
            },
        )

        page_jobs: list[dict] = []
        for i, page in enumerate(scored_pages):
            job = await _pkg().JobManager.create_job(
                db,
                tenant_id=tenant_id,
                url=page["url"],
                html_content=page.get("html_content"),
                scope=scope,
                style=style,
                page_type=page.get("page_type"),
                critique_findings=critique_findings,
                brand_info=brand_info,
                site_analysis=site_analysis,
                user_context=user_context,
                batch_id=batch.batch_id,
                priority_score=len(scored_pages) - i,  # Higher score = earlier in list
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
            )
            page_jobs.append(
                {
                    "job_id": job.job_id,
                    "url": page["url"],
                    "page_type": page.get("page_type"),
                    "priority_score": len(scored_pages) - i,
                }
            )

        return batch.batch_id, page_jobs

    @staticmethod
    def _select_batch_pages(
        batch_id: str, all_jobs: list[RedesignJob], job_ids: list[str] | None
    ) -> list[RedesignJob]:
        """Decide which of a batch's child jobs this run may render.

        Two independent restrictions, and the second is not redundant.

        ``job_ids`` is the CALLER's scope — the retry route naming the jobs it
        just reset. It is resolved against ``all_jobs`` rather than by querying
        the ids directly, which keeps the tenant predicate ``get_batch_jobs``
        already applied: a forged id from another tenant's batch cannot smuggle
        a job into this run.

        The terminal skip is the INVARIANT — a finished page is never
        re-rendered — and it holds even for a caller that forgets to pass a
        scope. ``retry-failed`` is not the only way to re-enter ``process_batch``
        (a second background dispatch, a future re-run path), and the harm is not
        a stale row: it is a paid LLM render that destroys an approved design and
        double-counts ``completed_pages``.
        """
        wanted = set(job_ids) if job_ids is not None else None
        jobs: list[RedesignJob] = []
        for job_ in all_jobs:
            if wanted is not None and job_.job_id not in wanted:
                continue
            if job_.status in TERMINAL_JOB_STATUSES:
                logger.info(
                    "Batch %s: skipping page %s, already %s — not re-rendering finished work",
                    batch_id,
                    job_.job_id,
                    job_.status,
                )
                continue
            jobs.append(job_)

        if wanted is not None and len(jobs) != len(wanted):
            # Loud: "you asked to retry 3 pages and I can run 2" is a fact the
            # operator needs, not something to absorb into a shorter loop.
            logger.warning(
                "Batch %s retry asked for %d pages but %d are runnable "
                "(the rest are missing, another tenant's, or already finished)",
                batch_id,
                len(wanted),
                len(jobs),
            )
        return jobs

    async def process_batch(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        batch_id: str,
        tenant_id: str,
        job_ids: list[str] | None = None,
    ) -> None:
        """Process pages in a batch with concurrency control.

        ``job_ids`` restricts the run to exactly those child jobs. It exists for
        ``POST /api/batch/{id}/retry-failed``, which reset 3 failed jobs and then
        called this with no scope at all: ``get_batch_jobs`` has no status
        filter and ``process_job`` had no status guard, so **3 failures in a
        20-page batch re-rendered all 20**. The 17 already-approved designs were
        overwritten with fresh LLM output, the tenant paid for 17 renders they
        did not ask for, and ``completed_pages`` finished at 37 of 20 because
        every page incremented the counter a second time.

        ``None`` means the whole batch, which is correct for the initial run.
        """
        t0 = time.monotonic()
        logger.info("Processing batch %s for tenant=%s", batch_id, tenant_id)
        settings = get_settings()

        async with session_factory() as db:
            if not await _pkg().BatchManager.mark_processing(db, batch_id):
                logger.info(
                    "Batch %s was cancelled before processing began; no pages will be "
                    "rendered and no batch webhook will be dispatched",
                    batch_id,
                )
                await db.commit()
                return
            await db.commit()
            all_jobs = await _pkg().BatchManager.get_batch_jobs(db, batch_id, tenant_id=tenant_id)
            batch = await _pkg().BatchManager.get_batch(db, batch_id, tenant_id=tenant_id)

        jobs = self._select_batch_pages(batch_id, all_jobs, job_ids)

        shared = (batch.shared_settings or {}) if batch else {}
        generate_screenshots = shared.get("generate_screenshots", True)
        generate_concept_images = shared.get("generate_concept_images", False)

        semaphore = asyncio.Semaphore(settings.redesign.batch_max_concurrency)

        async def process_page(job_: RedesignJob) -> None:
            async with semaphore, session_factory() as page_db:
                batch_check = await _pkg().BatchManager.get_batch(
                    page_db, batch_id, tenant_id=tenant_id
                )
                if batch_check and batch_check.status == "cancelled":
                    return

                async def _report_progress() -> None:
                    """Bump the batch's progress clock from inside this page.

                    ITS OWN SESSION AND ITS OWN COMMIT, deliberately. ``page_db``
                    is mid-render and will not commit until the whole page is
                    done — which is precisely the multi-hour window the reaper
                    cannot see into — so a bump written on that session would
                    become visible only at the moment it stopped being needed.
                    """
                    async with session_factory() as progress_db, progress_db.begin():
                        await _pkg().BatchManager.touch_progress(progress_db, batch_id)

                try:
                    with batch_progress(_report_progress):
                        await self.process_job(
                            page_db,
                            job_.job_id,
                            tenant_id,
                            generate_screenshots=generate_screenshots,
                            generate_concept_images=generate_concept_images,
                            session_factory=session_factory,
                        )
                    # ``process_job`` returns normally on cancellation too — its
                    # ``_check_cancelled`` gates are plain early returns, not
                    # exceptions — so "it did not raise" is NOT "the page was
                    # redesigned". Counting a cancelled page ``completed`` was
                    # the other half of the cancelled-batch defect: an in-flight
                    # page bumped ``completed_pages`` for work that never
                    # happened, which the tenant reads as progress and
                    # ``finalize_batch`` read as a clean run.
                    #
                    # ``expire_all`` is LOAD BEARING, not hygiene: this session
                    # already loaded the batch row for the pre-flight check
                    # above, so a second ``get_batch`` returns that identity-map
                    # instance and reports the status as it was BEFORE the
                    # render — which is the one status this check must not
                    # trust. The cancellation is committed by a different
                    # session (the route's), so only a fresh read can see it.
                    page_db.expire_all()
                    cancel_check = await _pkg().BatchManager.get_batch(
                        page_db, batch_id, tenant_id=tenant_id
                    )
                    if cancel_check is not None and cancel_check.status == "cancelled":
                        logger.info(
                            "Batch page %s stopped by cancellation — not counting it completed",
                            job_.job_id,
                        )
                        return
                    await _pkg().BatchManager.increment_completed(page_db, batch_id)
                    await page_db.commit()
                except Exception as exc:  # noqa: BLE001 — per-page absorb, see comment
                    # Catch every exception type, mirroring the variant fan-out in
                    # variants.py: the contract is "one bad page is counted failed
                    # and the batch moves on". The previous narrow tuple
                    # (OSError, ValueError, RuntimeError, TimeoutError) missed the
                    # types process_job really raises transitively — Playwright's
                    # Error/TimeoutError, redis's RedisError, sqlalchemy.exc.*,
                    # KeyError — all of which derive from Exception, not OSError.
                    # Those escaped into the gather(return_exceptions=True) below,
                    # which silently absorbed them, so increment_failed never ran:
                    # the batch counters under-counted and finalize_batch could
                    # never reach a terminal state. Counting the failure is the
                    # whole point of this handler, so it must be unmissable.
                    logger.error(
                        "Batch page %s failed (%s): %s", job_.job_id, type(exc).__name__, exc
                    )
                    await page_db.rollback()
                    await _pkg().BatchManager.increment_failed(page_db, batch_id)
                    await page_db.commit()

        tasks = [process_page(j) for j in jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

        async with session_factory() as db:
            final_status = await _pkg().BatchManager.finalize_batch(db, batch_id)
            await db.commit()
            elapsed = time.monotonic() - t0
            logger.info("Batch %s finalized as %s in %.2fs", batch_id, final_status, elapsed)

            # Dispatch batch-level webhooks
            if jobs:
                first_job = jobs[0]
                if final_status in ("completed", "completed_with_errors"):
                    await self._dispatch_webhook(
                        db, first_job, "batch.completed", batch_id=batch_id
                    )
                elif final_status == "failed":
                    await self._dispatch_webhook(db, first_job, "batch.failed", batch_id=batch_id)

            await db.commit()  # Persist webhook delivery logs
