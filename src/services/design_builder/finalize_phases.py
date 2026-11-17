"""Phase methods 6-11: a11y, screenshots, concept images, finalize."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from PIL.Image import DecompressionBombError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.net import SSRFError

from src.config import Settings
from src.database.models import RedesignJob
from src.llm_credentials import LLMConfigLike
from src.services.concept_image_generator import ConceptImageResult
from src.services.design_builder.helpers import capture_failure_types
from src.services.design_builder.types import (
    RenderOptions,
    _A11yResult,
    _GenerationResult,
    _ScreenshotsResult,
    _ThemeDecision,
    _TokenAccumulator,
)
from src.services.html_gen.section_run import GenerationBudget
from src.services.html_generator import estimate_cost
from src.services.job_manager import get_claim_owner
from src.services.upstream_failure import describe_failure, is_transient_upstream

logger = logging.getLogger(__name__)


def _pkg() -> Any:
    from src.services import design_builder as _p

    return _p


class RedesignFinalizeMixin:
    """Mixin: a11y, screenshots, concept-images, completion / failure phases."""

    _a11y_scorer: Any
    _storage: Any
    _concept_gen: Any
    _html_gen: Any
    _run_critique_adapter_revalidation: Callable[[str], dict]
    _dispatch_webhook: Callable[..., Awaitable[None]]
    # Provided by RedesignHelpersMixin. Sync + returns None: the cost-bus emit is
    # fire-and-forget and is deliberately NOT awaited into the render's success.
    _emit_render_cost_event: Callable[..., None]

    async def _score_or_fix_html(
        self,
        job_id: str,
        redesigned_html_path: str,
        *,
        auto_fix_accessibility: bool,
        accessibility_threshold: float,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> tuple[float | None, dict | None, str | None]:
        """Score (and optionally auto-fix) HTML; returns (score, report, new_path)."""
        html_for_check = await self._storage.retrieve_text(redesigned_html_path)
        if not html_for_check:
            return None, None, None
        new_html_path: str | None = None
        if auto_fix_accessibility:
            fixed_html, report = await self._a11y_scorer.score_and_fix(
                html_for_check,
                threshold=accessibility_threshold,
                llm_config=llm_config,
                budget=budget,
            )
            if report.html_was_fixed:
                new_html_path = await self._storage.store_html(job_id, fixed_html, "redesign.html")
        else:
            report = await self._a11y_scorer.score(
                html_for_check, llm_config=llm_config, budget=budget
            )
        return report.overall_score, report.to_dict(), new_html_path

    async def _augment_report_with_revalidation(
        self,
        current_path: str,
        a11y_report: dict | None,
        a11y_score: float | None,
        theme: _ThemeDecision,
    ) -> dict | None:
        """Run critique-adapter revalidation and merge metadata into the report."""
        html_for_revalidation = await self._storage.retrieve_text(current_path)
        critique_adapter_validation: dict | None = None
        if html_for_revalidation:
            critique_adapter_validation = self._run_critique_adapter_revalidation(
                html_for_revalidation
            )
        if a11y_report is None:
            a11y_report = {
                "overall_score": float(a11y_score or 0.0),
                "categories": [],
                "findings": [],
                "html_was_fixed": False,
            }
        if theme.auto_theme_metadata is not None:
            a11y_report["auto_theme"] = theme.auto_theme_metadata
        if critique_adapter_validation is not None:
            a11y_report["critique_adapter_validation"] = critique_adapter_validation
        return a11y_report

    async def _phase_accessibility(
        self,
        job_id: str,
        gen_result: _GenerationResult,
        theme: _ThemeDecision,
        *,
        run_accessibility_check: bool,
        auto_fix_accessibility: bool,
        accessibility_threshold: float,
        accessibility_first_auto_theme: bool,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> _A11yResult:
        """Phase 6: optional accessibility scoring + auto-fix + adapter revalidation.

        ``budget`` is the job's :class:`GenerationBudget` (see ``service.py``
        job setup) — threaded through so the scorer's LLM call(s) run under the
        same deadline/call-count seam as generation, instead of outside it
        (#130). ``None`` is accepted for callers with no job budget in scope;
        the scorer treats that as unbounded.
        """
        a11y_score: float | None = None
        a11y_report: dict | None = None
        new_html_path: str | None = None
        redesigned_html_path = gen_result.redesigned_html_path

        effective_run = run_accessibility_check or accessibility_first_auto_theme
        if effective_run and self._a11y_scorer and redesigned_html_path:
            a11y_score, a11y_report, new_html_path = await self._score_or_fix_html(
                job_id,
                redesigned_html_path,
                auto_fix_accessibility=auto_fix_accessibility,
                accessibility_threshold=accessibility_threshold,
                llm_config=llm_config,
                budget=budget,
            )

        if accessibility_first_auto_theme and (new_html_path or redesigned_html_path):
            current_path = new_html_path or redesigned_html_path
            if current_path is not None:
                a11y_report = await self._augment_report_with_revalidation(
                    current_path, a11y_report, a11y_score, theme
                )

        return _A11yResult(
            score=a11y_score,
            report=a11y_report,
            redesigned_html_path=new_html_path,
        )

    async def _capture_original_screenshots(
        self,
        job: RedesignJob,
        job_id: str,
        orig_dir: Path,
        viewports: list[str],
        settings: Settings,
        opts: RenderOptions,
    ) -> dict[str, str] | None:
        """Capture the live "before" screenshots, or ``None`` if that failed.

        Split out of :meth:`_phase_screenshots` purely to keep that method under
        the complexity ceiling; the behaviour is unchanged.

        The SSRF clause must precede the generic tuple: ``SSRFError`` subclasses
        ``ValueError``, so the generic handler would otherwise swallow a blocked
        internal target as an indistinguishable "capture failed" warning.
        Blocking is a security decision, not a transient render failure, so it is
        logged loudly with the job id. Non-fatal by design — this is the optional
        before/after original, and phase 1 (``_fetch_page_html``) already
        hard-fails the job for the same URL whenever there is no inline
        ``html_content``, so nothing internal is ever rendered.

        The generic clause resolves its types through
        :func:`~src.services.design_builder.helpers.capture_failure_types` so
        that Playwright's ``Error``/``TimeoutError`` — which derive from
        ``Exception``, not ``OSError`` — are actually caught. Before that, the
        ordinary nav-failure path was dead code: a tenant page that merely timed
        out escaped this "non-fatal by design" handler and lost a paid job.
        """
        try:
            return await _pkg().capture_url_screenshots(
                job.url,
                orig_dir,
                viewports,
                timeout_ms=settings.screenshot.timeout_ms,
                full_page=opts.full_page,
            )
        except SSRFError as exc:
            logger.error(
                "Refusing original capture for job %s — url %r is not a public target: %s",
                job_id,
                job.url,
                exc,
            )
        except capture_failure_types() as exc:
            logger.warning(
                "Failed to capture original screenshots (%s): %s", type(exc).__name__, exc
            )
        return None

    async def _phase_screenshots(
        self,
        job: RedesignJob,
        job_id: str,
        redesigned_html_path: str | None,
        settings: Settings,
        generate_screenshots: bool,
        render_options: RenderOptions | None = None,
    ) -> _ScreenshotsResult:
        """Phases 7-9: original capture, redesigned render, comparison.

        ``render_options`` shapes the work: which viewports, full-page vs
        viewport-framed, and whether to capture the live original + generate
        a before/after comparison. Defaults reproduce the full historical
        behavior; the promo fast-path opts into a lean single-shot.

        Per-job memory budget on the heavy (default) path: concurrent Chromium
        launches are capped process-wide by ``redesign.render_max_concurrency``
        (the screenshot_renderer semaphore), and ``generate_comparisons``
        down-scales captures past ``MAX_COMPARISON_HEIGHT`` before diffing and
        runs that PIL work off the event loop. Those two are the ceiling that
        keeps a tall-SPA redesign inside the pod's memory limit.
        """
        opts = render_options or RenderOptions()
        original_paths: dict[str, str] | None = None
        screenshot_paths: dict[str, str] | None = None
        comparison_paths: dict[str, str] | None = None
        diff_paths: dict[str, str] | None = None
        section_paths: dict[str, str] | None = None

        if not generate_screenshots or not settings.screenshot.enabled:
            return _ScreenshotsResult(None, None, None, None, 0)

        viewports = list(opts.viewports) if opts.viewports else settings.screenshot.viewports

        job_dir = self._storage.get_job_dir(job_id)
        # Greenfield (scope=landing, no source URL) has no original page to capture —
        # skip it rather than paying a doomed Page.goto(""). That Playwright Error is
        # now caught by _capture_original_screenshots, but there is still nothing to
        # screenshot, so the skip stays.
        if opts.capture_original and (job.url or "").strip():
            original_paths = await self._capture_original_screenshots(
                job, job_id, job_dir / "original_screenshots", viewports, settings, opts
            )

        if redesigned_html_path:
            html_for_render = await self._storage.retrieve_text(redesigned_html_path)
            if html_for_render:
                screenshots_dir = job_dir / "screenshots"
                screenshot_paths = await _pkg().render_from_html_string(
                    html_for_render,
                    screenshots_dir,
                    viewports,
                    timeout_ms=settings.screenshot.timeout_ms,
                    full_page=opts.full_page,
                )
                if opts.section_screenshots:
                    try:
                        section_viewport = viewports[0] if viewports else "desktop"
                        section_pairs = await _pkg().render_section_screenshots(
                            html_for_render,
                            job_dir / "sections",
                            section_viewport,
                            timeout_ms=settings.screenshot.timeout_ms,
                        )
                        section_paths = {name: path for name, path in section_pairs} or None
                    except capture_failure_types() as exc:
                        # Same Playwright sink, same soft contract as the original
                        # capture above: section stills are an enrichment, so a
                        # browser Error/TimeoutError degrades to "no sections"
                        # rather than escaping and failing the finished redesign.
                        logger.warning(
                            "Failed to capture section screenshots (%s): %s",
                            type(exc).__name__,
                            exc,
                        )

        if opts.generate_comparison and original_paths and screenshot_paths:
            try:
                comp_dir = job_dir / "comparisons"
                comparison_paths, diff_paths = await _pkg().generate_comparisons(
                    original_paths, screenshot_paths, comp_dir
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                TimeoutError,
                DecompressionBombError,
            ) as exc:
                # The before/after composite + pixel diff are enrichment: losing
                # them degrades to "no comparison", never to a lost render.
                #
                # This sink is pure PIL/CPU work (``comparison_generator``), so
                # the builtins genuinely do most of the job here — Pillow's
                # decode failures (``UnidentifiedImageError``, truncated file)
                # subclass ``OSError``. The one exception is
                # ``PIL.Image.DecompressionBombError``, which derives straight
                # from ``Exception``: it fires on a capture whose pixel count
                # exceeds twice ``MAX_IMAGE_PIXELS``, i.e. exactly the tall
                # full-page SPA screenshots this phase exists to compare (the
                # 8000px clip is a *height* cap, not a pixel-count cap, and
                # ``_open_capped`` only down-scales AFTER ``Image.open`` has
                # already tripped the bomb check). Without it named, a
                # bomb-sized before/after pair escaped this handler and
                # hard-failed a redesign that had already been generated,
                # rendered and paid for.
                logger.warning("Failed to generate comparisons: %s", exc)

        # Lift the locally-rendered PNGs into shared storage (no-op on the local
        # backend) so any replica can serve them. Paths are recorded unchanged.
        original_paths = await self._storage.persist_paths(original_paths)
        screenshot_paths = await self._storage.persist_paths(screenshot_paths)
        comparison_paths = await self._storage.persist_paths(comparison_paths)
        diff_paths = await self._storage.persist_paths(diff_paths)
        section_paths = await self._storage.persist_paths(section_paths)

        screenshots_count = len(screenshot_paths) if screenshot_paths else 0
        return _ScreenshotsResult(
            original_paths=original_paths,
            screenshot_paths=screenshot_paths,
            comparison_paths=comparison_paths,
            diff_paths=diff_paths,
            screenshots_count=screenshots_count,
            section_paths=section_paths,
        )

    async def _phase_concept_images(
        self,
        job: RedesignJob,
        job_id: str,
        style: str,
        settings: Settings,
        generate_concept_images: bool,
    ) -> ConceptImageResult:
        """Phase 10: optional concept image generation.

        Not-requested and feature-disabled are non-degraded empty results; a
        requested-and-attempted generation carries the generator's degraded flag
        so the caller can distinguish a real failure from "none requested".
        """
        if not generate_concept_images:
            return ConceptImageResult()
        if not await self._concept_gen.is_available():
            return ConceptImageResult()
        sa = job.site_analysis or {}
        bi = job.brand_info or {}
        concept_dir = self._storage.get_job_dir(job_id) / "concept_images"
        return await self._concept_gen.generate(
            site_description=sa.get("description"),
            target_audience=sa.get("target_audience"),
            brand_name=bi.get("name"),
            primary_color=bi.get("primary_color", "#2563EB"),
            style=style,
            count=settings.dalle.count,
            output_dir=concept_dir,
        )

    async def _phase_finalize_completed(
        self,
        db: AsyncSession,
        job: RedesignJob,
        job_id: str,
        tenant_id: str,
        *,
        gen_result: _GenerationResult,
        redesigned_html_path: str | None,
        a11y: _A11yResult,
        screenshots: _ScreenshotsResult,
        concept_image_assets: list[dict[str, object]] | None,
        concept_count: int,
        concept_image_degraded: bool = False,
        concept_image_reason: str | None = None,
        html_validation_report: dict | None,
        tokens: _TokenAccumulator,
    ) -> None:
        """Phase 11: persist completion, record usage, dispatch webhook.

        The three writes are ORDERED and CONDITIONAL, not a sequence. The DB
        write is fenced (``mark_completed`` refuses to overwrite ``cancelled``
        or another worker's verdict), and the two that follow it are the ones
        that actually reach the tenant: ``record_usage`` BILLS them and the
        webhook TELLS them. Fencing only the row would have left a cancelled
        job invoiced and announced as completed — the row would merely have
        stopped agreeing with the invoice. So a refused completion returns
        before either.
        """
        job_cost = estimate_cost(tokens.input, tokens.output, self._html_gen._model)
        completed = await _pkg().JobManager.mark_completed(
            db,
            job_id,
            redesigned_html_path=redesigned_html_path,
            section_html_paths=gen_result.section_html_paths,
            screenshot_paths=screenshots.screenshot_paths,
            section_screenshot_paths=screenshots.section_paths,
            concept_image_assets=concept_image_assets,
            concept_image_degraded=concept_image_degraded,
            concept_image_reason=concept_image_reason,
            pages_count=1,
            accessibility_score=a11y.score,
            accessibility_report=a11y.report,
            original_screenshot_paths=screenshots.original_paths,
            comparison_paths=screenshots.comparison_paths,
            diff_paths=screenshots.diff_paths,
            component_html_path=gen_result.component_html_path,
            html_validation_report=html_validation_report,
            input_tokens=tokens.input or None,
            output_tokens=tokens.output or None,
            estimated_cost=job_cost if job_cost > 0 else None,
            worker_id=get_claim_owner(),
        )
        if not completed:
            logger.warning(
                "Job %s completion was fenced off — not billing and not dispatching "
                "job.completed. The job was cancelled or another worker owns it.",
                job_id,
            )
            return

        await _pkg().JobManager.record_usage(
            db,
            tenant_id=tenant_id,
            pages_redesigned=1,
            screenshots_rendered=screenshots.screenshots_count,
            concept_images_generated=concept_count,
            llm_tokens_used=tokens.total,
        )
        # Bill the RENDER on the cost bus, on the same side of the fence as
        # ``record_usage``: a fenced-off completion returned above and bills
        # neither. Fire-and-forget — it cannot fail this finished design.
        self._emit_render_cost_event(
            job_id=job_id,
            tenant_id=tenant_id,
            stage="redesign",
            screenshots=screenshots.screenshots_count,
            concept_images=concept_count,
            llm_tokens=tokens.total,
        )

        await self._dispatch_webhook(db, job, "job.completed", accessibility_score=a11y.score)

    async def _phase_finalize_failed(
        self,
        db: AsyncSession,
        job: RedesignJob,
        job_id: str,
        tenant_id: str,
        exc: BaseException,
    ) -> None:
        """Failure path: persist error state, attempt webhook, commit.

        A *transient* upstream failure (an httpx timeout on the worker_management
        LLM call) is not a verdict — it is handed back to the claim queue instead
        of being stamped ``failed``, bounded by the queue's existing
        attempts/max_attempts poison gate. Neither the ``job.failed`` webhook nor
        the error-usage counter fires on that path: the job has not failed yet,
        and telling the tenant it did — then completing on the next attempt —
        would be a lie in both directions.
        """
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        error_msg = describe_failure(exc)
        worker_id = get_claim_owner()
        if is_transient_upstream(exc) and await _pkg().JobManager.mark_retryable(
            db, job_id, error_msg, worker_id=worker_id
        ):
            await db.commit()
            return
        # Tri-state on purpose: "the fence refused this write" (someone else owns
        # the outcome -> stay quiet) is NOT the same as "the write blew up"
        # (the job really did fail and the tenant is still owed the news).
        fenced_off = False
        try:
            fenced_off = not await _pkg().JobManager.mark_failed(
                db, job_id, error_msg, worker_id=worker_id
            )
            if not fenced_off:
                await _pkg().JobManager.record_usage(db, tenant_id=tenant_id, is_error=True)
        except Exception:
            # We are INSIDE `process_job`'s `except Exception as exc: ...; raise`
            # -- this whole method exists to persist `exc` before the bare
            # `raise` re-raises it. A DB error out of `mark_failed`/`record_usage`
            # (e.g. the backend was reaped, see the refine-call comment above)
            # would propagate from here instead, replacing `exc` with an
            # unrelated persistence failure and erasing the real cause from the
            # only exception the caller ever sees. Log-and-continue keeps `exc`
            # intact for that `raise`.
            #
            # Cost: if this write genuinely failed, the job is left in
            # `processing` with no `error_message` -- indistinguishable from a
            # crashed worker until the claim lease lapses and the poison gate
            # in `claim_next_render_job` reclaims it. Nothing else reconciles
            # this; the log line above is the only trace.
            logger.exception(
                "Failed to persist failure state for job %s (original error: %s)",
                job_id,
                error_msg,
            )
        if fenced_off:
            # `mark_failed` returning False means the error was DROPPED: the row
            # is cancelled/completed, or a sibling worker holds the claim and is
            # still rendering. Its result used to be discarded here, so the
            # tenant was told `job.failed` about a job that was about to arrive
            # as `job.completed` from the winning worker, and `error_count` was
            # inflated by a failure that was never recorded anywhere. A verdict
            # that did not land is not a verdict to announce.
            logger.warning(
                "Job %s failure was fenced off — not recording error usage and not "
                "dispatching job.failed. Someone else owns this job's outcome.",
                job_id,
            )
            await db.commit()
            return
        try:
            await self._dispatch_webhook(db, job, "job.failed", error=error_msg)
        except Exception:  # noqa: BLE001 — the failure record outranks the notification
            # Catch every exception type. We are already INSIDE the failure
            # recorder: `mark_failed` and `record_usage` have run and the
            # `db.commit()` below is what makes them durable. Letting a webhook
            # problem escape here would skip that commit and lose the recorded
            # reason — trading the thing that matters for the thing that does
            # not. The notification is best-effort; the record is not.
            logger.warning("Failed to dispatch failure webhook for job %s", job_id)
        await db.commit()
