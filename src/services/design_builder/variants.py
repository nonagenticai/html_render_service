"""A/B variant fan-out for ``RedesignService``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings
from src.database.models import RedesignJob
from src.services.context_resolver import ResolvedContext
from src.services.design_builder.types import RenderOptions
from src.services.html_gen.section_run import GenerationBudget
from src.services.html_generator import estimate_cost
from src.services.style_manager import StyleManager
from src.services.upstream_failure import describe_failure

logger = logging.getLogger(__name__)


def _pkg() -> Any:
    from src.services import design_builder as _p

    return _p


class RedesignVariantsMixin:
    """Mixin providing A/B variant processing for ``RedesignService``."""

    _storage: Any
    _html_gen: Any
    _check_cancelled: Callable[..., Awaitable[bool]]
    _dispatch_webhook: Callable[..., Awaitable[None]]

    async def _process_variants(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None,
        db: AsyncSession,
        job: RedesignJob,
        html: str,
        resolved_ctx: ResolvedContext,
        settings: Settings,
        generate_screenshots: bool = True,
        render_options: RenderOptions | None = None,
        budget: GenerationBudget | None = None,
        quality_warnings: dict[str, list[str]] | None = None,
    ) -> list[BaseException]:
        """Process multiple style variants concurrently.

        Returns the exception that ended each FAILED variant, in no particular
        order. Variant failures are absorbed (one bad style must never take out
        its siblings), so this list is the only place the caller can still see
        *what kind* of failure they were — and the parent job's verdict depends
        on it: an all-variants-failed job whose children all died of an upstream
        timeout is transient, exactly like the single-style path. Reading the
        persisted ``error_message`` strings back instead would mean classifying
        by substring, which is not a classification.

        When a ``session_factory`` is provided each variant gets its own
        independent database session so that ``asyncio.gather`` can safely
        run them in parallel.  If no factory is available the variants
        are processed sequentially using the caller's session (fallback).

        Concurrency is capped at ``settings.redesign.variant_concurrency``:
        each variant runs a multi-minute ``generate_page_chunked`` LLM call
        plus a full screenshot set, so an unbounded ``gather`` over every
        style hammers the redesign-html worker and multiplies render memory.

        ``budget`` is the JOB's single budget, not one per style — every variant below
        spends the SAME object (see the trade spelled out where it is created, in
        ``service.py``). Consequence worth knowing before debugging a short job: styles
        compete, so a late variant on a slow vLLM can be refused with
        ``BudgetExhausted`` where it previously overran the deadline and completed. It
        is recorded as that variant's failure and the job still completes on the
        variants that made it; it is a cost ceiling doing its job, not a bug.
        """
        styles = list(job.styles or [])
        failures: list[BaseException] = []
        if session_factory is not None:
            sem = asyncio.Semaphore(max(1, settings.redesign.variant_concurrency))

            async def _run(style: str) -> None:
                try:
                    async with sem, session_factory() as variant_db:
                        failure = await self._process_variant(
                            variant_db,
                            job,
                            html,
                            style,
                            resolved_ctx,
                            settings,
                            generate_screenshots,
                            render_options,
                            budget=budget,
                            quality_warnings=quality_warnings,
                        )
                    if failure is not None:
                        failures.append(failure)
                except Exception as exc:  # noqa: BLE001 — per-variant absorb, see comment
                    # Catch every exception type: the documented contract is "one
                    # bad variant logs + moves on, never takes out sibling
                    # variants". The previous narrow tuple (OSError, ValueError,
                    # RuntimeError, TimeoutError) let off-tuple exceptions
                    # (KeyError, sqlalchemy.exc.*, custom domain errors from
                    # _process_variant transitively) escape into gather() and
                    # cancel every sibling style render via task cancellation.
                    logger.error(
                        "Variant %s for job %s failed (%s): %s",
                        style,
                        job.job_id,
                        type(exc).__name__,
                        exc,
                    )
                    failures.append(exc)

            # return_exceptions=True is belt-and-braces against BaseException
            # slipping past the inner Exception absorb (e.g. CancelledError).
            results = await asyncio.gather(
                *[_run(s) for s in styles],
                return_exceptions=True,
            )
            for style, result in zip(styles, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "Variant %s for job %s raised past absorb (%s): %s",
                        style,
                        job.job_id,
                        type(result).__name__,
                        result,
                    )
                    failures.append(result)
        else:
            for style in styles:
                try:
                    failure = await self._process_variant(
                        db,
                        job,
                        html,
                        style,
                        resolved_ctx,
                        settings,
                        generate_screenshots,
                        render_options,
                        budget=budget,
                        quality_warnings=quality_warnings,
                    )
                    if failure is not None:
                        failures.append(failure)
                except Exception as exc:  # noqa: BLE001 — per-variant absorb, see comment
                    # Mirror the parallel branch above: narrow tuples here too
                    # would let off-tuple exceptions abort the whole sequential
                    # walk instead of moving on to the next variant.
                    logger.error(
                        "Variant %s for job %s failed (%s): %s",
                        style,
                        job.job_id,
                        type(exc).__name__,
                        exc,
                    )
                    failures.append(exc)
        return failures

    async def _process_variant(
        self,
        db: AsyncSession,
        job: RedesignJob,
        html: str,
        style: str,
        resolved_ctx: ResolvedContext,
        settings: Settings,
        generate_screenshots: bool = True,
        render_options: RenderOptions | None = None,
        *,
        budget: GenerationBudget | None = None,
        quality_warnings: dict[str, list[str]] | None = None,
    ) -> BaseException | None:
        """Process a single variant. Returns the exception that ended it, or None.

        The failure is ABSORBED here (see the recorder at the bottom) — nothing
        is raised out — so returning it is the only way the caller can learn
        whether the variant died of something transient. ``None`` means either
        success or a cancellation check, neither of which is a failure.

        DB transactions are scoped narrowly around the long LLM generation:
        the variant row plus any custom-style lookup happen in a short tx that
        is committed (releasing the connection) *before* generation, the
        generation runs with no connection held, and the result is persisted in
        a fresh tx. Holding a transaction open across the multi-minute generate
        call would let the backend be reaped and the final write hit a dead
        socket. The pre-generation commit is also what makes a variant row
        durable in the gather path — each variant has its own session, so
        without it the writes would roll back on session close.
        """
        variant = await _pkg().JobManager.create_variant(
            db,
            job_id=job.job_id,
            style=style,
        )
        variant_id = variant.variant_id

        # Cancellation + custom-style lookup share the variant-creation tx.
        if await self._check_cancelled(db, job.job_id):
            await db.rollback()
            return None

        style_description: str | None = None
        if style not in ("modern", "minimalist", "bold", "elegant"):
            custom_style = await StyleManager.get_style_by_slug(
                db,
                job.tenant_id,
                style,
            )
            if custom_style:
                style_description = StyleManager.resolve_style_description(custom_style)

        # Persist the pending variant and release the connection before the
        # long generation.
        await db.commit()

        try:
            llm_result = await self._html_gen.generate_page_chunked(
                url=job.url,
                html=html,
                page_type=job.page_type,
                style=style,
                findings=job.critique_findings,
                brand=job.brand_info,
                site_analysis=job.site_analysis,
                style_description=style_description,
                context=resolved_ctx,
                budget=budget,
            )
            warnings = list(getattr(llm_result, "quality_warnings", None) or [])
            if warnings and quality_warnings is not None:
                quality_warnings[style] = warnings

            html_path = await self._storage.store_html(
                job.job_id,
                llm_result.text,
                f"variant_{style}.html",
            )

            screenshot_paths: dict[str, str] | None = None
            if generate_screenshots and settings.screenshot.enabled:
                opts = render_options or RenderOptions()
                viewports = (
                    list(opts.viewports) if opts.viewports else settings.screenshot.viewports
                )
                job_dir = self._storage.get_job_dir(job.job_id)
                screenshots_dir = job_dir / f"screenshots_{style}"
                screenshot_paths = await _pkg().render_from_html_string(
                    llm_result.text,
                    screenshots_dir,
                    viewports,
                    timeout_ms=settings.screenshot.timeout_ms,
                    full_page=opts.full_page,
                )
                # Lift the variant PNGs into shared storage (no-op on local).
                screenshot_paths = await self._storage.persist_paths(screenshot_paths)

            # Persist the result in a fresh tx (the connection was released for
            # the duration of generation above).
            await _pkg().JobManager.mark_variant_completed(
                db,
                variant_id,
                redesigned_html_path=html_path,
                screenshot_paths=screenshot_paths,
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                estimated_cost=estimate_cost(llm_result.input_tokens, llm_result.output_tokens),
            )

            # Dispatch variant.completed webhook
            await self._dispatch_webhook(
                db,
                job,
                "variant.completed",
                variant_id=variant_id,
            )
            await db.commit()
            return None

        except Exception as exc:  # noqa: BLE001 — per-variant failure recorder, see comment
            # Catch every exception type, matching the absorbing handlers in
            # ``_process_variants`` above (:77, :121) that were already widened —
            # this one, the variant's own failure recorder, was missed.
            #
            # It is an ABSORB, not a record-and-re-raise: nothing is raised out
            # of here, so the caller sees a variant that merely "finished". That
            # makes ``mark_variant_failed`` the only durable trace of the
            # failure, so the handler must be unmissable. The previous narrow
            # tuple ``(OSError, ValueError, RuntimeError, TimeoutError)`` could
            # not see what the guarded block really raises: ``httpx.HTTPError``
            # from ``generate_page_chunked`` (worker_management),
            # ``playwright.async_api.Error``/``TimeoutError`` from
            # ``render_from_html_string``, ``redis.exceptions.RedisError``,
            # ``sqlalchemy.exc.*`` from the storage/commit work. All derive from
            # ``Exception``, not ``OSError``; Playwright's and redis-py's
            # ``TimeoutError`` only share the builtin's name. Those escaped
            # uncaught, so the variant row stayed ``pending`` forever and the
            # rollback that clears the half-open transaction never ran.
            #
            # ``CancelledError`` is a ``BaseException`` and still propagates, so
            # a shutdown drain is not mis-recorded as a variant failure.
            logger.error("Variant %s (style=%s) failed: %s", variant_id, style, exc)
            # Clear any half-open tx from the failed generation before writing
            # the failure state in a clean tx.
            await db.rollback()
            try:
                # `describe_failure`, not `str(exc)`: a bare `httpx.ReadTimeout`
                # stringifies to '', which wrote a variant row whose only record
                # of the failure was an empty string — and rolled up into the
                # parent as "all 2 redesign variants failed: modern: ; bold: ".
                await _pkg().JobManager.mark_variant_failed(db, variant_id, describe_failure(exc))
                await db.commit()
            except Exception:
                # `_process_variants`' own `_run`/sequential wrappers already
                # catch anything that escapes `_process_variant`
                # (`return_exceptions=True` on the gather is belt-and-braces on
                # top of that), so a `mark_variant_failed` DB error was already
                # contained -- it never crashed the fan-out or a sibling
                # variant. What it DID do: replace `exc` (e.g. the real
                # `httpx.ReadTimeout`) with the DB error in the caller's
                # `failures` list, which is what the parent job reads to decide
                # transient-vs-terminal (see this method's docstring). A
                # persistence hiccup would misclassify an upstream timeout as a
                # hard failure and deny it the retry it should get.
                #
                # Cost: if this write genuinely failed, this variant's row is
                # left `pending` (the pre-generation commit made the row itself
                # durable, but not a terminal status) instead of `failed`.
                # Nothing reconciles that row; only the parent's own
                # transient/terminal verdict is protected here.
                logger.exception(
                    "Failed to persist variant %s failure state (original error: %s)",
                    variant_id,
                    exc,
                )
            # Returned, not raised: the absorb contract is unchanged, but the
            # parent needs the exception TYPE to decide its own verdict.
            return exc
