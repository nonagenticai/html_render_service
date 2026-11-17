"""Phase methods 1-5 for ``RedesignService.process_job`` (HTML resolve, theme, generation)."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable
from html import escape as html_escape
from typing import Any

from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.net import SSRFError

from src.config import Settings
from src.database.models import RedesignJob
from src.llm_credentials import LLMConfigLike
from src.services.accessibility_theme import build_accessibility_theme_plan
from src.services.component_extractor import ComponentNotFoundError
from src.services.concept_image_generator import ConceptImageGenerator
from src.services.context_resolver import ResolvedContext
from src.services.design_builder.helpers import capture_failure_types
from src.services.design_builder.types import (
    RenderOptions,
    _GenerationResult,
    _ThemeDecision,
)
from src.services.hero_image import hero_data_uri
from src.services.html_gen.section_run import BudgetExhausted, GenerationBudget, SectionRun
from src.services.job_manager import get_claim_owner
from src.services.navigation_guard import PageFetchError
from src.services.style_manager import StyleManager
from src.services.upstream_failure import is_transient_upstream

logger = logging.getLogger(__name__)

# Hero image object extension by the downloaded content type.
_HERO_EXTENSIONS: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def _pkg() -> Any:
    from src.services import design_builder as _p

    return _p


class RedesignPhasesMixin:
    """Mixin: HTML resolve, theme resolution, HTML generation phases."""

    # Cross-mixin attributes/methods.
    _cache: Any
    _concept_gen: ConceptImageGenerator
    _component_extractor: Any
    _html_gen: Any
    _storage: Any
    _fetch_page_html: Callable[..., Awaitable[str]]
    _process_variants: Callable[..., Awaitable[list[BaseException]]]
    _dispatch_webhook: Callable[..., Awaitable[None]]

    async def _phase_resolve_html(self, job: RedesignJob) -> str:
        """Phase 1: return job HTML, fetching/caching as necessary.

        scope=landing with no source url/html generates from scratch — return
        an empty string rather than attempting to fetch an empty URL.

        Two-tier cache, and the ORDER is load bearing. A known-good body is
        returned before the failure keyspace is consulted, so a transient 503 can
        never shadow HTML we already have. Only when there is no good body do we
        ask whether this URL failed recently — if it did, fail immediately with
        the recorded reason instead of paying another 30s navigation to learn the
        same thing. A fetch that raises never reaches ``cache_html``, so an error
        document can never be pinned under the good key for its 3600s TTL; it is
        recorded in the short-TTL failure keyspace instead.
        """
        html = job.html_content
        if not html and not job.url:
            return ""
        if not html and self._cache:
            html = await self._cache.get_cached_html(job.url)
        if not html:
            if self._cache:
                recorded = await self._cache.get_failed_fetch(job.url)
                if recorded:
                    raise PageFetchError(
                        url=job.url,
                        reason="cached_failure",
                        message=f"recent fetch of {job.url} failed: {recorded}",
                    )
            try:
                html = await self._fetch_page_html(job.url)
            except PageFetchError as exc:
                if self._cache:
                    await self._cache.cache_failed_fetch(job.url, str(exc))
                raise
            if self._cache:
                await self._cache.cache_html(job.url, html)
        return html

    async def _phase_resolve_theme(
        self,
        db: AsyncSession,
        tenant_id: str,
        job: RedesignJob,
        resolved_ctx: ResolvedContext,
        *,
        accessibility_first_auto_theme: bool,
    ) -> _ThemeDecision:
        """Phase 2/3: resolve custom style description, optionally auto-theme for a11y."""
        style_description = await self._lookup_style_description(db, tenant_id, job.style)

        if not accessibility_first_auto_theme:
            return _ThemeDecision(
                style=job.style,
                style_description=style_description,
                brand_info=job.brand_info,
                context=resolved_ctx,
                auto_theme_metadata=None,
            )

        plan = build_accessibility_theme_plan(
            style=job.style,
            style_description=style_description,
            brand_info=job.brand_info,
            context=resolved_ctx,
            critique_findings=job.critique_findings,
        )
        return _ThemeDecision(
            style=plan.style,
            style_description=plan.style_description,
            brand_info=plan.brand_info,
            context=plan.context,
            auto_theme_metadata=plan.to_dict(),
        )

    async def _lookup_style_description(
        self, db: AsyncSession, tenant_id: str, style: str
    ) -> str | None:
        """Return a custom style description (cache-first), or None for built-in styles.

        ``style`` is a SLUG here (``RedesignJob.style`` holds the style token the
        caller submitted), and the cache is keyed by the same ``(tenant_id, slug)``
        pair the DB lookup below uses. Before that, the read looked up the slug
        while the write stored under ``custom_style.style_id`` — different keys,
        so the cache never once hit and every job paid for the DB round-trip. The
        tenant half of the key is load-bearing: a slug is unique only WITHIN a
        tenant, so a slug-only key would serve one tenant's brand description to
        another.
        """
        if style in ("modern", "minimalist", "bold", "elegant"):
            return None
        style_description: str | None = None
        if self._cache:
            style_description = await self._cache.get_cached_style_description(tenant_id, style)
        if style_description is not None:
            return style_description

        custom_style = await StyleManager.get_style_by_slug(db, tenant_id, style)
        if not custom_style:
            # Returning None means "built-in style" to the caller, so an
            # UNRESOLVABLE token silently downgrades the job to a built-in look:
            # right status, wrong design, no error anywhere. Reachable whenever a
            # stored token stops matching a row — the style was deleted or
            # deactivated after the job was queued, the caller sent a typo, or
            # migration 0009 re-slugged a duplicate out from under a queued job
            # (see that migration's DUPLICATE DATA note). Make it loud.
            logger.warning(
                "Style token '%s' did not resolve to a custom style for tenant=%s "
                "(deleted, deactivated, renamed, or never existed) — rendering with "
                "built-in styling instead",
                style,
                tenant_id,
            )
            return None
        style_description = StyleManager.resolve_style_description(custom_style)
        if self._cache and style_description:
            await self._cache.cache_style_description(
                custom_style.tenant_id, custom_style.slug, style_description
            )
        return style_description

    async def _capture_variant_original(
        self,
        job: RedesignJob,
        settings: Settings,
        render_options: RenderOptions | None,
        *,
        generate_screenshots: bool,
    ) -> dict[str, str] | None:
        """Capture the source page's 'before' screenshot for the variant path.

        The variant fan-out short-circuits ``_phase_screenshots``, so without this
        the redesign result carries NO original/before image at all — the
        website_review before act has nothing to show, and (with the desktop-only
        render the orchestrator now requests) a single transient variant-generation
        flake drops the result to one fetchable visual, failing the
        ``distinct_count_gte 2`` E2E assertion. The original capture is
        deterministic (just screenshots the live URL, no LLM), honors
        ``render_options`` viewports/full_page, and gives one before + N after
        visuals so a single flaked variant still yields >= 2 distinct images.
        Soft: any failure returns ``None`` (a missing before image must not fail
        an otherwise-good variant redesign).
        """
        opts = render_options or RenderOptions()
        # No source URL (greenfield design) → no original to capture. Page.goto("")
        # is now caught below (see ``capture_failure_types``), but there is still
        # nothing to screenshot, so skipping is the correct — and cheaper — answer.
        if (
            not generate_screenshots
            or not settings.screenshot.enabled
            or not opts.capture_original
            or not (job.url or "").strip()
        ):
            return None
        viewports = list(opts.viewports) if opts.viewports else settings.screenshot.viewports
        try:
            orig_dir = self._storage.get_job_dir(job.job_id) / "original_screenshots"
            original_paths = await _pkg().capture_url_screenshots(
                job.url,
                orig_dir,
                viewports,
                timeout_ms=settings.screenshot.timeout_ms,
                full_page=opts.full_page,
            )
            return await self._storage.persist_paths(original_paths)
        except SSRFError as exc:
            # Must precede the generic tuple: SSRFError subclasses ValueError,
            # so the warning below would otherwise swallow a blocked internal
            # target as an indistinguishable "capture failed". Same ordering and
            # rationale as `_capture_original_screenshots` in finalize_phases —
            # blocking is a security decision and is logged loudly. Reachable
            # from two raisers now: the up-front URL validation, and the
            # post-navigation cross-host redirect re-validation.
            logger.error(
                "Refusing variant original capture for job %s — url %r is not a public target: %s",
                job.job_id,
                job.url,
                exc,
            )
            return None
        except capture_failure_types() as exc:
            # Includes Playwright's own Error/TimeoutError: a tenant page that is
            # slow or dead must cost this job its optional before image, not the
            # whole (paid) redesign. See ``capture_failure_types``.
            logger.warning(
                "Failed to capture variant original screenshots (%s): %s",
                type(exc).__name__,
                exc,
            )
            return None

    async def _phase_process_variants_and_finalize(
        self,
        db: AsyncSession,
        job: RedesignJob,
        job_id: str,
        tenant_id: str,
        html: str,
        resolved_ctx: ResolvedContext,
        settings: Settings,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None,
        generate_screenshots: bool,
        render_options: RenderOptions | None = None,
        budget: GenerationBudget | None = None,
    ) -> None:
        """Phase 4: A/B variant fan-out and completion.

        The variants share the JOB's generation ``budget``; the quality compromises each
        made to fit it are persisted as ``html_validation_report.quality_warnings``
        (style -> warnings).
        """
        quality_warnings: dict[str, list[str]] = {}
        variant_failures = await self._process_variants(
            session_factory=session_factory,
            db=db,
            job=job,
            html=html,
            resolved_ctx=resolved_ctx,
            settings=settings,
            generate_screenshots=generate_screenshots,
            render_options=render_options,
            budget=budget,
            quality_warnings=quality_warnings,
        )
        variants_count = len(job.styles or [])

        variants = await _pkg().JobManager.get_variants(db, job_id, tenant_id=job.tenant_id)
        first_completed = next((v for v in variants if v.status == "completed"), None)

        # Fail loud, don't complete-empty: if every variant failed to generate,
        # the redesign produced no usable HTML/screenshots. Marking the job
        # `completed` here would hand downstream consumers an empty `visuals`
        # list (a silent bad video / failed website_review assertion) with no
        # cause attached. Surface it as a terminal failure with the per-variant
        # reasons instead.
        if variants_count > 1 and first_completed is None:
            reasons = (
                "; ".join(f"{v.style}: {v.error_message or v.status}" for v in variants)
                or "no variant rows were produced"
            )
            error = f"all {variants_count} redesign variants failed: {reasons}"
            logger.error("Variant job %s failed: %s", job_id, error)

            # Same transient/terminal split as the single-style path
            # (``RedesignFinalizeMixin._phase_finalize_failed``). Without it the
            # original defect survived intact for ``styles > 1``: every variant
            # dying of an upstream ``ReadTimeout`` still stamped the parent
            # terminally ``failed``, which the orchestrator (``retry_on=()``)
            # propagates immediately.
            #
            # The bar is deliberately unanimous AND complete: every style must
            # have produced a failure exception, and every one of them must be
            # transient. ``len(...) == variants_count`` is load-bearing, not
            # decoration — ``all()`` over an empty list is True, and a style that
            # never reported a failure (a cancellation check, a variant row that
            # never got written) is not evidence of a transient one. One
            # permanent child failure is enough to keep the whole job terminal:
            # re-running would pay for the same permanent failure again.
            all_children_transient = len(variant_failures) == variants_count and all(
                is_transient_upstream(exc) for exc in variant_failures
            )
            worker_id = get_claim_owner()
            if all_children_transient and await _pkg().JobManager.mark_retryable(
                db, job_id, error, worker_id=worker_id
            ):
                # No `job.failed` webhook: the job has not failed, it is going to
                # run again — telling the tenant otherwise and then completing
                # would be a lie in both directions.
                await db.commit()
                return
            # The result is USED, not discarded. Two ways this write is refused,
            # and neither may reach the tenant:
            #
            # * The job was CANCELLED mid fan-out. ``_process_variant`` returns
            #   early on its cancellation check without writing a variant row,
            #   so "zero variants completed" is exactly what a cancelled job
            #   looks like from here — and stamping ``failed`` over ``cancelled``
            #   turned the tenant's own stop request into a failure report.
            # * A sibling worker holds the claim and is still rendering; its
            #   ``job.completed`` would arrive right after our ``job.failed``.
            if not await _pkg().JobManager.mark_failed(db, job_id, error, worker_id=worker_id):
                logger.warning(
                    "Variant job %s failure was fenced off — not dispatching job.failed. "
                    "The job was cancelled or another worker owns its outcome.",
                    job_id,
                )
                await db.commit()
                return
            await self._dispatch_webhook(db, job, "job.failed")
            await db.commit()
            return

        # Capture the source page's 'before' screenshot (the variant fan-out
        # otherwise produces no original image; see _capture_variant_original).
        original_screenshot_paths = await self._capture_variant_original(
            job, settings, render_options, generate_screenshots=generate_screenshots
        )

        completed = await _pkg().JobManager.mark_completed(
            db,
            job_id,
            redesigned_html_path=first_completed.redesigned_html_path if first_completed else None,
            screenshot_paths=first_completed.screenshot_paths if first_completed else None,
            original_screenshot_paths=original_screenshot_paths,
            pages_count=1,
            html_validation_report=(
                {"quality_warnings": quality_warnings} if quality_warnings else None
            ),
            worker_id=get_claim_owner(),
        )
        if not completed:
            # Billing and the webhook are gated on the fenced write landing, not
            # merely attempted. `record_usage(variants_generated=N)` is money.
            logger.warning(
                "Variant job %s completion was fenced off — not billing %d variants and "
                "not dispatching job.completed. Cancelled, or owned by another worker.",
                job_id,
                variants_count,
            )
            await db.commit()
            return
        await _pkg().JobManager.record_usage(
            db,
            tenant_id=tenant_id,
            pages_redesigned=1,
            variants_generated=variants_count,
        )
        await self._dispatch_webhook(db, job, "job.completed")

        await db.commit()

    async def _phase_generate_html(
        self,
        job: RedesignJob,
        job_id: str,
        html: str,
        theme: _ThemeDecision,
        resolved_ctx: ResolvedContext,
        *,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> _GenerationResult:
        """Phase 5: dispatch by scope to component / full_page / section path.

        Scope vocabulary (any web design, not just redesign):
        - ``hero``: fast single hero-section treatment — the promo fast path.
          Auto-targets the hero component (no explicit ``component_target``
          needed) and reuses the component generation path (no full-page
          chunking, so it returns in well under the full-page latency).
        - ``component``: a single explicitly-targeted component.
        - ``full_page`` / ``redesign`` / ``landing``: whole-page chunked
          generation (``redesign`` is the critique-driven variant; ``landing``
          generates from scratch when no source url is given).
        - ``section`` / ``sections``: per-section generation when the site
          analysis carries ``content_sections``.

        When ``job.mode == 'assembly'`` all of the above are bypassed for the
        computed-assembly path (LLM selects catalog blocks; the assembler renders
        a self-contained page deterministically).
        """
        if getattr(job, "mode", "freeform") == "assembly":
            return await self._generate_assembly(
                job, job_id, theme, llm_config=llm_config, budget=budget
            )
        if job.scope == "hero":
            if not job.component_target:
                job.component_target = {"component_type": "hero"}
            return await self._generate_component(
                job, job_id, html, theme, resolved_ctx, llm_config=llm_config, budget=budget
            )
        if job.scope == "component" and job.component_target:
            return await self._generate_component(
                job, job_id, html, theme, resolved_ctx, llm_config=llm_config, budget=budget
            )
        if job.scope in ("full_page", "redesign", "landing"):
            return await self._generate_full_page(
                job, job_id, html, theme, resolved_ctx, llm_config=llm_config, budget=budget
            )
        return await self._generate_section_or_full(
            job, job_id, html, theme, resolved_ctx, llm_config=llm_config, budget=budget
        )

    async def _generate_assembly(
        self,
        job: RedesignJob,
        job_id: str,
        theme: _ThemeDecision,
        *,
        llm_config: LLMConfigLike | None,
        refine_findings: str | None = None,
        budget: GenerationBudget | None = None,
        hero_image_url: str | None = None,
        hero_image_check: dict[str, object] | None = None,
    ) -> _GenerationResult:
        """Computed-assembly path: LLM plans blocks; the assembler renders HTML.

        The brief lives in ``user_context.description`` (the web_design stage puts
        it there); brand/theme come from the resolved theme decision. Output is a
        single self-contained document stored as ``redesign.html``.
        ``refine_findings`` feed a prior verification failure back into the plan.

        ``budget`` is the JOB's generation budget — the SAME object the section and
        full-page paths spend. The plan draw runs under it, so it is refused when the
        job's deadline is reached and cut if the deadline passes in flight. A refine
        re-draw (``refine_findings`` present) is charged as an EXTRA call, which is
        what makes ``max_extra_llm_calls`` bind on the verify/refine loop.

        A ``SectionRun`` wraps that budget for the same reason the section path uses
        one: the split planner can run out of deadline mid-page and ship the remaining
        blocks on their defaults. That is a DEGRADED delivery, so it is collected from
        ``run.degradations`` into ``quality_warnings`` here and persisted by
        ``_attach_quality_warnings`` — not left as a pod log line on a job that reports
        ``succeeded``.

        The hero image is generated here, once per job (``_assembly_hero_image``):
        a refine re-plan passes the first attempt's ``hero_image_url`` back in so
        the paid generation is not repeated.
        """
        from src.services.catalog import assemble_page

        user_context = job.user_context or {}
        description = str(
            user_context.get("description") or job.feedback_text or job.page_type or ""
        ).strip()
        audience_list = user_context.get("target_audience")
        audience = (
            audience_list[0] if isinstance(audience_list, list) and audience_list else "general"
        )
        interactivity = getattr(job, "interactivity", "static") or "static"

        # This render's own state — never the shared generator's (see SectionRun).
        # ``run.budget`` IS the job's budget object, not a copy: the variants and the
        # refine loop must keep spending the same one.
        run = SectionRun(budget=budget or GenerationBudget.from_settings())
        plan, input_tokens, output_tokens = await self._html_gen.generate_assembly_plan(
            description=description,
            brand=theme.brand_info,
            page_type=job.page_type,
            audience=audience,
            interactivity=interactivity,
            refine_findings=refine_findings,
            llm_config=llm_config,
            budget=run.budget,
            extra=refine_findings is not None,
            run=run,
        )
        hero_image_url, hero_image_check = await self._assembly_hero_image(
            job_id,
            plan,
            theme,
            description,
            run=run,
            reuse=hero_image_url,
            reuse_check=hero_image_check,
        )
        assembled = assemble_page(
            plan,
            brand=theme.brand_info,
            # No ``title``: the assembler derives it from the ONE canonical brand
            # (+ hero headline). Falling back to ``job.page_type`` here shipped
            # ``<title>landing</title>`` on live run 01a156a6.
            interactive=interactivity == "interactive",
            embeds=list(job.embeds or []),
            brief=description,
            own_images=[hero_image_url] if hero_image_url else [],
        )
        redesigned_html_path = await self._storage.store_html(
            job_id, assembled.html, "redesign.html"
        )
        logger.info(
            "Assembly job %s composed %d blocks (%d embeds)",
            job_id,
            len(assembled.blocks_used),
            assembled.embeds_injected,
        )
        return _GenerationResult(
            redesigned_html=assembled.html,
            redesigned_html_path=redesigned_html_path,
            section_html_paths=None,
            component_html_path=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quality_warnings=list(run.degradations),
            hero_image_url=hero_image_url,
            hero_image_check=hero_image_check,
        )

    async def _assembly_hero_image(
        self,
        job_id: str,
        plan: list[dict],
        theme: _ThemeDecision,
        description: str,
        *,
        run: SectionRun,
        reuse: str | None = None,
        reuse_check: dict[str, object] | None = None,
    ) -> tuple[str | None, dict[str, object] | None]:
        """Generate the page's ONE hero image and set it on the plan's hero block.

        Owner decision (live web_design_v2 run): a hero is a real image, not a
        fallback card. The planner is told to leave ``image`` empty, so this
        overwrites whatever it wrote with our own generated image:

        1. ``ConceptImageGenerator.generate_hero`` — the existing ai_content_generation
           client, so the call carries the job's bound tenant + cell token exactly
           like the concept-image phase (it is a metered call);
        2. the original is kept in THIS service's bucket for provenance
           (``<job_id>/hero/original.<ext>``) — best effort, and never referenced by
           the page;
        3. ``plan.hero.image`` = the image downscaled/re-encoded (<=1600px, JPEG,
           <=300KB) as a ``data:`` URI, so the page stays self-contained: a
           ``minio.test`` URL is http, tailnet-only and expiring, and breaks
           under the served page's ``img-src https: data:`` CSP. The caller passes
           that exact value to ``assemble_page(own_images=...)``; it is the only
           ``data:`` image ``integrity.enforce_images`` keeps.

        Any failure leaves ``image`` empty, so the hero renders its brand-colour
        gradient. It NEVER fails the page; the reason is recorded as a quality
        warning on the job. Returns ``(url, check)``: the URL used, ``""`` when
        generation was attempted and failed (gradient), ``None`` when the plan has no
        hero; ``check`` is the visible-text check record (see
        :meth:`_generate_hero_image`), or ``None`` when no check ran.
        """
        hero = next((e for e in plan if isinstance(e, dict) and e.get("type") == "hero"), None)
        if hero is None:
            return reuse, reuse_check
        content = hero.get("content")
        if not isinstance(content, dict):
            content = {}
            hero["content"] = content
        content["image"] = ""  # never an LLM-written URL; ours or the gradient
        if reuse is not None:
            # A refine re-plan: reuse the first attempt's outcome, success ("<url>")
            # or failure (""), so a paid generation is never repeated in one job.
            content["image"] = reuse
            warning = (reuse_check or {}).get("warning")
            if warning:
                run.degrade(str(warning))
            return reuse, reuse_check
        check: dict[str, object] = {}
        try:
            url = await self._generate_hero_image(job_id, theme, description, run=run, check=check)
        except Exception as exc:  # noqa: BLE001 — a hero image must never fail the page
            logger.warning("Job %s hero image generation failed: %s", job_id, exc)
            run.degrade(f"hero image not generated ({type(exc).__name__}); gradient used")
            return "", check or None
        if not url:
            if not check.get("warning"):
                run.degrade("hero image not generated; gradient used")
            return "", check or None
        content["image"] = url
        return url, check or None

    async def _generate_hero_image(
        self,
        job_id: str,
        theme: _ThemeDecision,
        description: str,
        *,
        run: SectionRun,
        check: dict[str, object],
    ) -> str | None:
        """Generate -> text check -> (one abstract retry) -> inline data: URI (or None).

        Owner decision (live run 01a159b1, a garbled shop sign "Cafiiti Boker"): the
        hero is checked for visible text by a vision model (``HeroTextChecker``). Cap
        per job: 1 check + at most 1 regeneration + 1 re-check.

        * clean -> use it (``outcome=clean``);
        * text -> regenerate ONCE with a fresh seed and an abstract close-up prompt
          (product / material / hands, no storefront / signage / interior), re-check:
          clean -> ``regenerated``; text again -> ``fell_back`` (gradient);
        * the check itself failing (timeout / transport / unparseable / budget) counts
          as no-text — the page is never blocked — and is recorded as a quality
          warning (``check_failed`` / ``regenerated_unchecked``).

        ``check`` is filled in place with the outcome and persisted on the job as
        ``html_validation_report.hero_image_check``; degraded outcomes are also
        ``run.degrade``d into ``quality_warnings``.
        """
        data = await self._draw_hero(job_id, theme, description, name="original")
        if data is None:
            return None
        checker = getattr(self, "_hero_text_checker", None)
        if checker is None:
            return hero_data_uri(data)

        first = await checker.check(data, budget=run.budget)
        check["checks"] = 1
        if first.error:
            return self._hero_note(
                run, check, data, "check_failed",
                f"hero text check failed ({first.error}); image used unchecked",
            )  # fmt: skip
        if not first.has_text:
            check["outcome"] = "clean"
            return hero_data_uri(data)

        seen = first.text_seen[:80]
        check["text_seen"] = seen
        seed = secrets.randbelow(2**32)
        check["regenerated"] = True
        check["retry_seed"] = seed
        logger.info("Job %s hero image shows text (%r); regenerating once", job_id, seen)
        try:
            retry = await self._draw_hero(
                job_id, theme, description, name="retry", abstract=True, seed=seed
            )
        except Exception as exc:  # noqa: BLE001 — the retry must never fail the page
            logger.warning("Job %s hero image retry failed: %s", job_id, exc)
            retry = None
        if retry is None:
            return self._hero_note(
                run, check, None, "fell_back",
                f"hero image showed text ({seen!r}); regeneration failed; gradient used",
            )  # fmt: skip

        second = await checker.check(retry, budget=run.budget)
        check["checks"] = 2
        if second.error:
            return self._hero_note(
                run, check, retry, "regenerated_unchecked",
                f"hero image showed text ({seen!r}); regenerated once, re-check failed "
                f"({second.error}); retry used unchecked",
            )  # fmt: skip
        if second.has_text:
            check["retry_text_seen"] = second.text_seen[:80]
            return self._hero_note(
                run, check, None, "fell_back",
                f"hero image showed text twice ({seen!r}, {second.text_seen[:80]!r}); "
                "gradient used",
            )  # fmt: skip
        check["outcome"] = "regenerated"
        return hero_data_uri(retry)

    @staticmethod
    def _hero_note(
        run: SectionRun,
        check: dict[str, object],
        data: bytes | None,
        outcome: str,
        warning: str,
    ) -> str | None:
        """Record a DEGRADED hero outcome; return the image to use (``None`` = gradient)."""
        check["outcome"] = outcome
        check["warning"] = warning
        run.degrade(warning)
        return hero_data_uri(data) if data is not None else None

    async def _draw_hero(
        self,
        job_id: str,
        theme: _ThemeDecision,
        description: str,
        *,
        name: str,
        abstract: bool = False,
        seed: int | None = None,
    ) -> bytes | None:
        """One paid hero generation -> download -> keep a provenance copy -> raw bytes."""
        brand = theme.brand_info or {}
        result = await self._concept_gen.generate_hero(
            brand_name=brand.get("name"),
            brief=description,
            primary_color=str(brand.get("primary_color") or "#2563EB"),
            style=theme.style or "modern",
            abstract=abstract,
            seed=seed,
        )
        if not result.assets:
            logger.warning(
                "Job %s hero image: no asset (degraded=%s, reason=%s)",
                job_id,
                result.degraded,
                result.reason,
            )
            return None
        fetched = await self._concept_gen.download_asset(result.assets[0])
        if fetched is None:
            return None
        data, content_type = fetched
        ext = _HERO_EXTENSIONS.get(content_type, "png")
        try:
            await self._storage.store_bytes(job_id, data, f"hero/{name}.{ext}")
        except Exception as exc:  # noqa: BLE001 — provenance copy only; the page does not need it
            logger.warning("Job %s hero %s not stored: %s", job_id, name, exc)
        return data

    async def _phase_verify_and_refine(
        self,
        job: RedesignJob,
        job_id: str,
        gen_result: _GenerationResult,
        theme: _ThemeDecision,
        tokens: Any,
        *,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> tuple[_GenerationResult, dict | None]:
        """Assembly gate: verify the design; refine on hard failure (bounded).

        Returns ``(final_gen_result, report_dict)``. ``report_dict`` is None when
        verification is skipped (non-assembly or disabled). The caller persists the
        report and fails the job loud when the final status is ``failed`` — a
        design only completes if it passes.
        """
        from src.config import get_settings
        from src.services.verification import parse_axe_summary, verify_html

        if getattr(job, "mode", "freeform") != "assembly":
            return gen_result, None
        cfg = get_settings().catalog
        if not cfg.verify_enabled:
            return gen_result, None

        report = await verify_html(
            gen_result.redesigned_html, a11y_max_serious=cfg.verify_a11y_max_serious
        )
        attempts = 0
        while report.status == "failed" and attempts < cfg.verify_max_refine:
            attempts += 1
            logger.info(
                "Assembly job %s failed verification (attempt %d/%d): %s",
                job_id,
                attempts,
                cfg.verify_max_refine,
                report.summary(),
            )
            try:
                gen_result = await self._generate_assembly(
                    job,
                    job_id,
                    theme,
                    llm_config=llm_config,
                    refine_findings=parse_axe_summary(report),
                    budget=budget,
                    hero_image_url=gen_result.hero_image_url,
                    hero_image_check=gen_result.hero_image_check,
                )
            except BudgetExhausted as exc:
                # The job's deadline / extra-call cap, not this design. Stop refining
                # and hand back the last report: still ``failed``, so the caller fails
                # the job loud with findings rather than shipping an unverified page.
                logger.warning(
                    "Assembly job %s stopped refining after attempt %d: %s",
                    job_id,
                    attempts,
                    exc,
                )
                attempts -= 1
                break
            tokens.add(gen_result.input_tokens, gen_result.output_tokens)
            report = await verify_html(
                gen_result.redesigned_html, a11y_max_serious=cfg.verify_a11y_max_serious
            )

        report_dict = report.to_dict()
        report_dict["refine_attempts"] = attempts
        logger.info(
            "Assembly job %s verification %s after %d refine attempt(s)",
            job_id,
            report.status,
            attempts,
        )
        return gen_result, report_dict

    async def _generate_component(
        self,
        job: RedesignJob,
        job_id: str,
        html: str,
        theme: _ThemeDecision,
        resolved_ctx: ResolvedContext,
        *,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> _GenerationResult:
        component_target = job.component_target or {}
        component_type = component_target.get("component_type", "unknown")
        component_selector = component_target.get("component_selector")

        missing_hero = False
        context = theme.context
        try:
            component_html, _component_id = self._component_extractor.extract(
                html, component_type, component_selector
            )
        except ComponentNotFoundError:
            if job.scope != "hero":
                raise
            # A hero design may create a new section. Component replacement still
            # requires its explicit target to exist.
            content = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:12000]
            if not content:
                raise
            missing_hero = True
            # The source content travels INSIDE the placeholder section, which is the
            # part of the component prompt the model lays out ("Current HTML"). It must
            # never be folded into ``context``: that is the job's ResolvedContext (palette,
            # audience, fonts), and replacing it with a string crashed every hero job on
            # a page with no hero (``'str' object has no attribute 'target_audience'``).
            component_html = (
                "<section class='hero'><!-- No hero exists on the source page. Create a"
                " new hero from this source content only. --><p>"
                f"{html_escape(content)}</p></section>"
            )

        llm_result = await self._html_gen.generate_component(
            url=job.url,
            component_html=component_html,
            component_type=component_type,
            page_type=job.page_type,
            style=theme.style,
            findings=job.critique_findings,
            brand=theme.brand_info,
            site_analysis=job.site_analysis,
            style_description=theme.style_description,
            context=context,
            llm_config=llm_config,
            budget=budget,
        )
        component_html_path = await self._storage.store_html(
            job_id, llm_result.text, "component.html"
        )
        if missing_hero:
            composite = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{margin:0}</style></head><body>" + llm_result.text + "</body></html>"
            )
        else:
            composite = self._component_extractor.reinject(
                html, component_type, llm_result.text, component_selector
            )
        redesigned_html_path = await self._storage.store_html(job_id, composite, "redesign.html")
        return _GenerationResult(
            redesigned_html=composite,
            redesigned_html_path=redesigned_html_path,
            section_html_paths=None,
            component_html_path=component_html_path,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
        )

    async def _generate_full_page(
        self,
        job: RedesignJob,
        job_id: str,
        html: str,
        theme: _ThemeDecision,
        resolved_ctx: ResolvedContext,
        *,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> _GenerationResult:
        llm_result = await self._html_gen.generate_page_chunked(
            url=job.url,
            html=html,
            page_type=job.page_type,
            style=theme.style,
            findings=job.critique_findings,
            brand=theme.brand_info,
            site_analysis=job.site_analysis,
            style_description=theme.style_description,
            context=theme.context,
            llm_config=llm_config,
            budget=budget,
        )
        path = await self._storage.store_html(job_id, llm_result.text, "redesign.html")
        return _GenerationResult(
            redesigned_html=llm_result.text,
            redesigned_html_path=path,
            section_html_paths=None,
            component_html_path=None,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
            quality_warnings=list(getattr(llm_result, "quality_warnings", None) or []),
        )

    async def _generate_section_or_full(
        self,
        job: RedesignJob,
        job_id: str,
        html: str,
        theme: _ThemeDecision,
        resolved_ctx: ResolvedContext,
        *,
        llm_config: LLMConfigLike | None,
        budget: GenerationBudget | None = None,
    ) -> _GenerationResult:
        sections = job.site_analysis.get("content_sections", []) if job.site_analysis else []
        if not sections:
            return await self._generate_full_page(
                job, job_id, html, theme, resolved_ctx, llm_config=llm_config, budget=budget
            )

        # This render's own state — never the shared generator's (see SectionRun).
        run = SectionRun(budget=budget or GenerationBudget.from_settings())
        section_results = await self._html_gen.generate_sections(
            url=job.url,
            html=html,
            sections=sections,
            page_type=job.page_type,
            style=theme.style,
            findings=job.critique_findings,
            brand=theme.brand_info,
            site_analysis=job.site_analysis,
            style_description=theme.style_description,
            context=theme.context,
            llm_config=llm_config,
            run=run,
        )
        # Render the composed page: logos/contrast fixed, else regenerated, else degraded.
        section_results, composite = await self._html_gen.repair_rendered_page(
            section_results,
            compose=lambda results: self._html_gen.compose_section_page(
                results,
                style=theme.style,
                base_url=job.url,
                brand=theme.brand_info,
                context=theme.context,
                html=html,
                run=run,
            ),
            run=run,
        )
        paths: list[str] = []
        for section_name, section_html in section_results.items():
            path = await self._storage.store_html(
                job_id, section_html, f"section_{section_name}.html"
            )
            paths.append(path)

        redesigned_html_path = await self._storage.store_html(job_id, composite, "redesign.html")
        return _GenerationResult(
            redesigned_html=composite,
            redesigned_html_path=redesigned_html_path,
            section_html_paths=paths,
            component_html_path=None,
            input_tokens=run.prompt_tokens,
            output_tokens=run.completion_tokens,
            quality_warnings=list(run.degradations),
        )

    def _phase_validate_html(self, redesigned_html: str) -> dict | None:
        """Phase 5b: deterministic HTML validation."""
        if not redesigned_html:
            return None
        try:
            from src.services.html_validator import HtmlValidator

            return HtmlValidator.validate(redesigned_html).to_dict()
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            logger.warning("HTML validation failed: %s", exc)
            return None
