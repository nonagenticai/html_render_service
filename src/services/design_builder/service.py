"""Top-level ``RedesignService`` coordinator."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.llm_credentials import LLMConfigLike
from src.services.accessibility_scorer import AccessibilityScorer
from src.services.cache_service import CacheService
from src.services.component_extractor import ComponentExtractor
from src.services.concept_image_generator import ConceptImageGenerator
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.design_builder.lifecycle import RedesignLifecycleMixin
from src.services.design_builder.phases import RedesignPhasesMixin
from src.services.design_builder.types import RenderOptions, _A11yResult, _TokenAccumulator
from src.services.design_builder.variants import RedesignVariantsMixin
from src.services.hero_text_check import HeroTextChecker
from src.services.html_generator import HtmlGenerator
from src.services.metering import resolve_stream
from src.services.storage_service import StorageService
from src.services.webhook_dispatcher import WebhookDispatcher

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


def _pkg() -> Any:
    from src.services import design_builder as _p

    return _p


class RedesignService(
    RedesignLifecycleMixin,
    RedesignPhasesMixin,
    RedesignFinalizeMixin,
    RedesignVariantsMixin,
    RedesignHelpersMixin,
):
    """Orchestrates HTML redesign, screenshot rendering, and concept images."""

    def __init__(
        self,
        storage: StorageService,
        html_generator: HtmlGenerator,
        concept_generator: ConceptImageGenerator,
        accessibility_scorer: AccessibilityScorer | None = None,
        webhook_dispatcher: WebhookDispatcher | None = None,
        component_extractor: ComponentExtractor | None = None,
        cache_service: CacheService | None = None,
        metering_redis: aioredis.Redis | None = None,
        metering_stream: str | None = None,
        hero_text_checker: HeroTextChecker | None = None,
    ) -> None:
        self._storage = storage
        # Vision check of the generated hero (owner decision): None = no check.
        self._hero_text_checker = hero_text_checker
        self._html_gen = html_generator
        self._concept_gen = concept_generator
        self._a11y_scorer = accessibility_scorer
        self._webhook = webhook_dispatcher
        self._component_extractor = component_extractor or ComponentExtractor()
        self._cache = cache_service
        # Cost-bus connection for ``_emit_render_cost_event``. A SEPARATE redis
        # from ``settings.redis.url`` (cache/queue), built from
        # METERING_REDIS_URL. None when that is unset — emission then no-ops and
        # the render lane is unchanged.
        self._metering_redis = metering_redis
        # The cost-bus stream, resolved ONCE here and passed EXPLICITLY on every
        # emit — never left to ``w2v_common``'s internal default.
        #
        # Why explicit matters: the canonical name has already moved twice
        # (``metering:costs`` -> ``improvement-loop:costs`` -> ``billing:costs``).
        # A caller that omits ``stream=`` inherits whatever the linked library
        # version happens to think is canonical, so a routine dependency bump
        # can silently repoint every render's billing at a stream with no
        # consumer — and the emit still succeeds, so nothing goes red. Resolving
        # at construction also means the name is a fixed property of the process
        # (visible in one place, mirrored on ``app.state.metering_stream``)
        # rather than re-read from the environment on each emit.
        #
        # ``resolve_stream()`` reads ``METERING_STREAM`` — the env var the fleet
        # actually sets (``METERING_STREAM: "billing:costs"`` in the gitops
        # external-services manifests), falling back to the canonical name.
        self._metering_stream = metering_stream or resolve_stream()

    async def aclose(self) -> None:
        """Propagate aclose() to subordinate services that own AsyncClient handles.

        ``asgi._shutdown`` only knows about top-level ``app.state.*`` clients;
        nested clients (``concept_generator._client``) leak unless their
        owners forward the close call.
        """
        await self._concept_gen.aclose()

    @staticmethod
    def _a11y_from_verification(a11y: _A11yResult, verification_report: dict | None) -> _A11yResult:
        """Back-fill an axe-derived accessibility score when the LLM scorer didn't run.

        The assembly verify lane records a deterministic ``accessibility_score`` in
        the verification report's ``checks`` (see verification.verifier). Surface it
        as the job's score/report so an assembly design isn't reported as null.
        Returns ``a11y`` unchanged when it already has a score (never overrides the
        LLM scorer) or when no verification/axe score is available.
        """
        if a11y.score is not None:
            return a11y
        checks = (verification_report or {}).get("checks") or {}
        score = checks.get("accessibility_score")
        if score is None:
            return a11y
        return _A11yResult(
            score=float(score),
            report=a11y.report
            or {
                "overall_score": float(score),
                "source": "axe",
                "impact_counts": checks.get("a11y_impact_counts", {}),
                "categories": [],
                "findings": [],
                "html_was_fixed": False,
            },
            redesigned_html_path=a11y.redesigned_html_path,
        )

    @classmethod
    async def _backfill_a11y(
        cls, a11y: _A11yResult, verification_report: dict | None, html: str
    ) -> _A11yResult:
        """Fill a missing accessibility score: verify-lane axe first, then a fresh scan.

        Never overrides an LLM score, and never replaces an LLM ``unscored_reason``
        report (that already states why). Only a job that NOTHING measured — the
        freeform/section default: no verify lane, LLM scorer opt-in — gets the
        standalone axe scan.
        """
        if a11y.score is None:
            a11y = cls._a11y_from_verification(a11y, verification_report)
        if a11y.score is None and a11y.report is None:
            a11y = await cls._a11y_from_axe_scan(a11y, html)
        return a11y

    @staticmethod
    async def _a11y_from_axe_scan(a11y: _A11yResult, html: str) -> _A11yResult:
        """Measure a design nobody else scored with a standalone axe scan.

        Called only when neither the LLM scorer nor the assembly verify lane left
        a score OR a report. A successful scan yields a numeric ``source: axe``
        score; a failed scan yields ``score=None`` with an explicit
        ``unscored_reason`` in the report, so null is never mistaken for clean.
        """
        from src.services.verification import axe_score_html

        result = await axe_score_html(html)
        if result.score is None:
            logger.warning("accessibility not scored: %s", result.unscored_reason)
        return _A11yResult(
            score=result.score,
            report={
                "overall_score": result.score,
                "unscored_reason": result.unscored_reason,
                "source": "axe",
                "impact_counts": result.impact_counts,
                "categories": [],
                "findings": [],
                "html_was_fixed": False,
            },
            redesigned_html_path=a11y.redesigned_html_path,
        )

    async def _apply_verification_verdict(
        self, db: AsyncSession, job_id: str, verification_report: dict | None
    ) -> None:
        """Persist the assembly-gate verdict, and raise on a hard failure.

        Lifted out of ``process_job`` verbatim (no behaviour change) so the
        cancel-before-start gate could be added there without pushing it over
        the complexity ceiling. A design only completes if it passes — otherwise
        this fails loud with the findings, which are persisted for the
        UI/orchestrator first, rather than shipping an unverified design.
        """
        if verification_report is None:
            return
        await _pkg().JobManager.set_verification(
            db, job_id, verification_report.get("status", "failed"), verification_report
        )
        await db.commit()
        if verification_report.get("status") != "failed":
            return
        hard = [f for f in verification_report.get("findings", []) if f.get("severity") == "hard"]
        summary = (
            "; ".join(f"[{f.get('check')}] {f.get('message')}" for f in hard)
            or "verification failed"
        )
        raise RuntimeError(f"design failed verification: {summary}")

    @staticmethod
    def _attach_quality_warnings(
        job_id: str, report: dict | None, warnings: list[str]
    ) -> dict | None:
        """Put the generation's compromises on the job record (``html_validation_report``).

        A job that had to drop a section or leave a contrast failure unrepaired to fit
        its budget is DELIVERED — not failed, and not silently either.
        """
        if not warnings:
            return report
        logger.warning("Job %s completes DEGRADED: %s", job_id, " | ".join(warnings))
        return {**(report or {}), "quality_warnings": list(warnings)}

    @staticmethod
    def _attach_hero_image_check(
        report: dict | None, check: dict[str, object] | None
    ) -> dict | None:
        """Put the hero's visible-text check outcome on the job record (owner decision)."""
        if not check:
            return report
        return {**(report or {}), "hero_image_check": dict(check)}

    async def process_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        *,
        run_accessibility_check: bool = False,
        auto_fix_accessibility: bool = False,
        accessibility_threshold: float = 70.0,
        accessibility_first_auto_theme: bool = False,
        generate_screenshots: bool = True,
        generate_concept_images: bool = False,
        render_options: RenderOptions | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        llm_config: LLMConfigLike | None = None,
    ) -> None:
        """Process a pending redesign job through the full pipeline.

        Thin orchestrator. Phase methods (``_phase_*``) handle the work;
        each is independently testable.
        """
        t0 = time.monotonic()
        job = await _pkg().JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        logger.info("Processing job %s (scope=%s, tenant=%s)", job_id, job.scope, tenant_id)

        if not await _pkg().JobManager.mark_processing(db, job_id):
            # Cancelled before it ever started. Returning here — rather than
            # letting the fenced no-op fall through — is what stops the pipeline
            # rendering and BILLING a job the tenant already stopped, and it also
            # withholds the ``job.started`` webhook, which would otherwise be the
            # first thing they hear about a job they cancelled.
            logger.info("Job %s was cancelled before processing began; nothing to do", job_id)
            await db.commit()
            return

        await db.commit()

        settings = get_settings()
        tokens = _TokenAccumulator()
        # ONE generation budget for the whole job (its variants share it): the caller
        # waits on the job, not on a variant. See DesignSettings.generation_deadline_seconds.
        #
        # THE TRADE, stated because it is a behaviour change and not an obvious one.
        # Sharing means the styles compete: on a slow vLLM a 4-style job can spend the
        # whole deadline on styles 1-2 and have styles 3-4 refused outright with
        # ``BudgetExhausted`` (recorded per-variant; a job with at least one completed
        # variant still completes). Before the budget reached these paths those late
        # variants instead ran past the deadline and completed — which is precisely the
        # unbounded cost this exists to remove: the orchestrator stage waiting on the
        # job had already given up, and every one of those calls was still billed.
        #
        # A per-variant budget would restore that: N styles x the full deadline is N
        # times the cost ceiling, and the job still cannot return before the caller's
        # own timeout. Fewer variants delivered inside the deadline is the intended
        # outcome, not a regression — the lever is
        # DesignSettings.generation_deadline_seconds, not a budget per variant.
        from src.services.html_gen.section_run import GenerationBudget

        budget = GenerationBudget.from_settings()

        try:
            await self._dispatch_webhook(db, job, "job.started")

            if await self._check_cancelled(db, job_id):
                return

            # Release the DB connection before the long fetch/generation phases.
            # Holding an open transaction across a multi-minute LLM call lets
            # postgres/pgbouncer reap the idle backend, so the post-generation
            # write would land on a dead socket. Committing here returns the
            # connection to the pool; the next checkout is pool_pre_ping'd.
            await db.commit()

            html = await self._phase_resolve_html(job)
            resolved_ctx = _pkg().resolve_context(
                user_context=job.user_context,
                site_analysis=job.site_analysis,
                brand=job.brand_info,
                html=html,
            )

            theme = await self._phase_resolve_theme(
                db,
                tenant_id,
                job,
                resolved_ctx,
                accessibility_first_auto_theme=accessibility_first_auto_theme,
            )

            if await self._check_cancelled(db, job_id):
                return

            # Release the connection before generation (single-page path) and
            # before the variant gather (each variant uses its own session).
            await db.commit()

            # Variant fan-out short-circuits the rest of the pipeline.
            if job.styles and len(job.styles) > 1:
                await self._phase_process_variants_and_finalize(
                    db,
                    job,
                    job_id,
                    tenant_id,
                    html,
                    resolved_ctx,
                    settings,
                    session_factory=session_factory,
                    generate_screenshots=generate_screenshots,
                    render_options=render_options,
                    budget=budget,
                )
                return

            gen_result = await self._phase_generate_html(
                job, job_id, html, theme, resolved_ctx, llm_config=llm_config, budget=budget
            )
            tokens.add(gen_result.input_tokens, gen_result.output_tokens)

            # Assembly gate: verify the design; refine on hard failure (bounded).
            # A design only completes if it passes — otherwise fail loud with the
            # findings (persisted for the UI/orchestrator) rather than shipping it.
            gen_result, verification_report = await self._phase_verify_and_refine(
                job, job_id, gen_result, theme, tokens, llm_config=llm_config, budget=budget
            )
            await self._apply_verification_verdict(db, job_id, verification_report)

            html_validation_report = self._phase_validate_html(gen_result.redesigned_html)
            html_validation_report = self._attach_quality_warnings(
                job_id, html_validation_report, gen_result.quality_warnings
            )
            html_validation_report = self._attach_hero_image_check(
                html_validation_report, gen_result.hero_image_check
            )

            if await self._check_cancelled(db, job_id):
                return

            # Release the connection before the (potentially long) a11y auto-fix
            # LLM call, screenshot render, and concept-image generation.
            await db.commit()

            a11y = await self._phase_accessibility(
                job_id,
                gen_result,
                theme,
                run_accessibility_check=run_accessibility_check,
                auto_fix_accessibility=auto_fix_accessibility,
                accessibility_threshold=accessibility_threshold,
                accessibility_first_auto_theme=accessibility_first_auto_theme,
                llm_config=llm_config,
                budget=budget,
            )
            # web_design_v2 / assembly deliberately skips the LLM a11y scorer, but
            # its verify lane already ran axe. Derive a deterministic score from that
            # (no extra LLM cost) so the design reports a real accessibility_score
            # instead of null. Only fills the gap — never overrides an LLM score.
            # Freeform/section jobs have neither, so they are scored by a standalone
            # axe scan (see `_backfill_a11y`) rather than reported as a bare null.
            a11y = await self._backfill_a11y(a11y, verification_report, gen_result.redesigned_html)
            redesigned_html_path = a11y.redesigned_html_path or gen_result.redesigned_html_path

            screenshots = await self._phase_screenshots(
                job, job_id, redesigned_html_path, settings, generate_screenshots, render_options
            )

            concept_result = await self._phase_concept_images(
                job, job_id, theme.style, settings, generate_concept_images
            )
            concept_image_assets = concept_result.assets or None
            concept_count = len(concept_result.assets)

            await self._phase_finalize_completed(
                db,
                job,
                job_id,
                tenant_id,
                gen_result=gen_result,
                redesigned_html_path=redesigned_html_path,
                a11y=a11y,
                screenshots=screenshots,
                concept_image_assets=concept_image_assets,
                concept_count=concept_count,
                concept_image_degraded=concept_result.degraded,
                concept_image_reason=concept_result.reason,
                html_validation_report=html_validation_report,
                tokens=tokens,
            )

        except Exception as exc:
            # Catch EVERY exception type, deliberately. This is the job's only
            # record-the-failure seam: `_phase_finalize_failed` is what persists
            # the error and moves the row out of `processing`. The previous
            # narrow tuple could not see the exceptions these phases actually
            # raise -- `playwright.async_api.Error` (phase 1 `_fetch_page_html`,
            # every screenshot phase), `httpx.HTTPError` (WM, webhooks) and
            # `redis.exceptions.RedisError` all derive from `Exception`, not
            # `OSError` -- so the most common real failures skipped this handler
            # entirely and the tenant's job was left with NO recorded reason.
            #
            # Catching broadly is safe here precisely because this handler does
            # not swallow: it records and re-raises, so the caller's own
            # behaviour (incl. `CancelledError` propagation on shutdown drain)
            # is unchanged. Narrow tuples belong on the per-phase degrade
            # handlers, which choose to continue; this one only chooses to
            # remember.
            await self._phase_finalize_failed(db, job, job_id, tenant_id, exc)
            raise

        elapsed = time.monotonic() - t0
        logger.info(
            "Job %s completed successfully in %.2fs (tokens=%d)",
            job_id,
            elapsed,
            tokens.total,
        )
        await db.commit()
