"""Shared internal helpers for ``RedesignService``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from functools import cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from bs4.exceptions import ParserRejectedMarkup, StopParsing
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import (
    W2V_APP_KEY,
    get_attribution_subject_id_from_context,
    get_correlation_id_from_context,
    get_license_id_from_context,
)
from w2v_common.metering import cost_event_key, schedule_cost_event
from w2v_common.net import validate_public_url
from w2v_common.tenant import TenantError

from src.database.models import RedesignJob
from src.services.metering import COST_SOURCE, SERVICE_KEY

if TYPE_CHECKING:
    import redis.asyncio as aioredis
from src.services.navigation_guard import assert_not_challenge, require_ok_navigation
from src.services.webhook_dispatcher import WebhookDispatcher

#: WHO FUNDS a render: the Playwright/Chromium render runs on platform
#: hardware (billing ``LOCAL_PROVIDERS`` lists ``html-render``).
RENDER_FUNDING = "local_gpu"

logger = logging.getLogger(__name__)

# Runs in the fetched page: what the browser APPLIED, which the DOM string alone cannot
# say — a site's CSS normally lives in an external stylesheet the resolver never reads.
# Returns body vs first-heading ``font-family``, hex colour custom properties on the
# root (same-origin sheets + the computed root style), and the colours of links and
# buttons. Bounded and side-effect free.
SOURCE_STYLES_JS = """() => {
  const hex = (c) => {
    const m = /rgba?\\((\\d+)[,\\s]+(\\d+)[,\\s]+(\\d+)(?:[,\\s/]+([\\d.]+))?/.exec(c || '');
    if (!m || (m[4] !== undefined && parseFloat(m[4]) === 0)) return null;
    return '#' + [m[1], m[2], m[3]].map((v) => (+v).toString(16).padStart(2, '0')).join('');
  };
  const fam = (el) => (el ? getComputedStyle(el).fontFamily : null);
  const vars = [];
  const seen = new Set();
  const addVar = (name, value) => {
    const v = (value || '').trim();
    if (!name.startsWith('--') || seen.has(name) || !/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v)) return;
    seen.add(name);
    vars.push([name, v]);
  };
  for (const sheet of Array.from(document.styleSheets).slice(0, 20)) {
    let rules;
    try { rules = sheet.cssRules; } catch (e) { continue; }
    for (const r of Array.from(rules || []).slice(0, 400)) {
      if (!r.style || !/(^|,)\\s*(:root|html)\\s*(,|$)/.test(r.selectorText || '')) continue;
      for (const p of Array.from(r.style)) addVar(p, r.style.getPropertyValue(p));
    }
  }
  const rootStyle = getComputedStyle(document.documentElement);
  for (const p of Array.from(rootStyle)) {
    if (p.startsWith('--')) addVar(p, rootStyle.getPropertyValue(p));
  }
  const used = [];
  const els = document.querySelectorAll('a, button, .btn, [class*="button"], [class*="accent"]');
  for (const el of Array.from(els).slice(0, 60)) {
    const cs = getComputedStyle(el);
    for (const [k, v] of [['color', cs.color], ['background', cs.backgroundColor]]) {
      const h = hex(v);
      if (h) used.push([el.tagName.toLowerCase() + ':' + k, h]);
    }
  }
  return {
    body_font: fam(document.body),
    heading_font: fam(document.querySelector('h1, h2, h3')),
    vars: vars.slice(0, 60),
    used: used.slice(0, 60),
  };
}"""


def embed_source_styles(html: str, styles: object) -> str:
    """Append the computed-style snapshot to ``html`` as an inert JSON script block.

    Read back by ``context_resolver.extract_source_styles``. Carried IN the html so
    it rides the existing ``cache_html`` / ``job.html_content`` contract unchanged.
    """
    import json

    from src.services.context_resolver import SOURCE_STYLES_SCRIPT_ID

    if not isinstance(styles, dict) or not html:
        return html
    payload = json.dumps(styles, ensure_ascii=True).replace("</", "<\\/")
    block = f'<script type="application/json" id="{SOURCE_STYLES_SCRIPT_ID}">{payload}</script>'
    idx = html.lower().find("</head>")
    if idx == -1:
        return block + html
    return html[:idx] + block + html[idx:]


@cache
def capture_failure_types() -> tuple[type[Exception], ...]:
    """Return the exception classes a *soft* screenshot capture may degrade on.

    Why this exists rather than a literal tuple at each call site: every
    ``capture_url_screenshots`` caller used to write
    ``(OSError, ValueError, RuntimeError, TimeoutError)``, which reads broad but
    misses the one library the call actually drives. A ``Page.goto`` against a
    slow or dead tenant URL raises ``playwright.async_api.Error`` (and its
    subclass ``playwright.async_api.TimeoutError`` on a nav timeout), both of
    which derive straight from ``Exception`` and **not** from ``OSError``. The
    builtin ``TimeoutError`` in that tuple is an ``OSError`` subclass and does
    not alias Playwright's, so it gave no reach at all. The handlers were dead
    code: a tenant page that merely timed out escaped the "soft" capture and
    hard-failed a paid redesign job instead of degrading to "no before image".

    Why it is a lazy, cached function rather than a module-level constant:
    naming Playwright's class requires importing ``playwright.async_api``, a
    multi-second, browser-bindings-heavy import. This module is pulled in by the
    whole ``design_builder`` package — including unit tests that never launch
    Chromium — and the rest of the service (``screenshot_renderer``,
    ``browser_pool``) deliberately keeps that import lazy. Resolving on first
    failure and caching pays the cost once, only in a process that is really
    rendering. Playwright is a required dependency (``pyproject.toml``), so the
    ``ImportError`` arm guards a broken install, not a supported configuration.

    Deliberately NOT in the net: ``SSRFError`` handling stays as its own
    ``except`` clause *above* these, because it subclasses ``ValueError`` and a
    blocked internal target is a security decision that must be logged loudly,
    not folded into an indistinguishable "capture failed" warning.

    Returns:
        The exception classes a soft capture converts into "no screenshots".
    """
    failures: list[type[Exception]] = [
        OSError,
        ValueError,
        RuntimeError,
        # The builtin, aliased by ``asyncio.TimeoutError``. It is an ``OSError``
        # subclass — listed for the reader, not for extra reach.
        TimeoutError,
    ]
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    except ImportError:  # pragma: no cover — browser extras absent
        logger.debug("Playwright not importable; capture errors will not cover browser failures")
    else:
        # ``PlaywrightTimeoutError`` subclasses ``PlaywrightError``; naming both
        # keeps the nav-timeout case explicit for the next reader.
        failures.extend((PlaywrightError, PlaywrightTimeoutError))
    return tuple(failures)


def _pkg() -> Any:
    """Return the redesign_service package — used to resolve patchable names at call time."""
    from src.services import design_builder as _p

    return _p


class RedesignHelpersMixin:
    """Mixin: cancellation check, validator revalidation, fetch, webhook, cache."""

    # Attributes initialized by ``RedesignService.__init__``.
    _webhook: Any
    _cache: Any
    # The cost-bus connection, or None when METERING_REDIS_URL is unset.
    _metering_redis: aioredis.Redis | None
    # The cost-bus stream name, resolved once at construction from
    # METERING_STREAM. Always a real string — never None, never omitted from the
    # emit call (see the ``stream=`` note in ``_emit_render_cost_event``).
    _metering_stream: str

    async def _check_cancelled(self, db: AsyncSession, job_id: str) -> bool:
        """Check if the job has been cancelled. If so, log and return True."""
        if await _pkg().JobManager.is_cancelled(db, job_id):
            logger.info("Job %s was cancelled, stopping processing", job_id)
            return True
        return False

    def _run_critique_adapter_revalidation(self, html: str) -> dict:
        """Run deterministic post-fix accessibility validation adapters.

        Fail-soft: any validator failure degrades to a ``passed=False`` report
        carrying the error, because this is enrichment metadata merged into the
        a11y report — losing it must not lose a finished redesign.

        Why the handler names ``bs4`` classes: the enclosed call is
        ``HtmlValidator.validate``, which parses **LLM-generated, untrusted**
        markup with BeautifulSoup. When the underlying parser gives up, bs4 does
        not re-raise a builtin — ``HTMLParserTreeBuilder.feed`` converts the
        ``AssertionError`` that ``html.parser`` raises on fatal markup into
        ``bs4.exceptions.ParserRejectedMarkup``, and a tree builder that cannot
        continue raises ``StopParsing``. Both derive straight from ``Exception``
        and not from ``OSError``, so the previous
        ``(OSError, ValueError, RuntimeError, TimeoutError)`` tuple could never
        fire on the one input class that actually breaks this call. (bs4's other
        exception, ``FeatureNotFound``, is a ``ValueError`` and was already
        covered; ``RecursionError`` on deeply nested markup is a
        ``RuntimeError`` and likewise.)

        Args:
            html: The (LLM-authored) document to re-validate.

        Returns:
            The adapter report, or the degraded report when validation failed.
        """
        try:
            from src.services.html_validator import CATEGORY_A11Y, HtmlValidator

            report = HtmlValidator.validate(html)
            findings = [f for f in report.findings if f.category == CATEGORY_A11Y]
            # Severity aggregation and the blocking gate MUST consider EVERY
            # finding, not a truncated prefix — a `critical` ranked past the
            # display cutoff still fails accessibility. The `[:25]` slice is a
            # DISPLAY limit for the tenant-facing rows only, never the gate.
            severity_counts: dict[str, int] = {}
            for finding in findings:
                severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            finding_rows: list[dict] = [
                {
                    "severity": finding.severity,
                    "description": finding.description,
                    "recommendation": finding.recommendation,
                }
                for finding in findings[:25]
            ]

            # The validator only ever emits `critical`/`major`/`minor` severities
            # (see html_validator.SEVERITY_*); `major` is the blocking mid-tier,
            # NOT `high` (which is never produced — it made this gate a no-op).
            has_blocking = any(severity_counts.get(level, 0) > 0 for level in ("critical", "major"))
            return {
                "adapters_ran": ["deterministic_html_validator"],
                "passed": not has_blocking,
                "findings_count": len(findings),
                "severity_counts": severity_counts,
                "findings": finding_rows,
            }
        except (
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
            ParserRejectedMarkup,
            StopParsing,
        ) as exc:
            logger.warning("Critique adapter revalidation failed: %s", exc)
            return {
                "adapters_ran": [],
                "passed": False,
                "error": str(exc),
                "findings_count": 0,
                "severity_counts": {},
                "findings": [],
            }

    async def _fetch_page_html(self, url: str, timeout_ms: int = 30000) -> str:
        """Fetch live HTML via Playwright (using the shared browser pool).

        Args:
            url: Tenant-supplied page to navigate to.
            timeout_ms: Page load timeout in milliseconds.

        Returns:
            The rendered DOM of the target page.

        Raises:
            SSRFError: if ``url`` is not a public http(s) target. ``SSRFError``
                subclasses ``ValueError``, so ``process_job``'s existing
                ``except (OSError, ValueError, RuntimeError, TimeoutError)``
                catches it and routes the job through
                ``_phase_finalize_failed`` — the job is marked failed with the
                block reason, not a 500.
            PageFetchError: if the navigation returned HTTP >= 400, or rendered a
                bot-challenge interstitial. Subclasses ``RuntimeError``, so it
                rides the same ``process_job`` handler as ``SSRFError`` and fails
                the job loudly instead of handing an error document downstream as
                if it were the tenant's page. Because this raises, the caller's
                ``cache_html`` is never reached, so a bad fetch can never be
                pinned under the good-HTML key for its 3600s TTL.
        """
        from src.services.browser_pool import get_browser_pool

        # Sink-side SSRF guard, deliberately duplicating the
        # `RedesignRequest.url` admission check rather than trusting it. `url`
        # is read back from the `redesign_jobs.url` column at render time, so it
        # may be a row written before this guard existed, or by any writer that
        # does not go through the schema. Schema validation does not cover this
        # call; the sink does. Same defence-in-depth shape as
        # `webhook_dispatcher.dispatch` (pre-check, then re-validate in the
        # transport).
        #
        # Validating the STRING and then handing the same string to Chromium is
        # a TOCTOU: Chromium resolves the hostname itself, so a name that
        # answers public at validation time and private milliseconds later at
        # `goto` time gets its INTERNAL html fetched, cached under `cache_html`
        # for the 3600s TTL, and fed to the redesign model. Discarding the
        # vetted address (which this call site did) leaves that rebind window
        # wide open. Capture it and lease a resolver-pinned browser
        # (`--host-resolver-rules=MAP <host> <ip>`), exactly as the sibling sink
        # `screenshot_renderer.capture_url_screenshots` does — the pin is what
        # closes the window, the validation alone does not. `pinned_ip` is
        # `None` only for an allow-listed host (never passed here), which keeps
        # the shared pool path unchanged.
        pinned_ip = validate_public_url(url)
        host = urlparse(url).hostname

        pool = await get_browser_pool()
        lease_cm = pool.lease_pinned(host, pinned_ip) if pinned_ip and host else pool.lease()
        async with lease_cm as browser:
            page = await browser.new_page()
            try:
                response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                # The Response was previously discarded, so a 404/403/500 body was
                # returned to `_phase_resolve_html` as though it were the site --
                # non-empty, so `generate_full_page` set `greenfield=False` and the
                # model redesigned the error page. Check status BEFORE reading the
                # DOM so the failure names the status rather than the symptom.
                require_ok_navigation(response, url)
                html = await page.content()
                # A challenge interstitial is a 200, so the status check above
                # cannot see it; it is also non-empty, so nothing downstream can.
                assert_not_challenge(html, url)
                # Soft: the snapshot improves font/palette fidelity; a page that
                # refuses evaluate() must not fail the fetch.
                try:
                    styles = await page.evaluate(SOURCE_STYLES_JS)
                except Exception as exc:  # noqa: BLE001 — best-effort enrichment
                    logger.debug("source style snapshot failed for %s: %s", url, exc)
                else:
                    html = embed_source_styles(html, styles)
            finally:
                # Shielded + suppressed close — same contract as
                # ``screenshot_renderer._close_page_quietly`` (see its docstring).
                # ``suppress(Exception)`` stops a ``TargetClosedError`` from a dead
                # target masking the real fetch failure, and never catches
                # ``CancelledError``. The ``shield`` is required, NOT decoration:
                # ``close()`` suspends on a driver round-trip, so under a cancelled
                # scope a bare ``await page.close()`` raises at that suspension
                # before the page is torn down and the page leaks for the rest of
                # the pooled browser's 200-lease life.
                with contextlib.suppress(Exception):
                    await asyncio.shield(page.close())
        return html

    async def _dispatch_webhook(
        self,
        db: AsyncSession,
        job: RedesignJob,
        event_type: str,
        error: str | None = None,
        batch_id: str | None = None,
        variant_id: str | None = None,
        accessibility_score: float | None = None,
    ) -> None:
        """Dispatch webhook notification if configured. Non-fatal on failure.

        Fail-soft by contract: a tenant's unreachable webhook endpoint must
        never take down the job phase that is notifying them. That is load
        bearing on the failure path in particular — ``_phase_finalize_failed``
        dispatches ``job.failed`` *before* committing the failure state, so an
        escaping exception here would cost the job its recorded reason.

        Why ``httpx.HTTPError`` is named: the sink is
        ``WebhookDispatcher.dispatch``, whose transport is **httpx**. Every
        httpx transport and status failure (``ConnectError``, ``ReadTimeout``,
        ``ConnectTimeout``, ``HTTPStatusError``, …) subclasses
        ``httpx.HTTPError``, which derives from ``Exception`` and **not** from
        ``OSError`` — and ``httpx.ConnectTimeout`` is not the builtin
        ``TimeoutError``. So the previous
        ``(OSError, ValueError, RuntimeError, TimeoutError)`` tuple had no reach
        over the one library this call drives: a webhook to a resolvable but
        dead host escaped this "non-fatal" handler and skipped everything after
        the call. The dispatcher has its own inner retry/absorb loop, but that
        is the dispatcher's business — this handler must hold for any injected
        ``self._webhook``, so it names the transport's base class itself.

        Args:
            db: Session used for the delivery log rows.
            job: The job the event is about.
            event_type: Event name, e.g. ``job.failed``.
            error: Optional error string for failure events.
            batch_id: Optional batch the job belongs to.
            variant_id: Optional variant the event is about.
            accessibility_score: Optional score to include in the payload.
        """
        if not self._webhook:
            return
        webhook_url = getattr(job, "webhook_url", None)
        if not webhook_url:
            return
        try:
            payload = WebhookDispatcher.build_event_payload(
                event_type,
                job.job_id,
                status=job.status,
                error=error,
                result_url=f"/api/designs/jobs/{job.job_id}/result",
                batch_id=batch_id or getattr(job, "batch_id", None),
                variant_id=variant_id,
                accessibility_score=accessibility_score,
            )
            await self._webhook.dispatch(
                db,
                job_id=job.job_id,
                webhook_url=webhook_url,
                webhook_secret=getattr(job, "webhook_secret", None),
                event_type=event_type,
                payload=payload,
            )
        except (OSError, ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
            logger.warning("Webhook dispatch failed for job %s: %s", job.job_id, exc)

    def _emit_render_cost_event(
        self,
        *,
        job_id: str,
        tenant_id: str,
        stage: str,
        pages: int = 1,
        screenshots: int = 0,
        concept_images: int = 0,
        llm_tokens: int = 0,
    ) -> None:
        """Bill the RENDER itself — the work billing could not previously see.

        Until this existed, a web-design/HTML render emitted no cost event at
        all: the Playwright render, the screenshots and the concept images were
        free, and only the LLM slice billed, indirectly, as ``inference-router``.
        A 30-day sweep of ``billing_cost_events`` carried no render key of any
        kind. This is the emitter that puts ``w2v-html-render`` on the bus.

        **Where it is called from is the whole design.** Both call sites put it
        on the line immediately AFTER ``record_usage``, INSIDE the
        ``mark_completed`` CAS fence. That fence — not ``event_key`` — is the
        anti-double-bill barrier for a *retried job*: a worker that lost the
        claim, or a job cancelled mid-render, gets ``completed=False`` and
        returns before reaching here, so it neither bills the internal usage row
        nor the cost bus. (``event_key``, minted in ``w2v_common``, dedupes the
        narrower case of one logical emit whose XADD was retried.) Emitting
        before the fence would bill cancelled and stolen jobs.

        **A billing failure must never fail a render.** Nothing here is awaited
        into the render's success path: ``schedule_cost_event`` is fire-and-
        forget, retains a strong reference to its task (a bare ``create_task``
        can be GC'd mid-flight and silently drop the event), and never raises
        except ``TenantError`` — which is caught below, because an unattributable
        tenant is a billing defect, not a reason to fail a finished design.

        No-ops when ``self._metering_redis`` is None (``METERING_REDIS_URL``
        unset — local dev, tests that don't care), so the render lane is
        unchanged wherever metering is not configured.

        Args:
            job_id: The redesign job being billed, carried as event metadata.
            tenant_id: WHO PAYS. The job's tenant, never a substituted default.
            stage: Which lane produced the spend (``redesign`` / ``refinement``).
            pages: Billable unit — pages rendered. Carried as ``input_tokens``
                because that is the cost bus's only integer quantity field.
            screenshots: Screenshots rendered, for the usage breakdown.
            concept_images: Concept images generated, for the usage breakdown.
            llm_tokens: LLM tokens the job consumed. Reported for reconciliation
                against the ``inference-router`` events that actually bill them
                — NOT billed again here (``output_tokens=0``).
        """
        metering_redis: aioredis.Redis | None = getattr(self, "_metering_redis", None)
        if metering_redis is None:
            return
        try:
            schedule_cost_event(
                metering_redis,
                service_key=SERVICE_KEY,
                cost_source=COST_SOURCE,
                tenant_id=tenant_id,
                license_id=get_license_id_from_context(),
                subject_id=get_attribution_subject_id_from_context(),
                # G5: the app is stated, never left for billing to guess. The
                # licences app KEY; billing ``split_asserted_app`` resolves it.
                app_id=W2V_APP_KEY,
                # Local Chromium on platform hardware: no vendor invoice.
                funding=RENDER_FUNDING,
                # Deterministic per physical render: the mark_completed fence
                # admits exactly one completion per (job, stage), so a re-emit
                # of the same completion (restart, redelivery) dedupes at
                # billing instead of minting a fresh uuid7 and billing twice.
                event_key=cost_event_key(SERVICE_KEY, job_id, stage),
                provider="html-render",
                model="playwright",
                # EXPLICIT stream, never the library's internal default. The
                # canonical name has moved twice already (metering:costs ->
                # improvement-loop:costs -> billing:costs); a caller that omits
                # this inherits whatever the linked w2v_common happens to carry,
                # so an unrelated dependency bump can repoint every render's
                # billing at a consumerless stream while the XADD still
                # succeeds — a silent unbilling with nothing red to catch it.
                stream=self._metering_stream,
                input_tokens=pages,
                # The LLM tokens are already billed by inference-router. Billing
                # them again here would double-charge the tenant for the same
                # spend, so they travel in usage_extra as reconciliation data.
                output_tokens=0,
                latency_ms=0,
                cost_usd=0.0,  # local Chromium, no external quota
                usage_extra={
                    "unit": "pages",
                    "screenshots": screenshots,
                    "concept_images": concept_images,
                    "llm_tokens": llm_tokens,
                    # ⚠️ Carried HERE rather than as a top-level field because
                    # ``w2v_common.metering.cost_bus._build_fields`` does not
                    # emit a ``correlation_id`` field even though billing reads
                    # one. Widening w2v_common is the follow-up; until then this
                    # keeps the trace recoverable instead of losing it.
                    "correlation_id": get_correlation_id_from_context(),
                },
                metadata_extra={"job_id": job_id, "stage": stage},
            )
        except TenantError:
            # An unattributable tenant means this render cannot be billed. That
            # is a real defect and is logged loudly — but the design is already
            # finished and persisted, and refusing to return it would punish the
            # tenant for our accounting problem.
            logger.exception(
                "Render cost event for job %s could not be attributed to a tenant — "
                "this render is UNBILLED",
                job_id,
            )
