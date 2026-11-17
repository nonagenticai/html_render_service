"""Automated verification lane — the "only working designs complete" gate.

Loads a generated page in the shared Chromium pool and runs deterministic checks:

* **render** — the page loads without a navigation error;
* **console** — zero ``console.error`` / uncaught ``pageerror`` (JS actually runs);
* **assets** — no failed same-document network requests (broken inlined asset);
* **a11y** — axe-core serious/critical violations at or below threshold. An axe
  run that produced no result is a HARD finding (``a11y_probed: False``), not a
  pass: the violation counts are omitted rather than reported as zero;
* **links** — anchors resolve to a real target (advisory: every dead link — ``#``,
  empty, a ``#id`` that does not exist, a site-relative path — is listed);
* **images** — ``<img>`` that completed with no pixels (blocked hotlink, 404);
* **brand** — one brand across every ``[data-brand]`` element and a real ``<title>``.

Each finding is classified HARD (blocks completion → triggers the refine loop) or
ADVISORY (reported, non-blocking). Lighthouse perf/SEO is intentionally advisory
and left out of this deterministic lane (it needs a separate toolchain). Failures
of the checker itself fail SAFE — they surface as a hard finding, never a silent
pass, matching the service's "fail loud, don't complete-empty" philosophy.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)

_AXE_PATH = Path(__file__).resolve().parent / "assets" / "axe.min.js"

# axe impact levels that count against the a11y hard gate.
_BLOCKING_A11Y_IMPACTS = {"serious", "critical"}

# The raw axe-core run object as it arrives from the browser (or None when the
# run produced nothing). Aliased so the JSON-boundary `Any` is stated ONCE across
# the six signatures that carry it — see tests/test_any_ratchet.py, whose baseline
# may only ever be lowered.
type _AxeResult = dict[str, Any]


@dataclass
class Finding:
    check: str
    severity: str  # "hard" | "advisory"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    status: str  # "passed" | "failed"
    findings: list[Finding] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "hard"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": [asdict(f) for f in self.findings],
            "checks": self.checks,
        }

    def summary(self) -> str:
        hard = self.hard_findings
        if not hard:
            return "verification passed"
        return "; ".join(f"[{f.check}] {f.message}" for f in hard)


@dataclass
class _BrowserProbe:
    """Raw signals collected from loading the page in Chromium."""

    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    failed_requests: list[str] = field(default_factory=list)
    axe_result: _AxeResult | None = None
    render_error: str | None = None
    broken_images: list[str] = field(default_factory=list)


async def _probe_in_browser(html: str, *, timeout_ms: int, run_axe: bool) -> _BrowserProbe:
    """Load ``html`` in the shared pool and collect console/error/request/axe signals."""
    from src.services.browser_pool import get_browser_pool
    from src.services.screenshot_renderer import (
        VIEWPORT_SIZES,
        _minio_allow_hosts,
        block_private_subresources,
    )

    probe = _BrowserProbe()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(html)
            tmp_path = Path(fh.name)

        pool = await get_browser_pool()
        async with pool.lease() as browser:
            # Internal-only probe (console/pageerror/axe) — it captures no image,
            # so it gets no ``device_scale_factor``: DPR changes rasterisation,
            # not CSS layout, and nothing here is rasterised. The VIEWPORT is
            # shared with the capture path on purpose. It was hardcoded 1440x900,
            # i.e. a duplicate of the old ``VIEWPORT_SIZES["desktop"]``, so when
            # the delivery viewport moved to 1920x1080 this probe would have gone
            # on judging layout-dependent axe rules (target-size, overlap,
            # contrast over responsive backgrounds) at a breakpoint the shipped
            # page never renders at. Read the constant instead of re-pinning it.
            desktop = VIEWPORT_SIZES["desktop"]
            page = await browser.new_page(
                viewport={"width": desktop["width"], "height": desktop["height"]}
            )
            # Everything after new_page() lives under a finally that closes the
            # page. `pool.lease()` closes nothing on exit — it only re-queues the
            # slot — and the browser is only relaunched every `browser_max_renders`
            # (200) leases. So any exit path that skipped `page.close()` (a failed
            # navigation, an axe error, cancellation) orphaned a Chromium page plus
            # the three listeners below, which hold a reference to `probe`, for up
            # to 200 renders per slot: multi-GB, ending in an OOMKill.
            try:
                page.on(
                    "console",
                    lambda m: probe.console_errors.append(m.text) if m.type == "error" else None,
                )
                page.on("pageerror", lambda e: probe.page_errors.append(str(e)))
                page.on(
                    "requestfailed",
                    lambda r: probe.failed_requests.append(f"{r.url} ({r.failure})"),
                )
                # SSRF guard on the FIFTH ``file://`` sink. The four in
                # ``screenshot_renderer`` install this; this one did not, because it
                # lives in another module and spells the URL ``Path.as_uri()``
                # instead of ``f"file://{...}"`` — so the property test that pinned
                # "every file:// sink is guarded" never saw it. Same exposure as the
                # PDF sink (issue #104): ``html`` is the LLM-written assembly page,
                # navigated by the UNPINNED pool browser, so every ``<img>``/
                # ``<iframe>``/``fetch`` in it is issued by an in-cluster Chromium at
                # whatever address it names. Worse here than for the PDF, because
                # ``requestfailed``/``console`` are recorded into the report that
                # ``_phase_verify_and_refine`` persists and hands back to the tenant:
                # an unguarded document turns blind egress into a reachability
                # ORACLE for the estate. Presigned MinIO assets stay exempt via
                # ``allow_hosts`` so real designs still verify.
                blocked = await block_private_subresources(page, allow_hosts=_minio_allow_hosts())
                try:
                    await page.goto(tmp_path.as_uri(), wait_until="networkidle", timeout=timeout_ms)
                except Exception as exc:  # noqa: BLE001 — nav error is a hard fail, reported
                    probe.render_error = f"page failed to load: {exc}"
                    return probe
                if blocked:
                    # The other four sinks raise ``BlockedSubresourceError`` here.
                    # This one is the verification lane, whose documented contract
                    # is "surface as a hard finding, never a silent pass" — so a
                    # blocked reference becomes a render_error, which fails the
                    # design and feeds the bounded refine loop. Same fail-loud
                    # outcome, expressed in this module's own vocabulary. Never
                    # echo the blocked URLs into the report: the report is handed
                    # to the tenant, and the list is the oracle.
                    logger.error(
                        "Verification blocked %d non-public subresource request(s): %s",
                        len(blocked),
                        blocked[:5],
                    )
                    probe.render_error = f"page referenced {len(blocked)} non-public subresource(s)"
                    return probe
                probe.broken_images = await _broken_images(page)
                if run_axe:
                    probe.axe_result = await _run_axe(page)
            finally:
                # Never let cleanup mask the real outcome: Playwright raises if the
                # page/browser is already gone (`TargetClosedError`, and a plain
                # `Error` on a dead driver connection). Suppressing `Exception` —
                # the same shape browser_pool uses for its own closes — keeps
                # `CancelledError` (a BaseException) propagating, so a cancelled
                # probe still cancels.
                #
                # The `shield` is load-bearing, do NOT simplify it to a bare
                # `await page.close()`: `close()` is a driver round-trip, so it
                # suspends. Under a cancelled scope (SIGTERM drain, job
                # cancellation) a bare await re-raises `CancelledError` AT that
                # suspension — before Playwright has torn the page down — and the
                # page leaks exactly as it did before this fix. `shield` detaches
                # the close onto its own task, so the teardown runs to completion
                # even though this awaiter is cancelled; the `CancelledError` from
                # the shielded await still propagates (`suppress(Exception)` never
                # catches a `BaseException`), so a cancelled probe never degrades
                # into a normal return.
                with contextlib.suppress(Exception):
                    await asyncio.shield(page.close())
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return probe


async def verify_html(
    html: str,
    *,
    a11y_max_serious: int = 0,
    timeout_ms: int = 30000,
    run_axe: bool = True,
) -> VerificationReport:
    """Verify a self-contained HTML document. Returns a structured report.

    The check runs on the shared browser pool. Any internal failure is reported
    as a HARD finding (fail-safe) so a broken checker can never wave a design
    through.
    """
    findings: list[Finding] = []
    checks: dict[str, Any] = {}

    try:
        probe = await _probe_in_browser(html, timeout_ms=timeout_ms, run_axe=run_axe)
    except Exception as exc:  # noqa: BLE001 — checker failure must fail safe, not pass
        logger.warning("Verification checker error: %s", exc)
        findings.append(Finding("verifier", "hard", f"verification could not run: {exc}"))
        return VerificationReport(status="failed", findings=findings, checks=checks)

    if probe.render_error is not None:
        findings.append(Finding("render", "hard", probe.render_error))
        return VerificationReport(status="failed", findings=findings, checks=checks)

    # Chromium mirrors every failed subresource (a font/image blocked by pod
    # egress, a 404) into the console as ``Failed to load resource: ...`` with
    # type ``error``. Those are network outcomes, not "the page's JS is broken",
    # and the network-failure ones are already reported as ADVISORY via
    # ``requestfailed`` below. Counting them HARD failed every design that
    # referenced an unreachable font, burned both refine passes, then failed the
    # job. They stay visible in ``checks`` but do not gate.
    console_errors, console_resource_errors = _split_console_errors(probe.console_errors)
    page_errors = probe.page_errors
    failed_requests = probe.failed_requests
    axe_result = probe.axe_result

    # --- console / pageerror (HARD) ---
    checks["console_errors"] = console_errors
    checks["console_resource_errors"] = console_resource_errors
    checks["page_errors"] = page_errors
    if page_errors:
        findings.append(
            Finding(
                "console",
                "hard",
                f"{len(page_errors)} uncaught JS error(s)",
                {"errors": page_errors[:10]},
            )
        )
    if console_errors:
        findings.append(
            Finding(
                "console",
                "hard",
                f"{len(console_errors)} console error(s)",
                {"errors": console_errors[:10]},
            )
        )

    # --- failed network requests (ADVISORY) ---
    # A self-contained artifact inlines all its assets, so every outbound request
    # is external (Google Fonts, sandboxed embeds) and out of our control — and
    # would false-fail offline in CI. The render + console checks are the true
    # "it works" signal; failed external fetches are reported, not blocking.
    checks["failed_requests"] = failed_requests
    if failed_requests:
        findings.append(
            Finding(
                "assets",
                "advisory",
                f"{len(failed_requests)} external request(s) failed",
                {"requests": failed_requests[:10]},
            )
        )

    # --- subresource HTTP errors (ADVISORY) ---
    # ``requestfailed`` fires only for network-level failures (``net::ERR_*``).
    # A 4xx/5xx response is a *completed* request, so its only trace is the
    # console echo. Surface those as an advisory finding so a 404 is visible
    # in the report rather than living in ``checks`` alone.
    uncovered = [e for e in console_resource_errors if "net::ERR_" not in e]
    if uncovered:
        findings.append(
            Finding(
                "assets",
                "advisory",
                f"{len(uncovered)} subresource(s) returned an HTTP error",
                {"errors": uncovered[:10]},
            )
        )

    # --- a11y via axe (HARD if serious/critical over threshold) ---
    if run_axe:
        _apply_a11y_checks(axe_result, a11y_max_serious, checks, findings)

    _apply_integrity_checks(html, probe, checks, findings)

    status = "failed" if any(f.severity == "hard" for f in findings) else "passed"
    return VerificationReport(status=status, findings=findings, checks=checks)


def _apply_integrity_checks(
    html: str, probe: _BrowserProbe, checks: dict[str, object], findings: list[Finding]
) -> None:
    """Advisory page-integrity findings: dead links, broken images, brand/title."""
    # --- link integrity (ADVISORY) ---
    # ``placeholder_links`` (the ``#``/empty count) is kept for existing readers;
    # ``dead_links`` is the full list — it also catches a ``#id`` with no such id
    # and a site-relative path, both dead inside a single self-contained page.
    placeholder_links = _placeholder_links(html)
    checks["placeholder_links"] = placeholder_links
    dead_links = _dead_links(html)
    checks["dead_links"] = dead_links[:20]
    if dead_links:
        findings.append(
            Finding(
                "links",
                "advisory",
                f"{len(dead_links)} dead link(s) (# / empty / missing anchor / relative path)",
                {"links": dead_links[:10]},
            )
        )

    # --- rendered image integrity (ADVISORY) ---
    checks["broken_images"] = probe.broken_images[:20]
    if probe.broken_images:
        findings.append(
            Finding(
                "images",
                "advisory",
                f"{len(probe.broken_images)} image(s) failed to render",
                {"images": probe.broken_images[:10]},
            )
        )

    # --- brand / title consistency (ADVISORY) ---
    brand_problems = _brand_problems(html)
    checks["brand_consistent"] = not brand_problems
    if brand_problems:
        findings.append(
            Finding("brand", "advisory", "; ".join(brand_problems), {"problems": brand_problems})
        )

    # --- image integrity (ADVISORY) --- the missing half of the check above.
    placeholder_images = _placeholder_images(html)
    checks["placeholder_images"] = placeholder_images
    if placeholder_images:
        findings.append(
            Finding(
                "images",
                "advisory",
                f"{placeholder_images} placeholder image(s) (placehold.co and friends)",
            )
        )


# Prefix Chromium uses for the console echo of a failed subresource load.
_RESOURCE_LOAD_CONSOLE_PREFIX = "Failed to load resource"


def _split_console_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """Split console errors into (hard JS errors, subresource-load echoes)."""
    hard: list[str] = []
    resource: list[str] = []
    for text in errors:
        (resource if text.startswith(_RESOURCE_LOAD_CONSOLE_PREFIX) else hard).append(text)
    return hard, resource


def _apply_a11y_checks(
    axe_result: _AxeResult | None,
    a11y_max_serious: int,
    checks: dict[str, Any],
    findings: list[Finding],
) -> None:
    """Record the a11y verdict, distinguishing UNMEASURED from clean.

    An axe run that produced no result is NOT a clean page. Writing
    ``a11y_serious_violations: 0`` here made a failed checker byte-identical to a
    genuinely accessible document, and ``0 > a11y_max_serious`` is False — so the
    hard gate passed a page nobody ever looked at. Per this module's contract, a
    failure OF THE CHECKER surfaces as a hard finding, never a silent pass.

    The unmeasured state is expressible (``a11y_probed: False``) and the violation
    keys are OMITTED rather than zeroed — an absent key is honest, a 0 is a lie.
    Failing hard is safe: the refine loop that consumes this is bounded by
    ``cfg.verify_max_refine``, so it retries a bounded number of times and then
    fails loudly instead of looping.
    """
    if axe_result is None:
        checks["a11y_probed"] = False
        logger.warning(
            "a11y NOT MEASURED: axe produced no result; serious/critical violations, "
            "violation ids, impact counts and accessibility score are all UNKNOWN "
            "for this document"
        )
        findings.append(
            Finding(
                "a11y",
                "hard",
                "accessibility could not be measured: axe produced no result",
            )
        )
        return

    checks["a11y_probed"] = True
    serious = _axe_blocking(axe_result)
    checks["a11y_serious_violations"] = len(serious)
    checks["a11y_violation_ids"] = [v.get("id") for v in serious]
    # Deterministic 0-100 accessibility score derived from the axe run we already
    # did — no extra LLM call. Surfaced so callers (e.g. web_design_v2, whose LLM
    # a11y scorer is intentionally off) can report a real score instead of null.
    checks["a11y_impact_counts"] = _axe_impact_counts(axe_result)
    a11y_score = axe_accessibility_score(axe_result)
    if a11y_score is not None:
        checks["accessibility_score"] = a11y_score
    if len(serious) > a11y_max_serious:
        findings.append(
            Finding(
                "a11y",
                "hard",
                f"{len(serious)} serious/critical accessibility violation(s) "
                f"(max {a11y_max_serious})",
                {"violations": _axe_violation_details(serious)},
            )
        )


async def _run_axe(page: Any) -> _AxeResult | None:
    """Inject axe-core and run it against the loaded page."""
    try:
        await page.add_script_tag(content=_AXE_PATH.read_text(encoding="utf-8"))
        return await page.evaluate(
            "async () => await axe.run(document, "
            "{resultTypes:['violations'], "
            "runOnly:{type:'tag', values:['wcag2a','wcag2aa']}})"
        )
    except Exception as exc:  # noqa: BLE001 — a11y check is best-effort; absence != pass elsewhere
        logger.warning("axe run failed: %s", exc)
        return None


# The raw axe violation object as it arrives from the browser. Aliased (rather
# than spelled inline in two signatures) so the JSON-boundary `Any` is stated
# ONCE — see tests/test_any_ratchet.py, whose baseline may only ever be lowered.
type _AxeViolation = dict[str, Any]


class _AxeNodeDetail(TypedDict, total=False):
    """One offending element, as carried into the finding and the refine prompt."""

    target: str
    html: str
    failureSummary: str


class _AxeViolationDetail(TypedDict, total=False):
    """One axe rule plus the evidence needed to actually fix it."""

    id: str | None
    impact: str | None
    nodes: list[_AxeNodeDetail]


# Cap the evidence carried into `Finding.detail`. This dict is not just logged —
# `parse_axe_summary` serialises it verbatim into the refine prompt, so it is
# spend as well as diagnostics. Bounds are per-violation to keep one noisy rule
# (e.g. color-contrast on a repeated card) from crowding out the others.
_AXE_MAX_NODES_PER_RULE = 3
_AXE_MAX_HTML_CHARS = 200
_AXE_MAX_SUMMARY_CHARS = 300


def _axe_violation_details(violations: list[_AxeViolation]) -> list[_AxeViolationDetail]:
    """Keep the evidence a fixer actually needs, not just the rule name.

    Before this, both the check list and the hard Finding reduced each axe
    violation to ``v["id"]`` — so a failing run said only "color-contrast" and
    threw away axe's ``nodes`` array, which is where the offending selector, the
    element HTML, and (in ``failureSummary``) the measured contrast ratio and the
    two hex colours live.

    That was not merely undiagnosable. ``parse_axe_summary`` feeds this same
    ``detail`` into the refine loop, so the LLM was being asked to fix a contrast
    violation while being told nothing except the string "color-contrast" — it
    could not know which element or which colours. Measured on redesign job
    01a039f5 (2026-08-25): ``refine_attempts: 2``, both failed, and the stored
    report retained only ``a11y_violation_ids: ["color-contrast"]``.
    """
    details: list[_AxeViolationDetail] = []
    for v in violations[:15]:
        nodes = []
        for node in (v.get("nodes") or [])[:_AXE_MAX_NODES_PER_RULE]:
            target = node.get("target")
            entry: _AxeNodeDetail = {
                # axe's target is a list of selectors (one per frame level).
                "target": target if isinstance(target, str) else ", ".join(map(str, target or [])),
            }
            html = node.get("html")
            if html:
                entry["html"] = str(html)[:_AXE_MAX_HTML_CHARS]
            # failureSummary carries the contrast ratio and the fg/bg colours —
            # the single most actionable string axe produces for color-contrast.
            summary = node.get("failureSummary")
            if summary:
                entry["failureSummary"] = str(summary)[:_AXE_MAX_SUMMARY_CHARS]
            nodes.append(entry)
        detail: _AxeViolationDetail = {"id": v.get("id"), "impact": v.get("impact")}
        if nodes:
            detail["nodes"] = nodes
        details.append(detail)
    return details


def _axe_blocking(axe_result: _AxeResult | None) -> list[_AxeViolation]:
    if not axe_result:
        return []
    return [
        v for v in axe_result.get("violations", []) if v.get("impact") in _BLOCKING_A11Y_IMPACTS
    ]


# Penalty per violated rule, by axe impact level. Weights are ordered so a
# single critical rule dominates a handful of minor ones; a violation with an
# unknown/absent impact is treated as "minor" (least penalising, never zero).
_A11Y_IMPACT_WEIGHTS = {"critical": 25.0, "serious": 10.0, "moderate": 4.0, "minor": 1.0}


def _axe_impact_counts(axe_result: _AxeResult | None) -> dict[str, int]:
    """Count violated rules by axe impact level (unknown impact → 'minor')."""
    counts = {impact: 0 for impact in _A11Y_IMPACT_WEIGHTS}
    if not axe_result:
        return counts
    for v in axe_result.get("violations", []):
        impact = v.get("impact")
        key = impact if impact in _A11Y_IMPACT_WEIGHTS else "minor"
        counts[key] += 1
    return counts


def axe_accessibility_score(axe_result: _AxeResult | None) -> float | None:
    """Deterministic 0-100 accessibility score from an axe run.

    100 = no violations; each violated rule subtracts a weight by impact. Returns
    ``None`` when axe produced no result (absence is not a passing score). Counts
    violated rules (not per-node occurrences) to match how axe reports issues.
    """
    if axe_result is None:
        return None
    penalty = sum(
        _A11Y_IMPACT_WEIGHTS[impact] * count
        for impact, count in _axe_impact_counts(axe_result).items()
    )
    return round(max(0.0, 100.0 - penalty), 1)


@dataclass(frozen=True)
class AxeScore:
    """Outcome of a standalone axe scan: a number, or an explicit reason it has none.

    ``score is None`` always comes with ``unscored_reason`` — a scan that did not
    measure the page must never be read as a clean page (a default is not an
    observation).
    """

    score: float | None
    impact_counts: dict[str, int] = field(default_factory=dict)
    unscored_reason: str | None = None


async def axe_score_html(html: str, *, timeout_ms: int = 30000) -> AxeScore:
    """Deterministic axe score for ``html`` outside the assembly verify lane.

    Freeform/section designs never pass through :func:`verify_html`, and the LLM
    a11y scorer is opt-in, so without this their ``accessibility_score`` was null
    with no reason recorded. Never raises for a checker failure: every failure
    mode returns ``score=None`` plus ``unscored_reason``. ``CancelledError`` (a
    ``BaseException``) still propagates.
    """
    try:
        probe = await _probe_in_browser(html, timeout_ms=timeout_ms, run_axe=True)
    except Exception as exc:  # noqa: BLE001 — checker failure is reported, not raised
        logger.warning("axe scan failed: %s: %s", type(exc).__name__, exc)
        return AxeScore(score=None, unscored_reason=f"axe scan failed ({type(exc).__name__})")
    if probe.render_error:
        return AxeScore(score=None, unscored_reason=f"axe scan failed: {probe.render_error}")
    score = axe_accessibility_score(probe.axe_result)
    if score is None:
        return AxeScore(score=None, unscored_reason="axe scan produced no result")
    return AxeScore(score=score, impact_counts=_axe_impact_counts(probe.axe_result))


def _placeholder_links(html: str) -> int:
    """Count anchors whose href is a placeholder (#, empty). Advisory signal only."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    for a in soup.find_all("a"):
        href = str(a.get("href") or "").strip()
        if href in ("", "#"):
            count += 1
    return count


async def _broken_images(page: Page) -> list[str]:
    """``<img>`` elements that finished loading with no pixels (blocked / 404 / ORB).

    ``complete && naturalWidth === 0`` is the browser's own "this image is broken"
    state; a lazy image not yet requested is ``complete === false`` and is skipped.
    """
    try:
        result = await page.evaluate(
            "() => Array.from(document.images)"
            ".filter(i => i.complete && i.naturalWidth === 0)"
            ".map(i => (i.currentSrc || i.getAttribute('src') || '').slice(0, 200))"
        )
    except Exception as exc:  # noqa: BLE001 — advisory probe; absence is logged, not passed off
        logger.warning("broken-image probe failed: %s", exc)
        return []
    return [str(u) for u in result or []]


def _dead_links(html: str) -> list[dict[str, str]]:
    """Every anchor that goes nowhere inside this self-contained page.

    Dead: missing/empty href, bare ``#``, ``#id`` with no element carrying that id,
    and any site-relative path (``/menu``, ``about.html``) — a single generated
    document has no other pages. Absolute http(s), ``mailto:`` and ``tel:`` are
    taken as live (their reachability is not this check's job).
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return []
    ids = {str(el.get("id")) for el in soup.find_all(id=True)}
    dead: list[dict[str, str]] = []
    for a in soup.find_all("a"):
        href = str(a.get("href") or "").strip()
        low = href.lower()
        if low.startswith(("http://", "https://", "mailto:", "tel:")):
            continue
        if href.startswith("#") and len(href) > 1 and href[1:] in ids:
            continue
        dead.append({"href": href, "text": a.get_text(" ", strip=True)[:60]})
    return dead


# Page-type / scope slugs the caller used to fall back to as a <title>.
_GENERIC_TITLES = frozenset(
    {"", "landing", "hero", "section", "component", "full_page", "redesign", "web design"}
)


def _brand_problems(html: str) -> list[str]:
    """Brand/title inconsistencies: >1 brand spelling, generic or brandless <title>."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return []
    problems: list[str] = []
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if title.lower() in _GENERIC_TITLES:
        problems.append(f"generic <title> {title!r}")
    brands = [el.get_text(" ", strip=True) for el in soup.select("[data-brand]")]
    distinct = sorted({b for b in brands if b})
    if len(distinct) > 1:
        problems.append(f"{len(distinct)} different brand names: {', '.join(distinct)}")
    elif distinct and distinct[0].casefold() not in title.casefold():
        problems.append(f"<title> {title!r} does not name the brand {distinct[0]!r}")
    return problems


# Placeholder-image hosts the prompts name explicitly. `placehold.co` is the one
# this service instructs the model to use; the others are the near-neighbours a
# model substitutes from training data when it decides it needs a filler box.
_PLACEHOLDER_IMAGE_HOSTS = (
    "placehold.co",
    "placeholder.com",
    "placekitten.com",
    "picsum.photos",
    "dummyimage.com",
    "unsplash.it",
)


def _placeholder_images(html: str) -> int:
    """Count <img>/<source> elements pointing at a placeholder-image service.

    Guarded beside `_placeholder_links` because it is the same property — output
    that looks finished but is stuffed with filler. The link half was checked and
    the image half was not, inside the same function, which is exactly the shape
    that let the empty-image-manifest defect ship invisibly: with the manifest
    empty the prompt lost its "never blank out a real image" instruction, the
    model filled a real site's redesign with placehold.co boxes, and verification
    counted the links, saw zero, and passed it.

    Advisory only, matching `_placeholder_links`: greenfield designs are TOLD to
    use placehold.co, so a non-zero count is a signal to surface, not a hard fail.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    for tag in soup.find_all(("img", "source")):
        refs = f"{tag.get('src') or ''} {tag.get('srcset') or ''}".lower()
        if any(host in refs for host in _PLACEHOLDER_IMAGE_HOSTS):
            count += 1
    return count


def report_from_dict(data: dict[str, Any]) -> VerificationReport:
    """Rehydrate a VerificationReport from its persisted dict form."""
    findings = [Finding(**f) for f in data.get("findings", [])]
    return VerificationReport(
        status=str(data.get("status", "failed")),
        findings=findings,
        checks=data.get("checks", {}),
    )


def parse_axe_summary(report: VerificationReport) -> str:
    """One-line summary used to feed the refine loop with fixable findings."""
    return json.dumps(
        [{"check": f.check, "message": f.message, "detail": f.detail} for f in report.hard_findings]
    )
