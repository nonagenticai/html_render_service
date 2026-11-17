"""Status/shape guards for browser navigations at a REMOTE, tenant-supplied URL.

``Page.goto`` returns a ``Response``, and every remote call site in this service
used to throw it away. Chromium renders an error document exactly as happily as
it renders the real site, so a 404/403/500 — or a Cloudflare "Just a moment…"
interstitial served with a cheerful 200 — arrived downstream as a perfectly
ordinary, non-empty string. Nothing between the fetch and the model could tell
the difference:

* ``_phase_resolve_html`` cached it under the *good* HTML key for 3600s, so one
  blip poisoned every job for that URL for the next hour;
* ``generate_full_page`` saw a non-empty ``html`` and set ``greenfield=False``,
  so the model dutifully *redesigned the error page*;
* the job then reported ``completed``, with screenshots and a PDF of it.

The failure was silent at every layer because the gap was not expressible: there
was no value meaning "this fetch did not produce the site". These helpers make it
expressible (a typed error carrying status + reason) and loud (the job fails
naming the status, instead of billing the tenant for a redesigned 404).

Both helpers are deliberately kept in one module so the AST property test
``test_every_remote_navigation_checks_response_status`` has a single symbol to
enumerate against — see that test for the "a new sink cannot ship unchecked"
argument.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "NavigationResponse",
    "PageFetchError",
    "assert_not_challenge",
    "require_ok_navigation",
]


class NavigationResponse(Protocol):
    """The one member of ``playwright.async_api.Response`` this module reads.

    A structural type rather than the real class on purpose: naming
    ``playwright.async_api.Response`` at import time drags the multi-second,
    browser-bindings-heavy ``playwright.async_api`` import into modules the API
    process loads eagerly (see ``capture_failure_types`` for the same trade-off,
    made for the same reason). A Protocol keeps the signature honest — ``status``
    is an ``int``, which is exactly the assumption the comparison below rests on
    — without paying for the import or falling back to ``Any``.
    """

    @property
    def status(self) -> int: ...


# ── The non-OK rule ──────────────────────────────────────────────────────────
# We fail on ``status >= 400``, NOT on ``status != 200``. Those are very
# different rules and the difference is deliberate:
#
#   * 2xx is the ordinary success family — 200, but also 203 (transformed by a
#     proxy) and 206. All carry real content.
#   * 3xx never reaches us as a *final* response for an ordinary redirect —
#     Chromium follows those and ``Response.status`` reports the landed
#     document. A 3xx that survives to here is a redirect the browser chose not
#     to follow, and the body it did render is still what the user would see.
#   * 4xx/5xx is the line where the body stops being the site and starts being
#     the server's apology for the site. That is the entire defect: an apology
#     page is not something a tenant paid to have redesigned.
#
# Failing on every non-200 would break real sites that legitimately serve
# content outside 200, so the ceiling is 400 rather than an equality check.
_NOT_CONTENT_STATUS = 400


class PageFetchError(RuntimeError):
    """A remote navigation did not yield the tenant's page.

    Subclasses ``RuntimeError`` on purpose: ``RedesignService.process_job``
    already catches ``(OSError, ValueError, RuntimeError, TimeoutError)`` and
    routes it through ``_phase_finalize_failed``, so raising this fails the job
    *with the reason recorded* rather than escaping as a 500. It is the same
    contract ``SSRFError`` uses by subclassing ``ValueError``.

    ``reason`` is a stable, greppable slug (``http_status`` /
    ``challenge_interstitial``) so the failure class can be counted in logs
    without parsing prose. ``status`` is ``None`` for a shape failure, where the
    HTTP status was fine and the *body* was the problem.
    """

    def __init__(self, url: str, reason: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.reason = reason
        self.status = status


def require_ok_navigation(response: NavigationResponse | None, url: str) -> None:
    """Raise ``PageFetchError`` when *response* is not the tenant's page.

    ``response`` is whatever ``Page.goto`` returned; see ``NavigationResponse``
    for why it is a structural type rather than Playwright's own class.

    A ``None`` response is NOT treated as a failure. Playwright returns ``None``
    for a navigation that produced no response of its own (a same-document
    fragment nav, or a redirect to a document already in the cache); the page
    still rendered. Failing there would break real fetches to buy nothing, so
    the unmeasurable case is logged as unmeasurable rather than guessed either
    way — the point of this module is that a gap is *stated*, not that it is
    always fatal.
    """
    if response is None:
        logger.info(
            "navigation to %s produced no Response object; status unverified (same-document "
            "or cached navigation)",
            url,
        )
        return
    status = response.status
    if status >= _NOT_CONTENT_STATUS:
        raise PageFetchError(
            url=url,
            reason="http_status",
            status=status,
            message=(
                f"fetch of {url} returned HTTP {status} — the rendered document is an error "
                "page, not the site. Refusing to treat it as page content."
            ),
        )
    logger.debug("navigation to %s returned HTTP %s", url, status)


# ── Interstitial / challenge detection ───────────────────────────────────────
# Two tiers, because precision matters more than reach here: a false positive
# hard-fails a paid job for a site that merely *mentions* Cloudflare.
#
# Tier 1 — infrastructure tokens that cannot plausibly occur in ordinary prose.
# These are script ids, endpoint paths and vendor resource names emitted by the
# challenge page itself, so they are matched anywhere in the document.
_CHALLENGE_TOKENS = (
    "cf-browser-verification",
    "cf_chl_opt",
    "/cdn-cgi/challenge-platform",
    "_incapsula_resource",
    "distil_r_captcha",
    "__cf_chl_rt_tk",
)

# Tier 2 — human-readable banners. These ARE plausible prose ("Just a moment"
# is a normal English phrase), so they are matched ONLY inside <title>, which an
# article about bot protection would not set to the banner text.
_CHALLENGE_TITLES = (
    "just a moment",
    "checking your browser",
    "attention required",
    "access denied",
    "are you a robot",
    "one moment, please",
)

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def assert_not_challenge(html: str, url: str) -> None:
    """Raise ``PageFetchError`` when *html* is a bot-challenge interstitial.

    A challenge page is served with HTTP 200 — it is a successful fetch of the
    wrong document — so ``require_ok_navigation`` cannot see it. It is also
    short, generic and non-empty, which is precisely the shape that defeated
    every downstream emptiness check.

    Detection is intentionally conservative; see ``_CHALLENGE_TOKENS`` /
    ``_CHALLENGE_TITLES`` for why the two tiers are scoped differently. When in
    doubt this returns cleanly: a missed interstitial is the status quo, whereas
    a false positive fails a job that would otherwise have succeeded.
    """
    if not html:
        return
    lowered = html.lower()
    for token in _CHALLENGE_TOKENS:
        if token in lowered:
            raise PageFetchError(
                url=url,
                reason="challenge_interstitial",
                message=(
                    f"fetch of {url} returned a bot-challenge interstitial (matched "
                    f"{token!r}), not the site. Refusing to treat it as page content."
                ),
            )
    match = _TITLE_RE.search(html)
    if not match:
        return
    title = " ".join(match.group(1).split()).lower()
    for banner in _CHALLENGE_TITLES:
        if banner in title:
            raise PageFetchError(
                url=url,
                reason="challenge_interstitial",
                message=(
                    f"fetch of {url} returned a bot-challenge interstitial (title {title!r}), "
                    "not the site. Refusing to treat it as page content."
                ),
            )
