"""A remote fetch that did not return the site must never pass as page content.

``Page.goto`` returns a ``Response`` and every remote call site discarded it, so
a 404/403/500 body — or a Cloudflare interstitial served with a 200 — arrived
downstream as an ordinary non-empty string, was cached under the good HTML key
for 3600s, made ``generate_full_page`` pick the refine path
(``greenfield=False``), and got redesigned as though it were the tenant's site.
The job then reported ``completed``.

These tests drive an actually-non-OK response. A fake that always answers 200
leaves the fix unexercised, so every negative case here sets a real failing
status or a real interstitial body.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.navigation_guard import (
    PageFetchError,
    assert_not_challenge,
    require_ok_navigation,
)


def _response(status: int) -> MagicMock:
    """A minimal stand-in for ``playwright.async_api.Response``."""
    resp = MagicMock()
    resp.status = status
    return resp


def _fake_pool(status: int, html: str = "<html><body>real site</body></html>") -> AsyncMock:
    """A browser-pool factory whose single page answers *status* with *html*."""
    page = MagicMock()
    page.goto = AsyncMock(return_value=_response(status))
    page.content = AsyncMock(return_value=html)
    page.close = AsyncMock()

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)

    @asynccontextmanager
    async def _lease() -> AsyncIterator[MagicMock]:
        yield browser

    pool = MagicMock()
    pool.lease = _lease
    # `_fetch_page_html` and `capture_url_screenshots` both take the PINNED
    # lease for a resolvable public host (see
    # `test_fetch_page_html_leases_pinned.py`), so a fake that only models
    # `lease` would hand the sink a bare MagicMock. Model both; they yield the
    # same browser because the pin is a resolver rule, not a different page.
    pool.lease_pinned = lambda host, ip: _lease()
    return AsyncMock(return_value=pool)


# --------------------------------------------------------------------------- #
# The rule: >= 400 is not content. NOT "!= 200".
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 429, 500, 502, 503])
async def test_require_ok_navigation_rejects_error_statuses(status: int) -> None:
    """Every 4xx/5xx is refused, and the failure names the status."""
    with pytest.raises(PageFetchError) as exc:
        require_ok_navigation(_response(status), "https://example.com/")
    assert exc.value.status == status
    assert exc.value.reason == "http_status"
    assert f"HTTP {status}" in str(exc.value)


@pytest.mark.parametrize("status", [200, 203, 204, 206, 301, 302, 304, 399])
async def test_require_ok_navigation_allows_non_200_content_statuses(status: int) -> None:
    """The rule is ``>= 400``, not ``!= 200``.

    This is the deliberate half of the decision: failing on every non-200 would
    break sites that legitimately serve content outside 200 (a proxy-transformed
    203, a 206, a redirect the browser chose not to follow). If someone later
    "tightens" this to an equality check, these cases go red.
    """
    assert require_ok_navigation(_response(status), "https://example.com/") is None


async def test_require_ok_navigation_tolerates_a_missing_response() -> None:
    """``None`` is unmeasurable, not failed — Playwright returns it for a
    same-document or cached navigation, where the page still rendered."""
    assert require_ok_navigation(None, "https://example.com/") is None


# --------------------------------------------------------------------------- #
# Interstitials: a 200 that is not the site
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "html",
    [
        '<html><head><script src="/cdn-cgi/challenge-platform/h/b/orchestrate"></script></head></html>',
        '<html><body><div id="cf-browser-verification"></div></body></html>',
        "<html><head><title>Just a moment...</title></head><body></body></html>",
        "<html><head><title>Attention Required! | Cloudflare</title></head></html>",
        "<html><head><title>Checking your browser before accessing</title></head></html>",
    ],
)
async def test_assert_not_challenge_rejects_interstitials(html: str) -> None:
    """A challenge page is a successful fetch of the WRONG document."""
    with pytest.raises(PageFetchError) as exc:
        assert_not_challenge(html, "https://example.com/")
    assert exc.value.reason == "challenge_interstitial"


@pytest.mark.parametrize(
    "html",
    [
        "<html><head><title>Our Pricing</title></head><body><h1>Plans</h1></body></html>",
        # Prose ABOUT bot protection must not trip the tier-2 banners: they are
        # matched only inside <title>, never in body text.
        "<html><head><title>Blog</title></head><body>"
        "<p>Just a moment, checking your browser — we explain how Cloudflare works.</p>"
        "</body></html>",
        "",
    ],
)
async def test_assert_not_challenge_allows_real_pages(html: str) -> None:
    """Precision control: a false positive hard-fails a job that would succeed."""
    assert assert_not_challenge(html, "https://example.com/") is None


# --------------------------------------------------------------------------- #
# End to end through the real fetch helper
# --------------------------------------------------------------------------- #


async def test_fetch_page_html_raises_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defect, at its source: a 404 body must not be returned as the site."""
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", _fake_pool(404))
    helpers = RedesignHelpersMixin.__new__(RedesignHelpersMixin)

    with pytest.raises(PageFetchError) as exc:
        await helpers._fetch_page_html("https://example.com/gone")
    assert exc.value.status == 404


async def test_fetch_page_html_raises_on_interstitial(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200-with-a-challenge-body is refused too — status alone cannot see it."""
    monkeypatch.setattr(
        "src.services.browser_pool.get_browser_pool",
        _fake_pool(200, '<html><body><div id="cf-browser-verification"></div></body></html>'),
    )
    helpers = RedesignHelpersMixin.__new__(RedesignHelpersMixin)

    with pytest.raises(PageFetchError) as exc:
        await helpers._fetch_page_html("https://example.com/")
    assert exc.value.reason == "challenge_interstitial"


async def test_fetch_page_html_returns_html_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the guard does not break the ordinary path."""
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", _fake_pool(200))
    helpers = RedesignHelpersMixin.__new__(RedesignHelpersMixin)

    assert "real site" in await helpers._fetch_page_html("https://example.com/")


# --------------------------------------------------------------------------- #
# The cache is the amplifier — a bad fetch must never reach the good key
# --------------------------------------------------------------------------- #


class _RecordingCache:
    """Records which cache methods a resolve attempt reached."""

    def __init__(self, good: str | None = None, failed: str | None = None) -> None:
        self._good = good
        self._failed = failed
        self.cached_good: list[tuple[str, str]] = []
        self.cached_failed: list[tuple[str, str]] = []

    async def get_cached_html(self, url: str) -> str | None:
        return self._good

    async def get_failed_fetch(self, url: str) -> str | None:
        return self._failed

    async def cache_html(self, url: str, html: str) -> bool:
        self.cached_good.append((url, html))
        return True

    async def cache_failed_fetch(self, url: str, reason: str) -> bool:
        self.cached_failed.append((url, reason))
        return True


def _job(url: str) -> MagicMock:
    job = MagicMock()
    job.html_content = None
    job.url = url
    return job


async def test_failed_fetch_is_never_written_to_the_good_html_key() -> None:
    """The 3600s amplifier: an error document must not be pinned as the site.

    A 90s cache once turned one blip into 18x the damage; this key's TTL is
    3600s. The error body must land in the short-TTL failure keyspace instead.
    """
    from src.services.design_builder.phases import RedesignPhasesMixin

    cache = _RecordingCache()
    phases = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    phases._cache = cache
    phases._fetch_page_html = AsyncMock(
        side_effect=PageFetchError("https://example.com/", "http_status", "HTTP 500", 500)
    )

    with pytest.raises(PageFetchError):
        await phases._phase_resolve_html(_job("https://example.com/"))

    assert cache.cached_good == [], "an error document was cached under the GOOD html key"
    assert len(cache.cached_failed) == 1
    assert "HTTP 500" in cache.cached_failed[0][1]


async def test_a_recently_failed_url_fails_fast_without_refetching() -> None:
    """The negative cache short-circuits to a failure, never to content."""
    from src.services.design_builder.phases import RedesignPhasesMixin

    cache = _RecordingCache(failed="HTTP 404 — the rendered document is an error page")
    phases = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    phases._cache = cache
    phases._fetch_page_html = AsyncMock()

    with pytest.raises(PageFetchError) as exc:
        await phases._phase_resolve_html(_job("https://example.com/"))

    assert "HTTP 404" in str(exc.value)
    phases._fetch_page_html.assert_not_awaited()


async def test_a_good_cached_body_wins_over_a_recorded_failure() -> None:
    """Read ORDER is load bearing: a transient 503 must not shadow good HTML.

    Same property the two-tier split in
    ``modality_gateway/src/services/resolve_cache.py`` keeps structurally
    impossible rather than conditional.
    """
    from src.services.design_builder.phases import RedesignPhasesMixin

    cache = _RecordingCache(good="<html>the real site</html>", failed="HTTP 503")
    phases = RedesignPhasesMixin.__new__(RedesignPhasesMixin)
    phases._cache = cache
    phases._fetch_page_html = AsyncMock()

    assert await phases._phase_resolve_html(_job("https://example.com/")) == (
        "<html>the real site</html>"
    )
    phases._fetch_page_html.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Durability: no NEW remote navigation can ship without a status check
# --------------------------------------------------------------------------- #


def _remote_navigation_sinks() -> list[tuple[str, int, bool]]:
    """Every ``page.goto`` in ``src/`` that navigates at a REMOTE, tenant URL.

    Returns ``(relpath, goto_lineno, checked)``.

    Same shape, and the same argument, as
    ``_local_document_sinks`` in ``test_pdf_render_subresource_ssrf.py``: match
    the PROPERTY over the whole tree rather than the one spelling one author
    happened to use in one file. The property here is "a browser navigated at a
    caller-supplied remote URL checks what the server actually returned".

    A remote sink is identified by its enclosing function calling
    ``validate_public_url`` — that guard exists precisely and only for
    tenant-supplied http(s) targets, so it is an exact marker and needs no
    URL-shape heuristics. ``file://`` navigations (the PDF/verifier render paths)
    have no HTTP status and are correctly excluded by that same marker.
    """
    src_root = pathlib.Path(__file__).resolve().parents[2] / "src"
    sinks: list[tuple[str, int, bool]] = []

    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)]
            names = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
            if "validate_public_url" not in names:
                continue
            checks_status = "require_ok_navigation" in names
            for node in calls:
                fn = node.func
                if not (isinstance(fn, ast.Attribute) and fn.attr == "goto"):
                    continue
                arg_src = ast.unparse(node.args[0]) if node.args else ""
                # A local-document navigation inside a remote function (none
                # today) would have no status to check — exclude it explicitly.
                if "file://" in arg_src or "as_uri" in arg_src:
                    continue
                sinks.append((str(path.relative_to(src_root)), node.lineno, checks_status))
    return sinks


def test_every_remote_navigation_checks_response_status() -> None:
    """Fleet property: no browser navigates a remote URL without checking status.

    MUTATION KILLED: deleting the ``require_ok_navigation(response, url)`` call
    from EITHER sink — ``design_builder/helpers.py:_fetch_page_html`` or
    ``screenshot_renderer.py:capture_url_screenshots`` — lists that sink in the
    failure message. A third remote sink added later starts life listed here.
    """
    sinks = _remote_navigation_sinks()
    # Vacuity floor: 1 in design_builder/helpers.py (the HTML the model works
    # from) + 1 in screenshot_renderer.py (the "before" screenshot). A rename
    # that makes this enumeration find nothing fails here rather than passing
    # silently on an empty list.
    assert len(sinks) >= 2, f"expected >=2 remote navigation sinks, found {sinks}"
    unchecked = [(p, ln) for p, ln, ok in sinks if not ok]
    assert not unchecked, (
        f"remote navigation sink(s) {unchecked} render whatever the server returned "
        "without calling require_ok_navigation — a 404/403/500 body would be treated "
        "as the tenant's page"
    )
