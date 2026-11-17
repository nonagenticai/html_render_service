"""SSRF guard on the *render target* — the tenant-supplied URL this service
navigates to with headless Chromium.

WHY
---
``RedesignRequest.url`` / ``BatchPageInput.url`` / ``EmbedSpec.artifact_url`` are
tenant-controlled, and they end up as the argument to ``Page.goto`` inside the
cluster (``_fetch_page_html`` reads the DOM the LLM works from;
``capture_url_screenshots`` takes the before/after "original"; the assembler
mounts embeds as iframes in a page our own Chromium then rasterises). The
resulting HTML and PNGs are handed straight back to the tenant. An unguarded
value is therefore not "a bad request" — it is a **readable internal-network
oracle**, driven by a full browser engine, against every address this pod can
reach.

The address space that matters is easy to get wrong. RFC 6598
``100.64.0.0/10`` is this fleet's Tailscale tailnet — prod Postgres
``100.64.0.5``, the Docker registry ``100.64.0.6``, the k3s control plane
``100.64.0.7`` — and::

    >>> ipaddress.ip_address("100.64.0.5").is_private
    False

so the obvious ``is_private`` check, and every hand-rolled RFC1918 denylist,
waves the whole tailnet through. PR #64 pinned only the *webhook* egress path
(``webhook_dispatcher``); the render path stayed open. This module is the
regression proof for closing it, and it is written to fail against the unfixed
code.

WHAT EACH SECTION LOCKS DOWN
----------------------------
* **Admission (schema).** Blocked targets are refused at the API boundary as a
  422 — literal tailnet IP, the IPv4-mapped IPv6 spelling of that same address,
  cloud metadata, an ``*.internal`` name, loopback, and a public-*looking*
  hostname whose DNS answer is a tailnet address (the case a literal-IP check
  structurally cannot catch).
* **The deny is the /10, not a prefix string.** ``100.63.255.255`` and
  ``100.128.0.0`` bracket the range and must stay ALLOWED, which is what
  distinguishes a correct ``100.64.0.0/10`` check from ``startswith("100.")``.
  ``203.0.113.10`` (TEST-NET-3) must also stay allowed — the fleet's own test
  fixtures depend on it.
* **Empty stays legal.** ``url`` defaults to ``""`` and is genuinely optional
  for ``scope='landing'`` / ``mode='assembly'``; guarding it must not turn a
  greenfield brief into a 422.
* **The sinks re-validate.** ``url`` is read back out of the
  ``redesign_jobs.url`` column at render time, so a row written before the
  validator existed still reaches Chromium. The sink tests assert the strong
  property: ``goto`` is **never awaited**. Not "it raised" — *never fetched*.
* **A blocked embed degrades, it does not detonate.** One poisoned
  ``artifact_url`` is dropped and the rest of the design still assembles.
* **A block is not a shrug.** ``SSRFError`` subclasses ``ValueError``, so
  ``_phase_screenshots``' pre-existing ``except (OSError, ValueError, ...)``
  would have swallowed a security block as an indistinguishable
  "capture failed" warning. The explicit clause must log it at ERROR with the
  job id.

Hermetic: DNS comes from the autouse stub in ``tests/unit/conftest.py`` (see
that module for the patch target and the reasoning behind the map); no network,
no resolver, no browser.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from w2v_common.net import SSRFError

import src.services.design_builder as design_builder
from src.api.schemas import BatchPageInput, EmbedSpec, RedesignRequest
from src.config import get_settings
from src.services.catalog.assembler import assemble_page
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.design_builder.types import RenderOptions
from src.services.screenshot_renderer import capture_url_screenshots
from tests.unit.conftest import INTERNAL_REBIND_HOST

# --------------------------------------------------------------------------- #
# Fixtures shared by the whole module
# --------------------------------------------------------------------------- #

#: Targets that must never be navigated to. Each entry is a distinct bypass
#: shape, not a variation on one.
BLOCKED_TARGETS: list[str] = [
    "http://100.64.0.5/",  # tailnet literal — prod postgres; is_private is False
    "http://[::ffff:100.64.0.5]/",  # same host, IPv4-mapped IPv6 spelling
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata link-local
    "http://registry.internal/v2/_catalog",  # estate name -> tailnet address
    "http://localhost:8080/",  # loopback: the pod's own sidecars
    f"http://{INTERNAL_REBIND_HOST}/",  # public-looking name, tailnet answer
]

#: Targets that must keep working. The first two bracket ``100.64.0.0/10`` and
#: are the controls that prove the deny is the CIDR and not ``"100."``.
ALLOWED_TARGETS: list[str] = [
    "http://100.63.255.255/",  # one below the /10
    "http://100.128.0.0/",  # one above the /10
    "http://8.8.8.8/",  # ordinary public literal
    "http://203.0.113.10/",  # TEST-NET-3 — the fleet suite depends on this
    "https://example.com/",  # ordinary public hostname
]


def _fake_browser_pool() -> tuple[AsyncMock, MagicMock]:
    """Build a browser-pool factory mock and expose the page it would hand out.

    Returns:
        ``(get_browser_pool, page)``. ``get_browser_pool`` is an ``AsyncMock``
        suitable for patching ``src.services.browser_pool.get_browser_pool``;
        ``page`` is the ``MagicMock`` whose ``goto``/``screenshot`` are the
        never-should-be-awaited assertions.
    """
    page = MagicMock()
    # `goto` must return a Response with an INT status: `require_ok_navigation`
    # compares it against 400, and a bare AsyncMock return value raises TypeError
    # there. A real Playwright Response always carries an int, so modelling 200
    # here keeps the fake honest rather than papering over the guard.
    _ok_response = MagicMock()
    _ok_response.status = 200
    page.goto = AsyncMock(return_value=_ok_response)
    page.content = AsyncMock(return_value="<html>leaked</html>")
    page.close = AsyncMock()
    page.screenshot = AsyncMock()
    page.set_viewport_size = AsyncMock()

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
    return AsyncMock(return_value=pool), page


# --------------------------------------------------------------------------- #
# Admission: RedesignRequest.url
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", BLOCKED_TARGETS)
def test_redesign_request_rejects_internal_render_target(url: str) -> None:
    """Every internal-target shape is refused at the API boundary (422).

    Args:
        url: A render target that must never reach Chromium.
    """
    with pytest.raises(ValidationError) as exc:
        RedesignRequest(url=url)

    # Assert the *guard* rejected it, not some unrelated field constraint —
    # otherwise this test would still pass if the validator were deleted and a
    # max_length rule happened to fire.
    assert "not a valid public http(s) URL" in str(exc.value)


@pytest.mark.parametrize("url", ALLOWED_TARGETS)
def test_redesign_request_allows_public_render_target(url: str) -> None:
    """Public targets — including both /10 boundary neighbours and TEST-NET-3 —
    are untouched.

    These are the controls that make the blocked cases meaningful: they prove
    the guard denies the ``100.64.0.0/10`` CIDR rather than any address whose
    text starts with ``100.``, and that the fleet's TEST-NET fixtures survive.

    Args:
        url: A render target that must keep working.
    """
    assert RedesignRequest(url=url).url == url


@pytest.mark.parametrize("url", ["", "   "])
def test_redesign_request_allows_empty_render_target(url: str) -> None:
    """An empty/whitespace ``url`` is legal and must pass through untouched.

    ``scope='landing'`` (and ``mode='assembly'``) generate from a brief with no
    page to fetch, and ``_phase_resolve_html`` short-circuits on a falsy url for
    exactly that reason. The guard must not resolve ``""`` and must not rewrite
    it — a regression here turns every greenfield design into a 422.

    Args:
        url: The empty or whitespace-only value under test.
    """
    req = RedesignRequest(url=url, scope="landing")
    assert req.url == url  # preserved verbatim, not stripped


def test_redesign_request_rejects_non_http_scheme() -> None:
    """A non-http(s) scheme is refused before any resolution is attempted."""
    with pytest.raises(ValidationError) as exc:
        RedesignRequest(url="file:///etc/passwd")
    assert "not a valid public http(s) URL" in str(exc.value)


# --------------------------------------------------------------------------- #
# Admission: EmbedSpec.artifact_url and BatchPageInput.url
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", BLOCKED_TARGETS)
def test_embed_spec_rejects_internal_artifact_url(url: str) -> None:
    """The embed seam gets the same treatment as the render target.

    The old check was ``startswith("http")`` — a *scheme* check, not an SSRF
    guard: ``http://100.64.0.5:5432/`` sailed through it, and the embed is
    mounted as an iframe in a page our own Chromium rasterises, so the target's
    pixels come back to the tenant.

    Args:
        url: An artifact URL that must be refused.
    """
    with pytest.raises(ValidationError) as exc:
        EmbedSpec(artifact_url=url)
    assert "not a valid public http(s) URL" in str(exc.value)


@pytest.mark.parametrize("url", ALLOWED_TARGETS)
def test_embed_spec_allows_public_artifact_url(url: str) -> None:
    """A public artifact URL still constructs, with defaults intact.

    Args:
        url: An artifact URL that must keep working.
    """
    spec = EmbedSpec(artifact_url=url)
    assert spec.artifact_url == url
    assert spec.sandbox is True


@pytest.mark.parametrize("url", BLOCKED_TARGETS)
def test_batch_page_input_rejects_internal_url(url: str) -> None:
    """The batch route is guarded too, or it is a scriptable bypass.

    Each ``BatchPageInput`` becomes its own job row and reaches the same
    Chromium sinks; ``BatchRedesignRequest`` accepts up to 20 of them, so an
    unguarded field here is a 20-target-per-request internal port scanner.

    Args:
        url: A page URL that must be refused.
    """
    with pytest.raises(ValidationError) as exc:
        BatchPageInput(url=url)
    assert "not a valid public http(s) URL" in str(exc.value)


@pytest.mark.parametrize("url", ALLOWED_TARGETS)
def test_batch_page_input_allows_public_url(url: str) -> None:
    """A public batch page URL is untouched.

    Args:
        url: A page URL that must keep working.
    """
    assert BatchPageInput(url=url).url == url


# --------------------------------------------------------------------------- #
# Sinks: the browser must never be pointed at the target at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", BLOCKED_TARGETS)
async def test_fetch_page_html_never_navigates_to_blocked_target(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_fetch_page_html`` refuses **before** Playwright is touched.

    This is the load-bearing assertion of the whole module. "It raised" is not
    good enough: if the guard ran after ``page.goto`` the request would already
    have been made and the internal service already probed. So the proof is
    that ``goto`` was never awaited — and, tighter still, that a browser was
    never even leased from the pool.

    Args:
        url: A render target that must never be navigated to.
        monkeypatch: Replaces the browser-pool factory.
    """
    get_pool, page = _fake_browser_pool()
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", get_pool)

    helpers = RedesignHelpersMixin.__new__(RedesignHelpersMixin)

    with pytest.raises(SSRFError):
        await helpers._fetch_page_html(url)

    page.goto.assert_not_awaited()  # never fetched
    get_pool.assert_not_awaited()  # never even leased a browser


async def test_fetch_page_html_does_navigate_to_public_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control for the test above.

    Without this, ``goto.assert_not_awaited()`` would still pass if
    ``_fetch_page_html`` were broken in some unrelated way, and the guard test
    would be proving nothing.

    Args:
        monkeypatch: Replaces the browser-pool factory.
    """
    get_pool, page = _fake_browser_pool()
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", get_pool)

    helpers = RedesignHelpersMixin.__new__(RedesignHelpersMixin)
    html = await helpers._fetch_page_html("https://example.com/")

    assert html == "<html>leaked</html>"
    page.goto.assert_awaited_once()
    assert page.goto.await_args is not None
    assert page.goto.await_args.args[0] == "https://example.com/"


@pytest.mark.parametrize("url", BLOCKED_TARGETS)
async def test_capture_url_screenshots_never_navigates_to_blocked_target(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``capture_url_screenshots`` refuses before Playwright *and* before I/O.

    Same "never fetched" property as ``_fetch_page_html``, plus a cheap
    ordering proof: the guard sits above ``output_dir.mkdir()``, so a refused
    capture leaves no directory behind. If the validation were bolted on lower
    down, this directory would exist.

    Args:
        url: A render target that must never be navigated to.
        tmp_path: Parent for the (never-to-be-created) output directory.
        monkeypatch: Replaces the browser-pool factory.
    """
    get_pool, page = _fake_browser_pool()
    monkeypatch.setattr("src.services.browser_pool.get_browser_pool", get_pool)
    out_dir = tmp_path / "original_screenshots"

    with pytest.raises(SSRFError):
        await capture_url_screenshots(url, out_dir, ["desktop"])

    page.goto.assert_not_awaited()  # never fetched
    page.screenshot.assert_not_awaited()  # nothing rasterised for the tenant
    get_pool.assert_not_awaited()
    assert not out_dir.exists()  # guard runs before mkdir


# --------------------------------------------------------------------------- #
# Assembler sink: a poisoned embed is dropped, not fatal
# --------------------------------------------------------------------------- #


def test_assembler_drops_blocked_embed_and_still_assembles() -> None:
    """A blocked ``artifact_url`` is skipped; the surrounding design survives.

    ``embeds`` is read back from the ``redesign_jobs.embeds`` JSONB column at
    render time, so schema validation does not cover this call — a row written
    before the validator existed still reaches the assembler. Dropping is the
    right failure mode: one bad artifact must not fail an otherwise-good design.
    The count must reflect the drop, and the internal URL must not appear
    anywhere in the emitted document (not even inertly, in a comment or a
    ``title``).
    """
    good_a = "https://good-a.example/a.html"
    good_b = "https://good-b.example/b.html"
    poisoned = "http://100.64.0.5/oracle.html"
    res = assemble_page(
        [{"type": "hero", "content": {"headline": "hi"}}],
        embeds=[
            {"type": "threejs", "artifact_url": good_a, "mount": "hero"},
            {"type": "threejs", "artifact_url": poisoned, "mount": "hero"},
            {"type": "threejs", "artifact_url": good_b, "mount": "hero"},
        ],
    )

    assert res.embeds_injected == 2  # the count reflects the drop
    assert good_a in res.html
    assert good_b in res.html
    assert "100.64.0.5" not in res.html
    assert "oracle.html" not in res.html


def test_assembler_drops_embed_whose_hostname_resolves_internal() -> None:
    """The drop is driven by *resolution*, not by the literal text of the URL.

    ``INTERNAL_REBIND_HOST`` looks like an ordinary public CDN name; only its
    DNS answer betrays it. A literal-IP denylist in the assembler would inject
    this embed.
    """
    res = assemble_page(
        [{"type": "hero", "content": {"headline": "hi"}}],
        embeds=[{"type": "threejs", "artifact_url": f"https://{INTERNAL_REBIND_HOST}/s.html"}],
    )

    assert res.embeds_injected == 0
    assert INTERNAL_REBIND_HOST not in res.html


# --------------------------------------------------------------------------- #
# _phase_screenshots: a block is logged as a block, not as a shrug
# --------------------------------------------------------------------------- #


def _finalize(tmp_path: Path) -> RedesignFinalizeMixin:
    """Build a bare ``RedesignFinalizeMixin`` with a stubbed storage service.

    Args:
        tmp_path: Directory handed back as the job dir.

    Returns:
        A mixin instance ready for ``_phase_screenshots``.
    """
    obj = RedesignFinalizeMixin.__new__(RedesignFinalizeMixin)
    storage = MagicMock()
    storage.get_job_dir.return_value = tmp_path
    storage.retrieve_text = AsyncMock(return_value="<html><body>hi</body></html>")
    storage.persist_paths = AsyncMock(side_effect=lambda paths: paths)
    obj._storage = storage  # type: ignore[attr-defined]
    return obj


async def test_phase_screenshots_logs_ssrf_block_distinctly_from_a_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An ``SSRFError`` is handled by the explicit clause, not the generic tuple.

    ``SSRFError`` subclasses ``ValueError``, so the pre-existing
    ``except (OSError, ValueError, RuntimeError, TimeoutError)`` would have
    caught it and logged ``"Failed to capture original screenshots"`` at
    WARNING — a security block rendered indistinguishable from a flaky page
    load, in the one log line an operator would use to tell them apart. The
    observable difference this locks down: ERROR level, the job id, and the
    refusal wording.

    The block is deliberately non-fatal (this is the optional before/after
    original), so the phase must still return with the redesigned-page
    screenshots and no original.

    Args:
        tmp_path: Job directory.
        monkeypatch: Replaces the package-level renderer functions.
        caplog: Captures the emitted log records.
    """
    blocked_url = "http://100.64.0.5/"
    cap = AsyncMock(side_effect=SSRFError(f"URL {blocked_url} resolves to blocked address"))
    rend = AsyncMock(return_value={"desktop": str(tmp_path / "redesign.png")})
    comp = AsyncMock(return_value=({"desktop": "c.png"}, {"desktop": "d.png"}))
    monkeypatch.setattr(design_builder, "capture_url_screenshots", cap)
    monkeypatch.setattr(design_builder, "render_from_html_string", rend)
    monkeypatch.setattr(design_builder, "generate_comparisons", comp)

    job = MagicMock()
    job.url = blocked_url
    finalize = _finalize(tmp_path)

    with caplog.at_level(logging.DEBUG, logger="src.services.design_builder.finalize_phases"):
        result = await finalize._phase_screenshots(
            job, "job-ssrf", "redesign.html", get_settings(), True, RenderOptions()
        )

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "Refusing original capture" in message
    assert "job-ssrf" in message  # the operator can find the job

    # NOT downgraded into the generic "something went wrong" warning.
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("Failed to capture original screenshots" in m for m in warnings)

    # Non-fatal: no original, but the design itself still rendered.
    assert result.original_paths is None
    assert result.screenshot_paths == {"desktop": str(tmp_path / "redesign.png")}
    comp.assert_not_awaited()  # no original -> nothing to compare


async def test_phase_screenshots_still_warns_generically_for_a_real_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Control for the test above: a plain failure keeps its WARNING.

    Without this, the ERROR-level assertion could be satisfied by a change that
    simply promoted *every* capture failure to ERROR, which would destroy the
    very distinction the fix exists to create.

    Args:
        tmp_path: Job directory.
        monkeypatch: Replaces the package-level renderer functions.
        caplog: Captures the emitted log records.
    """
    cap = AsyncMock(side_effect=TimeoutError("page load timed out"))
    rend = AsyncMock(return_value={"desktop": str(tmp_path / "redesign.png")})
    comp = AsyncMock(return_value=({"desktop": "c.png"}, {"desktop": "d.png"}))
    monkeypatch.setattr(design_builder, "capture_url_screenshots", cap)
    monkeypatch.setattr(design_builder, "render_from_html_string", rend)
    monkeypatch.setattr(design_builder, "generate_comparisons", comp)

    job = MagicMock()
    job.url = "https://example.com/"
    finalize = _finalize(tmp_path)

    with caplog.at_level(logging.DEBUG, logger="src.services.design_builder.finalize_phases"):
        await finalize._phase_screenshots(
            job, "job-slow", "redesign.html", get_settings(), True, RenderOptions()
        )

    assert not [r for r in caplog.records if r.levelno == logging.ERROR]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Failed to capture original screenshots" in m for m in warnings)
