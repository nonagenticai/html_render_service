"""SSRF guard on the PDF print sink — the subresources a caller-supplied document loads.

WHY THIS SINK IS DIFFERENT FROM THE ONES ``test_render_target_ssrf`` COVERS
--------------------------------------------------------------------------
Those sinks guard a tenant-supplied *navigation target*: a URL that
``validate_public_url`` vets and ``lease_pinned`` pins before ``Page.goto``.
``render_pdf_from_html_string`` has no such URL. It writes the caller's HTML to a
temp file and navigates the **unpinned** pool browser to ``file://…`` — so the
target check has nothing to check, the resolver pin is not installed, and every
``<img>``/``<iframe>``/``fetch`` inside that document was issued by an in-cluster
Chromium against whatever address it named. MinIO on the tailnet, a ClusterIP,
``169.254.169.254`` — the response is painted into the page and returned to the
caller **as PDF bytes**. ``PdfRenderRequest`` promised "no external assets" in a
docstring and enforced nothing.

WHAT IS PINNED HERE
-------------------
* The guard is an **interception**: the blocked request is ``abort``ed, so the
  GET is never issued. These tests assert on the aborted request — the recorded
  URL and the ``abort`` call — not on "no exception was raised", which would
  pass against a guard that fires after the bytes are already home.
* Public remote assets still render. A fonts/CDN URL is ``continue_``d; a rule
  that blocked all remote subresources would break ``/jobs/{id}/pdf``, whose
  documents are LLM-written pages that legitimately reference them.
* The route pattern is scheme-scoped to http(s), so ``file:``/``data:`` never
  reach the handler and the main-frame navigation is untouched. ``file://``
  subresource reads are explicitly NOT covered — see the guard's docstring.
* ``POST /render/pdf`` resolves a tenant. It was the only route in
  ``artifacts.py`` that did not, i.e. anonymous, unmetered Chromium work.

Hermetic: DNS comes from the autouse stub in ``tests/unit/conftest.py``; no
network, no browser.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api.routes.artifacts import PdfRenderRequest, render_pdf
from src.services.screenshot_renderer import block_private_subresources
from tests.tenants import TENANT_1

pytestmark = pytest.mark.asyncio

#: Subresource URLs a rendered document must never be allowed to fetch. The
#: tailnet /10 entries are the ones a hand-rolled RFC1918 denylist waves through.
BLOCKED_SUBRESOURCES = [
    "http://100.64.0.5:9000/artifacts/secret.png",  # MinIO / prod postgres host
    "http://100.64.0.6:5000/v2/_catalog",  # docker registry, tailnet
    "http://169.254.169.254/latest/meta-data/iam/",  # cloud metadata
    "http://10.43.0.1/api/v1/namespaces",  # ClusterIP
    "http://127.0.0.1:6319/api/designs/jobs",  # loopback — our own API
    "https://postgres.internal/",  # estate-internal name
    "https://cdn-lookalike.example/logo.png",  # public NAME, tailnet answer
]

ALLOWED_SUBRESOURCES = [
    "https://fonts.example.com/inter.woff2",
    "http://203.0.113.10/hero.png",  # TEST-NET-3
]


class _FakeRoute:
    """A Playwright ``Route`` double recording which verdict the guard reached."""

    def __init__(self, url: str) -> None:
        self.request = MagicMock()
        self.request.url = url
        self.aborted: str | None = None
        self.continued = False

    async def abort(self, error_code: str = "failed") -> None:
        self.aborted = error_code

    async def continue_(self, **_kwargs: Any) -> None:
        self.continued = True


async def _install(page: MagicMock) -> tuple[list[str], Any]:
    """Run the guard against ``page`` and return ``(blocked_list, handler)``."""
    blocked = await block_private_subresources(page)
    # Both patterns must register, and both must share one handler.
    patterns = [call.args[0] for call in page.route.await_args_list]
    assert patterns == ["http://**", "https://**"], patterns
    handlers = {id(call.args[1]) for call in page.route.await_args_list}
    assert len(handlers) == 1
    return blocked, page.route.await_args_list[0].args[1]


def _page() -> MagicMock:
    page = MagicMock()
    page.route = AsyncMock()
    return page


@pytest.mark.parametrize("url", BLOCKED_SUBRESOURCES)
async def test_private_subresource_request_is_aborted_not_merely_flagged(url: str) -> None:
    """The request is ABORTED — never issued — and recorded.

    MUTATION KILLED: replacing ``route.abort(...)`` with ``route.continue_()``
    (i.e. validate-and-shrug), or dropping the ``except (SSRFError, ValueError)``
    arm — ``route.continued`` becomes True / the call raises, and the blocked
    list stays empty.

    Args:
        url: A subresource address the rendered document must not reach.
    """
    page = _page()
    blocked, handler = await _install(page)

    route = _FakeRoute(url)
    await handler(route)

    assert route.aborted == "blockedbyclient", f"{url} was not aborted"
    assert route.continued is False, f"{url} was allowed through"
    assert blocked == [url]


@pytest.mark.parametrize("url", ALLOWED_SUBRESOURCES)
async def test_public_subresource_still_loads(url: str) -> None:
    """A public font/image must still render, or every real PDF regresses.

    MUTATION KILLED: aborting unconditionally — ``continued`` is False and the
    blocked list is non-empty.

    Args:
        url: An ordinary public asset URL.
    """
    page = _page()
    blocked, handler = await _install(page)

    route = _FakeRoute(url)
    await handler(route)

    assert route.continued is True, f"{url} was wrongly blocked"
    assert route.aborted is None
    assert blocked == []


async def test_guard_is_installed_before_navigation() -> None:
    """``page.route`` must be awaited BEFORE ``page.goto``.

    Ordering is the whole guard: a handler registered after navigation misses
    the subresources the initial parse already kicked off.

    MUTATION KILLED: moving ``block_private_subresources`` below ``page.goto``
    in ``render_pdf_from_html_string`` — the recorded call order flips.
    """
    import src.services.browser_pool as browser_pool
    import src.services.screenshot_renderer as sr

    calls: list[str] = []
    page = MagicMock()
    page.route = AsyncMock(side_effect=lambda *_a, **_k: calls.append("route"))
    page.goto = AsyncMock(side_effect=lambda *_a, **_k: calls.append("goto"))
    page.pdf = AsyncMock(return_value=b"%PDF-1.4")
    page.close = AsyncMock()

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)

    class _Lease:
        async def __aenter__(self) -> MagicMock:
            return browser

        async def __aexit__(self, *_exc: object) -> None:
            return None

    pool = MagicMock()
    pool.lease = lambda: _Lease()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(browser_pool, "get_browser_pool", AsyncMock(return_value=pool))
        out = await sr.render_pdf_from_html_string("<html><body>x</body></html>")

    assert out == b"%PDF-1.4"
    assert calls == ["route", "route", "goto"], calls


async def _render_pdf_and_capture_guard_handler() -> Any:
    """Run the REAL ``render_pdf_from_html_string`` call site and return its
    installed ``page.route`` handler.

    Exercises the wired call site (``block_private_subresources(page,
    allow_hosts=_minio_allow_hosts())``), not the helper in isolation — the
    returned handler is exactly what production installs.
    """
    import src.services.browser_pool as browser_pool
    import src.services.screenshot_renderer as sr

    page = MagicMock()
    page.route = AsyncMock()
    page.goto = AsyncMock()
    page.pdf = AsyncMock(return_value=b"%PDF-1.4")
    page.close = AsyncMock()

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)

    class _Lease:
        async def __aenter__(self) -> MagicMock:
            return browser

        async def __aexit__(self, *_exc: object) -> None:
            return None

    pool = MagicMock()
    pool.lease = lambda: _Lease()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(browser_pool, "get_browser_pool", AsyncMock(return_value=pool))
        await sr.render_pdf_from_html_string("<html><body>x</body></html>")

    return page.route.await_args_list[0].args[1]


async def test_presigned_minio_subresource_allowed_through_the_wired_call_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A presigned ``minio.internal`` asset must load — issue #104's fix.

    Exercises the REAL call site in ``render_pdf_from_html_string``, not
    ``block_private_subresources``/``_minio_allow_hosts`` in isolation: sets
    ``STORAGE_MINIO_ENDPOINT`` so ``_minio_allow_hosts()`` resolves to
    ``["minio.internal:8389"]``, captures the actual guard handler the call
    site installs via ``page.route``, and fires it at a presigned MinIO URL.

    MUTATION KILLED: dropping ``allow_hosts=_minio_allow_hosts()`` from the
    ``block_private_subresources(page, ...)`` call in
    ``render_pdf_from_html_string`` (or any of the other 3 sinks) turns this
    RED — the handler falls back to no exemption and aborts the request.
    """
    import src.config as config

    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.internal:8389")
    config.get_settings.cache_clear()
    try:
        handler = await _render_pdf_and_capture_guard_handler()
    finally:
        config.get_settings.cache_clear()

    route = _FakeRoute(
        "http://minio.internal:8389/html-render-service/job-1/desktop.png?X-Amz-Signature=abc"
    )
    await handler(route)

    assert route.continued is True, "presigned minio.internal asset was wrongly blocked"
    assert route.aborted is None


async def test_arbitrary_private_host_still_blocked_when_minio_is_allow_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MinIO exemption must not widen into a general private-network bypass.

    Same wiring as the allow test above (real call site, ``STORAGE_MINIO_ENDPOINT``
    set) but fires the handler at a CGNAT host that is NOT the configured MinIO
    endpoint — proving the fix did not just disable the guard.

    MUTATION KILLED: any change that lets a non-allow-listed private/CGNAT host
    through once ``allow_hosts`` is non-empty (e.g. treating a non-empty
    ``allow_hosts`` as "skip the private-range check entirely") turns this RED.
    """
    import src.config as config

    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.internal:8389")
    config.get_settings.cache_clear()
    try:
        handler = await _render_pdf_and_capture_guard_handler()
    finally:
        config.get_settings.cache_clear()

    route = _FakeRoute("http://100.64.0.5:9000/unrelated-bucket/secret.png")
    await handler(route)

    assert route.aborted == "blockedbyclient", "arbitrary CGNAT host was wrongly allowed"
    assert route.continued is False


async def test_render_pdf_route_requires_a_tenant() -> None:
    """No established identity → 401, not an anonymous Chromium job.

    MUTATION KILLED: deleting the ``_get_tenant_id(request)`` call from
    ``render_pdf`` — the handler returns a PDF for a request that never proved
    who it was.
    """
    request = MagicMock()
    request.state = MagicMock(spec=[])  # no tenant_id attribute

    with pytest.raises(HTTPException) as exc:
        await render_pdf(PdfRenderRequest(html="<html></html>"), request)
    assert exc.value.status_code == 401


async def test_render_pdf_route_renders_for_an_authenticated_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legitimate orchestrator call (which sends ``X-Tenant-Id``) still works."""
    import src.services.screenshot_renderer as sr

    async def fake_render(html: str, timeout_ms: int = 30000) -> bytes:
        return b"%PDF-1.4 ok"

    monkeypatch.setattr(sr, "render_pdf_from_html_string", fake_render)

    request = MagicMock()
    request.state.tenant_id = TENANT_1

    resp = await render_pdf(PdfRenderRequest(html="<html></html>"), request)
    assert resp.body == b"%PDF-1.4 ok"


def test_every_file_url_sink_installs_the_subresource_guard() -> None:
    """A new ``file://`` sink must not ship without the interception.

    The original fix guarded ONLY ``render_pdf_from_html_string``, leaving
    ``render_screenshots``, ``render_component_screenshot`` and
    ``render_section_screenshots`` open -- all three navigate the *unpinned* pool
    browser at caller-influenced HTML (``html_for_render`` is the LLM-generated
    redesign, retrieved in ``design_builder/finalize_phases.py``). That is the
    "guarded beside unguarded" shape that keeps recurring here, so pin the
    PROPERTY rather than trusting the four call sites to stay in sync: every
    ``page.goto(file_url ...)`` must be immediately preceded by installing the
    guard.
    """
    import ast
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[2] / "src" / "services" / "screenshot_renderer.py"
    )
    lines = src.read_text().splitlines()

    goto_lines = [i for i, ln in enumerate(lines) if "page.goto(file_url" in ln]
    assert goto_lines, "no file:// sinks found -- has the module moved?"

    # Substring, not an exact call match: issue #104 added `allow_hosts=...` to
    # every call site (`block_private_subresources(page, allow_hosts=...)`), so
    # this must not require the closing paren immediately after `page`.
    unguarded = [i + 1 for i in goto_lines if "block_private_subresources(page" not in lines[i - 1]]
    assert not unguarded, (
        f"file:// sink(s) at line(s) {unguarded} navigate caller-supplied HTML "
        "without installing block_private_subresources first"
    )
    # Guard against the assertion going vacuous if the sinks are ever renamed.
    assert len(goto_lines) >= 4, f"expected >=4 file:// sinks, found {len(goto_lines)}"
    ast.parse(src.read_text())


def _local_document_sinks() -> list[tuple[str, int, bool]]:
    """Every ``page.goto`` in ``src/`` that navigates at a LOCAL document.

    Returns ``(relpath, goto_lineno, guarded)`` per sink, where ``guarded`` means
    the enclosing function installs ``block_private_subresources`` on a line
    ABOVE the navigation.

    Why AST over the whole tree and not a substring over one file: the previous
    version of this property scanned only ``screenshot_renderer.py`` for the
    literal ``page.goto(file_url``. ``verification/verifier.py`` is a fifth sink
    in another module that spells the same thing ``page.goto(tmp_path.as_uri())``
    — invisible to both halves of that check, and it shipped unguarded. The
    property is "a browser navigated at a local document intercepts its
    subresources", so match the property (any local-file navigation, anywhere in
    ``src``), not the one spelling one author happened to use.
    """
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[2] / "src"
    sinks: list[tuple[str, int, bool]] = []

    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            body_src = ast.unparse(func)
            # A local-document navigation: the function builds a file:// URL,
            # either as an f-string/literal or via Path.as_uri().
            if "file://" not in body_src and ".as_uri()" not in body_src:
                continue
            guard_lines = [
                n.lineno
                for n in ast.walk(func)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "block_private_subresources"
            ]
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not (isinstance(fn, ast.Attribute) and fn.attr == "goto"):
                    continue
                arg_src = ast.unparse(node.args[0]) if node.args else ""
                # Either the argument is itself a file:// expression, or it is a
                # local name (`file_url`) assigned from one earlier in the body.
                names_a_local_doc = "file://" in arg_src or "as_uri" in arg_src
                if not names_a_local_doc:
                    names_a_local_doc = any(
                        isinstance(a, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == arg_src for t in a.targets)
                        and "file://" in ast.unparse(a.value)
                        for a in ast.walk(func)
                    )
                if not names_a_local_doc:
                    continue
                guarded = any(g < node.lineno for g in guard_lines)
                sinks.append((str(path.relative_to(src_root)), node.lineno, guarded))
    return sinks


def test_every_local_document_sink_in_src_installs_the_subresource_guard() -> None:
    """Fleet property: no browser navigates a local document unintercepted.

    MUTATION KILLED: deleting the ``block_private_subresources(page, ...)`` call
    from ANY sink -- including ``verification/verifier.py:_probe_in_browser``,
    which the old single-file substring check could not see -- lists that sink in
    the failure message.
    """
    sinks = _local_document_sinks()
    # Vacuity guard: 4 in screenshot_renderer + 1 in verification/verifier.py.
    assert len(sinks) >= 5, f"expected >=5 local-document sinks, found {sinks}"
    unguarded = [(p, ln) for p, ln, ok in sinks if not ok]
    assert not unguarded, (
        f"local-document sink(s) {unguarded} navigate caller-supplied HTML "
        "without installing block_private_subresources first"
    )
