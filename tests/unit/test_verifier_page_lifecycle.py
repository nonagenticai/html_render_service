"""``_probe_in_browser`` must close its page on *every* exit path.

``BrowserPool.lease()`` closes nothing when the lease ends — it only re-queues
the slot — and the slot's browser is relaunched once every ``browser_max_renders``
(200) leases. So a probe that returns or raises without calling ``page.close()``
orphans a live Chromium page (plus its three event listeners, which hold a
reference to the probe) for up to 200 renders per slot. That is the multi-GB
OOMKill path, and it was reachable from the assembly verification loop in
``design_builder/phases.py`` on every failed navigation.

The fakes here implement a real async ``close()`` that counts its calls, rather
than an ``AsyncMock``: a bare mock records ``.close()`` as "called" even if
production code invoked some other method entirely, which is exactly the class
of false green this test exists to prevent.

``close()`` counts **completions, not entries**, and suspends in between. A real
``Page.close()`` is a driver round-trip, so it has a suspension point; counting at
entry would score a close that was cancelled half-way — with the page still very
much alive — as a successful teardown, which is precisely the hole
``test_page_is_closed_when_the_probe_is_cancelled`` exists to catch.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest

from src.services import browser_pool as browser_pool_module
from src.services.verification import verifier


class _FakePage:
    """Minimal Playwright page stand-in that records real close() calls."""

    def __init__(
        self,
        *,
        goto_error: BaseException | None = None,
        close_error: BaseException | None = None,
        goto_hangs: bool = False,
    ) -> None:
        self._goto_error = goto_error
        self._close_error = close_error
        self._goto_hangs = goto_hangs
        self.close_entries = 0
        self.close_calls = 0
        self.listeners: list[str] = []
        self.routes: list[str] = []

    def on(self, event: str, handler: Callable[[Any], Any]) -> None:
        self.listeners.append(event)

    async def route(self, pattern: str, handler: Callable[[Any], Any]) -> None:
        # `_probe_in_browser` installs `block_private_subresources` before
        # navigating, and that guard is `await page.route(...)`. A double that
        # omits it does not merely under-model the page -- it makes every
        # lifecycle test here fail with AttributeError the moment the sink is
        # guarded, which is exactly what happened when the guard was added.
        self.routes.append(pattern)

    async def goto(self, url: str, **kwargs: Any) -> None:
        if self._goto_error is not None:
            raise self._goto_error
        if self._goto_hangs:
            # Suspend forever so the caller can be cancelled mid-navigation,
            # which is what a SIGTERM drain does to an in-flight probe.
            await asyncio.Event().wait()

    async def close(self) -> None:
        self.close_entries += 1
        # A real Page.close() is a driver round-trip, so it suspends. This
        # sleep is the whole point of the test: an unshielded close inside a
        # cancelled scope raises HERE, leaving the page alive, so close_calls
        # separates "we called close" from "the page actually went away".
        await asyncio.sleep(0)
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.new_page_calls = 0

    async def new_page(self, **kwargs: Any) -> _FakePage:
        self.new_page_calls += 1
        return self._page


class _FakePool:
    """Mirrors BrowserPool.lease(): yields a browser, closes nothing on exit."""

    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser
        self.leases = 0

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[_FakeBrowser]:
        self.leases += 1
        # Exactly like the real pool: on exit the slot is re-queued and nothing
        # is closed, so the page's lifetime is entirely the caller's problem.
        yield self._browser


def _install(monkeypatch: pytest.MonkeyPatch, page: _FakePage) -> _FakePool:
    pool = _FakePool(_FakeBrowser(page))

    async def _fake_get_browser_pool() -> _FakePool:
        return pool

    monkeypatch.setattr(browser_pool_module, "get_browser_pool", _fake_get_browser_pool)
    return pool


_HTML = "<html><body><h1>hi</h1></body></html>"


async def test_page_is_closed_when_navigation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage(goto_error=RuntimeError("nav boom"))
    _install(monkeypatch, page)

    probe = await verifier._probe_in_browser(_HTML, timeout_ms=1000, run_axe=False)

    assert page.close_calls == 1, "failed navigation must still close the page"
    # The reported contract is unchanged by the lifetime fix.
    assert probe.render_error == "page failed to load: nav boom"
    assert probe.axe_result is None


async def test_page_is_closed_when_axe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage()
    _install(monkeypatch, page)

    async def _boom(_page: Any) -> dict[str, Any] | None:
        raise RuntimeError("axe exploded")

    monkeypatch.setattr(verifier, "_run_axe", _boom)

    with pytest.raises(RuntimeError, match="axe exploded"):
        await verifier._probe_in_browser(_HTML, timeout_ms=1000, run_axe=True)

    assert page.close_calls == 1, "an axe failure must still close the page"


async def test_page_is_closed_exactly_once_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _FakePage()
    _install(monkeypatch, page)

    async def _axe(_page: Any) -> dict[str, Any] | None:
        return {"violations": []}

    monkeypatch.setattr(verifier, "_run_axe", _axe)

    probe = await verifier._probe_in_browser(_HTML, timeout_ms=1000, run_axe=True)

    assert page.close_calls == 1, "success path must close once, not zero or twice"
    assert probe.render_error is None
    assert probe.axe_result == {"violations": []}
    assert page.listeners == ["console", "pageerror", "requestfailed"]


async def test_close_failure_does_not_change_reported_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Playwright raises from close() when the target is already gone; that must
    # not turn a reported render failure into a checker crash.
    page = _FakePage(goto_error=RuntimeError("nav boom"), close_error=RuntimeError("already gone"))
    _install(monkeypatch, page)

    probe = await verifier._probe_in_browser(_HTML, timeout_ms=1000, run_axe=False)

    assert page.close_calls == 1
    assert probe.render_error == "page failed to load: nav boom"


async def test_page_is_closed_when_the_probe_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled probe must still tear the page down — the SIGTERM-drain path.

    This is the one an unshielded ``await page.close()`` fails. The probe runs
    under an anyio cancel scope (the same shape Starlette/uvicorn wrap a request
    handler in), and that scope stays cancelled: every subsequent await re-raises
    immediately. So a bare close raises at its own suspension point, before
    Playwright has torn the page down, and the page leaks for the rest of the
    pooled browser's 200-lease life. ``asyncio.shield`` detaches the close so it
    completes anyway.
    """
    page = _FakePage(goto_hangs=True)
    _install(monkeypatch, page)

    with anyio.move_on_after(0.05) as scope:
        await verifier._probe_in_browser(_HTML, timeout_ms=1000, run_axe=False)

    assert scope.cancelled_caught, "the probe should have been cancelled mid-navigation"
    assert page.close_entries == 1, "the finally must always reach close()"

    # Let the shielded close task finish; it is deliberately detached from the
    # cancelled scope, so it lands a tick later.
    for _ in range(10):
        if page.close_calls:
            break
        await asyncio.sleep(0)

    assert page.close_calls == 1, (
        "close() was entered but never completed — the page leaked. "
        "asyncio.shield around page.close() is what makes this pass."
    )


async def test_close_failure_does_not_mask_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _FakePage(close_error=RuntimeError("already gone"))
    _install(monkeypatch, page)

    async def _boom(_page: Any) -> dict[str, Any] | None:
        raise ValueError("original failure")

    monkeypatch.setattr(verifier, "_run_axe", _boom)

    # The caller must see the real cause, not the cleanup error.
    with pytest.raises(ValueError, match="original failure"):
        await verifier._probe_in_browser(_HTML, timeout_ms=1000, run_axe=True)

    assert page.close_calls == 1
