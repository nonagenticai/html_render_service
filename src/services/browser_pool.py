"""Pooled, long-lived Chromium browsers (Phase 3).

Replaces the launch-a-browser-per-render model. A bounded pool of long-lived
Chromium processes is created once; each render leases a browser, opens a fresh
page on it, and returns it. This removes the per-call launch cost (≈0.5-1s + RAM
churn) that made heavy redesigns slow, while the bounded pool size keeps the
same concurrency ceiling the old process-wide semaphore provided.

A long-lived Chromium can ratchet memory over the pod's lifetime, so each
browser is **relaunched after N leases** (``browser_max_renders``) — the same
bounded-lifetime trick used for the MLX/onnxruntime arenas. Combined with the
Phase 0 full-page height cap, peak per-render memory stays bounded.

The pool is a module global (like the semaphore it replaces) so the existing
module-level render functions can use it without threading it through every
caller. The ASGI lifespan starts/stops it explicitly; tests that call a render
function directly trigger a lazy start.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class PoolStatus(NamedTuple):
    """A non-mutating snapshot of the pool, for /health.

    ``capacity`` is the number of slots that exist — the render concurrency this
    pod can actually deliver. It is 0 whenever the pool never started, which is
    the state the old health check could not see: DB and Redis answer, `/health`
    says ``healthy``, and every render in the process is waiting on a Chromium
    that does not exist.

    ``live`` counts slots currently holding a *connected* browser. It is
    deliberately NOT part of the healthy/unhealthy verdict: a slot is launched
    lazily on first lease and relaunched when found stale, so ``live < capacity``
    is the normal steady state of an idle pod, not a fault.
    """

    started: bool
    capacity: int
    idle: int
    live: int


_LAUNCH_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]


def host_resolver_rule(host: str, ip: str) -> str:
    """Build the Chromium ``--host-resolver-rules`` value pinning ``host`` to ``ip``.

    Form: ``MAP <host> <ip>`` — deliberately **scoped to the single host being
    rendered** and with **no port**, which matters twice over:

    * No trailing ``:port`` in the replacement means Chromium keeps the port
      from the request, so ``http://h:8123/`` still reaches ``<ip>:8123``.
    * No ``MAP *`` wildcard and no ``~NOTFOUND`` catch-all, so every *other*
      host — the CDNs serving the page's CSS/JS/fonts/images — resolves
      normally. A blanket rule would render the page blank, which is worse
      than the hole being closed.

    An IPv6 literal **must be bracketed**: Chromium rejects the whole rule for a
    bare ``::1`` (verified — the rule is dropped and navigation then fails), so
    an unbracketed address would break rendering for every AAAA-resolved target.

    An **IDN host must be punycode**, and this one is a fail-*open* trap rather
    than a fail-closed one. ``urlparse().hostname`` hands back the Unicode form
    (``bücher.example``) while Chromium resolves the A-label
    (``xn--bcher-kva.example``). Verified against real Chromium: a Unicode rule
    simply **does not match**, so the browser resolves the host freely and the
    pin is silently inert — a guard that looks present and protects nothing. An
    unencodable name raises rather than rendering unpinned; the caller degrades
    that to "no before image" (see ``capture_failure_types``), never a blank one.
    ASCII hosts (including a trailing-root-dot form like ``example.com.``, which
    Chromium matches as-is — verified) pass through untouched.

    Args:
        host: Hostname from the validated URL (already lowercased by ``urlparse``).
        ip: The address ``validate_public_url`` resolved and vetted.

    Returns:
        The switch value, e.g. ``MAP example.com 93.184.216.34`` or
        ``MAP example.com [2606:2800:220:1:248:1893:25c8:1946]``.

    Raises:
        ValueError: if ``host`` is a non-ASCII name that cannot be IDNA-encoded.
            Failing closed is deliberate: the alternative is an unpinned render.
    """
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"Cannot IDNA-encode render host {host!r} to pin it") from exc
    try:
        literal = ipaddress.ip_address(ip)
    except ValueError:
        # Not an IP literal. validate_public_url only ever returns one, so this
        # is unreachable in production; pass it through rather than inventing
        # brackets around something that is not an address.
        return f"MAP {host} {ip}"
    return f"MAP {host} [{ip}]" if literal.version == 6 else f"MAP {host} {ip}"


class _Slot:
    """A reusable pool slot holding one browser and its lease count."""

    __slots__ = ("browser", "renders")

    def __init__(self) -> None:
        self.browser: Any = None
        self.renders: int = 0


class BrowserPool:
    """A fixed-size pool of long-lived Chromium browsers."""

    def __init__(self, *, size: int, max_renders: int) -> None:
        self._size = max(1, size)
        self._max_renders = max(1, max_renders)
        self._pw: Any = None
        self._idle: asyncio.Queue[_Slot] | None = None
        self._slots: list[_Slot] = []
        self._start_lock = asyncio.Lock()
        # In-flight lease count for `lease()` (the pooled path direct PDF routes
        # use outside JobWorkerPool's queue). `_drained` is set whenever the
        # count is 0 so stop() can wait on it without polling.
        self._active_leases = 0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def started(self) -> bool:
        return self._idle is not None

    def status(self) -> PoolStatus:
        """Snapshot the pool for /health. Never starts it and never blocks."""
        live = 0
        for slot in self._slots:
            browser = slot.browser
            if browser is None:
                continue
            try:
                if browser.is_connected():
                    live += 1
            except Exception:  # a probe must never raise into /health
                logger.debug("Browser slot connectivity probe failed", exc_info=True)
        return PoolStatus(
            started=self._idle is not None,
            capacity=len(self._slots),
            idle=self._idle.qsize() if self._idle is not None else 0,
            live=live,
        )

    async def start(self) -> None:
        """Start the Playwright driver and seed the (lazily-launched) slots."""
        async with self._start_lock:
            if self._idle is not None:
                return
            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            queue: asyncio.Queue[_Slot] = asyncio.Queue()
            for _ in range(self._size):
                slot = _Slot()
                self._slots.append(slot)
                queue.put_nowait(slot)
            self._idle = queue
            logger.info(
                "Browser pool started: size=%d, relaunch every %d leases",
                self._size,
                self._max_renders,
            )

    async def _launch(self) -> Any:
        return await self._pw.chromium.launch(args=_LAUNCH_ARGS)

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[Any]:
        """Lease a connected browser; blocks when all ``size`` are in use.

        (Re)launches the slot's browser if it has none, is disconnected (crashed),
        or has hit ``max_renders`` — bounding the long-lived-Chromium memory
        ratchet. Acquiring is the concurrency gate (queue depth == pool size),
        replacing the old render semaphore.
        """
        if self._idle is None:
            await self.start()
        idle = self._idle
        if idle is None:  # pragma: no cover - start() always sets it
            raise RuntimeError("Browser pool failed to start")
        slot = await idle.get()
        self._active_leases += 1
        self._drained.clear()
        try:
            stale = (
                slot.browser is None
                or not slot.browser.is_connected()
                or slot.renders >= self._max_renders
            )
            if stale:
                if slot.browser is not None:
                    with contextlib.suppress(Exception):
                        await slot.browser.close()
                slot.browser = await self._launch()
                slot.renders = 0
            slot.renders += 1
            yield slot.browser
        finally:
            idle.put_nowait(slot)
            self._active_leases -= 1
            if self._active_leases <= 0:
                self._active_leases = 0
                self._drained.set()

    @asynccontextmanager
    async def lease_pinned(self, host: str, ip: str) -> AsyncIterator[Any]:
        """Lease a dedicated, non-pooled browser with DNS for ``host`` pinned to ``ip``.

        The pooled ``lease`` browsers are long-lived and launched once with a
        fixed arg set, but ``--host-resolver-rules`` is a *launch* arg and the
        pinned IP is per-render, so a pinned target needs its own short-lived
        browser. This closes the DNS-rebind window that IP-pinning at the httpx
        layer cannot reach: Chromium resolves the hostname itself, and this
        forces that resolution to the already-validated address instead of
        re-resolving (which, after a rebind, could land on an internal IP).

        **Cost, measured:** a pinned launch is ~0.9-1.6s (3 samples, warm
        driver). It is paid **once per redesign job**, not per viewport — all
        the job's viewports share this one browser, exactly as they shared one
        pooled browser before — and only on the ``capture_original`` path,
        which is a single leg of a multi-minute LLM redesign. The pooled
        ``lease`` (and therefore the hot ``file://`` render path, which has no
        untrusted host to pin) is completely untouched. Trading ~1s per job for
        closing the rebind window is the right side of that ledger.

        The browser is closed on exit; it is never returned to the pool, so a
        tenant-specific resolver rule can never leak into another render.
        """
        if self._idle is None:
            await self.start()
        pw = self._pw
        if pw is None:  # pragma: no cover - start() always sets it
            raise RuntimeError("Browser pool failed to start")
        args = [*_LAUNCH_ARGS, f"--host-resolver-rules={host_resolver_rule(host, ip)}"]
        browser = await pw.chromium.launch(args=args)
        try:
            yield browser
        finally:
            with contextlib.suppress(Exception):
                await browser.close()

    async def stop(self) -> None:
        """Drain outstanding leases, then close all browsers and stop Playwright.

        Direct PDF routes (``GET /jobs/{id}/pdf``, ``POST /render/pdf``) lease a
        browser via :meth:`lease` straight from the request handler, outside
        ``JobWorkerPool``'s queue — so that pool's own drain never sees this
        lease, and closing browsers unconditionally on SIGTERM used to kill an
        in-flight ``page.pdf()``/``page.goto()`` out from under the caller
        (``TargetClosedError`` -> 500), exactly the failure
        ``JobWorkerPool.stop()`` was written to avoid for queued renders.

        So: wait for ``_active_leases`` to reach 0, bounded by
        ``browser_drain_timeout_seconds`` so shutdown can never hang forever. A
        lease that never releases (a wedged render) is logged and the browsers
        are closed anyway — better than blocking the pod's exit past its
        terminationGracePeriodSeconds and eating a SIGKILL. No leases in flight
        is the common case (an idle pod, or between renders) and adds no wait at
        all: ``_drained`` is already set then.
        """
        if self._idle is None:
            return
        if self._active_leases > 0:
            from src.config import get_settings

            timeout = max(0, get_settings().redesign.browser_drain_timeout_seconds)
            try:
                await asyncio.wait_for(self._drained.wait(), timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "Browser pool drain deadline (%ds) passed with %d lease(s) still "
                    "active — closing anyway; any in-flight render will fail",
                    timeout,
                    self._active_leases,
                )
        for slot in self._slots:
            if slot.browser is not None:
                with contextlib.suppress(Exception):
                    await slot.browser.close()
        if self._pw is not None:
            with contextlib.suppress(Exception):
                await self._pw.stop()
        self._idle = None
        self._slots = []
        self._pw = None
        logger.info("Browser pool stopped")


_pool: BrowserPool | None = None


def _build_pool() -> BrowserPool:
    from src.config import get_settings

    cfg = get_settings().redesign
    return BrowserPool(size=cfg.render_max_concurrency, max_renders=cfg.browser_max_renders)


async def get_browser_pool() -> BrowserPool:
    """Return the started module-global pool, creating + starting it on first use."""
    global _pool
    if _pool is None:
        _pool = _build_pool()
    if not _pool.started:
        await _pool.start()
    return _pool


def pool_status() -> PoolStatus:
    """Snapshot the module-global pool WITHOUT creating or starting it.

    ``get_browser_pool`` is start-on-demand, so a health check must not go
    through it: probing would *create* the very thing it is meant to report on
    and the probe would always pass.
    """
    pool = _pool
    if pool is None:
        return PoolStatus(started=False, capacity=0, idle=0, live=0)
    return pool.status()


async def init_browser_pool() -> None:
    """Eagerly start the pool (ASGI startup) so the first request isn't slow."""
    await get_browser_pool()


async def close_browser_pool() -> None:
    """Stop and drop the module-global pool (ASGI shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.stop()
        _pool = None
