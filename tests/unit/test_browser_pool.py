"""BrowserPool logic (Phase 3) with fake browsers — no real chromium.

The real-chromium path is covered end-to-end by test_screenshot_renderer (which
now leases from the pool via lazy init); here we pin the reuse / relaunch-after-N
/ bounded-concurrency / crash-reconnect behavior deterministically.
"""

from __future__ import annotations

import asyncio
import logging

from src.config import get_settings
from src.services.browser_pool import BrowserPool, _Slot


class _FakeBrowser:
    def __init__(self, ident: int) -> None:
        self.id = ident
        self.connected = True
        self.closed = False

    def is_connected(self) -> bool:
        return self.connected

    async def close(self) -> None:
        self.closed = True
        self.connected = False


def _seed(pool: BrowserPool, monkeypatch) -> list[_FakeBrowser]:
    """Put the pool in a started state with a fake launcher; return launch log."""
    launched: list[_FakeBrowser] = []

    async def _fake_launch() -> _FakeBrowser:
        b = _FakeBrowser(len(launched))
        launched.append(b)
        return b

    monkeypatch.setattr(pool, "_launch", _fake_launch)
    queue: asyncio.Queue[_Slot] = asyncio.Queue()
    for _ in range(pool._size):
        slot = _Slot()
        pool._slots.append(slot)
        queue.put_nowait(slot)
    pool._idle = queue
    return launched


async def test_browser_is_reused_then_relaunched_after_max(monkeypatch) -> None:
    pool = BrowserPool(size=1, max_renders=3)
    launched = _seed(pool, monkeypatch)

    seen: list[int] = []
    for _ in range(7):
        async with pool.lease() as browser:
            seen.append(browser.id)

    # 3 leases per browser → ids 0,0,0,1,1,1,2; exactly 3 launches.
    assert seen == [0, 0, 0, 1, 1, 1, 2]
    assert len(launched) == 3
    assert launched[0].closed and launched[1].closed  # retired ones were closed
    assert not launched[2].closed


async def test_disconnected_browser_is_relaunched(monkeypatch) -> None:
    pool = BrowserPool(size=1, max_renders=100)
    launched = _seed(pool, monkeypatch)

    async with pool.lease() as first:
        first_id = first.id
    launched[first_id].connected = False  # simulate a crash between leases

    async with pool.lease() as second:
        assert second.id != first_id
    assert len(launched) == 2


async def test_pool_bounds_concurrency(monkeypatch) -> None:
    pool = BrowserPool(size=2, max_renders=100)
    _seed(pool, monkeypatch)

    cm1, cm2, cm3 = pool.lease(), pool.lease(), pool.lease()
    await cm1.__aenter__()
    await cm2.__aenter__()

    # Both slots are leased; a third acquire must block until one is returned.
    third = asyncio.create_task(cm3.__aenter__())
    await asyncio.sleep(0.05)
    assert not third.done()

    await cm1.__aexit__(None, None, None)
    browser3 = await asyncio.wait_for(third, timeout=1.0)
    assert browser3 is not None

    await cm2.__aexit__(None, None, None)
    await cm3.__aexit__(None, None, None)


async def test_stop_closes_immediately_with_no_active_leases(monkeypatch) -> None:
    """The common case (idle pod, or between renders) must add no shutdown latency."""
    pool = BrowserPool(size=1, max_renders=100)
    launched = _seed(pool, monkeypatch)

    async with pool.lease():
        pass  # lease released before stop() — no in-flight work

    await asyncio.wait_for(pool.stop(), timeout=0.2)
    assert launched[0].closed


async def test_stop_waits_for_an_active_lease_then_closes(monkeypatch) -> None:
    """SIGTERM must not close a browser out from under an in-flight direct-PDF render.

    Mirrors JobWorkerPool.stop()'s drain: the direct PDF routes lease a browser
    via BrowserPool.lease() straight from the request handler, outside
    JobWorkerPool's queue, so this is the only drain that can protect them.
    """
    pool = BrowserPool(size=1, max_renders=100)
    launched = _seed(pool, monkeypatch)
    settings = get_settings()
    monkeypatch.setattr(settings.redesign, "browser_drain_timeout_seconds", 5)

    started = asyncio.Event()
    release = asyncio.Event()

    async def _in_flight_render() -> None:
        async with pool.lease():
            started.set()  # genuinely holding the lease when stop() is called
            await release.wait()

    task = asyncio.create_task(_in_flight_render())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    stop_task = asyncio.create_task(pool.stop())
    await asyncio.sleep(0.05)
    assert not stop_task.done(), "stop() closed the browser while a lease was still active"
    assert not launched[0].closed

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    await asyncio.wait_for(stop_task, timeout=1.0)
    assert launched[0].closed


async def test_stop_closes_anyway_after_drain_timeout_and_logs(monkeypatch, caplog) -> None:
    """The drain is bounded — it cannot hold the pod past its termination grace period."""
    pool = BrowserPool(size=1, max_renders=100)
    launched = _seed(pool, monkeypatch)
    settings = get_settings()
    monkeypatch.setattr(settings.redesign, "browser_drain_timeout_seconds", 0)

    started = asyncio.Event()
    never_release = asyncio.Event()  # deliberately never set — lease never returns

    async def _wedged_render() -> None:
        async with pool.lease():
            started.set()
            await never_release.wait()

    task = asyncio.create_task(_wedged_render())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(pool.stop(), timeout=1.0)

    assert launched[0].closed, "stop() must close anyway once the drain deadline passes"
    assert any("still active" in r.message for r in caplog.records)

    task.cancel()
    import contextlib

    with contextlib.suppress(asyncio.CancelledError):
        await task
