"""Progress reporting from inside a batch page render.

WHY THIS EXISTS. ``batch_jobs.last_progress_at`` was bumped once per PAGE, by
``increment_completed`` / ``increment_failed``. That made the reaper's window
depend on how long one page takes — and one page is unbounded:
``generate_sections`` (``html_gen/generator.py``) loops
``sections = detect_sections(html)`` SEQUENTIALLY, one ``_call_claude`` per DOM
landmark, and ``detect_sections`` is an uncapped
``soup.find_all(["header","nav","main","footer","aside","section"])``. Thirteen
landmarks is 13 x 600s = 7800s of real work emitting ZERO progress signal, so a
perfectly healthy page got reaped. Bounding the page with a timeout instead
would have fixed the signal by KILLING the work, which is a product regression
dressed as a monitoring fix.

So the clock is bumped at the granularity the work actually happens: after every
section. The window then only has to exceed ONE WM call, which IS bounded —
3 attempts x ``wm.timeout_seconds`` plus backoff (w2v_common ``_MAX_RETRIES`` /
``_RETRY_DELAY``; the ``wm.max_retries`` field that used to sit in this repo's
config was never read by anything and has been deleted) — and no longer depends
on the page's DOM at all.

WHY A CONTEXTVAR. Same reason ``job_manager.claim_owner`` is one: the reporter
would otherwise have to be threaded through ~6 generator signatures that have no
business knowing what a batch is. The page render is owned by one task and every
section is reached by ``await`` from it, so a ContextVar reaches all of them —
and ``generate_sections`` stays a pure HTML-generation function that simply
announces progress to whoever is listening.

Unbound (the queue path, or any direct call) ``report_progress`` is a no-op, so
this is inert everywhere except inside a batch page.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

#: The bound reporter, or ``None`` outside a batch page render.
_PROGRESS_REPORTER: ContextVar[Callable[[], Awaitable[None]] | None] = ContextVar(
    "html_render_batch_progress", default=None
)


@contextmanager
def batch_progress(reporter: Callable[[], Awaitable[None]]) -> Iterator[None]:
    """Bind *reporter* as the progress sink for the duration of one page render.

    Reset on the way out so the next page on a reused event loop cannot inherit
    the previous page's batch — the same discipline ``claim_owner`` applies to a
    reused worker loop.
    """
    token = _PROGRESS_REPORTER.set(reporter)
    try:
        yield
    finally:
        _PROGRESS_REPORTER.reset(token)


#: Failure counters. A swallowed exception that only reaches a log line is
#: indistinguishable from having no clock at all: the reporter dies, every bump
#: silently does nothing, and the batch gets REAPED WHILE ALIVE with a warning as
#: the only trace. Swallowing is still right — a liveness signal must never take
#: down a render that is succeeding — but it has to be COUNTABLE, so /health can
#: say "the progress clock is not being written" instead of nothing at all.
_report_failures_total = 0
_report_failures_consecutive = 0

#: Consecutive failures after which the situation is no longer a blip: at this
#: many misses the elapsed un-reported time is comparable to a whole section, so
#: the batch is measurably at risk of a false reap.
CONSECUTIVE_FAILURE_ALERT = 3


def progress_report_failures() -> tuple[int, int]:
    """Return ``(total, consecutive)`` progress-report failures for this process."""
    return _report_failures_total, _report_failures_consecutive


def reset_progress_report_failures() -> None:
    """Zero the counters. For tests and for a deliberate operational reset."""
    global _report_failures_total, _report_failures_consecutive
    _report_failures_total = 0
    _report_failures_consecutive = 0


async def report_progress() -> None:
    """Announce that a unit of work finished. No-op when nothing is listening.

    BEST-EFFORT BY DESIGN, and the ordering matters: this is a liveness signal,
    not part of the render. A failed bump must never take down a page that is
    genuinely succeeding — the worst case is one missed beat, which shortens the
    window exactly like a missed lease heartbeat does on the queue path
    (``job_worker._heartbeat`` makes the same trade for the same reason).

    But best-effort is not the same as invisible: every failure is COUNTED, and a
    run of them escalates to ERROR naming the count, because a reporter that is
    persistently failing means this batch is heading for a reap while alive.
    """
    global _report_failures_total, _report_failures_consecutive

    reporter = _PROGRESS_REPORTER.get()
    if reporter is None:
        return
    try:
        await reporter()
    except Exception:
        _report_failures_total += 1
        _report_failures_consecutive += 1
        if _report_failures_consecutive >= CONSECUTIVE_FAILURE_ALERT:
            logger.error(
                "Batch progress report has failed %d times in a row (%d total) — the "
                "progress clock is NOT being written, so a live batch will be reaped "
                "as abandoned. The render itself is unaffected and still running.",
                _report_failures_consecutive,
                _report_failures_total,
                exc_info=True,
            )
        else:
            logger.warning(
                "Batch progress report failed (%d in a row, %d total)",
                _report_failures_consecutive,
                _report_failures_total,
                exc_info=True,
            )
        return
    _report_failures_consecutive = 0
