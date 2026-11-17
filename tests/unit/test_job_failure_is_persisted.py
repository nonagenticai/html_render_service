"""A failing job must ALWAYS record why it failed.

``RedesignService.process_job`` has exactly one seam that persists a failure:
the ``except`` around the phase pipeline, which calls
``_phase_finalize_failed`` and then re-raises. That handler used to catch
``(OSError, ValueError, RuntimeError, TimeoutError)``.

None of the exceptions these phases actually raise are in that set:

* ``playwright.async_api.Error`` / ``TimeoutError`` -- phase 1
  ``_fetch_page_html`` and every screenshot phase. Playwright's ``TimeoutError``
  is **not** the builtin; it derives from Playwright's ``Error(Exception)``.
* ``httpx.HTTPError`` -- the WM call and webhook dispatch.
* ``redis.exceptions.RedisError`` -- the cancel check and job lock. redis-py's
  ``ConnectionError``/``TimeoutError`` merely *share their names* with the
  builtins; they derive from ``RedisError(Exception)``.

All derive from ``Exception``, not ``OSError``. So the most common real-world
failures skipped the handler entirely and the tenant's job was left with **no
recorded reason** -- the row never got its error, and the failure was invisible
to anyone reading the job.

These tests pin the outcome, not the tuple: whatever a phase raises, the failure
is recorded AND still propagates. The propagation half matters as much as the
recording -- this handler is `record and re-raise`, never `swallow`, so the
caller's behaviour (including ``CancelledError`` on a shutdown drain) is
unchanged.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import redis.exceptions
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.services.design_builder.service import RedesignService

# Exactly the exception types the phases raise that the old tuple could not see.
# Each is paired with the phase that raises it so a failure names the real path.
OFF_TUPLE_FAILURES = [
    pytest.param(PlaywrightError("nav failed"), id="playwright-error"),
    pytest.param(PlaywrightTimeoutError("nav timeout"), id="playwright-timeout"),
    pytest.param(httpx.ConnectError("wm unreachable"), id="httpx-connect"),
    pytest.param(
        httpx.HTTPStatusError(
            "502",
            request=httpx.Request("POST", "https://wm.invalid/"),
            response=httpx.Response(502),
        ),
        id="httpx-status",
    ),
    pytest.param(redis.exceptions.ConnectionError("redis down"), id="redis-connection"),
    pytest.param(redis.exceptions.RedisError("redis error"), id="redis-error"),
]

# Types the OLD tuple already caught. Kept so a regression that narrows the
# handler the other way -- catching only the new types -- is also caught.
IN_TUPLE_FAILURES = [
    pytest.param(OSError("disk gone"), id="oserror"),
    pytest.param(ValueError("bad shape"), id="valueerror"),
    pytest.param(RuntimeError("boom"), id="runtimeerror"),
    pytest.param(TimeoutError("builtin timeout"), id="builtin-timeout"),
]


def _service_with_failing_first_phase(exc: BaseException) -> tuple[RedesignService, AsyncMock]:
    """Build a service whose first pipeline phase raises *exc*.

    ``_phase_resolve_html`` is the first call inside the guarded block, so it is
    the earliest point at which a real phase failure can occur — and the one the
    old tuple most conspicuously missed, since it is where Playwright runs.

    Args:
        exc: The exception the phase should raise.

    Returns:
        The service, and the ``_phase_finalize_failed`` mock to assert on.
    """
    service = RedesignService.__new__(RedesignService)
    finalize_failed = AsyncMock()
    service._phase_finalize_failed = finalize_failed  # type: ignore[method-assign]
    service._dispatch_webhook = AsyncMock()  # type: ignore[method-assign]
    service._check_cancelled = AsyncMock(return_value=False)  # type: ignore[method-assign]
    service._phase_resolve_html = AsyncMock(side_effect=exc)  # type: ignore[method-assign]
    return service, finalize_failed


def _job() -> MagicMock:
    job = MagicMock()
    job.scope = "redesign"
    job.user_context = None
    job.site_analysis = None
    job.brand_info = None
    return job


async def _run(service: RedesignService) -> None:
    db = AsyncMock()
    with patch("src.services.design_builder.service._pkg") as pkg:
        pkg.return_value.JobManager.get_job = AsyncMock(return_value=_job())
        pkg.return_value.JobManager.mark_processing = AsyncMock()
        await service.process_job(db, "job-1", "tenant-1")


class TestFailureIsAlwaysRecorded:
    """Whatever a phase raises, the job's failure reaches the database."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", OFF_TUPLE_FAILURES)
    async def test_off_tuple_failure_is_still_finalized(self, exc: BaseException) -> None:
        """The exceptions the old tuple could not see are now recorded.

        Each of these derives from ``Exception`` but not ``OSError``, which is
        exactly why the previous handler skipped them and the job lost its
        reason.
        """
        service, finalize_failed = _service_with_failing_first_phase(exc)

        with pytest.raises(type(exc)):
            await _run(service)

        finalize_failed.assert_awaited_once()
        assert finalize_failed.await_args is not None
        assert finalize_failed.await_args.args[-1] is exc

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", IN_TUPLE_FAILURES)
    async def test_previously_caught_types_still_finalized(self, exc: BaseException) -> None:
        """Widening must not drop what the old tuple already handled."""
        service, finalize_failed = _service_with_failing_first_phase(exc)

        with pytest.raises(type(exc)):
            await _run(service)

        finalize_failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates_without_being_swallowed(self) -> None:
        """A shutdown drain must not be turned into a recorded job failure.

        ``CancelledError`` derives from ``BaseException``, not ``Exception``, so
        the broad clause does not catch it — the drain cancels the task and the
        row is left for startup reclaim rather than being stamped ``failed`` on
        work that was merely cut short.
        """
        service, finalize_failed = _service_with_failing_first_phase(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await _run(service)

        finalize_failed.assert_not_awaited()


class TestTheHierarchyThatMadeTheOldTupleDead:
    """Executable documentation for why the previous handler was unreachable."""

    @pytest.mark.parametrize(
        "exc_type",
        [PlaywrightError, PlaywrightTimeoutError, httpx.HTTPError, redis.exceptions.RedisError],
    )
    def test_phase_exceptions_are_not_oserror(self, exc_type: type[BaseException]) -> None:
        """None of these is an ``OSError``, so an OSError-based tuple misses all."""
        assert not issubclass(exc_type, OSError)
        assert issubclass(exc_type, Exception)

    def test_the_same_named_classes_are_not_the_builtins(self) -> None:
        """The name collision is what made this invisible on reading."""
        assert redis.exceptions.ConnectionError is not ConnectionError
        assert redis.exceptions.TimeoutError is not TimeoutError
        assert PlaywrightTimeoutError is not TimeoutError
