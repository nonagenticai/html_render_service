"""The lease heartbeat must not renew a job past ``max_job_lifetime_seconds``.

``claim_next_job`` already refuses to re-claim a job that has been running longer
than the ceiling — but it only ever sees a job whose lease LAPSED. ``_heartbeat``
renewed the lease unconditionally for as long as ``_dispatch`` was awaiting, and
nothing up that chain has a timeout, so a hung render kept its lease alive
forever, was never re-claimable, and never reached the ceiling. With
``worker_pool_size`` defaulting to 1 that wedges the pod's whole queue, silently:
no exception, no ERROR line, ``/health`` green, and no reaper exists to notice.

These tests are deliberately DB-free: they drive ``_heartbeat`` directly with a
fake session factory and count the renewals it issues.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace

import pytest

from src.services import job_worker


class _FakeSessionFactory:
    """Async-context-manager session factory whose sessions do nothing."""

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self


async def _run_heartbeat(
    monkeypatch,
    *,
    max_lifetime_seconds: int,
    clock: list[float],
    beats_before_stop: int = 4,
) -> list[str]:
    """Drive ``_heartbeat`` over a fake clock; return the job ids it renewed."""
    renewed: list[str] = []

    async def _fake_renew(db, job_id, lease_seconds):
        renewed.append(job_id)

    monkeypatch.setattr(job_worker.JobManager, "renew_lease", _fake_renew)

    # Advance a synthetic monotonic clock by one step per beat, so the test never
    # actually waits and elapsed time is fully deterministic. The shims replace
    # the NAMES in job_worker's own globals, never attributes on the shared
    # stdlib modules — patching ``time.monotonic`` or ``asyncio.wait_for``
    # globally would reach into pytest-asyncio's own event loop machinery.
    ticks = iter(clock)
    last = [clock[0]]

    def _fake_monotonic() -> float:
        value = next(ticks, None)
        if value is None:
            return last[0]
        last[0] = value
        return value

    monkeypatch.setattr(job_worker, "time", SimpleNamespace(monotonic=_fake_monotonic))

    beats = [0]

    # ``**_kw`` swallows the ``timeout=`` keyword rather than naming it: a
    # parameter literally called ``timeout`` on an async def trips ruff ASYNC109.
    async def _never_stops(coro, **_kw):
        # Stand in for ``asyncio.wait_for(stop.wait(), ...)``: always "times out"
        # (i.e. the job is still running) until the test has seen enough beats.
        coro.close()
        beats[0] += 1
        if beats[0] > beats_before_stop:
            return True  # stop set → heartbeat returns
        raise TimeoutError

    monkeypatch.setattr(job_worker, "asyncio", SimpleNamespace(wait_for=_never_stops))

    stop = asyncio.Event()
    with contextlib.suppress(TimeoutError):
        await job_worker._heartbeat(
            _FakeSessionFactory(),
            "job-hung",
            30,
            stop,
            max_lifetime_seconds,
        )
    return renewed


@pytest.mark.asyncio
async def test_heartbeat_stops_renewing_once_past_the_max_lifetime(monkeypatch, caplog) -> None:
    """A job still running past the ceiling must have its lease ABANDONED, loudly.

    The clock jumps to 100s of elapsed time on the second beat, well past the 60s
    cap. Exactly one renewal (the first, still inside the cap) may be issued;
    after that the heartbeat must return so the lease can lapse and the existing
    ``claim_next_job`` ceiling can fire and write the real failure reason.
    """
    with caplog.at_level(logging.ERROR, logger=job_worker.logger.name):
        renewed = await _run_heartbeat(
            monkeypatch,
            max_lifetime_seconds=60,
            # start, beat-1 elapsed=10s (renew), beat-2 elapsed=100s (abandon)
            clock=[0.0, 10.0, 100.0, 200.0, 300.0],
        )

    assert renewed == ["job-hung"], (
        "the heartbeat renewed the lease past max_job_lifetime_seconds; a hung "
        f"job stays un-reclaimable forever (renewals: {renewed})"
    )
    loud = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert loud, "abandoning the lease must be logged at ERROR, not silently"
    message = loud[0].getMessage()
    assert "job-hung" in message and "100" in message, (
        f"the log line must name the job and its elapsed seconds; got: {message!r}"
    )


@pytest.mark.asyncio
async def test_heartbeat_keeps_renewing_a_job_inside_the_lifetime(monkeypatch) -> None:
    """The bound must not clip a healthy long render that is still under the cap."""
    renewed = await _run_heartbeat(
        monkeypatch,
        max_lifetime_seconds=3600,
        clock=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        beats_before_stop=4,
    )
    assert renewed == ["job-hung"] * 4


@pytest.mark.asyncio
async def test_heartbeat_is_unbounded_when_the_ceiling_is_disabled(monkeypatch) -> None:
    """``max_lifetime_seconds <= 0`` means "no ceiling" — same convention as claim_next_job."""
    renewed = await _run_heartbeat(
        monkeypatch,
        max_lifetime_seconds=0,
        clock=[0.0, 10.0, 100_000.0, 200_000.0, 300_000.0],
        beats_before_stop=4,
    )
    assert renewed == ["job-hung"] * 4
