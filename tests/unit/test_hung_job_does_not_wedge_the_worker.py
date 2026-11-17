"""A job hung inside ``_dispatch`` must not consume the pod's only worker loop.

WHY THIS FILE EXISTS. ``_worker_loop`` calls ``claim_next_job`` only after
``_run_claimed`` returns, ``worker_pool_size`` defaults to 1, and the deployment
is pinned ``minReplicas: 1 / maxReplicas: 1``. So ONE await that never returns
used to stop the pod draining its queue for the life of the process — with no
exception, no ERROR line, and ``/health`` legitimately green (``status()`` counts
``not task.done()``, and a blocked task is not done).

#112 stopped the heartbeat renewing such a job's lease past
``max_job_lifetime_seconds``, so the lease lapses and the existing ceiling in
``claim_next_job`` writes the real failure reason. That recovery needs SOME
worker to reach ``claim_next_job``. The only worker is the wedged one. These
tests pin the other half: at the SAME ceiling, ``_run_claimed`` stops waiting.

At the ceiling the dispatch is now CANCELLED (it used to be left running: its writes
are fenced off by then, so it only kept spending LLM calls and a browser lease). A
dispatch that swallows the cancellation is abandoned after ``CANCEL_GRACE_SECONDS``.
Every test here also pins what the fix must NOT do — fire on a healthy job, wait
forever on a task that refuses to stop, or let the abandoned dispatch's identity leak
into the next claim.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace

import pytest
from w2v_common.auth import clear_identity_context, get_tenant_id_from_context

from src.services import job_worker
from src.services.job_manager import ClaimedJob, JobManager
from tests.tenants import TENANT_1, TENANT_2

# The smallest ceiling the production knob (an int-seconds Field) can express.
# Everything here is driven by real elapsed time, so keep it at 1.
CEILING_S = 1

# The DB-backed test needs a bigger one. ``claim_next_job`` shares this setting
# and measures ``running_for`` as host-clock ``now()`` minus a ``started_at``
# stamped by the POSTGRES clock. At 1s the sub-second skew between the host and
# the test postgres container is enough to age out the *healthy* job the instant
# it is claimed, which reddens this test for a reason that is not the code's.
POOL_CEILING_S = 4


@pytest.fixture(autouse=True)
def _short_cancel_grace(monkeypatch):
    monkeypatch.setattr(job_worker, "CANCEL_GRACE_SECONDS", 0.5)


async def _uncancellable_wait(event: asyncio.Event) -> None:
    """Work that swallows cancellation — the shape the grace period exists for."""
    while not event.is_set():
        with contextlib.suppress(asyncio.CancelledError):
            await event.wait()


@pytest.fixture(autouse=True)
def _clean_identity():
    clear_identity_context()
    yield
    clear_identity_context()


def _app():
    # session_factory=None is safe: with lease_seconds=30 the heartbeat's first
    # renewal is 10s away and every test here finishes long before that.
    return SimpleNamespace(state=SimpleNamespace(session_factory=None))


async def _run_abandoning(claimed: ClaimedJob, worker_id: str = "pod") -> None:
    """``_run_claimed`` on a job whose dispatch never returns, under a hard bound.

    The bound is generously above ``CEILING_S`` so it can only be hit by the
    defect itself: WITHOUT the fix this call never returns at all, and the bound
    turns a permanently wedged worker into a readable failure rather than a hung
    suite. It is a test harness, not part of the assertion.
    """
    await asyncio.wait_for(
        job_worker._run_claimed(
            _app(),
            claimed,
            lease_seconds=30,
            worker_id=worker_id,
            max_lifetime_seconds=CEILING_S,
        ),
        timeout=15,
    )


# ---------------------------------------------------------------------------
# The core assertion, end to end through the real pool + real queue.
# ---------------------------------------------------------------------------


async def test_a_hung_job_does_not_stop_the_loop_claiming_the_next_job(
    db_session_factory, monkeypatch
) -> None:
    """THE defect. Job A hangs forever; job B must still be claimed and run.

    Driven through ``JobWorkerPool`` with the production default of ONE worker,
    so it fails for exactly the reason production fails: the single loop is
    parked inside ``_run_claimed``.
    """
    async with db_session_factory() as s:
        job_a = await JobManager.create_job(
            s, tenant_id=TENANT_1, url="https://hangs.example", scope="hero"
        )
        await JobManager.set_job_options(s, job_a.job_id, {"kind": "redesign", "hang": True})
        job_b = await JobManager.create_job(
            s, tenant_id=TENANT_1, url="https://healthy.example", scope="hero"
        )
        await JobManager.set_job_options(s, job_b.job_id, {"kind": "redesign"})
        await s.commit()

    hung_started = asyncio.Event()
    second_ran = asyncio.Event()
    hung_task: dict[str, asyncio.Task] = {}

    async def _dispatch(app, claimed) -> None:
        if claimed.options.get("hang"):
            task = asyncio.current_task()
            assert task is not None
            hung_task["t"] = task
            hung_started.set()
            await asyncio.Event().wait()  # never returns, exactly like the bug
            return
        second_ran.set()

    monkeypatch.setattr(job_worker, "_dispatch", _dispatch)
    settings = job_worker.get_settings()
    monkeypatch.setattr(settings.redesign, "max_job_lifetime_seconds", POOL_CEILING_S)
    monkeypatch.setattr(settings.redesign, "worker_pool_size", 1)

    pool = job_worker.JobWorkerPool(
        SimpleNamespace(state=SimpleNamespace(session_factory=db_session_factory))
    )
    await pool.start()
    try:
        await asyncio.wait_for(hung_started.wait(), timeout=10)
        # Before the fix this wait_for times out: the one worker is parked in
        # `await _dispatch(...)` and never reaches claim_next_job again.
        await asyncio.wait_for(second_ran.wait(), timeout=20)
    finally:
        await pool.stop()
        t = hung_task.get("t")
        if t is not None:
            t.cancel()

    # ...and the wedged job was CANCELLED at the ceiling, not left spending.
    assert hung_task["t"].cancelled() is True


# ---------------------------------------------------------------------------
# The cancellation at the ceiling, isolated.
# ---------------------------------------------------------------------------


async def test_the_expired_dispatch_is_cancelled() -> None:
    observed: dict[str, object] = {}

    async def _dispatch(app, claimed) -> None:
        observed["task"] = asyncio.current_task()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed["cancelled"] = True
            raise

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(job_worker, "_dispatch", _dispatch)
        await _run_abandoning(ClaimedJob(job_id="hung", tenant_id=TENANT_1, options={}))

        task = observed["task"]
        assert isinstance(task, asyncio.Task)
        assert task.done() and task.cancelled()
        assert observed.get("cancelled") is True


async def test_a_dispatch_that_ignores_cancellation_is_abandoned_not_awaited_forever(
    caplog,
) -> None:
    released = asyncio.Event()
    observed: dict[str, object] = {}

    async def _dispatch(app, claimed) -> None:
        observed["task"] = asyncio.current_task()
        await _uncancellable_wait(released)

    with pytest.MonkeyPatch.context() as mp, caplog.at_level(logging.ERROR):
        mp.setattr(job_worker, "_dispatch", _dispatch)
        await _run_abandoning(ClaimedJob(job_id="stubborn", tenant_id=TENANT_1, options={}))

        task = observed["task"]
        assert isinstance(task, asyncio.Task) and not task.done()
        assert any("ABANDONING" in r.getMessage() for r in caplog.records)
        released.set()
        await asyncio.wait_for(task, timeout=5)


# ---------------------------------------------------------------------------
# The regression guard that matters most.
# ---------------------------------------------------------------------------


async def test_a_normal_job_is_untouched_and_logs_no_abandonment(caplog) -> None:
    """A healthy job must complete, return promptly, and produce no ERROR.

    A ceiling that fires on healthy work would be far worse than the bug: it
    would abandon (and then double-count) every render that merely takes a while.
    """
    ran = asyncio.Event()

    async def _dispatch(app, claimed) -> None:
        ran.set()

    with pytest.MonkeyPatch.context() as mp, caplog.at_level(logging.ERROR):
        mp.setattr(job_worker, "_dispatch", _dispatch)
        await job_worker._run_claimed(
            _app(),
            ClaimedJob(job_id="fine", tenant_id=TENANT_1, options={}),
            lease_seconds=30,
            worker_id="pod",
            max_lifetime_seconds=CEILING_S,
        )

    assert ran.is_set()
    assert not [r for r in caplog.records if "ABANDONING" in r.getMessage()]


async def test_a_dispatch_that_raises_still_propagates_to_the_loop() -> None:
    """Running the dispatch as a task must not swallow its exception.

    ``_worker_loop`` relies on the raise to log and back off; a task whose
    exception is never retrieved would turn every render failure into silence.
    """

    async def _dispatch(app, claimed) -> None:
        raise RuntimeError("render blew up")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(job_worker, "_dispatch", _dispatch)
        with pytest.raises(RuntimeError, match="render blew up"):
            await job_worker._run_claimed(
                _app(),
                ClaimedJob(job_id="boom", tenant_id=TENANT_1, options={}),
                lease_seconds=30,
                worker_id="pod",
                max_lifetime_seconds=CEILING_S,
            )


async def test_a_dispatch_raising_timeouterror_is_not_mistaken_for_the_ceiling() -> None:
    """``wait_for`` would conflate these two. The ceiling must not eat a real one.

    If the ceiling seam cannot tell "my deadline fired" from "the render itself
    raised TimeoutError", a plain render timeout gets logged as a wedge and — far
    worse — swallowed instead of raised, so the loop never marks the job failed.
    """

    async def _dispatch(app, claimed) -> None:
        raise TimeoutError("the render's own client timed out")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(job_worker, "_dispatch", _dispatch)
        with pytest.raises(TimeoutError, match="own client timed out"):
            await job_worker._run_claimed(
                _app(),
                ClaimedJob(job_id="inner-timeout", tenant_id=TENANT_1, options={}),
                lease_seconds=30,
                worker_id="pod",
                max_lifetime_seconds=CEILING_S,
            )


# ---------------------------------------------------------------------------
# Context hygiene across the abandonment.
# ---------------------------------------------------------------------------


async def test_identity_does_not_leak_from_an_abandoned_dispatch_into_the_next_claim() -> None:
    """The abandoned task keeps ITS tenant; the worker context is left clean.

    Two directions, both load-bearing. If ``clear_identity_context`` in the
    ``finally`` reached the orphan, the orphan's next outbound WM call would go
    out with an unbound tenant (``UnvouchedTenantError``, or worse, misattributed
    spend). If it did NOT clear the worker's own context, tenant 1's identity
    would still be bound while the loop dispatches tenant 2's job.
    """
    orphan_saw_after: dict[str, object] = {}
    released = asyncio.Event()
    orphan_done = asyncio.Event()
    second_saw: dict[str, object] = {}

    async def _dispatch(app, claimed) -> None:
        if claimed.job_id == "hung":
            await _uncancellable_wait(released)  # still running while the next job dispatches
            orphan_saw_after["tenant"] = get_tenant_id_from_context()
            orphan_done.set()
            return
        second_saw["tenant"] = get_tenant_id_from_context()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(job_worker, "_dispatch", _dispatch)
        await _run_abandoning(ClaimedJob(job_id="hung", tenant_id=TENANT_1, options={}))

        # The worker context is clean the moment the abandonment returns.
        assert get_tenant_id_from_context() is None

        await job_worker._run_claimed(
            _app(),
            ClaimedJob(job_id="next", tenant_id=TENANT_2, options={}),
            lease_seconds=30,
            worker_id="pod",
            max_lifetime_seconds=CEILING_S,
        )
        assert str(second_saw["tenant"]) == TENANT_2

        # Only now let the orphan finish: it must still hold ITS OWN tenant,
        # neither cleared by the finally nor overwritten by the second job.
        released.set()
        await asyncio.wait_for(orphan_done.wait(), timeout=5)
        assert str(orphan_saw_after["tenant"]) == TENANT_1


async def test_the_abandoned_dispatch_keeps_the_claim_owner_fence() -> None:
    """An orphan's terminal writes must stay fenced on the worker id it claimed
    under. ``claim_owner`` is a ContextVar reset by the ``finally``; if the
    dispatch did not run in its own context copy, the reset would unbind the
    orphan's owner and its ``mark_failed`` would degrade to "unclaimed rows
    only" — or, worse, land on a row a sibling had re-claimed."""
    from src.services.job_manager import get_claim_owner

    seen: dict[str, object] = {}
    released = asyncio.Event()
    done = asyncio.Event()

    async def _dispatch(app, claimed) -> None:
        await _uncancellable_wait(released)
        seen["owner"] = get_claim_owner()
        done.set()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(job_worker, "_dispatch", _dispatch)
        await _run_abandoning(
            ClaimedJob(job_id="hung", tenant_id=TENANT_1, options={}), worker_id="pod-7"
        )
        assert get_claim_owner() is None  # the worker loop's own context is clean

        released.set()
        await asyncio.wait_for(done.wait(), timeout=5)
        assert seen["owner"] == "pod-7"
