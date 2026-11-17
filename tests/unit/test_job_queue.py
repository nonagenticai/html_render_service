"""Durable claim-queue (Phase 1): lease-based claiming + reclaim.

DB-backed (real postgres, per the repo's conftest fixtures). Replaces the old
startup-reap tests: orphaned jobs are no longer blanket-failed at boot, they are
reclaimed via an expired lease by the next worker.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.config import get_settings
from src.services.design_builder.types import RenderOptions
from src.services.job_manager import JobManager
from src.services.job_worker import _render_options_from_dict, build_redesign_options
from tests.tenants import NIL_TENANT_SPELLINGS, TENANT_1


async def _enqueue(db, url: str, options: dict | None = None) -> str:
    """Insert a queue-owned job (job_options set) and commit."""
    job = await JobManager.create_job(db, tenant_id=TENANT_1, url=url, scope="hero")
    await JobManager.set_job_options(db, job.job_id, options or {"kind": "redesign"})
    await db.commit()
    return job.job_id


async def _get(db, job_id: str):
    """Fetch a job, asserting it exists (narrows the Optional for the type checker)."""
    job = await JobManager.get_job(db, job_id, tenant_id=None)
    assert job is not None
    return job


async def test_claim_marks_processing_with_lease(db_session) -> None:
    jid = await _enqueue(db_session, "https://a.example", {"kind": "redesign", "x": 1})
    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=180, max_attempts=3
    )
    assert claimed is not None
    assert claimed.job_id == jid
    assert claimed.options == {"kind": "redesign", "x": 1}

    # ``ClaimedJob.tenant_id`` MUST be a plain ``str``, equal to the tenant that
    # enqueued the job. Since 0010_tenant_id_uuid the SECURITY DEFINER resolver
    # declares ``tenant_id uuid`` and ``claim_next_job`` reads it through a raw
    # ``text()`` query, so psycopg hands back a ``uuid.UUID`` — which
    # ``claim_next_job`` coerces. Without that coercion this assertion is False
    # (``uuid.UUID(t) != t``), and in production the worker would thread a UUID
    # object into every per-job scoped read and match nothing, silently, with no
    # type error to catch it. This is the regression test for that coercion.
    assert isinstance(claimed.tenant_id, str)
    assert claimed.tenant_id == TENANT_1

    job = await _get(db_session, jid)
    assert job is not None
    assert job.status == "processing"
    assert job.claimed_by == "pod"
    assert job.attempts == 1
    assert job.lease_expires_at is not None


async def test_job_without_options_is_not_claimed(db_session) -> None:
    # Batch child rows carry no job_options and must be left to their own
    # orchestration — the worker pool must not touch them.
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="https://b.example", scope="hero"
    )
    await db_session.commit()
    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=180, max_attempts=3
    )
    assert claimed is None
    assert (await _get(db_session, job.job_id)).status == "pending"


async def test_fresh_lease_blocks_reclaim_expired_allows_it(db_session) -> None:
    jid = await _enqueue(db_session, "https://c.example")
    first = await JobManager.claim_next_job(
        db_session, worker_id="podA", lease_seconds=180, max_attempts=3
    )
    assert first is not None

    # Fresh lease → not reclaimable by another worker.
    assert (
        await JobManager.claim_next_job(
            db_session, worker_id="podB", lease_seconds=180, max_attempts=3
        )
        is None
    )

    # Expire the lease (simulate the claiming pod crashing).
    job = await _get(db_session, jid)
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    reclaimed = await JobManager.claim_next_job(
        db_session, worker_id="podB", lease_seconds=180, max_attempts=3
    )
    assert reclaimed is not None
    assert reclaimed.job_id == jid
    job = await _get(db_session, jid)
    assert job.claimed_by == "podB"
    assert job.attempts == 2  # reclaim counts as a new attempt


async def test_poison_job_is_failed_after_max_attempts(db_session) -> None:
    jid = await _enqueue(db_session, "https://d.example")
    job = await _get(db_session, jid)
    job.status = "processing"
    job.attempts = 3
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=180, max_attempts=3
    )
    assert claimed is None
    job = await _get(db_session, jid)
    assert job.status == "failed"
    assert "poison" in (job.error_message or "")
    assert job.completed_at is not None


async def test_renew_lease_extends(db_session) -> None:
    jid = await _enqueue(db_session, "https://e.example")
    await JobManager.claim_next_job(db_session, worker_id="pod", lease_seconds=10, max_attempts=3)
    before = (await _get(db_session, jid)).lease_expires_at
    await JobManager.renew_lease(db_session, jid, 600)
    await db_session.commit()
    after = (await _get(db_session, jid)).lease_expires_at
    assert before is not None and after is not None
    assert after > before


async def test_release_claims_makes_jobs_reclaimable(db_session) -> None:
    jid = await _enqueue(db_session, "https://f.example")
    await JobManager.claim_next_job(db_session, worker_id="podX", lease_seconds=600, max_attempts=3)
    released = await JobManager.release_claims(db_session, "podX")
    await db_session.commit()
    assert released == 1

    reclaimed = await JobManager.claim_next_job(
        db_session, worker_id="podY", lease_seconds=600, max_attempts=3
    )
    assert reclaimed is not None
    assert reclaimed.job_id == jid


async def test_graceful_release_refunds_the_attempt(db_session) -> None:
    """A shutdown handoff is not a failed attempt, so it must not be charged as one.

    ``attempts`` is a *crash* counter (it gates the poison verdict). A graceful
    release surrenders the claim without a verdict — the job never got its try.
    """
    jid = await _enqueue(db_session, "https://refund.example")
    await JobManager.claim_next_job(db_session, worker_id="podX", lease_seconds=600, max_attempts=3)
    assert (await _get(db_session, jid)).attempts == 1  # the claim charged one

    await JobManager.release_claims(db_session, "podX")
    await db_session.commit()
    assert (await _get(db_session, jid)).attempts == 0  # ...and the release refunded it


async def test_rollout_churn_never_poisons_a_healthy_job(db_session) -> None:
    """The real website_redesign poison bug: deploys must not kill in-flight jobs.

    A redesign runs for minutes (16k-token WM call + multi-viewport Chromium) on a
    ``replicas:2 + HPA`` deployment, so it is routinely caught mid-render by a
    rollout / scale-down / node drain. Before the refund, each such graceful handoff
    burned one of the job's three lives and a perfectly healthy job died as
    "exceeded max claim attempts" after the third deploy. Far more churn than that
    must now be survivable.
    """
    jid = await _enqueue(db_session, "https://churn.example")

    for i in range(6):  # 2x the max_attempts=3 that used to poison it
        claimed = await JobManager.claim_next_job(
            db_session, worker_id=f"pod{i}", lease_seconds=600, max_attempts=3
        )
        assert claimed is not None, f"healthy job was poisoned by rollout #{i + 1}"
        assert claimed.job_id == jid
        # The pod is rolled: ASGI lifespan shutdown → JobWorkerPool.stop() → release.
        await JobManager.release_claims(db_session, f"pod{i}")
        await db_session.commit()

    job = await _get(db_session, jid)
    assert job.status == "processing"  # still alive and reclaimable, never poisoned
    assert job.error_message is None
    assert job.attempts == 0  # every graceful handoff refunded


async def test_max_lifetime_fails_a_job_that_can_never_finish(
    db_session, db_session_factory, monkeypatch
) -> None:
    """The bound that replaces the one the attempt-refund removes — reached HONESTLY.

    Refunding graceful handoffs means sustained churn could otherwise re-claim a job
    forever, replaying the (paid) WM call every time. The wall-clock ceiling stops it.

    ⚠️ This test used to hand-write ``job.lease_expires_at = now - 1s`` before the
    second claim. That manufactured a precondition production could NOT reach: the
    ceiling in ``claim_next_job`` only ever sees a job whose lease has LAPSED, and
    ``_heartbeat`` renewed the lease unconditionally for as long as ``_dispatch``
    was awaiting. So the branch this test asserts was, in the real system,
    unreachable for exactly the job it exists to kill — a hung one. The test was
    green throughout; the pod wedged anyway (fixed in #112).

    With the heartbeat now bounded, the lapse is produced by the PRODUCTION path:
    the real ``_heartbeat`` is driven over the job's real row and must stop calling
    ``renew_lease`` once past the ceiling, the real lease then expires on its own
    wall clock, and only then does ``claim_next_job`` get its turn. The single
    remaining hand-write is ``started_at`` — a test cannot wait two hours, and that
    column is an input to the ceiling, not the state the bug made unreachable.
    """
    from src.services import job_worker

    jid = await _enqueue(db_session, "https://forever.example")
    # A 1s lease so the genuine expiry is observable inside a test; the ceiling
    # is what decides the outcome, the lease length only decides how long we wait.
    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=1, max_attempts=3, max_lifetime_seconds=3600
    )
    assert claimed is not None

    # It has been running (across churn) for longer than the cap allows.
    job = await _get(db_session, jid)
    job.started_at = datetime.now(UTC) - timedelta(seconds=7200)
    await db_session.commit()

    # --- the heartbeat must ABANDON the lease rather than renew it forever ----
    # renew_lease is wrapped, not replaced: every call still performs the real DB
    # write, we only count them.
    renewals: list[str] = []
    real_renew = JobManager.renew_lease

    async def _counting_renew(db, job_id: str, lease_seconds: int) -> None:
        renewals.append(job_id)
        await real_renew(db, job_id, lease_seconds)

    monkeypatch.setattr(job_worker.JobManager, "renew_lease", _counting_renew)

    # Synthetic monotonic clock: beat 1 is 10s into the run (inside the cap, so it
    # renews), beat 2 is 7200s in (past it, so it must give up). Patching the NAMES
    # in job_worker's own globals, never the shared stdlib modules — see
    # tests/unit/test_heartbeat_lifetime_bound.py for why.
    ticks = iter([0.0, 10.0, 7200.0, 7300.0, 7400.0, 7500.0])
    last = [0.0]

    def _fake_monotonic() -> float:
        value = next(ticks, None)
        if value is not None:
            last[0] = value
        return last[0]

    monkeypatch.setattr(job_worker, "time", SimpleNamespace(monotonic=_fake_monotonic))

    beats = [0]

    async def _still_running(coro, **_kw):
        # Stand in for `asyncio.wait_for(stop.wait(), ...)`: the job never finishes,
        # so every wait times out. Bounded so an unbounded heartbeat cannot hang the
        # suite — it would fail the renewal assertion below instead.
        coro.close()
        beats[0] += 1
        if beats[0] > 5:
            return True
        raise TimeoutError

    monkeypatch.setattr(job_worker, "asyncio", SimpleNamespace(wait_for=_still_running))

    await job_worker._heartbeat(db_session_factory, jid, 1, asyncio.Event(), 3600)
    assert renewals == [jid], (
        "the heartbeat kept renewing a job past max_job_lifetime_seconds, so its "
        f"lease can never lapse and the ceiling below is unreachable (renewals: {renewals})"
    )
    monkeypatch.undo()

    # --- the lease now expires on its own; no timestamp is written by hand ------
    db_session.expunge_all()
    job = await _get(db_session, jid)
    lease = job.lease_expires_at
    assert lease is not None
    if lease.tzinfo is None:
        lease = lease.replace(tzinfo=UTC)
    await asyncio.sleep(max(0.0, (lease - datetime.now(UTC)).total_seconds()) + 0.2)
    # End the read transaction before claiming. The resolver's `lease_expires_at <
    # now()` uses postgres `now()` == TRANSACTION start time, so a transaction
    # opened by the read above would evaluate the lease against a clock reading
    # from before the sleep and find nothing expired.
    await db_session.commit()

    reclaimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=600, max_attempts=3, max_lifetime_seconds=3600
    )
    assert reclaimed is None  # not handed to a worker again
    job = await _get(db_session, jid)
    assert job.status == "failed"
    assert "max job lifetime" in (job.error_message or "")


async def test_backlogged_job_is_never_aged_out_before_it_starts(db_session) -> None:
    """The ceiling is measured from first claim, not creation.

    A job queued behind a backlog has consumed nothing. Ageing it out on ``created_at``
    would mass-fail a queue that is merely behind — the opposite of the bug being fixed.
    """
    jid = await _enqueue(db_session, "https://backlog.example")
    job = await _get(db_session, jid)
    job.created_at = datetime.now(UTC) - timedelta(seconds=7200)  # long queued, never started
    await db_session.commit()

    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod", lease_seconds=600, max_attempts=3, max_lifetime_seconds=3600
    )
    assert claimed is not None  # still runnable
    assert claimed.job_id == jid
    assert (await _get(db_session, jid)).status == "processing"


async def test_crashes_still_poison_after_max_attempts(db_session) -> None:
    """The refund must not defang real poison detection.

    A crash (OOMKill/SIGKILL) never reaches ``release_claims`` — the lease just
    expires unrefunded — so a payload that keeps killing its worker still gets the
    poison verdict on the claim after ``max_attempts``.
    """
    jid = await _enqueue(db_session, "https://crash.example")

    for _ in range(3):
        claimed = await JobManager.claim_next_job(
            db_session, worker_id="pod", lease_seconds=600, max_attempts=3
        )
        assert claimed is not None
        # Hard death: no graceful release, the lease merely lapses.
        job = await _get(db_session, jid)
        job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db_session.commit()

    assert (await _get(db_session, jid)).attempts == 3
    assert (
        await JobManager.claim_next_job(
            db_session, worker_id="pod", lease_seconds=600, max_attempts=3
        )
        is None
    )
    job = await _get(db_session, jid)
    assert job.status == "failed"
    assert "poison" in (job.error_message or "")


async def test_drain_lets_an_in_flight_render_finish(db_session_factory, monkeypatch) -> None:
    """SIGTERM must not throw away a render that is already running (and already paid for).

    The old stop() cancelled workers outright, so an in-flight job lost its 16k-token
    WM call and the next claim replayed it. The drain lets it finish.
    """
    from src.services import job_worker
    from src.services.job_worker import JobWorkerPool

    async with db_session_factory() as s:
        await _enqueue(s, "https://drain.example")

    started = asyncio.Event()
    finished = asyncio.Event()

    async def _slow_dispatch(app, claimed) -> None:
        started.set()  # we are now genuinely mid-render
        await asyncio.sleep(0.3)  # ...and still rendering when stop() is called
        finished.set()

    monkeypatch.setattr(job_worker, "_dispatch", _slow_dispatch)

    app = SimpleNamespace(state=SimpleNamespace(session_factory=db_session_factory))
    pool = JobWorkerPool(app)
    await pool.start()
    # Synchronise on the render actually starting. Sleeping a fixed amount here races
    # on a slow runner: stop() could land before the worker has even claimed, and the
    # test would then prove nothing.
    await asyncio.wait_for(started.wait(), timeout=10)

    await pool.stop()  # drain_timeout default (25s) >> the 0.3s render

    assert finished.is_set(), "drain cancelled an in-flight render instead of letting it finish"


async def test_drain_cancels_work_that_outlives_the_deadline(
    db_session_factory, monkeypatch
) -> None:
    """The drain is bounded — it cannot hold the pod past its termination grace period."""
    from src.services import job_worker
    from src.services.job_worker import JobWorkerPool

    async with db_session_factory() as s:
        await _enqueue(s, "https://slowdrain.example")

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _endless_dispatch(app, claimed) -> None:
        started.set()  # genuinely mid-render before stop() lands
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(job_worker, "_dispatch", _endless_dispatch)
    settings = get_settings()
    monkeypatch.setattr(settings.redesign, "drain_timeout_seconds", 0)  # deadline already passed

    app = SimpleNamespace(state=SimpleNamespace(session_factory=db_session_factory))
    pool = JobWorkerPool(app)
    await pool.start()
    # Wait for the render to actually be in flight. A fixed sleep races on a slow
    # runner: if stop() lands before the claim, the worker exits cleanly with nothing
    # to cancel and the assertion fails for a reason that is not the code's fault.
    # (This is exactly how it failed in CI while passing locally.)
    await asyncio.wait_for(started.wait(), timeout=10)

    await pool.stop()

    assert cancelled.is_set(), "a job outliving the drain deadline must be cancelled"


async def test_concurrent_claims_never_collide(db_session_factory) -> None:
    async with db_session_factory() as s:
        await _enqueue(s, "https://g1.example")
        await _enqueue(s, "https://g2.example")

    async def _claim() -> str | None:
        async with db_session_factory() as s:
            c = await JobManager.claim_next_job(
                s, worker_id="pod", lease_seconds=180, max_attempts=3
            )
            return c.job_id if c else None

    a, b = await asyncio.gather(_claim(), _claim())
    ids = {x for x in (a, b) if x is not None}
    assert len(ids) == 2  # FOR UPDATE SKIP LOCKED: never the same row twice


async def test_worker_restores_then_clears_identity_context(monkeypatch) -> None:
    from types import SimpleNamespace

    from w2v_common.auth import (
        clear_identity_context,
        get_subject_id_from_context,
        get_tenant_id_from_context,
    )

    from src.services import job_worker
    from src.services.job_manager import ClaimedJob

    # ``get_tenant_id_from_context`` returns ``uuid.UUID | None`` since
    # w2v_common 0.67.0 — the ContextVar is no longer a ``str`` defaulting to
    # ``""``. Typing this dict ``str`` is what ``ty check .`` caught, and the
    # honest repair is to assert the UUID, not to stringify it away: the whole
    # point of the change is that "unbound" is ``None``, distinguishable from
    # "bound to nothing".
    seen: dict[str, uuid.UUID | str | None] = {}

    async def fake_dispatch(app, claimed) -> None:
        seen["tenant"] = get_tenant_id_from_context()
        seen["subject"] = get_subject_id_from_context()

    monkeypatch.setattr(job_worker, "_dispatch", fake_dispatch)
    app = SimpleNamespace(state=SimpleNamespace(session_factory=None))
    claimed = ClaimedJob(
        job_id="j1",
        tenant_id=TENANT_1,
        options={"identity": {"license_id": "lic-1", "subject_id": "subj-1"}},
    )

    clear_identity_context()
    await job_worker._run_claimed(app, claimed, lease_seconds=30, worker_id="pod")

    # Identity was visible to the dispatch (audit headers reproduce the old
    # create_task context inheritance) and is cleared afterwards.
    assert seen["tenant"] == uuid.UUID(TENANT_1)
    assert seen["subject"] == "subj-1"
    # UNBOUND is ``None`` now, not ``""``. Asserting ``is None`` is stricter
    # than the old ``== ""``: an empty string was itself a sentinel that made
    # "no tenant" indistinguishable from "tenant bound to nothing", and it is
    # what used to render as a blank ``X-Tenant-Id`` on outbound calls.
    assert get_tenant_id_from_context() is None


async def test_worker_restores_then_clears_correlation_context(monkeypatch) -> None:
    """Cost-correlation (run_id + stage) survives the enqueue→detached-worker hop.

    The detached worker loop doesn't inherit the enqueue request's ContextVars, so
    correlation must round-trip through the persisted ``identity`` bundle. Without
    this, the WM LLM call the worker makes emits a cost_event with correlation_id
    NULL and billing can't tie the spend to the orchestrator run.
    """
    from types import SimpleNamespace

    from w2v_common.auth import (
        clear_identity_context,
        get_correlation_id_from_context,
        get_stage_from_context,
        set_correlation_context,
    )

    from src.services import job_worker
    from src.services.job_manager import ClaimedJob

    # Enqueue-time: AuthMiddleware has populated correlation from the inbound
    # X-Correlation-Id/X-Stage. _capture_identity snapshots it into the bundle.
    set_correlation_context("run-xyz", "web_design")
    ident = job_worker._capture_identity()
    assert ident["correlation_id"] == "run-xyz"
    assert ident["stage"] == "web_design"
    clear_identity_context()

    seen: dict[str, str] = {}

    async def fake_dispatch(app, claimed) -> None:
        seen["correlation_id"] = get_correlation_id_from_context()
        seen["stage"] = get_stage_from_context()

    monkeypatch.setattr(job_worker, "_dispatch", fake_dispatch)
    app = SimpleNamespace(state=SimpleNamespace(session_factory=None))
    claimed = ClaimedJob(job_id="j1", tenant_id=TENANT_1, options={"identity": ident})

    await job_worker._run_claimed(app, claimed, lease_seconds=30, worker_id="pod")

    # Correlation was visible to the detached dispatch (so the WM call propagates
    # it) and is cleared afterwards (no leak into the shared event loop).
    assert seen["correlation_id"] == "run-xyz"
    assert seen["stage"] == "web_design"
    assert get_correlation_id_from_context() == ""
    assert get_stage_from_context() == ""


def test_render_options_roundtrip_through_bundle() -> None:
    ro = RenderOptions(
        capture_original=False,
        generate_comparison=False,
        viewports=("desktop",),
        full_page=False,
        section_screenshots=True,
    )
    opts = build_redesign_options(
        run_accessibility_check=True,
        auto_fix_accessibility=False,
        accessibility_threshold=80.0,
        accessibility_first_auto_theme=False,
        generate_screenshots=True,
        generate_concept_images=False,
        render_options=ro,
        llm_config=None,
    )
    assert opts["kind"] == "redesign"
    assert opts["run_accessibility_check"] is True
    assert _render_options_from_dict(opts["render_options"]) == ro


async def test_mark_completed_persists_section_screenshot_paths(db_session) -> None:
    # The new column round-trips through the migration-built (create_all) schema:
    # opt-in section stills land as a section_name → path JSONB map on the row.
    jid = await _enqueue(db_session, "https://sections.example")
    sections = {"hero": "jobs/x/sections/00_hero.png", "pricing": "jobs/x/sections/01_pricing.png"}
    await JobManager.mark_completed(
        db_session,
        jid,
        redesigned_html_path="jobs/x/redesign.html",
        section_screenshot_paths=sections,
    )
    await db_session.commit()

    job = await _get(db_session, jid)
    assert job.section_screenshot_paths == sections


async def test_mark_completed_without_sections_leaves_column_null(db_session) -> None:
    jid = await _enqueue(db_session, "https://nosections.example")
    await JobManager.mark_completed(db_session, jid, redesigned_html_path="jobs/y/redesign.html")
    await db_session.commit()

    job = await _get(db_session, jid)
    assert job.section_screenshot_paths is None


# ---------------------------------------------------------------------------
# Ownership fence on the verdict writes
# ---------------------------------------------------------------------------
#
# The claim writes `claimed_by`; every write that ENDS a job must predicate on
# it. Without that agreement a worker whose claim has already moved on can still
# overwrite the row, and two workers render (and charge for) the same job.


async def test_requeue_from_a_worker_that_lost_the_claim_affects_zero_rows(db_session) -> None:
    """The concrete double-render this fence exists to prevent.

    podA requeues on a timeout, podB claims the freed row and starts rendering,
    then podA's OUTER failure seam unwinds with the same exception. Its second
    requeue must touch nothing: parking the row back in `pending` would let podC
    claim it while podB is still mid-render — two paid renders, racing on
    `mark_completed`.
    """
    jid = await _enqueue(db_session, "https://fence-a.example")
    assert (
        await JobManager.claim_next_job(
            db_session, worker_id="podA", lease_seconds=600, max_attempts=5
        )
    ) is not None

    # Inner seam: the owner's requeue is legal and frees the row.
    assert await JobManager.mark_retryable(db_session, jid, "ReadTimeout", worker_id="podA") is True
    await db_session.commit()
    db_session.expunge_all()
    assert (await _get(db_session, jid)).status == "pending"

    # A sibling claims it on its next poll and is now the owner.
    second = await JobManager.claim_next_job(
        db_session, worker_id="podB", lease_seconds=600, max_attempts=5
    )
    assert second is not None and second.job_id == jid

    # Outer seam of the LOSING worker.
    assert (
        await JobManager.mark_retryable(db_session, jid, "ReadTimeout", worker_id="podA") is False
    )
    await db_session.commit()
    db_session.expunge_all()

    job = await _get(db_session, jid)
    assert job.status == "processing"
    assert job.claimed_by == "podB"


async def test_terminal_failure_from_a_worker_that_lost_the_claim_affects_zero_rows(
    db_session,
) -> None:
    """The fall-back path is fenced too, or the fence just moves the bug.

    A fenced-off requeue makes the caller fall back to `mark_failed`. If THAT
    write were unfenced it would stamp `failed` on the job podB is actively
    rendering — the same lost-race stomp, one line lower down.
    """
    jid = await _enqueue(db_session, "https://fence-b.example")
    assert (
        await JobManager.claim_next_job(
            db_session, worker_id="podA", lease_seconds=600, max_attempts=5
        )
    ) is not None
    await JobManager.mark_retryable(db_session, jid, "ReadTimeout", worker_id="podA")
    await db_session.commit()
    assert (
        await JobManager.claim_next_job(
            db_session, worker_id="podB", lease_seconds=600, max_attempts=5
        )
    ) is not None

    assert await JobManager.mark_failed(db_session, jid, "boom", worker_id="podA") is False
    await db_session.commit()
    db_session.expunge_all()

    job = await _get(db_session, jid)
    assert job.status == "processing"
    assert job.error_message != "boom"


async def test_the_owning_worker_can_still_requeue_and_still_fail(db_session) -> None:
    """The fence must not break the honest case it is wrapped around."""
    jid = await _enqueue(db_session, "https://fence-c.example")
    assert (
        await JobManager.claim_next_job(
            db_session, worker_id="podA", lease_seconds=600, max_attempts=5
        )
    ) is not None

    assert await JobManager.mark_retryable(db_session, jid, "ReadTimeout", worker_id="podA") is True
    # Idempotent at the second seam: still podA's row, now `pending` + unclaimed.
    assert await JobManager.mark_retryable(db_session, jid, "ReadTimeout", worker_id="podA") is True
    await db_session.commit()
    db_session.expunge_all()
    assert (await _get(db_session, jid)).status == "pending"

    assert await JobManager.mark_failed(db_session, jid, "permanent", worker_id="podA") is True
    await db_session.commit()
    db_session.expunge_all()
    job = await _get(db_session, jid)
    assert job.status == "failed"
    assert job.error_message == "permanent"


async def test_an_unclaimed_batch_child_is_still_markable(db_session) -> None:
    """`claimed_by IS NULL` keeps rows no worker ever claimed writable.

    A batch child (`job_options` NULL) is never claimed, so an equality-only
    fence would leave it un-failable forever.
    """
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="https://fence-d.example", scope="hero"
    )
    await db_session.commit()

    assert await JobManager.mark_failed(db_session, job.job_id, "child boom", worker_id="podA")
    await db_session.commit()
    db_session.expunge_all()
    assert (await _get(db_session, job.job_id)).status == "failed"


async def test_the_dispatch_runs_with_the_claim_owner_bound(monkeypatch) -> None:
    """The last link in the fence: the seams read the owner off a ContextVar the
    worker binds around the dispatch. Bound but never set = a fence that is
    always open; set but never reset = the NEXT job on this reused loop inherits
    a stale owner and its own verdict writes get fenced off.
    """
    from types import SimpleNamespace

    from src.services import job_worker
    from src.services.job_manager import ClaimedJob, get_claim_owner

    seen: dict[str, str | None] = {}

    async def fake_dispatch(app, claimed) -> None:
        seen["owner"] = get_claim_owner()

    monkeypatch.setattr(job_worker, "_dispatch", fake_dispatch)
    app = SimpleNamespace(state=SimpleNamespace(session_factory=None))
    claimed = ClaimedJob(job_id="j1", tenant_id=TENANT_1, options={})

    await job_worker._run_claimed(app, claimed, lease_seconds=30, worker_id="pod-7")

    assert seen["owner"] == "pod-7"
    assert get_claim_owner() is None


@pytest.mark.parametrize(
    "bad_tenant",
    [
        None,
        "",
        "   ",
        "anonymous",
        "tenant-abc",
        *NIL_TENANT_SPELLINGS,
    ],
)
async def test_worker_refuses_a_job_whose_tenant_is_not_a_real_tenant(
    monkeypatch, bad_tenant
) -> None:
    """A job that cannot name a real tenant is never dispatched.

    Since w2v_common 0.67.0 there is no fallback tenant: an absent, slug or
    nil-UUID tenant RAISES instead of being substituted. This asserts the refusal
    happens BEFORE ``_dispatch`` — i.e. before any render work, any storage write
    and any outbound LLM spend — so the failure is attributed to the job row that
    carries the bad tenant rather than surfacing later from an unattributed
    cost event.

    The nil UUID is parametrised in all four spellings ``uuid.UUID`` accepts,
    because they normalise to one value and a string-matching guard would only
    close one of the four doors.
    """
    from types import SimpleNamespace

    from w2v_common.tenant import TenantError

    from src.services import job_worker
    from src.services.job_manager import ClaimedJob

    dispatched: list[str] = []

    async def fake_dispatch(app, claimed) -> None:
        dispatched.append(claimed.job_id)

    monkeypatch.setattr(job_worker, "_dispatch", fake_dispatch)
    app = SimpleNamespace(state=SimpleNamespace(session_factory=None))
    claimed = ClaimedJob(job_id="j-bad", tenant_id=bad_tenant, options={})

    with pytest.raises(TenantError):
        await job_worker._run_claimed(app, claimed, lease_seconds=30, worker_id="pod")

    assert dispatched == []


async def test_captured_identity_is_json_safe_and_never_invents_a_tenant() -> None:
    """``_capture_identity`` is persisted into the ``job_options`` jsonb column.

    Two things must hold and both were broken by the 0.67.0 bump: the tenant is
    now a ``uuid.UUID``, which ``json.dumps`` cannot serialise, and an UNBOUND
    tenant is ``None``, which must NOT become a placeholder here — the job row's
    own ``tenant_id`` is the authority and inventing one in the snapshot would
    re-open the substitution from the other side.
    """
    import json

    from w2v_common.auth import clear_identity_context, set_identity_context

    from src.services import job_worker

    set_identity_context(TENANT_1, "lic-1", "subj-1")
    try:
        ident = job_worker._capture_identity()
        assert ident["tenant_id"] == TENANT_1
        assert json.dumps(ident)  # jsonb-serialisable, i.e. str not UUID
    finally:
        clear_identity_context()

    unbound = job_worker._capture_identity()
    assert unbound["tenant_id"] == ""
    assert json.dumps(unbound)
