"""Abandoned batches are reaped; re-claimed jobs do not leave immortal variants.

DB-backed (real postgres, per the repo's conftest fixtures) — every assertion
below reads the ROW BACK. None of them can be satisfied by a mock's call count:
the defect was that no code path ever wrote these rows again, so only the
persisted state proves it is fixed.

The defect these cover:

``batch_jobs`` rows were immortal. The table has no lease columns and is never
claimed — a batch is driven by an in-process ``asyncio.create_task`` — so a pod
restart strands it in ``pending``/``processing`` with nothing to advance it. The
lease reclaim (``0002_job_lease``) is ``redesign_jobs``-only, ``CleanupService``
has no caller anywhere in the repo, and ``RetentionSettings.enabled`` defaults to
``False``. The batch status endpoint was wedged forever for that client.

One table down, the same shape: ``redesign_variants.status`` starts ``pending``
and ``create_variant`` only ever INSERTs, so the rows an interrupted attempt left
behind were never written again by anyone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from src.database.models import BatchJob, RedesignJob, RedesignVariant
from src.services.batch_manager import BatchManager
from src.services.job_manager import JobManager
from tests.tenants import TENANT_1, TENANT_2

STALE_AFTER = 3600


async def _make_batch(
    db,
    *,
    status: str,
    age_seconds: int,
    tenant_id: str = TENANT_1,
    started: bool = True,
    total_pages: int = 1,
    progress_age_seconds: int | None = None,
) -> str:
    """Create a batch and back-date it so the reaper's window sees it as stale.

    ``progress_age_seconds`` back-dates ``last_progress_at`` independently, which
    is how a LONG-RUNNING BUT HEALTHY batch is spelled: old, still working.
    """
    batch = await BatchManager.create_batch(db, tenant_id=tenant_id, total_pages=total_pages)
    ts = datetime.now(UTC) - timedelta(seconds=age_seconds)
    progress = (
        None
        if progress_age_seconds is None
        else datetime.now(UTC) - timedelta(seconds=progress_age_seconds)
    )
    await db.execute(
        update(BatchJob)
        .where(BatchJob.batch_id == batch.batch_id)
        .values(
            status=status,
            created_at=ts,
            started_at=ts if started else None,
            last_progress_at=progress,
        )
    )
    await db.commit()
    return batch.batch_id


async def _batch(db, batch_id: str) -> BatchJob:
    db.expunge_all()
    row = (
        await db.execute(select(BatchJob).where(BatchJob.batch_id == batch_id))
    ).scalar_one_or_none()
    assert row is not None
    return row


async def test_abandoned_processing_batch_reaches_a_terminal_state_with_a_reason(
    db_session,
) -> None:
    """THE defect. A stale ``processing`` batch must not stay ``processing``."""
    batch_id = await _make_batch(db_session, status="processing", age_seconds=STALE_AFTER * 3)

    reaped = await BatchManager.reap_stale_batches(db_session, STALE_AFTER)
    await db_session.commit()

    assert reaped == 1
    row = await _batch(db_session, batch_id)
    assert row.status == "failed", "abandoned batch is still non-terminal — endpoint wedged"
    assert row.completed_at is not None
    # A terminal state with no reason is only half an answer: the client polling
    # the status endpoint has to be able to see WHY.
    assert row.error_message is not None
    assert "abandoned" in row.error_message


async def test_abandoned_pending_batch_is_reaped_too(db_session) -> None:
    """A batch whose task died before ``mark_processing`` is just as dead.

    Unlike a queue job — where an unstarted ``pending`` row is merely backlogged
    and must NOT be aged out — nobody will ever pick this one up.
    """
    batch_id = await _make_batch(
        db_session, status="pending", age_seconds=STALE_AFTER * 3, started=False
    )

    assert await BatchManager.reap_stale_batches(db_session, STALE_AFTER) == 1
    await db_session.commit()
    assert (await _batch(db_session, batch_id)).status == "failed"


async def test_reaper_leaves_young_and_terminal_batches_alone(db_session) -> None:
    """The ceiling must not fail live work, and must not rewrite finished work."""
    young = await _make_batch(db_session, status="processing", age_seconds=5)
    done = await _make_batch(db_session, status="completed", age_seconds=STALE_AFTER * 3)

    assert await BatchManager.reap_stale_batches(db_session, STALE_AFTER) == 0
    await db_session.commit()

    assert (await _batch(db_session, young)).status == "processing"
    assert (await _batch(db_session, done)).status == "completed"


async def test_reaper_is_cross_tenant(db_session) -> None:
    """The sweep binds no tenant, so it must reach every tenant's stale rows."""
    a = await _make_batch(db_session, status="processing", age_seconds=STALE_AFTER * 3)
    b = await _make_batch(
        db_session, status="processing", age_seconds=STALE_AFTER * 3, tenant_id=TENANT_2
    )

    assert await BatchManager.reap_stale_batches(db_session, STALE_AFTER) == 2
    await db_session.commit()
    assert (await _batch(db_session, a)).status == "failed"
    assert (await _batch(db_session, b)).status == "failed"


async def test_reaping_a_batch_also_ends_its_stranded_children_but_not_queue_owned_ones(
    db_session,
) -> None:
    """Child rows die with the batch — except the ones the worker pool owns.

    A batch child (``job_options IS NULL``) is orchestrated by the same dead
    task, so it is stranded too. A queue-owned row belongs to the claim queue and
    its own lease/poison path must stay the only thing that ends it.
    """
    batch_id = await _make_batch(db_session, status="processing", age_seconds=STALE_AFTER * 3)
    child = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="https://child.example", batch_id=batch_id
    )
    queue_owned = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="https://queued.example", batch_id=batch_id
    )
    await JobManager.set_job_options(db_session, queue_owned.job_id, {"kind": "redesign"})
    variant = await JobManager.create_variant(db_session, job_id=child.job_id, style="modern")
    await db_session.commit()

    await BatchManager.reap_stale_batches(db_session, STALE_AFTER)
    await db_session.commit()
    db_session.expunge_all()

    child_row = await JobManager.get_job(db_session, child.job_id, tenant_id=None)
    assert child_row is not None
    assert child_row.status == "failed"
    assert child_row.error_message is not None

    queued_row = await JobManager.get_job(db_session, queue_owned.job_id, tenant_id=None)
    assert queued_row is not None
    assert queued_row.status == "pending", "the worker pool's row was reaped out from under it"

    variant_row = (
        await db_session.execute(
            select(RedesignVariant).where(RedesignVariant.variant_id == variant.variant_id)
        )
    ).scalar_one()
    assert variant_row.status == "failed"
    assert variant_row.error_message is not None


async def test_mark_batch_failed_persists_its_reason(db_session) -> None:
    """It accepted an ``error_message`` and dropped it on the floor."""
    batch = await BatchManager.create_batch(db_session, tenant_id=TENANT_1, total_pages=1)
    await BatchManager.mark_batch_failed(db_session, batch.batch_id, "upstream exploded")
    await db_session.commit()

    row = await _batch(db_session, batch.batch_id)
    assert row.status == "failed"
    assert row.error_message == "upstream exploded"


async def test_reclaim_retires_the_previous_attempts_pending_variants(db_session) -> None:
    """The second finding: a variant left ``pending`` by an interrupted attempt.

    The parent reclaim re-runs the job, and the re-run INSERTs fresh variant rows
    — it never touches the old ones. Nothing else does either, so the variant
    list showed a permanent ``pending``.
    """
    job = await JobManager.create_job(db_session, tenant_id=TENANT_1, url="https://v.example")
    await JobManager.set_job_options(db_session, job.job_id, {"kind": "redesign"})
    await db_session.commit()

    claimed = await JobManager.claim_next_job(
        db_session, worker_id="pod-a", lease_seconds=180, max_attempts=3
    )
    assert claimed is not None

    # The attempt gets as far as creating its variants, then its pod dies: the
    # lease lapses with the variants still ``pending``.
    stale_variant = await JobManager.create_variant(db_session, job_id=job.job_id, style="modern")
    done_variant = await JobManager.create_variant(db_session, job_id=job.job_id, style="bold")
    await JobManager.mark_variant_completed(db_session, done_variant.variant_id)
    await db_session.execute(
        update(RedesignJob)
        .where(RedesignJob.job_id == job.job_id)
        .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
    )
    await db_session.commit()

    reclaimed = await JobManager.claim_next_job(
        db_session, worker_id="pod-b", lease_seconds=180, max_attempts=3
    )
    assert reclaimed is not None and reclaimed.job_id == job.job_id

    db_session.expunge_all()
    rows = {
        v.variant_id: v
        for v in (
            await db_session.execute(
                select(RedesignVariant).where(RedesignVariant.job_id == job.job_id)
            )
        )
        .scalars()
        .all()
    }
    assert rows[stale_variant.variant_id].status == "failed", (
        "the interrupted attempt's variant is still 'pending' — nothing ever writes it again"
    )
    assert rows[stale_variant.variant_id].error_message is not None
    # A finished variant is a result, not an orphan: never rewrite it.
    assert rows[done_variant.variant_id].status == "completed"


# ---------------------------------------------------------------------------
# The window is NO-PROGRESS, not wall-clock — a healthy long batch is not reapable
# ---------------------------------------------------------------------------


async def test_a_long_running_batch_that_is_still_making_progress_is_not_reaped(
    db_session,
) -> None:
    """The false positive a wall-clock ceiling could not avoid.

    ``BatchRedesignRequest.pages`` allows 20 pages and ``batch_max_concurrency``
    is 3, so a legitimate batch runs 7 sequential waves — and an in-process batch
    child has NO wall-clock cap (``max_job_lifetime_seconds`` bounds only the
    queue path), so there is no per-page number to multiply into an honest
    ceiling. This batch has been running for 10x the window and is perfectly
    healthy: it completed a page a minute ago.
    """
    batch_id = await _make_batch(
        db_session,
        status="processing",
        age_seconds=STALE_AFTER * 10,
        total_pages=20,
        progress_age_seconds=60,
    )

    reaped = await BatchManager.reap_stale_batches(db_session, STALE_AFTER)
    await db_session.commit()
    assert (await _batch(db_session, batch_id)).status == "processing", (
        "a healthy batch was reaped for being OLD rather than for being STUCK"
    )
    assert reaped == 0


async def test_a_page_outcome_moves_the_progress_clock(db_session) -> None:
    """The counters ARE the progress signal — assert they move the clock.

    Both arms: a completed page and a FAILED page. A batch failing its way
    through 20 pages is still being worked and must not be reaped.
    """
    for bump, label in (
        (BatchManager.increment_completed, "completed"),
        (BatchManager.increment_failed, "failed"),
    ):
        batch_id = await _make_batch(
            db_session,
            status="processing",
            age_seconds=STALE_AFTER * 10,
            progress_age_seconds=STALE_AFTER * 10,
        )
        assert await BatchManager.reap_stale_batches(db_session, STALE_AFTER) == 1, label
        await db_session.commit()

        fresh = await _make_batch(
            db_session,
            status="processing",
            age_seconds=STALE_AFTER * 10,
            progress_age_seconds=STALE_AFTER * 10,
        )
        await bump(db_session, fresh)
        await db_session.commit()

        assert await BatchManager.reap_stale_batches(db_session, STALE_AFTER) == 0, (
            f"a {label} page did not move the progress clock — the batch is still reapable"
        )
        await db_session.commit()
        assert (await _batch(db_session, fresh)).status == "processing"
        assert (await _batch(db_session, batch_id)).status == "failed"


# ---------------------------------------------------------------------------
# A reaped batch keeps running — the reason must not outlive the verdict
# ---------------------------------------------------------------------------


async def test_finalize_clears_a_reaper_reason_it_overwrites(db_session) -> None:
    """Reaping does NOT stop the work, so the batch can still finish.

    ``process_page`` aborts only on status ``cancelled``. A reaped batch whose
    pages are still in flight lands in ``finalize_batch``, and overwriting
    ``status`` while leaving ``error_message`` set makes the row report a
    permanent contradiction on the field the status endpoint surfaces.
    """
    batch_id = await _make_batch(db_session, status="processing", age_seconds=STALE_AFTER * 3)
    assert await BatchManager.reap_stale_batches(db_session, STALE_AFTER) == 1
    await db_session.commit()
    assert (await _batch(db_session, batch_id)).error_message is not None

    # The work the reaper did not stop now finishes.
    await BatchManager.increment_completed(db_session, batch_id)
    status = await BatchManager.finalize_batch(db_session, batch_id)
    await db_session.commit()

    row = await _batch(db_session, batch_id)
    assert status == "completed"
    assert row.status == "completed"
    assert row.error_message is None, (
        f"status={row.status!r} contradicts error={row.error_message!r} — permanently"
    )


async def test_cancel_clears_a_reaper_reason_it_overwrites(db_session) -> None:
    """A cancel supersedes the reaper's verdict, so it owns the reason too."""
    batch_id = await _make_batch(db_session, status="processing", age_seconds=STALE_AFTER * 3)
    await BatchManager.reap_stale_batches(db_session, STALE_AFTER)
    # Reaped rows are terminal, so make it cancellable again to reach the path.
    await db_session.execute(
        update(BatchJob).where(BatchJob.batch_id == batch_id).values(status="processing")
    )
    await db_session.commit()

    assert await BatchManager.cancel_batch(db_session, batch_id, tenant_id=TENANT_1) is True
    await db_session.commit()

    row = await _batch(db_session, batch_id)
    assert row.status == "cancelled"
    assert row.error_message is None


# ---------------------------------------------------------------------------
# The resolver is SECURITY DEFINER — it cannot take the caller's word
# ---------------------------------------------------------------------------


async def test_a_tiny_window_is_clamped_and_cannot_reap_live_batches(db_session) -> None:
    """The app role can call this resolver with any argument it likes.

    It bypasses RLS and terminates work across every tenant, so ``1`` would fail
    every in-flight batch in the fleet in one statement. The floor lives in the
    function, not only in config.
    """
    live = await _make_batch(
        db_session, status="processing", age_seconds=600, progress_age_seconds=600
    )
    genuinely_stuck = await _make_batch(
        db_session,
        status="processing",
        age_seconds=STALE_AFTER * 5,
        progress_age_seconds=STALE_AFTER * 5,
    )

    reaped = await BatchManager.reap_stale_batches(db_session, 1)
    await db_session.commit()

    assert (await _batch(db_session, live)).status == "processing", (
        "a 10-minute-old live batch was reaped by a 1-second window — floor not enforced"
    )
    assert reaped == 1
    assert (await _batch(db_session, genuinely_stuck)).status == "failed"


async def test_zero_disables_the_reaper(db_session) -> None:
    """0 is off, and must not be clamped up into an aggressive sweep."""
    batch_id = await _make_batch(
        db_session,
        status="processing",
        age_seconds=STALE_AFTER * 10,
        progress_age_seconds=STALE_AFTER * 10,
    )
    assert await BatchManager.reap_stale_batches(db_session, 0) == 0
    await db_session.commit()
    assert (await _batch(db_session, batch_id)).status == "processing"


# ---------------------------------------------------------------------------
# The clock ticks per SECTION, because one page is unbounded
# ---------------------------------------------------------------------------


async def test_a_section_heavy_page_reports_progress_and_is_never_reapable(
    db_session, db_session_factory, monkeypatch
) -> None:
    """The defect the previous round still had: one PAGE is not a bounded unit.

    ``generate_sections`` loops ``detect_sections(html)`` SEQUENTIALLY, one
    ``_call_claude`` per DOM landmark, and ``detect_sections`` is an uncapped
    ``find_all([...])``. At 600s per call, 13 landmarks is 7800s of healthy work
    — so a per-PAGE progress bump left a live page looking dead, whatever window
    was chosen. Bounding the page with a timeout would have fixed the signal by
    killing the work.

    This drives the REAL loop with a stubbed LLM call, back-dating the clock
    between sections to stand in for a 600s call, and asserts the batch is never
    reapable at any point across a span far longer than the window.
    """
    from src.services.batch_progress import batch_progress
    from src.services.html_gen.generator import HtmlGenerator
    from src.services.html_gen.types import LLMResult

    window = 2100  # the resolver's floor; one section gap must stay under it
    # One WM call: 3 attempts x wm.timeout_seconds + backoff, per w2v_common
    # _MAX_RETRIES/_RETRY_DELAY. NOT 600: a read timeout is retried.
    section_call_s = 1803
    # created_at/started_at are ANCIENT: if the fallback arm of the COALESCE were
    # doing the work this batch would be reaped instantly, so anything that keeps
    # it alive below is the progress clock and nothing else.
    batch_id = await _make_batch(
        db_session,
        status="processing",
        age_seconds=window * 10,
        progress_age_seconds=0,
    )

    # 13 landmarks — the verifier's own counter-example, and 13 * 600 = 7800s.
    html = "<html><body>" + "".join(f"<section><h2>S{i}</h2>p</section>" for i in range(13))
    html += "</body></html>"
    from src.services.context_resolver import detect_sections

    sections = detect_sections(html)
    assert len(sections) >= 10, f"fixture must be section-heavy, got {len(sections)}"

    async def _report() -> None:
        async with db_session_factory() as pdb, pdb.begin():
            await BatchManager.touch_progress(pdb, batch_id)

    generator = HtmlGenerator.__new__(HtmlGenerator)
    generator._generation_temperature = 0.7
    generator._template_loader = None
    generator._max_tokens = 8000
    generator._output_ceiling = 12069
    generator._model = "test"

    calls = {"n": 0}

    async def _fake_call(prompt, **kwargs):
        calls["n"] += 1
        # Stand in for one 600s WM call. The ageing is RELATIVE — it pushes the
        # existing clock further into the past — because that is what real time
        # does. An absolute back-date would silently refresh the clock every
        # iteration and the test could never fail, however broken the code.
        async with db_session_factory() as adb, adb.begin():
            await adb.execute(
                update(BatchJob)
                .where(BatchJob.batch_id == batch_id)
                .values(
                    last_progress_at=BatchJob.last_progress_at - timedelta(seconds=section_call_s)
                )
            )
        # A sweep runs mid-render, exactly as worker 0 does every 5 minutes. It
        # lands BEFORE this section's report, i.e. at the worst moment.
        async with db_session_factory() as sdb, sdb.begin():
            await BatchManager.reap_stale_batches(sdb, window)
        return LLMResult(
            text="<html><section class='s'><style>.s{color:var(--color-ink)}</style>"
            "<h2>x</h2><p>y</p></section></html>"
        )

    # monkeypatch, not bare assignment: `# type: ignore[method-assign]` is a MYPY
    # code and `ty` -- the gate this repo actually runs -- does not map it, so the
    # suppression was inert and the assignment failed CI as invalid-assignment.
    monkeypatch.setattr(generator, "_call_claude", _fake_call)
    monkeypatch.setattr(generator, "_build_section_prompt", lambda **kw: "prompt")

    with batch_progress(_report):
        await generator.generate_sections(url="https://x.example", html=html, sections=sections)

    assert calls["n"] == len(sections)
    # Sanity on the fixture itself: without a per-section signal the accumulated
    # gap MUST exceed the window, or this test proves nothing.
    assert len(sections) * section_call_s > window, "fixture cannot exercise the defect"
    # ...and one gap alone must NOT, or the test would pass for the wrong reason
    # (a window so small that even a working clock cannot keep up).
    assert section_call_s < window, "one section gap already exceeds the window"

    row = await _batch(db_session, batch_id)
    assert row.status == "processing", (
        f"a live {len(sections)}-section page was reaped mid-render after "
        f"{calls['n']} sections of real progress"
    )


async def test_touch_progress_moves_only_the_clock_and_not_a_terminal_batch(
    db_session,
) -> None:
    """It is a liveness signal: no counter moves, and the dead stay dead."""
    batch_id = await _make_batch(
        db_session, status="processing", age_seconds=600, progress_age_seconds=600
    )
    await BatchManager.touch_progress(db_session, batch_id)
    await db_session.commit()
    row = await _batch(db_session, batch_id)
    assert row.completed_pages == 0 and row.failed_pages == 0
    assert row.last_progress_at is not None

    # A late report from a page still in flight must not un-finalize the batch,
    # nor push the progress clock of a row that already reached a verdict.
    await BatchManager.finalize_batch(db_session, batch_id)
    await db_session.commit()
    finalized = await _batch(db_session, batch_id)
    before = finalized.last_progress_at

    await BatchManager.touch_progress(db_session, batch_id)
    await db_session.commit()
    after = await _batch(db_session, batch_id)
    assert after.status == finalized.status
    assert after.last_progress_at == before


async def test_report_progress_is_inert_outside_a_batch(db_session) -> None:
    """The queue path shares ``generate_sections``; it must not need a reporter."""
    from src.services.batch_progress import (
        progress_report_failures,
        report_progress,
        reset_progress_report_failures,
    )

    reset_progress_report_failures()
    assert await report_progress() is None  # unbound: no-op, must not raise
    # The load-bearing half. Returning None is also what a reporter that RAISED
    # and got swallowed looks like, so the return value alone cannot tell the two
    # apart -- and "silently failed" is the exact mode this module warns about.
    # Zero failures is what distinguishes "never invoked" from "invoked and lost".
    assert progress_report_failures() == (0, 0)


async def test_process_batch_binds_the_progress_reporter_for_each_page(
    db_session, db_session_factory
) -> None:
    """The production wiring, not just the mechanism.

    The test above binds the reporter itself, so it would still pass if
    ``process_page`` never bound one — i.e. if the whole signal were dead in
    production. This drives the REAL ``process_batch`` and asserts that a
    ``report_progress()`` issued from inside ``process_job`` (where
    ``generate_sections`` lives) reaches the batch row.
    """
    from src.services.batch_progress import report_progress
    from src.services.design_builder.lifecycle import RedesignLifecycleMixin

    batch_id = await _make_batch(
        db_session, status="pending", age_seconds=0, progress_age_seconds=0
    )
    job = await JobManager.create_job(
        db_session, tenant_id=TENANT_1, url="https://p.example", batch_id=batch_id
    )
    await db_session.commit()

    seen: dict[str, object] = {}

    async def _fake_process_job(page_db, job_id, tenant_id, **kwargs):
        # The ageing has to happen INSIDE the page, after ``mark_processing`` has
        # seeded the clock — otherwise ``mark_processing`` refreshes it and the
        # assertion below passes whether or not a reporter was ever bound.
        async with db_session_factory() as adb, adb.begin():
            await adb.execute(
                update(BatchJob)
                .where(BatchJob.batch_id == batch_id)
                .values(last_progress_at=datetime.now(UTC) - timedelta(seconds=9000))
            )
        # Stands in for generate_sections announcing a finished section.
        await report_progress()
        async with db_session_factory() as rdb:
            row = (
                await rdb.execute(select(BatchJob).where(BatchJob.batch_id == batch_id))
            ).scalar_one()
            seen["progress_during_page"] = row.last_progress_at

    service = RedesignLifecycleMixin.__new__(RedesignLifecycleMixin)
    service.process_job = _fake_process_job  # type: ignore[method-assign]
    service._dispatch_webhook = _noop_webhook  # type: ignore[method-assign]

    await service.process_batch(db_session_factory, batch_id, TENANT_1)

    during = seen.get("progress_during_page")
    assert during is not None, "process_job never ran"
    # `seen` is dict[str, object], so `.get` narrows to object, not datetime.
    assert isinstance(during, datetime)
    age = (datetime.now(UTC) - during).total_seconds()
    assert age < 60, (
        "the progress clock was NOT bumped from inside the page — process_page "
        f"did not bind the reporter (clock is {age:.0f}s old, was 9000s)"
    )
    assert job.job_id


async def _noop_webhook(*args, **kwargs) -> None:
    return None


# ---------------------------------------------------------------------------
# A failing reporter is countable, not just loggable
# ---------------------------------------------------------------------------


async def test_a_failing_reporter_never_breaks_the_render_but_is_counted() -> None:
    """``report_progress`` swallows — so the swallow itself needs a witness.

    A reporter that fails every time is indistinguishable from having no clock
    at all: every bump silently does nothing and the batch is reaped WHILE
    ALIVE. Swallowing stays correct (a liveness signal must not take down a
    render that is succeeding), but it has to leave a countable trace.
    """
    from src.services.batch_progress import (
        CONSECUTIVE_FAILURE_ALERT,
        batch_progress,
        progress_report_failures,
        report_progress,
        reset_progress_report_failures,
    )

    reset_progress_report_failures()
    try:

        async def _broken() -> None:
            raise RuntimeError("progress DB unreachable")

        with batch_progress(_broken):
            for _ in range(CONSECUTIVE_FAILURE_ALERT):
                # Must NOT raise: the render is fine, only the signal is lost.
                await report_progress()

        total, consecutive = progress_report_failures()
        assert total == CONSECUTIVE_FAILURE_ALERT, (
            "a failing progress report left no countable trace — it is "
            "indistinguishable from having no clock at all"
        )
        assert consecutive == CONSECUTIVE_FAILURE_ALERT

        # And /health must say so, rather than reporting a healthy pod that is
        # quietly about to have its live batches reaped.
        from src.api.health import _batch_progress_component

        message, level = _batch_progress_component()
        assert level == "degraded", f"health hid a dead progress clock: {message!r}"
        assert "reaped" in message

        # A single success clears the streak: this is a liveness signal, not a
        # permanent verdict, and a transient blip must not latch degraded.
        async def _ok() -> None:
            return None

        with batch_progress(_ok):
            await report_progress()
        total_after, consecutive_after = progress_report_failures()
        assert consecutive_after == 0
        assert total_after == CONSECUTIVE_FAILURE_ALERT, "the total must not be forgotten"
        assert _batch_progress_component()[1] == "healthy"
    finally:
        reset_progress_report_failures()


#: The longest legitimate gap between two progress signals, derived — the gap
#: BEFORE the first section, which is longer than any inter-section gap:
#:   job.started webhook (<=~60s)
#: + page fetch (browser lease + 30s goto, `screenshot_timeout_ms`)
#: + first WM call: 3 attempts x 600s (`wm.timeout_seconds`) + 1.0s + 2.0s
#:   backoff = 1803s  (w2v_common `_MAX_RETRIES`/`_RETRY_DELAY`; a read timeout
#:   IS a TransportError and IS replayed)
#: ~= 1893s. NOT 1202s — that figure came from `wm.max_retries`/`wm.retry_delay`,
#: two config fields nothing ever read, now deleted.
WORST_CASE_PROGRESS_GAP = 1893


async def test_the_resolver_floor_covers_the_worst_case_progress_gap(db_session) -> None:
    """The floor must sit ABOVE the single gap it exists to bound.

    A previous revision put it at 1800 against a mis-derived 1202s worst case —
    i.e. BELOW the real 1893s gap. That is harmless at the shipped default and
    lethal the moment an operator lowers REDESIGN_BATCH_NO_PROGRESS_SECONDS: the
    number the code calls a safety bound would itself reap live batches.

    This batch is mid-first-WM-call: it reported progress 1833s ago, which is
    inside one legitimate gap. It must survive a sweep whose caller asked for a
    window below the floor.
    """
    healthy = await _make_batch(
        db_session,
        status="processing",
        age_seconds=STALE_AFTER * 10,
        progress_age_seconds=WORST_CASE_PROGRESS_GAP - 60,
    )

    reaped = await BatchManager.reap_stale_batches(db_session, 1800)
    await db_session.commit()

    assert (await _batch(db_session, healthy)).status == "processing", (
        f"the floor is below the worst-case progress gap ({WORST_CASE_PROGRESS_GAP}s): "
        "a batch waiting on its first WM call was reaped alive"
    )
    assert reaped == 0


def test_wm_settings_carries_no_dead_retry_knobs() -> None:
    """Retries belong to w2v_common, and a local copy is worse than none.

    ``WMSettings.max_retries=1`` / ``retry_delay=2.0`` sat in this repo's config
    and were never read: ``llm_client.get_wm_client`` builds
    ``WMClient(timeout=...)`` and passes nothing else. They read as authoritative
    and were not, which is exactly how this branch first derived a 1202s
    worst-case gap instead of the real 1803s. Deleted; this keeps them deleted.
    """
    from src.config import WorkerManagementSettings

    fields = set(WorkerManagementSettings.model_fields)
    assert not fields & {"max_retries", "retry_delay"}, (
        "a retry knob is back on WMSettings but nothing passes it to WMClient — "
        "either wire it to the client or delete it"
    )
