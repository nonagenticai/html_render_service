"""A terminal status is a fence, and the side effects are fenced with it.

DB-backed (real postgres, per the repo's conftest fixtures). Every assertion
here reads back PERSISTED STATE or a recorded delivery — none of them is a call
count, because a call count cannot tell the difference between the defect and
the fix. The four defects these cover all had the same shape: the row was
already ``cancelled``, later code wrote over it, and the writes that reach the
TENANT went along for the ride.

* A cancelled batch reported ``completed`` and dispatched ``batch.completed``.
* A job cancelled during the long a11y-autofix / screenshot / concept-image tail
  was stamped ``completed`` **and billed** (``record_usage(pages_redesigned=1,
  llm_tokens_used=…)``).
* ``mark_failed``'s fence already existed and its result was DISCARDED, so a
  worker that had lost the claim still ran ``record_usage(is_error=True)`` and
  dispatched ``job.failed`` — the tenant got ``job.failed`` immediately followed
  by ``job.completed`` from the worker that actually owned the job.
* A cancelled variant job (zero variant rows, because the fan-out returned on
  its cancellation check) was stamped ``failed`` by the "all N variants failed"
  verdict.

Fixing only the DB write fixes almost nothing: the invoice and the webhook are
the part the tenant experiences. So most tests below assert on the
``redesign_usage`` row and on the dispatcher's recorded events, and treat the
job/batch status as the supporting evidence rather than the point.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, update

from src.database.models import BatchJob, RedesignJob, RedesignUsage
from src.services.batch_manager import BatchManager
from src.services.design_builder.finalize_phases import RedesignFinalizeMixin
from src.services.design_builder.helpers import RedesignHelpersMixin
from src.services.design_builder.lifecycle import RedesignLifecycleMixin
from src.services.design_builder.types import (
    _A11yResult,
    _GenerationResult,
    _ScreenshotsResult,
    _TokenAccumulator,
)
from src.services.job_manager import JobManager, claim_owner
from tests.tenants import TENANT_1

WEBHOOK = "https://tenant.example.com/hook"


# ---------------------------------------------------------------------------
# Helpers — real rows, real reads, a recording webhook sink
# ---------------------------------------------------------------------------


class RecordingWebhook:
    """Stands in for ``WebhookDispatcher`` and REMEMBERS what the tenant was told.

    Deliberately not a ``MagicMock``: the assertions read ``.events``, which is
    the tenant-visible fact ("you were notified your cancelled job completed"),
    not the internal fact ("dispatch was called once").
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    async def dispatch(self, db, **kwargs) -> None:
        self.events.append(kwargs["event_type"])


async def _job(db, *, status: str = "processing", **kw) -> RedesignJob:
    job = await JobManager.create_job(
        db,
        tenant_id=TENANT_1,
        url="https://example.com/pricing",
        webhook_url=WEBHOOK,
        **kw,
    )
    await db.execute(
        update(RedesignJob).where(RedesignJob.job_id == job.job_id).values(status=status)
    )
    await db.commit()
    return job


async def _row(db, job_id: str) -> RedesignJob:
    db.expunge_all()
    row = (
        await db.execute(select(RedesignJob).where(RedesignJob.job_id == job_id))
    ).scalar_one_or_none()
    assert row is not None
    return row


async def _usage(db) -> RedesignUsage | None:
    """The tenant's billing row for today, or None if they were never billed."""
    db.expunge_all()
    return (
        await db.execute(select(RedesignUsage).where(RedesignUsage.tenant_id == TENANT_1))
    ).scalar_one_or_none()


class _Finalizer(RedesignFinalizeMixin, RedesignHelpersMixin):
    """The real phase-11/failure code plus the real ``_dispatch_webhook``.

    Composed exactly as ``RedesignService`` composes them, so the webhook goes
    through the production ``_dispatch_webhook`` (which reads ``job.webhook_url``
    and builds the real payload) rather than a shortcut the test invented.
    """


def _finalize_mixin(webhook: RecordingWebhook) -> _Finalizer:
    """A real finalizer over the REAL ``JobManager``.

    ``_pkg()`` is deliberately not patched: the fence under test lives in
    ``JobManager.mark_completed`` / ``mark_failed`` and the gating lives in the
    finalize mixin, so stubbing either end would test the stub.
    """
    obj = _Finalizer()
    obj._webhook = webhook  # type: ignore[attr-defined]
    obj._html_gen = SimpleNamespace(_model="test-model")  # type: ignore[attr-defined]
    return obj


async def _finalize_completed(mixin, db, job) -> None:
    """Drive phase 11 with a plausible successful render."""
    await mixin._phase_finalize_completed(
        db,
        job,
        job.job_id,
        TENANT_1,
        gen_result=_GenerationResult(
            redesigned_html="<html></html>",
            redesigned_html_path="jobs/x/redesign.html",
            section_html_paths=None,
            component_html_path=None,
            input_tokens=4000,
            output_tokens=8000,
        ),
        redesigned_html_path="jobs/x/redesign.html",
        a11y=_A11yResult(score=88.0, report={"passed": True}, redesigned_html_path=None),
        screenshots=_ScreenshotsResult(
            original_paths=None,
            screenshot_paths={"desktop": "jobs/x/desktop.png"},
            comparison_paths=None,
            diff_paths=None,
            screenshots_count=1,
        ),
        concept_image_assets=None,
        concept_count=0,
        html_validation_report=None,
        tokens=_TokenAccumulator(input=4000, output=8000),
    )


# ---------------------------------------------------------------------------
# 1. The job-level fence itself
# ---------------------------------------------------------------------------


async def test_mark_completed_refuses_to_overwrite_a_cancelled_job(db_session) -> None:
    """Cancel-mid-flight: the row keeps the tenant's verdict, not the worker's."""
    job = await _job(db_session)
    assert await JobManager.cancel_job(db_session, job.job_id, tenant_id=TENANT_1) is True
    await db_session.commit()

    marked = await JobManager.mark_completed(
        db_session, job.job_id, redesigned_html_path="jobs/x/redesign.html"
    )
    await db_session.commit()

    assert marked is False, "a cancelled job must not be completable"
    row = await _row(db_session, job.job_id)
    assert row.status == "cancelled"
    assert row.redesigned_html_path is None, "the cancelled row was written to anyway"


async def test_mark_failed_refuses_to_overwrite_a_cancelled_job(db_session) -> None:
    """Finding 6's primitive: cancelled is already a verdict; ``failed`` is not an upgrade."""
    job = await _job(db_session)
    assert await JobManager.cancel_job(db_session, job.job_id, tenant_id=TENANT_1) is True
    await db_session.commit()

    marked = await JobManager.mark_failed(db_session, job.job_id, "all 2 variants failed")
    await db_session.commit()

    assert marked is False
    row = await _row(db_session, job.job_id)
    assert row.status == "cancelled"
    assert row.error_message is None, "a cancelled job was given a failure reason"


async def test_mark_completed_refuses_a_job_a_sibling_worker_claimed(db_session) -> None:
    """Lost claim, two workers, one row — on the COMPLETION side.

    ``mark_failed`` has had this fence for a while; ``mark_completed`` had none
    at all, which is the more expensive direction: it also bills.
    """
    job = await _job(db_session)
    await db_session.execute(
        update(RedesignJob).where(RedesignJob.job_id == job.job_id).values(claimed_by="podB")
    )
    await db_session.commit()

    marked = await JobManager.mark_completed(
        db_session, job.job_id, redesigned_html_path="jobs/podA/redesign.html", worker_id="podA"
    )
    await db_session.commit()

    assert marked is False
    row = await _row(db_session, job.job_id)
    assert row.status == "processing", "podA completed a job podB is still rendering"
    assert row.redesigned_html_path is None


async def test_the_fence_blocks_only_a_different_terminal_status(db_session) -> None:
    """The invariant is "not by a DIFFERENT one", and the difference matters.

    ``_record_background_failure`` is the OUTER of two failure seams and re-marks
    a row the inner seam already marked ``failed``. That is a legitimate double
    record, not a lost race, and blocking it would make every ordinary failure
    log a "lost the race" warning — training the reader to ignore the warning
    that means something.
    """
    job = await _job(db_session)
    assert await JobManager.mark_failed(db_session, job.job_id, "boom") is True
    await db_session.commit()

    assert await JobManager.mark_failed(db_session, job.job_id, "boom") is True, (
        "the outer failure seam's idempotent re-record must still land"
    )
    assert await JobManager.mark_completed(db_session, job.job_id) is False, (
        "a failed job must not be completable"
    )
    await db_session.commit()
    assert (await _row(db_session, job.job_id)).status == "failed"


# ---------------------------------------------------------------------------
# 2. The SIDE EFFECTS — billing and the webhook, which is the actual harm
# ---------------------------------------------------------------------------


async def test_a_job_cancelled_in_the_tail_is_not_billed_and_not_announced(db_session) -> None:
    """The money test.

    The last ``_check_cancelled`` is in ``service.py``; the a11y auto-fix LLM
    call, the screenshot renders and the concept-image poll all run after it —
    minutes in which ``POST /api/designs/jobs/{id}/cancel`` lands. Phase 11 then
    ran unconditionally: ``mark_completed`` over ``cancelled``,
    ``record_usage(pages_redesigned=1, llm_tokens_used=12000)`` and
    ``job.completed`` at the tenant's webhook.
    """
    job = await _job(db_session)
    assert await JobManager.cancel_job(db_session, job.job_id, tenant_id=TENANT_1) is True
    await db_session.commit()

    webhook = RecordingWebhook()
    await _finalize_completed(_finalize_mixin(webhook), db_session, job)
    await db_session.commit()

    assert await _usage(db_session) is None, "the tenant was billed for a job they cancelled"
    assert webhook.events == [], "the tenant was told a cancelled job completed"
    assert (await _row(db_session, job.job_id)).status == "cancelled"


async def test_a_job_that_really_completes_is_billed_and_announced(db_session) -> None:
    """The control. Without it the test above passes if phase 11 does nothing at all."""
    job = await _job(db_session)

    webhook = RecordingWebhook()
    await _finalize_completed(_finalize_mixin(webhook), db_session, job)
    await db_session.commit()

    usage = await _usage(db_session)
    assert usage is not None and usage.pages_redesigned == 1
    assert usage.llm_tokens_used == 12000
    assert webhook.events == ["job.completed"]
    assert (await _row(db_session, job.job_id)).status == "completed"


async def test_a_lost_claim_failure_neither_counts_an_error_nor_notifies(db_session) -> None:
    """Finding 3, at the seam that matters.

    ``mark_failed`` correctly returned False — "error DROPPED, another worker
    owns this row" — and ``_phase_finalize_failed`` ignored it, running
    ``record_usage(is_error=True)`` and dispatching ``job.failed``. The tenant
    got ``job.failed`` and then ``job.completed`` from the winning worker, and
    ``error_count`` was inflated by a failure that was never recorded anywhere.
    """
    job = await _job(db_session)
    await db_session.execute(
        update(RedesignJob).where(RedesignJob.job_id == job.job_id).values(claimed_by="podB")
    )
    await db_session.commit()

    webhook = RecordingWebhook()
    mixin = _finalize_mixin(webhook)
    with claim_owner("podA"):
        await mixin._phase_finalize_failed(
            db_session, job, job.job_id, TENANT_1, RuntimeError("podA lost the race")
        )
    await db_session.commit()

    assert await _usage(db_session) is None, "error_count inflated by a dropped failure"
    assert webhook.events == [], "the tenant was told a job failed that another worker owns"
    row = await _row(db_session, job.job_id)
    assert row.status == "processing" and row.error_message is None


async def test_a_real_failure_is_still_counted_and_announced(db_session) -> None:
    """The control for the fence above — it must not have silenced real failures."""
    job = await _job(db_session)

    webhook = RecordingWebhook()
    mixin = _finalize_mixin(webhook)
    with claim_owner("podA"):
        await mixin._phase_finalize_failed(
            db_session, job, job.job_id, TENANT_1, RuntimeError("the render really died")
        )
    await db_session.commit()

    usage = await _usage(db_session)
    assert usage is not None and usage.error_count == 1
    assert webhook.events == ["job.failed"]
    assert (await _row(db_session, job.job_id)).status == "failed"


async def test_a_persistence_error_still_announces_the_failure(db_session) -> None:
    """ "Fenced off" and "the write blew up" are NOT the same thing.

    A refused write means someone else owns the outcome, so stay quiet. A write
    that RAISED means the job really did fail and the tenant is still owed the
    news — collapsing the two into one boolean would have silenced the second.
    """
    job = await _job(db_session)
    webhook = RecordingWebhook()
    mixin = _finalize_mixin(webhook)

    import src.services.design_builder as pkg

    real = pkg.JobManager
    try:
        pkg.JobManager = MagicMock(  # type: ignore[misc]
            mark_retryable=AsyncMock(return_value=False),
            mark_failed=AsyncMock(side_effect=OSError("db socket died")),
            record_usage=AsyncMock(),
        )
        await mixin._phase_finalize_failed(
            db_session, job, job.job_id, TENANT_1, RuntimeError("the render really died")
        )
    finally:
        pkg.JobManager = real  # type: ignore[misc]

    assert webhook.events == ["job.failed"], (
        "a failed persistence write must not be mistaken for a lost race"
    )


# ---------------------------------------------------------------------------
# 3. The batch fence — cancel-before-start and cancel-mid-flight
# ---------------------------------------------------------------------------


class _Batcher(RedesignLifecycleMixin, RedesignHelpersMixin):
    """``process_batch`` plus the real ``_dispatch_webhook``, as composed in prod.

    Both mixins, deliberately: with only the lifecycle half the batch-webhook
    call raises ``AttributeError`` and the test goes red on the WRONG thing --
    it would look like proof while proving nothing about the fence.
    """


async def _batch_row(db, batch_id: str) -> BatchJob:
    db.expunge_all()
    row = (
        await db.execute(select(BatchJob).where(BatchJob.batch_id == batch_id))
    ).scalar_one_or_none()
    assert row is not None
    return row


async def test_finalize_batch_refuses_a_cancelled_batch_and_reports_the_truth(
    db_session,
) -> None:
    """It must return the status the batch ACTUALLY has.

    The caller dispatches ``batch.completed`` / ``batch.failed`` off this return
    value, so fixing the row and handing back the derived status would leave the
    webhook lying — the half-fix that fixes nothing.
    """
    batch = await BatchManager.create_batch(db_session, tenant_id=TENANT_1, total_pages=3)
    await db_session.commit()
    assert await BatchManager.cancel_batch(db_session, batch.batch_id, tenant_id=TENANT_1) is True
    await db_session.commit()

    status = await BatchManager.finalize_batch(db_session, batch.batch_id)
    await db_session.commit()

    assert status == "cancelled", f"finalize_batch reported {status!r} for a cancelled batch"
    assert (await _batch_row(db_session, batch.batch_id)).status == "cancelled"


async def test_finalize_batch_will_not_call_a_batch_with_unaccounted_pages_completed(
    db_session,
) -> None:
    """Silence is not success.

    Status was derived from ``failed_pages`` ALONE — it never checked
    ``completed + failed == total`` — so a batch whose pages all returned early
    derived a clean ``completed`` out of zero successes and zero failures.
    """
    batch = await BatchManager.create_batch(db_session, tenant_id=TENANT_1, total_pages=3)
    await BatchManager.increment_completed(db_session, batch.batch_id)
    await db_session.commit()

    status = await BatchManager.finalize_batch(db_session, batch.batch_id)
    await db_session.commit()

    assert status == "completed_with_errors", f"1 of 3 pages done reported as {status!r}"
    row = await _batch_row(db_session, batch.batch_id)
    assert row.error_message is not None and "2 of 3" in row.error_message


@pytest.mark.parametrize("cancel_before_start", [True, False])
async def test_a_cancelled_batch_stays_cancelled_and_fires_no_webhook(
    db_session, db_session_factory, cancel_before_start: bool
) -> None:
    """The whole of finding 1, end to end, through the real ``process_batch``.

    ``cancel_before_start=True``  — pages early-return at the batch check, so
    NEITHER counter moves and the derived status was ``completed`` out of 0/0/3.
    ``cancel_before_start=False`` — pages are in flight and return via
    ``_check_cancelled``; ``process_job`` returns None rather than raising, so
    every one of them was counted ``completed``.

    Both landed in an unfenced ``finalize_batch`` that overwrote ``cancelled``
    and dispatched ``batch.completed`` at the tenant.
    """
    batch = await BatchManager.create_batch(db_session, tenant_id=TENANT_1, total_pages=3)
    for _ in range(3):
        await _job(db_session, status="pending", batch_id=batch.batch_id)
    if cancel_before_start:
        assert (
            await BatchManager.cancel_batch(db_session, batch.batch_id, tenant_id=TENANT_1) is True
        )
    await db_session.commit()

    webhook = RecordingWebhook()
    service = _Batcher()
    service._webhook = webhook  # type: ignore[attr-defined]

    async def _process_job(page_db, job_id, tenant_id, **kw):
        """Stand in for the real render: it RETURNS on cancellation, never raises."""
        if not cancel_before_start:
            async with db_session_factory() as canceller:
                await BatchManager.cancel_batch(canceller, batch.batch_id, tenant_id=TENANT_1)
                await canceller.commit()

    service.process_job = _process_job  # type: ignore[attr-defined,method-assign]

    await service.process_batch(db_session_factory, batch.batch_id, TENANT_1)

    row = await _batch_row(db_session, batch.batch_id)
    assert row.status == "cancelled", f"a cancelled batch reported {row.status!r}"
    assert row.completed_pages == 0, (
        f"{row.completed_pages} pages counted completed for a cancelled batch"
    )
    assert webhook.events == [], f"the tenant was told {webhook.events} about a cancelled batch"
